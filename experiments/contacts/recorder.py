"""Log MuJoCo contacts at every simulation substep of a rollout.

An ``env.step`` of the dynamic3d environments runs 200 ``mj_step`` substeps (0.5 ms
each), and kinder offers no hook between them. ``recording()`` wraps
``MjSim.step`` and ``MujocoEnv.step`` at class level (the pattern of
``render_run.scale_mujoco_cameras``), so a ``ContactRecorder`` sees every substep
without changing the simulation.

What is kept, per **contact key** = (geom A, surface of A, geom B, surface of B):

* ``intervals``: exact on/off times in substep ticks, so impacts shorter than one
  env step are not lost;
* ``steps``: one row per env step the key was active in, with the active fraction,
  mean / peak normal force, mean tangential force, and the force-weighted mean
  contact point on each side, in normalised local coordinates (see ``entities``).

Plus, per env step, the full ``qpos`` (to re-render anything later without running
the policy again), the gripper pinch-site position and the simulation time.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from typing import Any

import mujoco
import numpy as np

from experiments.contacts.entities import EntityMap, build_entity_map, local_points

STEP_COLUMNS = [
    "key",
    "step",
    "fraction",  # share of the step's substeps in which the key was active
    "fn_mean",  # mean total normal force while active [N]
    "fn_max",  # peak total normal force [N]
    "ft_mean",  # mean total tangential force while active [N]
    "ax",
    "ay",
    "az",  # contact point on side A, normalised local coordinates
    "bx",
    "by",
    "bz",  # contact point on side B
]


class BaseRecorder:
    """The part of a recorder that does not depend on the simulator.

    A subclass reads the simulator: ``substep`` appends to ``_buffer`` one tuple of
    arrays per substep (tick, geom A, face A, geom B, face B, normal force,
    tangential force, point on A, point on B), and ``_snapshot`` appends the state
    of one env step to ``_qpos`` (whatever a replay needs), ``_time`` and ``_ee``.
    It sets ``emap`` and ``dt`` before it calls ``BaseRecorder.__init__``.
    """

    backend = "mujoco"  # name of the simulator family, written to ``meta.json``
    emap: EntityMap
    dt: float

    def __init__(self) -> None:
        self.tick = 0
        self._step_start = 0
        self._buffer: list[tuple[np.ndarray, ...]] = []
        self._keys: dict[tuple[int, int, int, int], int] = {}
        self._open: dict[int, list[int]] = {}  # key -> [start tick, last tick]
        self._intervals: list[tuple[int, int, int]] = []
        self._rows: list[np.ndarray] = []
        self._qpos: list[np.ndarray] = []
        self._ee: list[np.ndarray] = []
        self._time: list[float] = []
        self._snapshot()

    def substep(self) -> None:
        """Read the contacts after one simulation substep."""
        raise NotImplementedError

    def _snapshot(self) -> None:
        raise NotImplementedError

    def extra_meta(self) -> dict[str, Any]:
        """What a replay of this simulator needs besides the entity tables."""
        return {}

    # ----------------------------------------------------------- per env step

    def end_step(self) -> None:
        """Aggregate the substeps of the env step that just finished."""
        step, start, ticks = len(self._qpos) - 1, self._step_start, self.tick
        self._step_start = ticks
        self._snapshot()
        if not self._buffer:
            self._close_inactive(ticks)
            return
        tick, g_a, f_a, g_b, f_b, f_n, f_t, p_a, p_b = (
            np.concatenate(col) for col in zip(*self._buffer)
        )
        self._buffer.clear()
        code = (
            (g_a.astype(np.int64) << 40)
            | (f_a.astype(np.int64) << 32)
            | (g_b.astype(np.int64) << 8)
            | f_b.astype(np.int64)
        )
        uniq, first, inverse = np.unique(code, return_index=True, return_inverse=True)
        n_ticks = max(ticks - start, 1)
        for k in range(len(uniq)):
            i = int(first[k])
            key_tuple = (int(g_a[i]), int(f_a[i]), int(g_b[i]), int(f_b[i]))
            key = self._keys.setdefault(key_tuple, len(self._keys))
            mask = inverse == k
            offsets = tick[mask] - start - 1
            per_tick = np.bincount(offsets, weights=f_n[mask], minlength=n_ticks)
            active = np.flatnonzero(np.bincount(offsets, minlength=n_ticks))
            weight = f_n[mask][:, None]
            total = float(weight.sum())
            self._rows.append(
                np.array(
                    [
                        key,
                        step,
                        len(active) / n_ticks,
                        total / len(active),
                        per_tick.max(),
                        float(f_t[mask].sum()) / len(active),
                        *_mean_point(p_a[mask], weight, total),
                        *_mean_point(p_b[mask], weight, total),
                    ],
                    dtype=np.float32,
                )
            )
            self._extend_intervals(key, active + start + 1)
        self._close_inactive(ticks)

    def _extend_intervals(self, key: int, active_ticks: np.ndarray) -> None:
        breaks = np.flatnonzero(np.diff(active_ticks) > 1)
        starts = np.concatenate(([active_ticks[0]], active_ticks[breaks + 1]))
        ends = np.concatenate((active_ticks[breaks], [active_ticks[-1]]))
        for first, last in zip(starts.tolist(), ends.tolist()):
            current = self._open.get(key)
            if current is not None and first == current[1] + 1:
                current[1] = last
                continue
            if current is not None:
                self._intervals.append((key, current[0], current[1] + 1))
            self._open[key] = [first, last]

    def _close_inactive(self, ticks: int) -> None:
        """Close the intervals that were not active in the step's last substep."""
        for key in [k for k, cur in self._open.items() if cur[1] < ticks]:
            start, last = self._open.pop(key)
            self._intervals.append((key, start, last + 1))

    # ------------------------------------------------------------------ result

    def finish(self) -> dict[str, np.ndarray]:
        """Close open intervals and return the episode's arrays (for ``np.savez``)."""
        if self.tick > self._step_start:  # substeps after the last full env step
            self.end_step()
        for key, (start, last) in self._open.items():
            self._intervals.append((key, start, last + 1))
        self._open.clear()
        keys = np.zeros((len(self._keys), 4), dtype=np.int32)
        for key_tuple, key in self._keys.items():
            keys[key] = key_tuple
        intervals = np.array(sorted(self._intervals), dtype=np.int32).reshape(-1, 3)
        return {
            "keys": keys,  # [K, 4] geom A, face A, geom B, face B
            "intervals": intervals,  # [M, 3] key, start tick, end tick (exclusive)
            "steps": (
                np.stack(self._rows)
                if self._rows
                else np.zeros((0, len(STEP_COLUMNS)), dtype=np.float32)
            ),
            "qpos": np.stack(self._qpos),  # [T + 1, nq]
            "ee": np.stack(self._ee) if self._ee else np.zeros((0, 3)),
            "time": np.array(self._time),
            "step_ticks": np.array(
                [round((t - self._time[0]) / self.dt) for t in self._time],
                dtype=np.int64,
            ),
        }


def _mean_point(points: np.ndarray, weight: np.ndarray, total: float) -> np.ndarray:
    """Force-weighted mean point (the plain mean when no force is known)."""
    if total <= 0.0:
        return points.mean(axis=0)
    return (points * weight).sum(axis=0) / total


class ContactRecorder(BaseRecorder):
    """Accumulates the contacts of one episode of one ``MujocoEnv``."""

    def __init__(self, renv: Any, min_force: float = 1e-3) -> None:
        self.sim = renv.sim
        self.model = renv.sim.model.mj_model
        self.data = renv.sim.data.mj_data
        self.emap = build_entity_map(self.model)
        self.min_force = min_force
        self.dt = float(self.model.opt.timestep)
        self._rank = self.emap.rank()
        site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site"
        )
        self._site = site if site >= 0 else None
        super().__init__()

    def substep(self) -> None:
        """Read the contacts after one ``mj_step``."""
        self.tick += 1
        data, n = self.data, self.data.ncon
        if not n:
            return
        force = np.zeros((n, 6))
        for i in range(n):
            mujoco.mj_contactForce(self.model, data, i, force[i])
        keep = force[:, 0] > self.min_force
        if not keep.any():
            return
        geoms = np.array(data.contact.geom[:n])[keep]
        points = np.array(data.contact.pos[:n])[keep]
        # Order each pair: robot before object before fixture before world.
        swap = self._rank[geoms[:, 0]] > self._rank[geoms[:, 1]]
        geoms[swap] = geoms[swap][:, ::-1]
        normals = np.array(data.contact.frame[:n])[keep, :3]
        sides = [
            local_points(
                self.emap, geoms[:, k], points, normals, data.geom_xpos, data.geom_xmat
            )
            for k in (0, 1)
        ]
        self._buffer.append(
            (
                np.full(len(geoms), self.tick, dtype=np.int64),
                geoms[:, 0],
                sides[0][1],
                geoms[:, 1],
                sides[1][1],
                force[keep, 0],
                np.hypot(force[keep, 1], force[keep, 2]),
                sides[0][0],
                sides[1][0],
            )
        )

    def _snapshot(self) -> None:
        self._qpos.append(np.array(self.data.qpos))
        self._time.append(float(self.data.time))
        if self._site is not None:
            self._ee.append(np.array(self.data.site_xpos[self._site]))


@contextlib.contextmanager
def recording() -> Iterator[list[ContactRecorder | None]]:
    """Route every ``MjSim.step`` / ``MujocoEnv.step`` to the active recorder.

    Yields a one-slot list: put a ``ContactRecorder`` in ``slot[0]`` after a reset
    (the sim is rebuilt on every reset) and set it back to None when the episode
    ends. Steps of other sims, and the settle steps of a reset, are ignored.
    """
    # pylint: disable-next=import-outside-toplevel
    from kinder.envs.dynamic3d.mujoco_utils import MjSim, MujocoEnv

    slot: list[ContactRecorder | None] = [None]
    sim_step, env_step = MjSim.step, MujocoEnv.step

    @functools.wraps(sim_step)
    def patched_sim_step(self: Any) -> None:
        sim_step(self)
        recorder = slot[0]
        if recorder is not None and self is recorder.sim:
            recorder.substep()

    @functools.wraps(env_step)
    def patched_env_step(self: Any, action: Any) -> Any:
        result = env_step(self, action)
        recorder = slot[0]
        if recorder is not None and self.sim is recorder.sim:
            recorder.end_step()
        return result

    MjSim.step = patched_sim_step  # type: ignore[method-assign]
    MujocoEnv.step = patched_env_step  # type: ignore[method-assign]
    try:
        yield slot
    finally:
        MjSim.step = sim_step  # type: ignore[method-assign]
        MujocoEnv.step = env_step  # type: ignore[method-assign]
