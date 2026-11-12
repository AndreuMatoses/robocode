"""Task-space motion of a trace, and distances between the motions of two traces.

The contact analysis says *what touches what*. This module says *where things go*.
A trace stores the full state of every env step. From it we take the position of a
few bodies, the **channels** of the motion:

* ``ee``: the end effector (stored in the trace);
* ``base``: the robot base;
* ``obj:<name>``: every movable object.

Positions only; orientations are not compared. Two traces are compared on the
channels that both have, so they must come from the same start scene. ``BODIES``
selects a part of the channels (the robot only, the objects only, ...).

The cost between a state of A and a state of B is the mean distance between the same
body in the two states, in metres: "each tracked body is 4 cm from its place in the
other run". The distances differ in what they do with time. Each one is a standard
measure of the literature on trajectory similarity:

* ``locked``: lock-step distance. The states at the same time are compared; a run
  that ended holds its last state. **The speed counts.**
* ``warp``: dynamic time warping (Sakoe and Chiba 1978). A state can match several
  states of the other run, the order is kept; the mean cost of the best match.
  **The path and its order count, the speed does not.**
* ``frechet``: discrete Frechet distance (Eiter and Mannila 1994). The same match,
  but the *largest* cost counts: the worst moment of the best match. A true metric
  on paths; one large detour decides it.
* ``arc``: both paths are sampled at equal steps of path length and compared point
  by point. No match is searched. **The shape of the path counts; pauses do not.**
* ``places``: average Hausdorff (Chamfer) distance. Every state is compared with
  the nearest state of the other run. **Only the places count, not their order.**
* ``phase``: dynamic time warping on position and velocity together. The cost is
  ``|dx| + PHASE_SECONDS * |dv|``: the difference of the positions plus the
  difference of the velocities times a time constant. **The path and the speed on
  the path count, the start time of a motion does not.**
* ``events`` (``events_distance``): the episode is cut at events, here the contact
  modes, and the segments that an alignment of the events links are compared. Each
  link has the same weight. **A short phase counts as much as a long one.** It
  needs the contact view, so ``views.py`` computes it.

``scale`` of a group of traces is the diagonal of the box around all their end
effector positions; a distance divided by it is a share of the workspace.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METRICS = {
    "locked": {
        "name": "lock-step distance",
        "source": "the baseline of the literature on trajectory similarity (Euclidean distance at equal times)",
        "text": "the mean distance between the same body in the two runs at the same "
        "time (a run that ended holds its last state)",
        "sees": "the path and the speed",
        "ignores": "nothing",
    },
    "warp": {
        "name": "dynamic time warping",
        "source": "Sakoe and Chiba 1978",
        "text": "the mean distance between the same body after the best "
        "order-keeping match of the states (a state can match several states)",
        "sees": "the path and its order",
        "ignores": "the speed, pauses",
    },
    "frechet": {
        "name": "Frechet distance",
        "source": "Eiter and Mannila 1994 (discrete form); Alt and Godau 1995",
        "text": "the largest distance in the best order-keeping match of the states "
        "(the worst moment of the best match)",
        "sees": "the largest detour",
        "ignores": "the speed, how long the paths differ",
    },
    "arc": {
        "name": "arc-length distance",
        "source": "arc-length parametrisation, standard in shape analysis",
        "text": "the mean distance between the same body when both paths are sampled "
        "at equal steps of path length (no match is searched)",
        "sees": "the shape of the path",
        "ignores": "the speed, pauses",
    },
    "places": {
        "name": "average Hausdorff distance",
        "source": "Hausdorff distance in its mean form (Chamfer distance)",
        "text": "the mean distance from a state to the nearest state of the other run",
        "sees": "the places",
        "ignores": "the order, the speed",
    },
    "phase": {
        "name": "position and velocity (warped)",
        "source": "a first-order Sobolev cost; near to derivative dynamic time warping (Keogh and Pazzani 2001)",
        "text": "dynamic time warping where the cost is the distance of the positions "
        "plus 0.5 s times the difference of the velocities",
        "sees": "the path and the speed on the path",
        "ignores": "when a motion starts",
    },
}
# Other measures of the literature that this module does not have (the viewer lists
# them, so that the choice of the six distances above is visible).
OPTIONS = [
    {
        "name": "elastic shape distance (square-root velocity function)",
        "source": "Srivastava, Klassen, Joshi and Jermyn 2011",
        "idea": "The path is written as its velocity divided by the root of its "
        "speed. The distance between two such functions, with the best "
        "re-timing, is a true metric on paths that does not depend on the timing.",
        "note": "One term for position and velocity, and no weight to select. "
        "The expected result is near to the arc-length distance.",
    },
    {
        "name": "homotopy class of the path",
        "source": "Bhattacharya, Likhachev and Kumar 2012 (H-signature); winding "
        "numbers",
        "idea": "Two paths are in one class if one can change into the other and "
        "not cross an obstacle. The class is a discrete label of the route.",
        "note": "It fits the navigation scenarios. For manipulation, the sequence "
        "of contact modes is already the discrete description.",
    },
    {
        "name": "persistent homology of the paths",
        "source": "Pokorny, Hawasly and Ramamoorthy 2016",
        "idea": "The classes of a group of paths come from the holes of the free "
        "space at many scales, with no model of the obstacles.",
        "note": "It needs many paths for each scene.",
    },
    {
        "name": "longest common subsequence, edit distance on real sequences",
        "source": "Vlachos, Kollios and Gunopulos 2002; Chen, Ozsu and Oria 2005",
        "idea": "Two states match if they are nearer than a threshold. The "
        "distance counts the states with no partner.",
        "note": "Robust to outliers; it has a threshold to select. The mode edit "
        "distance of the contact view is of this family.",
    },
    {
        "name": "optimal transport between the visited states",
        "source": "Wasserstein distance (see Peyre and Cuturi 2019)",
        "idea": "Each run is the set of its states with no order. The distance is "
        "the least work that moves one set onto the other.",
        "note": "Order-free, as the average Hausdorff distance, but each state "
        "has one partner.",
    },
]
# A distance that needs the contact view for its alignment: ``views.py`` computes it
# with ``events_distance`` and adds it to the other motion distances.
EVENTS = {
    "name": "event-aligned distance",
    "source": "this study: the contact modes give the events",
    "text": "the episode of each run is cut at its contact modes; the elastic "
    "alignment of the contact view links the modes of the two runs; the distance is "
    "the mean distance between the same body over the linked segments, each link "
    "with the same weight",
    "sees": "the motion in each phase, short phases as much as long ones",
    "ignores": "how long a phase is",
}
BODIES = {
    "all": "the end effector, the base and every movable object",
    "robot": "the end effector and the base",
    "ee": "the end effector",
    "base": "the robot base",
    "objects": "the movable objects",
}
CACHE = "motion.npz"
MAX_POINTS = 160  # states of a trace that a distance uses
EVENT_POINTS = 12  # states of a segment that the event-aligned distance compares
PHASE_SECONDS = 0.5  # weight of a velocity difference in "phase": 1 m/s counts as 0.5 m


def _in_bodies(channel: str, bodies: str) -> bool:
    if bodies == "all":
        return True
    if bodies == "robot":
        return channel == "ee" or channel.split(":")[0] == "base"
    if bodies == "base":
        return channel.split(":")[0] == "base"
    if bodies == "objects":
        return channel.startswith("obj:")
    return channel == bodies


@dataclasses.dataclass
class Motion:
    """The positions of the tracked bodies of one episode."""

    channels: dict[str, np.ndarray]  # name -> [T, d] positions in metres
    step_seconds: float

    @property
    def steps(self) -> int:
        """Number of states (env steps + 1)."""
        return len(next(iter(self.channels.values())))

    def select(self, bodies: str) -> "Motion":
        """The channels of a group of bodies (see ``BODIES``)."""
        if bodies not in BODIES:
            raise ValueError(f"unknown bodies {bodies!r}")
        keep = {k: v for k, v in self.channels.items() if _in_bodies(k, bodies)}
        return Motion(keep, self.step_seconds)

    def resampled(self, by: str = "time", count: int = MAX_POINTS) -> "Samples":
        """``count`` states at equal steps of time or of path length ("arc").

        The path length of a step is the mean distance that the bodies move in it.
        """
        names, steps = sorted(self.channels), self.steps
        grid = np.arange(steps)
        at = np.linspace(0, steps - 1, count)
        if by == "arc" and steps > 1:
            moved = sum(
                np.linalg.norm(np.diff(self.channels[k], axis=0), axis=1) for k in names
            ) / len(names)
            length = np.concatenate([[0.0], np.cumsum(moved)])
            if length[-1] > 0:
                at = np.interp(np.linspace(0, length[-1], count), length, grid)

        def take(values: np.ndarray) -> np.ndarray:
            return np.stack(
                [np.interp(at, grid, values[:, d]) for d in range(values.shape[1])], 1
            )

        pos, vel = {}, {}
        for name in names:
            values = self.channels[name]
            speed = (
                np.gradient(values, self.step_seconds, axis=0)
                if steps > 1
                else np.zeros_like(values)
            )
            pos[name], vel[name] = take(values), take(speed)
        return Samples(pos, vel)


@dataclasses.dataclass
class Samples:
    """Positions (m) and velocities (m/s) of the bodies at the sampled states."""

    pos: dict[str, np.ndarray]
    vel: dict[str, np.ndarray]


# ------------------------------------------------------------------ extraction


def _entity_geoms(meta: dict[str, Any]) -> dict[int, list[int]]:
    geoms: dict[int, list[int]] = defaultdict(list)
    for geom in meta["geoms"]:
        geoms[geom["entity"]].append(geom["id"])
    return geoms


def _tracked(meta: dict[str, Any]) -> dict[str, int]:
    """Channel name -> entity id: the robot base and every movable object."""
    out = {}
    for entity in meta["entities"]:
        if entity["name"] == "robot.base":
            out["base"] = entity["id"]
        elif entity["name"].startswith("robot.rover"):
            out["base:" + entity["name"].removeprefix("robot.")] = entity["id"]
        elif entity["kind"] == "object":
            out["obj:" + entity["name"].removeprefix("env.")] = entity["id"]
    return out


def _from_outlines(meta: dict[str, Any], data: Any) -> dict[str, np.ndarray]:
    """Kinematic 2D: the centre of the outline of every tracked entity."""
    centres = np.nanmean(data["verts"], axis=2)  # [T, geom, 2]
    geoms = _entity_geoms(meta)
    out = {"ee": np.asarray(data["ee"])[:, :2]}
    for name, entity in _tracked(meta).items():
        out[name] = np.nanmean(centres[:, geoms[entity]], axis=1)
    return out


def _from_poses(meta: dict[str, Any], data: Any) -> dict[str, np.ndarray]:
    """pymunk: the stored pose ``x, y, theta`` of every entity."""
    poses = np.asarray(data["qpos"]).reshape(len(data["qpos"]), -1, 3)
    out = {"ee": np.asarray(data["ee"])[:, :2]}
    for name, entity in _tracked(meta).items():
        out[name] = poses[:, entity, :2]
    return out


def _from_mujoco(directories: list[Path]) -> dict[Path, dict[str, np.ndarray]]:
    """MuJoCo: forward kinematics of every recorded state, one scene for all traces."""
    # pylint: disable=import-outside-toplevel
    from experiments.contacts.entities import robot_env
    from experiments.contacts.replay import open_trace_env, set_qpos

    out: dict[Path, dict[str, np.ndarray]] = {}
    env, _, first = open_trace_env(directories[0])
    try:
        renv = robot_env(env)
        data = renv.sim.data.mj_data
        for directory in directories:
            meta = json.loads((directory / "meta.json").read_text())
            if meta["signature"] != first["signature"]:
                raise ValueError(f"{directory}: not the scene of {directories[0]}")
            geoms, tracked = _entity_geoms(meta), _tracked(meta)
            with np.load(directory / "trace.npz") as trace:
                qpos, ee = trace["qpos"], trace["ee"]
            rows: dict[str, list[np.ndarray]] = {name: [] for name in tracked}
            for state in qpos:
                set_qpos(renv, state)
                for name, entity in tracked.items():
                    rows[name].append(data.geom_xpos[geoms[entity]].mean(axis=0))
            channels = {name: np.array(values) for name, values in rows.items()}
            channels["ee"] = np.asarray(ee)
            out[directory] = channels
    finally:
        env.close()
    return out


def _from_bullet(directories: list[Path]) -> dict[Path, dict[str, np.ndarray]]:
    """PyBullet: the link positions of every recorded state, one scene for all."""
    # pylint: disable=import-outside-toplevel
    import pybullet as p

    from experiments.contacts.replay_bullet import load_states, open_scene

    out: dict[Path, dict[str, np.ndarray]] = {}
    env, scene, first = open_scene(directories[0])
    cid = scene.client

    def position(body: int, link: int) -> np.ndarray:
        if link < 0:
            return np.array(
                p.getBasePositionAndOrientation(body, physicsClientId=cid)[0]
            )
        return np.array(p.getLinkState(body, link, physicsClientId=cid)[4])

    try:
        for directory in directories:
            meta = json.loads((directory / "meta.json").read_text())
            if meta["signature"] != first["signature"]:
                raise ValueError(f"{directory}: not the scene of {directories[0]}")
            geoms, tracked = _entity_geoms(meta), _tracked(meta)
            rows: dict[str, list[np.ndarray]] = {name: [] for name in tracked}
            for state in load_states(directory):
                scene.restore(state)
                for name, entity in tracked.items():
                    points = [position(*meta["bodies"][g]) for g in geoms[entity]]
                    rows[name].append(np.mean(points, axis=0))
            channels = {name: np.array(values) for name, values in rows.items()}
            if "base" in channels or not channels:  # the rovers have no tool
                with np.load(directory / "trace.npz") as trace:
                    channels["ee"] = np.asarray(trace["ee"])
            out[directory] = channels
    finally:
        env.close()
    return out


def ensure(directories: list[Path]) -> None:
    """Write ``motion.npz`` for every trace that has none (newer than its trace)."""
    todo: dict[tuple[Any, ...], list[Path]] = defaultdict(list)
    for directory in directories:
        cache, trace = directory / CACHE, directory / "trace.npz"
        if cache.exists() and cache.stat().st_mtime >= trace.stat().st_mtime:
            continue
        meta = json.loads((directory / "meta.json").read_text())
        backend = meta.get("backend", "mujoco")
        todo[backend, meta["environment"], meta["signature"]].append(directory)
    for (backend, _, _), group in todo.items():
        if backend == "mujoco":
            found = _from_mujoco(group)
        elif backend == "bullet":
            found = _from_bullet(group)
        elif backend in ("kin2d", "pymunk"):
            read = _from_outlines if backend == "kin2d" else _from_poses
            found = {}
            for directory in group:
                meta = json.loads((directory / "meta.json").read_text())
                with np.load(directory / "trace.npz") as data:
                    found[directory] = read(meta, data)
        else:
            raise NotImplementedError(f"no task-space motion for backend {backend}")
        for directory, channels in found.items():
            np.savez_compressed(directory / CACHE, **channels)


def load(directory: Path) -> Motion:
    """The motion of a trace (``ensure`` first)."""
    meta = json.loads((directory / "meta.json").read_text())
    seconds = meta.get("step_seconds") or 1.0 / meta.get("fps", 10)
    with np.load(directory / CACHE) as data:
        return Motion({name: data[name] for name in data.files}, float(seconds))


# -------------------------------------------------------------------- distances


def cost_matrix(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> np.ndarray:
    """[Ta, Tb] mean distance between the same body in a state of A and one of B."""
    names = sorted(set(a) & set(b))
    if not names:
        raise ValueError("the two traces have no body in common")
    total = np.zeros((len(a[names[0]]), len(b[names[0]])))
    for name in names:
        total += np.linalg.norm(a[name][:, None, :] - b[name][None, :, :], axis=-1)
    return total / len(names)


def locked_distance(a: Motion, b: Motion) -> float:
    """Mean cost at equal times; the shorter run holds its last state."""
    names = sorted(set(a.channels) & set(b.channels))
    steps = max(a.steps, b.steps)
    total = np.zeros(steps)
    for name in names:
        x, y = a.channels[name], b.channels[name]
        x = np.vstack([x, np.repeat(x[-1:], steps - len(x), axis=0)])
        y = np.vstack([y, np.repeat(y[-1:], steps - len(y), axis=0)])
        total += np.linalg.norm(x - y, axis=-1)
    return float(total.mean() / len(names))


def match_distance(cost: np.ndarray, largest: bool = False) -> float:
    """The cheapest order-keeping match of the rows and the columns of ``cost``.

    Every state has a partner, a state can have several, the order is kept (dynamic
    programming). The result is the mean cost of the links (dynamic time warping) or,
    with ``largest``, the largest cost of a link (discrete Frechet distance).
    """
    inf = float("inf")
    cols = cost.shape[1]
    above, above_links = [inf] * (cols + 1), [0] * (cols + 1)
    above[0] = 0.0
    for row in cost.tolist():
        line, links = [inf] * (cols + 1), [0] * (cols + 1)
        for j in range(1, cols + 1):
            best, count = above[j - 1], above_links[j - 1]
            if above[j] < best:
                best, count = above[j], above_links[j]
            if line[j - 1] < best:
                best, count = line[j - 1], links[j - 1]
            value = row[j - 1]
            line[j] = max(best, value) if largest else best + value
            links[j] = count + 1
        above, above_links = line, links
    return float(above[cols] if largest else above[cols] / above_links[cols])


def places_distance(cost: np.ndarray) -> float:
    """Symmetric mean distance to the nearest state of the other run."""
    return float((cost.min(axis=1).mean() + cost.min(axis=0).mean()) / 2.0)


def segment(
    item: Motion, start: float, end: float, count: int
) -> dict[str, np.ndarray]:
    """``count`` states of a motion between two times (seconds), evenly spaced."""
    last = item.steps - 1
    at = np.clip(np.linspace(start, end, count) / item.step_seconds, 0, last)
    grid = np.arange(item.steps)
    return {
        name: np.stack(
            [np.interp(at, grid, values[:, d]) for d in range(values.shape[1])], 1
        )
        for name, values in item.channels.items()
    }


def events_distance(
    a: Motion,
    b: Motion,
    spans_a: list[tuple[float, float]],
    spans_b: list[tuple[float, float]],
    links: list[tuple[int, int]],
    count: int = EVENT_POINTS,
) -> float:
    """Mean distance between the segments of two runs that an event alignment links.

    ``spans_a`` and ``spans_b`` are the time spans (seconds) of the segments of each
    run, for example its contact modes. ``links`` says which segment of A goes with
    which segment of B. Each linked pair of segments is compared at ``count`` states
    at equal shares of the segment time, and each link has the same weight: a short
    event (a throw) counts as much as a long one (a drive).
    """
    names = sorted(set(a.channels) & set(b.channels))
    if not links or not names:
        return locked_distance(a, b)
    total = 0.0
    for i, j in links:
        part_a = segment(a, *spans_a[i], count)
        part_b = segment(b, *spans_b[j], count)
        cost = sum(
            np.linalg.norm(part_a[name] - part_b[name], axis=-1) for name in names
        )
        total += float(np.mean(cost)) / len(names)
    return total / len(links)


def workspace_scale(motions: list[Motion]) -> float:
    """Diagonal of the box around all end-effector positions of a group of traces.

    Robots without a tool (the rovers): the box around all base positions.
    """
    points = np.vstack(
        [
            values
            for m in motions
            for name, values in m.channels.items()
            if name == "ee" or ("ee" not in m.channels and name.startswith("base"))
        ]
    )
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def distance_matrices(directories: list[Path], bodies: str = "all") -> dict[str, Any]:
    """Every motion distance between every two traces, in metres, and the scale."""
    ensure(directories)
    full = [load(d) for d in directories]
    part = [m.select(bodies) for m in full]
    if part and not part[0].channels:
        raise ValueError(f"the traces have no channel of the bodies {bodies!r}")
    by_time = [m.resampled("time") for m in part]
    by_arc = [m.resampled("arc") for m in part]
    count = len(full)
    out = {key: np.zeros((count, count)) for key in METRICS}
    for i in range(count):
        for j in range(i + 1, count):
            cost = cost_matrix(by_time[i].pos, by_time[j].pos)
            speed = cost_matrix(by_time[i].vel, by_time[j].vel)
            along = cost_matrix(by_arc[i].pos, by_arc[j].pos)
            values = {
                "locked": locked_distance(part[i], part[j]),
                "warp": match_distance(cost),
                "frechet": match_distance(cost, largest=True),
                "arc": float(np.diag(along).mean()),
                "places": places_distance(cost),
                "phase": match_distance(cost + PHASE_SECONDS * speed),
            }
            for key, value in values.items():
                out[key][i, j] = out[key][j, i] = value
    return {"metres": out, "scale": workspace_scale(full), "motions": full}
