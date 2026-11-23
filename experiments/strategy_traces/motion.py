"""The positions of the bodies of a trace over the episode (``motion.npz``).

A trace has the state of each env step, as the replay of its simulator needs it. This
module turns the states into positions: one array ``[T + 1, 2 or 3]`` for each
**channel**: ``base`` (the robot base), ``ee`` (the tool of the robot) and
``obj:<name>`` (each movable object). A tool that compares the motion of two episodes
reads this file and needs no simulator.

MuJoCo and PyBullet give the positions by the forward kinematics of the scene. The 2D
backends have them in the trace. No policy runs.

python -m experiments.traces.motion <trace folder> ...
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CACHE = "motion.npz"


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
    """Pymunk: the stored pose ``x, y, theta`` of every entity."""
    poses = np.asarray(data["qpos"]).reshape(len(data["qpos"]), -1, 3)
    out = {"ee": np.asarray(data["ee"])[:, :2]}
    for name, entity in _tracked(meta).items():
        out[name] = poses[:, entity, :2]
    return out


def _from_mujoco(directories: list[Path]) -> dict[Path, dict[str, np.ndarray]]:
    """MuJoCo: forward kinematics of every recorded state, one scene for all traces."""
    # pylint: disable=import-outside-toplevel
    from experiments.traces.entities import robot_env
    from experiments.traces.replay import open_trace_env, set_qpos

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

    from experiments.traces.replay_bullet import load_states, open_scene

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


def main() -> None:
    """Write ``motion.npz`` for each trace folder of the command line."""
    ensure([Path(arg).resolve() for arg in sys.argv[1:]])


if __name__ == "__main__":
    main()
