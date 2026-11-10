"""Load recorded traces and summarise them at the level a person reads.

The recorder keys contacts by (geom, surface) pairs. Here they are grouped into
**entity pairs** (``robot.gripper.left_pad`` and ``env.cube_0``), each keeping its
surface-level detail, with times in seconds since the start of the episode.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.contacts.entities import FACES

# A pair that is on for (almost) the whole episode and involves no robot part is a
# resting contact (a cube on the floor): background, hidden by default.
RESTING_FRACTION = 0.98


def trace_dirs(root: Path) -> list[Path]:
    """Every recorded instance under ``root`` (directories with a ``meta.json``)."""
    return sorted(p.parent for p in root.glob("*/*/seed*/commit*/*/meta.json"))


def trace_id(root: Path, directory: Path) -> str:
    """The id used in URLs: the path relative to the traces root."""
    return directory.relative_to(root).as_posix()


@functools.lru_cache(maxsize=16384)
def load_meta(directory: Path) -> dict[str, Any]:
    """The ``meta.json`` of a trace (cached)."""
    return json.loads((directory / "meta.json").read_text())


def merge_intervals(intervals: np.ndarray) -> list[list[int]]:
    """Union of ``[start, end)`` tick intervals."""
    merged: list[list[int]] = []
    for start, end in sorted(map(tuple, intervals.tolist())):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _category(kind_a: str, kind_b: str) -> str:
    return f"{kind_a}-{kind_b}" if kind_a != kind_b else f"{kind_a}-self"


@functools.lru_cache(maxsize=64)
def summarise(directory: Path) -> dict[str, Any]:
    """Entity-pair view of one trace, JSON-ready."""
    meta = load_meta(directory)
    with np.load(directory / "trace.npz") as data:
        keys, intervals, steps = data["keys"], data["intervals"], data["steps"]
        step_ticks = data["step_ticks"]
    dt, total = meta["dt"], max(int(meta["num_ticks"]), 1)
    geoms = {g["id"]: g for g in meta["geoms"]}
    entities = meta["entities"]
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for key, (g_a, f_a, g_b, f_b) in enumerate(keys.tolist()):
        e_a, e_b = geoms[g_a]["entity"], geoms[g_b]["entity"]
        pair = pairs.setdefault(
            (e_a, e_b), {"a": e_a, "b": e_b, "surfaces": [], "_iv": [], "_rows": []}
        )
        own = intervals[intervals[:, 0] == key][:, 1:]
        rows = steps[steps[:, 0] == key]
        pair["_iv"].append(own)
        pair["_rows"].append(rows)
        pair["surfaces"].append(
            {
                "geom_a": geoms[g_a]["index"],
                "face_a": FACES[f_a],
                "type_a": geoms[g_a]["type"],
                "extent_a": geoms[g_a]["extent"],
                "geom_b": geoms[g_b]["index"],
                "face_b": FACES[f_b],
                "type_b": geoms[g_b]["type"],
                "extent_b": geoms[g_b]["extent"],
                "intervals": [
                    [round(s * dt, 4), round(e * dt, 4)]
                    for s, e in merge_intervals(own)
                ],
                # step, peak force, point on A (3), point on B (3)
                "points": np.round(rows[:, [1, 4, 6, 7, 8, 9, 10, 11]], 3).tolist(),
            }
        )
    out = []
    for pair in pairs.values():
        merged = merge_intervals(np.concatenate(pair.pop("_iv")))
        rows = np.concatenate(pair.pop("_rows"))
        ent_a, ent_b = entities[pair["a"]], entities[pair["b"]]
        active = sum(e - s for s, e in merged)
        n_steps = len(step_ticks) - 1
        force = np.zeros(n_steps)
        np.add.at(force, rows[:, 1].astype(int).clip(0, n_steps - 1), rows[:, 4])
        pair.update(
            name_a=ent_a["name"],
            name_b=ent_b["name"],
            cls_a=ent_a["cls"],
            cls_b=ent_b["cls"],
            category=_category(ent_a["kind"], ent_b["kind"]),
            resting=ent_a["kind"] != "robot" and active >= RESTING_FRACTION * total,
            intervals=[[round(s * dt, 4), round(e * dt, 4)] for s, e in merged],
            active_s=round(active * dt, 3),
            peak_force=round(float(force.max(initial=0.0)), 3),
            force=np.round(force, 2).tolist(),  # summed peak force per env step
        )
        out.append(pair)
    out.sort(key=lambda p: (p["intervals"][0][0] if p["intervals"] else 0, p["name_a"]))
    return {
        "meta": {k: v for k, v in meta.items() if k not in ("geoms", "entities")},
        "entities": [
            {k: e[k] for k in ("id", "name", "cls", "kind")} for e in entities
        ],
        "duration": round(total * dt, 3),
        "step_times": np.round(step_ticks * dt, 4).tolist(),
        "pairs": out,
    }


def active_entities_per_step(directory: Path) -> list[set[int]]:
    """For each env step, the entities in a non-resting contact during it."""
    summary = summarise(directory)
    times = summary["step_times"]
    active: list[set[int]] = [set() for _ in times]
    for pair in summary["pairs"]:
        if pair["resting"] or pair["category"] == "fixture-self":
            continue
        for start, end in pair["intervals"]:
            first = int(np.searchsorted(times, start, side="right")) - 1
            last = int(np.searchsorted(times, end, side="left"))
            for step in range(max(first, 0), min(last + 1, len(times))):
                active[step].update((pair["a"], pair["b"]))
    return active
