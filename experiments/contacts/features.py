"""Measured numbers of a final policy: the evidence behind the semantic tags.

A ``Run`` is the final policy of one agent run on one start scene: its motion
(``motion.py``), its contact pairs (``strategy.py``) and the text of its program.
The rule files in ``tag_rules/`` read a ``Run`` and give the tags.

Command line: ``python -m experiments.contacts.features <scenario>`` prints one
block for each run (base path, each object: shift, lift, first and last motion; the
contact pairs with a robot part). Read it before you write the rules of a scenario.
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from experiments.contacts import motion, strategy

MOVE = 0.004  # [m] a body moves in an env step when it goes farther than this


def short(pair: str) -> str:
    """``robot.gripper.pad <> env.can_0`` -> ``gripper.pad~can_0``."""
    for prefix in ("robot.", "env.", "world."):
        pair = pair.replace(prefix, "")
    return pair.replace(" <> ", "~")


@dataclasses.dataclass
class Run:
    """The final policy of one agent run on one start scene."""

    run_id: str
    method: str
    seed: str
    steps: int
    directory: Path  # the trace
    sandbox: Path  # the program of the agent
    motion: motion.Motion
    modes: tuple[strategy.Mode, ...]  # contact modes in order (class level)

    def ch(self, name: str) -> np.ndarray:
        """[T, d] positions of a channel (``ee``, ``base``, ``obj:<name>``)."""
        return self.motion.channels[name]

    @property
    def objects(self) -> dict[str, np.ndarray]:
        """Positions of every movable object, by its name."""
        return {
            k[4:]: v for k, v in self.motion.channels.items() if k.startswith("obj:")
        }

    @property
    def dt(self) -> float:
        """Seconds of one env step."""
        return self.motion.step_seconds

    @functools.cached_property
    def meta(self) -> dict[str, Any]:
        """``meta.json`` of the trace."""
        return json.loads((self.directory / "meta.json").read_text())

    @functools.cached_property
    def pairs(self) -> list[str]:
        """Every contact pair of the episode, in the short form of ``short``."""
        return sorted({short(p) for mode in self.modes for p in mode})

    def touches(self, pattern: str) -> bool:
        """True if a contact pair matches the regular expression."""
        return any(re.search(pattern, pair) for pair in self.pairs)

    def first_contact(self, pattern: str) -> int | None:
        """Index of the first contact mode with a pair that matches."""
        for index, mode in enumerate(self.modes):
            if any(re.search(pattern, short(pair)) for pair in mode):
                return index
        return None

    @functools.cached_property
    def program(self) -> dict[str, str]:
        """Text of ``approach.py`` and of the sibling modules that it imports."""
        seen: dict[str, str] = {}
        todo = ["approach"]
        while todo:
            name = todo.pop()
            path = self.sandbox / f"{name}.py"
            if name in seen or not path.exists():
                continue
            seen[name] = path.read_text(errors="replace")
            try:
                tree = ast.parse(seen[name])
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    todo += [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    todo.append(node.module.split(".")[0])
        return seen

    def program_has(self, pattern: str) -> int:
        """Number of matches of a regular expression in the program text."""
        text = "\n".join(self.program.values())
        return len(re.findall(pattern, text, flags=re.I))


def nearest_gap(run: Run, entity: str, point: np.ndarray) -> float:
    """Smallest distance from the outline of an entity to a point, over the episode.

    Kinematic 2D traces only (they store the outline of every shape). Use it when a
    touch can be the last instant of the episode, which the contact modes can miss.
    """
    wanted = next(e["id"] for e in run.meta["entities"] if e["name"] == entity)
    geoms = [g["id"] for g in run.meta["geoms"] if g["entity"] == wanted]
    with np.load(run.directory / "trace.npz") as data:
        verts = data["verts"][:, geoms]  # [T, geom, V, 2]
    best = np.inf
    for geom in range(verts.shape[1]):
        shape = verts[:, geom]
        shape = shape[:, ~np.isnan(shape[0, :, 0])]
        ahead = np.roll(shape, -1, axis=1)
        for share in np.linspace(0.0, 1.0, 12, endpoint=False):
            points = shape + share * (ahead - shape)
            best = min(best, float(np.linalg.norm(points - point, axis=-1).min()))
    return best


def path_length(pos: np.ndarray) -> float:
    """Length of a path in metres."""
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def moving(pos: np.ndarray, threshold: float = MOVE) -> np.ndarray:
    """The env steps in which a body moves."""
    return np.flatnonzero(np.linalg.norm(np.diff(pos, axis=0), axis=1) > threshold)


def summary(pos: np.ndarray) -> dict[str, Any]:
    """Shift, lift, drop, path and the first and last motion of one body."""
    steps = moving(pos)
    up = pos[:, 2] if pos.shape[1] == 3 else np.zeros(len(pos))
    return {
        "start": np.asarray(pos[0], dtype=float).round(2).tolist(),
        "end": np.asarray(pos[-1], dtype=float).round(2).tolist(),
        "shift": float(np.linalg.norm(pos[-1] - pos[0])),
        "lift": float(up.max() - up[0]),
        "drop": float(up[0] - up.min()),
        "path": path_length(pos),
        "first": int(steps[0]) if len(steps) else None,
        "last": int(steps[-1]) if len(steps) else None,
    }


def moved_order(run: Run, shift: float = 0.03, path: float = 0.1) -> list[str]:
    """The objects that move, in the order of their first motion."""
    found = []
    for name, pos in run.objects.items():
        info = summary(pos)
        if info["first"] is not None and (info["shift"] > shift or info["path"] > path):
            found.append((info["first"], name))
    return [name for _, name in sorted(found)]


def stops(pos: np.ndarray, threshold: float = MOVE, least: int = 2) -> int:
    """Number of pauses of a body between its first and its last motion."""
    steps = moving(pos, threshold)
    if len(steps) < 2:
        return 0
    return int((np.diff(steps) > least).sum())


def bouts(pos: np.ndarray, gap: int = 10, threshold: float = MOVE) -> int:
    """Number of separate motions of a body (a pause longer than ``gap`` steps splits)."""
    steps = moving(pos, threshold)
    return 0 if not len(steps) else int(1 + (np.diff(steps) > gap).sum())


def moves_with(run: Run, body: str, other: str) -> float:
    """Share of the moving steps of ``body`` in which ``other`` moves too (channels)."""
    first = np.linalg.norm(np.diff(run.ch(body), axis=0), axis=1) > MOVE
    second = np.linalg.norm(np.diff(run.ch(other), axis=0), axis=1) > MOVE
    return float((first & second).sum() / first.sum()) if first.sum() else 0.0


def load_runs(scenario: str, instance: str = "inst0") -> list[Run]:
    """The final policy of every run of a scenario that solves the instance."""
    # pylint: disable=import-outside-toplevel,protected-access
    from experiments.contacts import metrics, viewer

    rows = metrics._final_solved(
        viewer.CONFIG.traces,
        viewer._scenario_runs(scenario),
        instance,
        viewer._versions,
    )
    dirs = [row["dir"] for row in rows]
    motion.ensure(dirs)
    opts = viewer._check("class", instance)
    return [
        Run(
            row["run_id"],
            row["method"],
            row["seed"],
            int(row["steps"] or 0),
            row["dir"],
            viewer.CONFIG.data / "runs" / row["run_id"] / "sandbox",
            motion.load(row["dir"]),
            strategy.strategy_modes(row["dir"], opts),
        )
        for row in rows
    ]


def load_history(
    scenario: str, instance: str = "inst0"
) -> dict[str, list[dict[str, Any]]]:
    """Every commit of every run of a scenario, with a ``Run`` for the solved ones.

    One row for each commit: ``commit``, ``elapsed`` (seconds since the first
    commit), ``state`` (solved, failed, crashed, broken, none), ``version`` (true if
    the commit changes the policy) and ``run`` (a ``Run`` if the episode is solved
    and recorded, else ``None``). Commits with the same policy share one ``Run``.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from experiments.contacts import histories, viewer
    from experiments.contacts.commits import versions_path

    traces = viewer.CONFIG.traces
    opts = viewer._check("class", instance)
    out: dict[str, list[dict[str, Any]]] = {}
    dirs: list[Path] = []
    for run_id in viewer._scenario_runs(scenario):
        if not versions_path(traces, run_id).exists():
            continue
        rows = histories.run_history(traces, run_id, viewer._versions(run_id), instance)
        out[run_id] = rows
        dirs += [r["dir"] for r in rows if r["state"] == "solved" and r["dir"]]
    dirs = list(dict.fromkeys(dirs))
    motion.ensure(dirs)
    made: dict[Path, Run] = {}
    for run_id, rows in out.items():
        method, _, seed = run_id.split("/")
        for row in rows:
            directory = row.pop("dir")
            row["run"] = None
            if row["state"] != "solved" or directory is None:
                continue
            if directory not in made:
                made[directory] = Run(
                    run_id,
                    method,
                    seed,
                    int(row.get("num_steps") or 0),
                    directory,
                    viewer.CONFIG.data / "runs" / run_id / "sandbox",
                    motion.load(directory),
                    strategy.strategy_modes(directory, opts),
                )
            row["run"] = made[directory]
    return out


def describe(run: Run, index: int) -> str:
    """One text block with the generic numbers of a run."""
    lines = [f"{index:2d} {run.method[:13]:13s} {run.seed:7s} {run.steps:4d} steps"]
    for name in ("ee", "base"):
        if name in run.motion.channels:
            pos = run.ch(name)
            lines[0] += (
                f" | {name} path {path_length(pos):.2f} m, stops {stops(pos)}, "
                f"{np.asarray(pos[0], dtype=float).round(2).tolist()} -> "
                f"{np.asarray(pos[-1], dtype=float).round(2).tolist()}"
            )
    for name, pos in run.motion.channels.items():
        if name in ("ee", "base"):
            continue
        info = summary(pos)
        if info["first"] is None:
            continue
        lines.append(
            f"      {name.removeprefix('obj:'):18s} {info['start']} -> {info['end']} "
            f"shift {info['shift']:.2f} lift {info['lift']:.2f} drop {info['drop']:.2f} "
            f"path {info['path']:.2f} steps {info['first']}-{info['last']}"
        )
    lines.append("      order: " + " > ".join(moved_order(run)))
    lines.append("      pairs: " + ", ".join(run.pairs))
    return "\n".join(lines)


def _spread(values: list[float]) -> str:
    if not values:
        return "-"
    q = np.percentile(values, [0, 25, 50, 75, 100])
    return " / ".join(f"{v:.2f}" for v in q)


def overview(runs: list[Run]) -> str:
    """The spread of the generic numbers over all runs (min / quartiles / max)."""
    from collections import Counter  # pylint: disable=import-outside-toplevel

    count = len(runs)
    lines = [f"steps: {_spread([r.steps for r in runs])}"]
    lines.append(f"contact modes: {_spread([len(r.modes) for r in runs])}")
    for name in sorted(runs[0].motion.channels):
        infos = [summary(r.ch(name)) for r in runs if name in r.motion.channels]
        moved = [i for i in infos if i["first"] is not None and i["path"] > 0.02]
        line = f"{name:22s} moves in {len(moved):2d}/{count}"
        if moved:
            line += (
                f" | start {infos[0]['start']} | path {_spread([i['path'] for i in moved])}"
                f" | shift {_spread([i['shift'] for i in moved])}"
                f" | lift {_spread([i['lift'] for i in moved])}"
                f" | drop {_spread([i['drop'] for i in moved])}"
                f" | first step {_spread([i['first'] for i in moved])}"
            )
        if name in ("ee", "base"):
            line += f" | stops {_spread([stops(r.ch(name)) for r in runs])}"
        lines.append(line)
    orders = Counter(" > ".join(moved_order(r)) for r in runs)
    lines.append("order of first motion:")
    lines += [
        f"   {n:2d} x {order or '(nothing moves)'}"
        for order, n in orders.most_common(12)
    ]
    pairs = Counter(pair for r in runs for pair in r.pairs)
    lines.append(
        "contact pairs (runs): " + ", ".join(f"{p} {n}" for p, n in pairs.most_common())
    )
    return "\n".join(lines)


def main() -> None:
    """Print the generic numbers of every final policy of the given scenarios."""
    import sys  # pylint: disable=import-outside-toplevel

    for name in sys.argv[1:]:
        runs = load_runs(name)
        print(f"=== {name}: {len(runs)} final policies solve inst0")
        if runs:
            still = [
                k for k, v in runs[0].motion.channels.items() if not len(moving(v))
            ]
            print(
                f"channels: {sorted(runs[0].motion.channels)}; still in run 0: {still}"
            )
        print(overview(runs))
        limit = int(os.environ.get("FEATURE_RUNS", len(runs)))
        for index, run in enumerate(runs[:limit]):
            print(describe(run, index))


if __name__ == "__main__":
    main()
