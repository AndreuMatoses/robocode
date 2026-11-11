"""Compare the commit histories of several runs of one scenario.

Runs do not share an axis (one has 9 commits, another 155), so the histories are
compared through a **strategy vocabulary**: every distinct contact strategy
(``strategy.strategy_modes``) of a *solved* episode gets an identifier, shared by all
runs of the scenario. A history is then a sequence of identifiers over the commits,
placed either by commit fraction or by the time elapsed since the run's first commit.

Identifiers can be grouped (``group``): ``exact`` = one per distinct mode sequence;
``similar`` = sequences within an edit distance of ``tau`` of each other share one
(complete-linkage clusters, so every two members are that close); ``pairs`` = one
per set of contact pairs, whatever their order. The class is named after its most
common member.

Unsolved episodes are long and noisy, so they are not given identifiers: a commit
is ``failed``, ``crashed``, ``broken`` (the policy does not load, or is not recorded)
or ``none`` (no policy yet).
"""

from __future__ import annotations

import dataclasses
import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from experiments.contacts import strategy
from experiments.contacts.commits import versions_path
from experiments.contacts.trace import load_meta, trace_id


def _seconds(stamp: str) -> float:
    return datetime.datetime.fromisoformat(stamp).timestamp()


def _trace(traces: Path, run_id: str, commit: int | None, instance: str) -> Path | None:
    if commit is None:
        return None
    path = traces / run_id / f"commit{commit}" / instance
    return path if (path / "meta.json").exists() else None


def run_history(
    traces: Path, run_id: str, versions: list[dict[str, Any]], instance: str
) -> list[dict[str, Any]]:
    """One row per commit: state on the instance and the trace that shows it."""
    start = _seconds(versions[0]["time"]) if versions else 0.0
    rows = []
    for v in versions:
        shown = _trace(traces, run_id, v["commit"], instance) or _trace(
            traces, run_id, v["same_as"], instance
        )
        row: dict[str, Any] = {
            "commit": v["commit"],
            "elapsed": round(_seconds(v["time"]) - start, 1),
            "subject": v["subject"],
            "solve_rate": v["reeval_solve_rate"],
            "version": v["same_as"] == v["commit"],
            "trace": trace_id(traces, shown) if shown else None,
            "dir": shown,
        }
        if shown is None:
            row["state"] = "broken" if v["key"] else "none"
        else:
            metrics = load_meta(shown)["metrics"]
            row["num_steps"] = metrics.get("num_steps")
            row["state"] = (
                "crashed"
                if metrics.get("crashed")
                else "solved" if metrics.get("solved") else "failed"
            )
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Milestones of one history, as commit indices (``None`` if never)."""

    def tag(r: dict[str, Any]) -> str:
        return r.get("sid") or r["state"]

    first_solve = next((r["commit"] for r in rows if r["state"] == "solved"), None)
    half = next((r["commit"] for r in rows if (r["solve_rate"] or 0.0) >= 0.5), None)
    settled = None
    if rows and rows[-1]["state"] == "solved":
        settled = rows[-1]["commit"]
        for r in reversed(rows):
            if tag(r) != tag(rows[-1]):
                break
            settled = r["commit"]
    visited = [tag(r) for r in rows if r["version"] and r.get("sid")]
    order = [s for i, s in enumerate(visited) if i == 0 or visited[i - 1] != s]
    return {
        "first_solve": first_solve,
        "half": half,
        "settled": settled,
        "distinct": len(set(order)),
        "returns": len(order) - len(set(order)),
        "final": tag(rows[-1]) if rows else None,
    }


Key = tuple[strategy.Mode, ...]


def group_strategies(
    keys: list[Key], group: str, tau: float, metric: str = "edit"
) -> dict[Key, Any]:
    """Map every distinct strategy to the label of its class."""
    if group == "exact" or len(keys) < 2:
        return {key: key for key in keys}
    if group != "similar":
        raise ValueError(f"unknown grouping {group!r}")
    dist = strategy.key_distance_matrix(keys, metric)
    labels = fcluster(linkage(squareform(dist), "complete"), tau + 1e-9, "distance")
    return {key: int(label) for key, label in zip(keys, labels)}


def scenario_histories(
    traces: Path,
    run_ids: list[str],
    instance: str,
    opts: strategy.Options,
    load_versions: Any,
    group: str = "similar",
    tau: float = 0.35,
    defn: strategy.Definition = strategy.Definition(),
) -> dict[str, Any]:
    """Histories of the runs that have a recorded commit table, on one instance."""
    if group == "pairs":  # the old name of ``Definition(order="set")``
        defn, group = dataclasses.replace(defn, order="set"), "exact"
    histories, missing = {}, []
    for run_id in run_ids:
        if not versions_path(traces, run_id).exists():
            missing.append(run_id)
            continue
        histories[run_id] = run_history(traces, run_id, load_versions(run_id), instance)
    # every distinct strategy of a solved episode
    exact: dict[Key, dict[str, Any]] = {}
    for run_id, rows in histories.items():
        for row in rows:
            if row["state"] != "solved":
                continue
            key = strategy.strategy_key(row["dir"], opts, defn)
            entry = exact.setdefault(key, {"runs": set(), "versions": 0})
            entry["runs"].add(run_id)
            entry["versions"] += row["version"]
            row["_key"] = key
    # classes of strategies, named after their most common member
    # the time-free distances work on the keys; the others fall back to "edit"
    label_of = group_strategies(
        list(exact), group, tau, defn.metric if defn.metric == "warp" else "edit"
    )
    classes: dict[Any, dict[str, Any]] = {}
    for key, entry in exact.items():
        cls = classes.setdefault(
            label_of[key], {"runs": set(), "versions": 0, "members": []}
        )
        cls["runs"] |= entry["runs"]
        cls["versions"] += entry["versions"]
        cls["members"].append((len(entry["runs"]), entry["versions"], key))
    ranked = sorted(
        classes.items(), key=lambda kv: (-len(kv[1]["runs"]), -kv[1]["versions"])
    )
    ids = {label: f"S{i}" for i, (label, _) in enumerate(ranked, 1)}
    out_runs = []
    for run_id, rows in histories.items():
        for row in rows:
            row.pop("dir")
            if "_key" in row:
                row["sid"] = ids[label_of[row.pop("_key")]]
        out_runs.append(
            {
                "run_id": run_id,
                "method": run_id.split("/")[0],
                "seed": run_id.rsplit("/", 1)[-1],
                "commits": rows,
                "elapsed": rows[-1]["elapsed"] if rows else 0.0,
                **_summary(rows),
            }
        )
    strategies = []
    for label, cls in ranked:
        best = max(cls["members"], key=lambda m: (m[0], m[1]))[2]
        strategies.append(
            {
                "id": ids[label],
                "modes": [sorted(m) for m in best],
                "pairs": sorted(
                    frozenset().union(*[mode for m in cls["members"] for mode in m[2]])
                ),
                "variants": len(cls["members"]),
                "runs": len(cls["runs"]),
                "methods": len({r.split("/")[0] for r in cls["runs"]}),
                "versions": cls["versions"],
            }
        )
    return {
        "instance": instance,
        "level": opts.level,
        "group": group,
        "tau": tau,
        "definition": dataclasses.asdict(defn),
        "definition_text": defn.text(group, tau),
        "exact": len(exact),
        "runs": out_runs,
        "missing": missing,
        "strategies": strategies,
    }


def history_pair(
    traces: Path,
    run_a: str,
    run_b: str,
    instance: str,
    opts: strategy.Options,
    load_versions: Any,
) -> dict[str, Any]:
    """Sequence distance between every policy version of two runs."""
    sides = []
    for run_id in (run_a, run_b):
        rows = run_history(traces, run_id, load_versions(run_id), instance)
        sides.append([r for r in rows if r["version"] and r["dir"] is not None])
    matrix = [
        [
            round(strategy.sequence_distance(a["dir"], b["dir"], opts), 3)
            for b in sides[1]
        ]
        for a in sides[0]
    ]
    for side in sides:
        for row in side:
            row["modes"] = len(strategy.strategy_modes(row.pop("dir"), opts))
    return {"a": run_a, "b": run_b, "rows": sides[0], "cols": sides[1], "d": matrix}
