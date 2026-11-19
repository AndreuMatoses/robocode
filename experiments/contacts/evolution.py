"""How the solution of an agent changes over its commits, in two views.

For each run, the **solved policy versions** are the commits that change the policy
and solve start scene 0. Each one has

* a **semantic strategy**: the values of the plan tag groups (``semantic.py``);
* a **contact strategy**: the class of its mode sequence (``histories.py``), with
  the tuned definition of the simulator;
* two measures of quality: the **episode length** on start scene 0 (steps; shorter
  is better) and the **solve rate** of the commit on the evaluation scenes.

A **transition** is two solved versions of one run that follow each other. It is a
*change* if the strategy differs, else a *keep*. This module answers:

* does an agent end with the strategy of its first solved version? (``kept``)
* how far is its final motion from its first one? (``motion_shift``)
* is a change better or worse than a keep? (``transition_stats``)
* does a run that changes end better than a run that does not? (``run_stats``)
* what does an agent do after a change that makes the episode longer? (``after``)

``python -m experiments.contacts.evolution`` prints the numbers of all scenarios and
writes ``<traces>/_scenarios/evolution.json``. ``--scenario NAME`` prints one.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from experiments.contacts import catalog, motion, semantic, strategy, viewer

BETTER, WORSE = 0.98, 1.02  # ratio of the episode lengths: outside is a real change
VIEWS = ("semantic", "contact")


def sequences(name: str, instance: str = "inst0") -> dict[str, list[dict[str, Any]]]:
    """The solved policy versions of each run of a scenario, in commit order."""
    tags = semantic.load_history(name)
    if tags is None:
        raise FileNotFoundError(f"no tag history of {name}: build it first")
    plan = [k for k, g in enumerate(tags["groups"]) if g.get("role") == "plan"]
    preset = strategy.PRESETS[viewer._preset(name)]  # pylint: disable=protected-access
    defn = strategy.Definition(
        preset.get("scope", "all"),
        preset.get("identity", "keep"),
        preset.get("order", "sequence"),
        preset.get("fingers", "all"),
        preset.get("metric", "edit"),
    )
    grouping = (defn, preset.get("group", "similar"), preset.get("tau", 0.35))
    contact = viewer.scenario_histories(name, instance, "class", grouping)
    out: dict[str, list[dict[str, Any]]] = {}
    for run in contact["runs"]:
        by_commit = {row["commit"]: row for row in tags["runs"].get(run["run_id"], [])}
        rows = []
        for row in run["commits"]:
            mark = by_commit.get(row["commit"], {})
            if not row["version"] or row["state"] != "solved" or "tags" not in mark:
                continue
            rows.append(
                {
                    "commit": row["commit"],
                    "commits": len(run["commits"]),
                    "elapsed": row["elapsed"],
                    "trace": row["trace"],
                    "steps": row["num_steps"],
                    "solve_rate": row["solve_rate"],
                    "semantic": " / ".join(mark["tags"][k] for k in plan) or "one plan",
                    "contact": row["sid"],
                    "subject": row["subject"],
                }
            )
        out[run["run_id"]] = rows
    return out


def coverage(name: str, instance: str = "inst0") -> dict[str, int]:
    """How many policy versions of a scenario have a recorded episode."""
    preset = strategy.PRESETS[viewer._preset(name)]  # pylint: disable=protected-access
    defn = strategy.Definition(metric=preset.get("metric", "edit"))
    contact = viewer.scenario_histories(
        name, instance, "class", (defn, "similar", 0.35)
    )
    states = Counter(
        row["state"]
        for run in contact["runs"]
        for row in run["commits"]
        if row["version"]
    )
    return dict(states)


def kept(runs: dict[str, list[dict[str, Any]]], view: str) -> dict[str, int]:
    """Runs with two solved versions or more: first strategy == last, and no change."""
    count = same = constant = returned = 0
    for rows in runs.values():
        seq = [row[view] for row in rows]
        if len(seq) < 2:
            continue
        count += 1
        same += seq[0] == seq[-1]
        constant += len(set(seq)) == 1
        order = [s for k, s in enumerate(seq) if k == 0 or seq[k - 1] != s]
        returned += len(order) > len(set(order))
    return {"runs": count, "kept": same, "constant": constant, "returned": returned}


def transitions(
    name: str, runs: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Every pair of solved versions of one run that follow each other."""
    out = []
    for run_id, rows in runs.items():
        for k in range(1, len(rows)):
            a, b = rows[k - 1], rows[k]
            item = {
                "scenario": name,
                "run_id": run_id,
                "index": k,
                "last": k == len(rows) - 1,
                "a": a,
                "b": b,
                "ratio": b["steps"] / a["steps"] if a["steps"] else None,
                "rate": (
                    None
                    if a["solve_rate"] is None or b["solve_rate"] is None
                    else b["solve_rate"] - a["solve_rate"]
                ),
            }
            for view in VIEWS:
                item[view] = a[view] != b[view]
            out.append(item)
    return out


def _outcome(items: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = np.array([t["ratio"] for t in items if t["ratio"]])
    rates = np.array([t["rate"] for t in items if t["rate"] is not None])
    if not len(ratios):
        return {"n": 0}
    return {
        "n": len(items),
        "shorter": round(float((ratios < BETTER).mean()), 3),
        "longer": round(float((ratios > WORSE).mean()), 3),
        "median_ratio": round(float(np.median(ratios)), 3),
        "rate_up": round(float((rates > 0.005).mean()), 3) if len(rates) else None,
        "rate_down": round(float((rates < -0.005).mean()), 3) if len(rates) else None,
        "mean_rate": round(float(rates.mean()), 4) if len(rates) else None,
    }


def transition_stats(items: list[dict[str, Any]], view: str) -> dict[str, Any]:
    """Outcome of the transitions with a change of strategy and of those without."""
    return {
        "change": _outcome([t for t in items if t[view]]),
        "keep": _outcome([t for t in items if not t[view]]),
    }


def run_stats(all_runs: dict[str, dict[str, list[dict[str, Any]]]], view: str) -> dict:
    """Runs that change their strategy against runs that do not.

    ``gain``: episode length of the last solved version / of the first one.
    ``first`` and ``final``: episode length of the first and of the last solved
    version / the median final episode length of the scenario.
    ``distinct_against_final``: rank correlation between the number of strategies of
    a run and ``final`` (negative: more strategies, shorter final episode).
    """
    rows = []
    for name, runs in all_runs.items():
        finals = [
            r[-1]["steps"] for r in runs.values() if len(r) >= 2 and r[-1]["steps"]
        ]
        if not finals:
            continue
        middle = float(np.median(finals))
        for seq in runs.values():
            if len(seq) < 2 or not seq[0]["steps"] or not seq[-1]["steps"]:
                continue
            rows.append(
                {
                    "scenario": name,
                    "changes": len({row[view] for row in seq}) > 1,
                    "moved": seq[0][view] != seq[-1][view],
                    "gain": seq[-1]["steps"] / seq[0]["steps"],
                    "first": seq[0]["steps"] / middle,
                    "distinct": len({row[view] for row in seq}),
                    "final": seq[-1]["steps"] / middle,
                    "rate": seq[-1]["solve_rate"],
                    "versions": len(seq),
                }
            )
    out = {}
    for label, pick in (("changes", True), ("constant", False)):
        part = [r for r in rows if r["changes"] is pick]
        rates = [r["rate"] for r in part if r["rate"] is not None]
        out[label] = {
            "runs": len(part),
            "median_gain": round(float(np.median([r["gain"] for r in part])), 3),
            "median_first": round(float(np.median([r["first"] for r in part])), 3),
            "median_final": round(float(np.median([r["final"] for r in part])), 3),
            "mean_final_rate": round(float(np.mean(rates)), 3),
            "median_versions": float(np.median([r["versions"] for r in part])),
        }
    # more strategies in a run and a shorter final episode? (rank correlation)
    out["distinct_against_final"] = round(
        float(spearmanr([r["distinct"] for r in rows], [r["final"] for r in rows])[0]),
        3,
    )
    out["versions_against_final"] = round(
        float(spearmanr([r["versions"] for r in rows], [r["final"] for r in rows])[0]),
        3,
    )
    return out


def after(all_runs: dict[str, dict[str, list[dict[str, Any]]]], view: str) -> dict:
    """What a run does after a change of strategy, by the result of the change.

    A change is *worse* if the episode is longer than before (ratio above 1.02),
    *better* if it is shorter (below 0.98), else *equal*. Then, in the later solved
    versions of the run: ``back`` = the next change goes back to the strategy before;
    ``again`` = the next change goes to another strategy; ``stays`` = no more change;
    ``last`` = the change is in the last version. ``recovers`` counts the runs whose
    last version is shorter than the version before a worse change.
    """
    out = {k: Counter() for k in ("worse", "better", "equal")}
    recovers = {"worse": [0, 0]}
    for runs in all_runs.values():
        for seq in runs.values():
            for k in range(1, len(seq)):
                a, b = seq[k - 1], seq[k]
                if a[view] == b[view] or not a["steps"]:
                    continue
                ratio = b["steps"] / a["steps"]
                kind = (
                    "worse"
                    if ratio > WORSE
                    else "better" if ratio < BETTER else "equal"
                )
                later = [row[view] for row in seq[k + 1 :]]
                nxt = next((s for s in later if s != b[view]), None)
                if not later:
                    out[kind]["last"] += 1
                elif nxt is None:
                    out[kind]["stays"] += 1
                elif nxt == a[view]:
                    out[kind]["back"] += 1
                else:
                    out[kind]["again"] += 1
                if kind == "worse" and later:
                    recovers["worse"][0] += 1
                    recovers["worse"][1] += seq[-1]["steps"] < a["steps"]
    return {
        **{kind: dict(counts) for kind, counts in out.items()},
        "worse_followed": recovers["worse"][0],
        "worse_recovers": recovers["worse"][1],
    }


def motion_shift(
    all_runs: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """How far the motion of the last solved version is from the first one.

    The distance is dynamic time warping on the positions of the bodies
    (``motion.py``), divided by the mean distance between the final policies of two
    runs of the scenario: 1 = as far as another run. The median over the runs, for
    all runs and by the semantic strategy (changed or not).
    """
    rows = []
    for name, runs in all_runs.items():
        between = next(
            d["mean"]
            for d in viewer.views(name)["distances"]
            if d["key"] == "motion:warp"
        )
        for seq in runs.values():
            if len(seq) < 2 or not between:
                continue
            ends = [
                viewer._trace_dir(seq[k]["trace"]) for k in (0, -1)
            ]  # pylint: disable=protected-access
            motion.ensure(ends)
            first, last = (motion.load(d).resampled("time") for d in ends)
            cost = motion.cost_matrix(first.pos, last.pos)
            rows.append(
                (
                    motion.match_distance(cost) / between,
                    seq[0]["semantic"] != seq[-1]["semantic"],
                )
            )
    values = np.array([r[0] for r in rows])
    moved = np.array([r[1] for r in rows])
    return {
        "runs": len(rows),
        "median": round(float(np.median(values)), 3),
        "median_strategy_kept": round(float(np.median(values[~moved])), 3),
        "median_strategy_changed": round(float(np.median(values[moved])), 3),
        "below_half": round(float((values < 0.5).mean()), 3),
    }


def bootstrap(flags: list[bool], rounds: int = 2000, seed: int = 0) -> list[float]:
    """95 % interval of a share, from a resample of the runs."""
    rng = np.random.default_rng(seed)
    values = np.array(flags, dtype=float)
    means = rng.choice(values, (rounds, len(values))).mean(axis=1)
    return [round(float(v), 3) for v in np.percentile(means, [2.5, 97.5])]


def analyse(names: list[str]) -> dict[str, Any]:
    """All numbers for a list of scenarios."""
    all_runs = {name: sequences(name) for name in names}
    items = [t for name, runs in all_runs.items() for t in transitions(name, runs)]
    out: dict[str, Any] = {"scenarios": len(names), "transitions": len(items)}
    for view in VIEWS:
        per = {name: kept(runs, view) for name, runs in all_runs.items()}
        total = {
            k: sum(p[k] for p in per.values())
            for k in ("runs", "kept", "constant", "returned")
        }
        flags = [
            [row[view] for row in seq]
            for runs in all_runs.values()
            for seq in runs.values()
            if len(seq) >= 2
        ]
        out[view] = {
            "kept": total,
            "kept_interval": bootstrap([s[0] == s[-1] for s in flags]),
            "constant_interval": bootstrap([len(set(s)) == 1 for s in flags]),
            "per_scenario": per,
            "transitions": transition_stats(items, view),
            "runs": run_stats(all_runs, view),
            "after": after(all_runs, view),
        }
    out["motion"] = motion_shift(all_runs)
    out["items"] = items
    return out


def main() -> None:
    """Print the numbers and write them to a file."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scenario", action="append")
    args = parser.parse_args()
    names = args.scenario or sorted(p.stem for p in semantic.HISTORY.glob("*.json"))
    data = analyse(names)
    print(
        f"{data['scenarios']} scenarios, {data['transitions']} transitions between solved versions"
    )
    for view in VIEWS:
        d = data[view]
        k = d["kept"]
        print(f"\n== {view} strategy")
        print(
            f"runs with two solved versions or more: {k['runs']}; first == last: {k['kept']} "
            f"({k['kept'] / k['runs']:.1%}, interval {d['kept_interval']}); never changes: "
            f"{k['constant']} ({k['constant'] / k['runs']:.1%}, interval {d['constant_interval']}); "
            f"goes back to an earlier strategy: {k['returned']}"
        )
        for kind in ("change", "keep"):
            print(f"  transition {kind:6s}: {d['transitions'][kind]}")
        print(f"  runs: {d['runs']}")
        print(f"  after a change: {d['after']}")
        shares = [
            p["kept"] / p["runs"] for p in d["per_scenario"].values() if p["runs"] >= 5
        ]
        print(f"  first == last by scenario: {min(shares):.0%} to {max(shares):.0%}")
    print(f"\n== motion: {data['motion']}")
    path = viewer.CONFIG.traces / catalog.SCENARIO_DIR / "evolution.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"\n{path}")


if __name__ == "__main__":
    main()
