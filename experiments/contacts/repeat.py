"""Repeatability of a strategy definition: one policy on several start scenes.

The final policy of a run is recorded on more than one instance (start scene). Two
episodes of ONE policy on two instances are a *within* pair; two episodes of TWO
runs on two instances are a *between* pair. Only instances with the same number of
objects are compared: another number of objects is another task.

A definition of "strategy" is useful when a policy has the same strategy as itself
on another start scene, so the within distances give the threshold: the distance
below which most within pairs fall. ``auc`` is the probability that a between pair
is farther apart than a within pair (0.5 = the definition cannot tell one policy
from another; this also happens when all runs of a scenario do the same thing).

Record the instances first:

    python -m experiments.contacts.record '*/tossing3d/*' --commit -1 \
        --instances 1 2 3 --no-video --jobs 12
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import rankdata

from experiments.contacts import histories, metrics, strategy
from experiments.contacts.commits import versions_path
from experiments.contacts.trace import load_meta

INSTANCES = (0, 1, 2, 3)
METRICS = ("edit", "warp", "overlap")
QUANTILES = (0.5, 0.8, 0.9)
BETWEEN_LIMIT = 1500  # between pairs per scenario (a fixed random sample)


def population(
    traces: Path,
    run_ids: list[str],
    load_versions: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Per run: the solved episodes of its final policy (trace dir by instance)."""
    runs = []
    for run_id in run_ids:
        if not versions_path(traces, run_id).exists():
            continue
        versions = load_versions(run_id)
        rows = histories.run_history(traces, run_id, versions, "inst0")
        if not rows:
            continue
        last = versions[-1]["commit"]
        episodes: dict[int, Path] = {}
        count: dict[int, Any] = {}
        for k in INSTANCES:
            # instance 0 can be the trace of an earlier commit with the same policy
            directory = (
                rows[-1]["dir"]
                if k == 0
                else traces / run_id / f"commit{last}" / f"inst{k}"
            )
            if directory is None or not (directory / "meta.json").exists():
                continue
            result = load_meta(directory)["metrics"]
            if result.get("solved") and not result.get("crashed"):
                episodes[k] = directory
                count[k] = result.get("object_count")
        if episodes:
            runs.append({"run_id": run_id, "episodes": episodes, "count": count})
    return runs


def _pairs(runs: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    within, between = [], []
    for a, run in enumerate(runs):
        own = sorted(run["episodes"])
        within += [
            ((a, i), (a, j))
            for x, i in enumerate(own)
            for j in own[x + 1 :]
            if run["count"][i] == run["count"][j]
        ]
        for b in range(a):
            other = runs[b]
            between += [
                ((a, i), (b, j))
                for i in own
                for j in other["episodes"]
                if i != j and run["count"][i] == other["count"][j]
            ]
    if len(between) > BETWEEN_LIMIT:
        between = random.Random(0).sample(between, BETWEEN_LIMIT)
    return within, between


def auc(within: np.ndarray, between: np.ndarray) -> float | None:
    """Probability that a between pair is farther apart than a within pair."""
    if not len(within) or not len(between):
        return None
    ranks = rankdata(np.concatenate([within, between]))
    above = ranks[len(within) :].sum() - len(between) * (len(between) + 1) / 2
    return float(above / (len(within) * len(between)))


def scenario_repeatability(
    traces: Path,
    run_ids: list[str],
    opts: strategy.Options,
    load_versions: Callable[[str], list[dict[str, Any]]],
    defn: strategy.Definition = strategy.Definition(),
    which: tuple[str, ...] = METRICS,
) -> dict[str, Any]:
    """Within and between distances of every metric in ``which``."""
    runs = population(traces, run_ids, load_versions)
    within, between = _pairs(runs)
    need = sorted({e for group in (within, between) for pair in group for e in pair})
    dirs = {e: runs[e[0]]["episodes"][e[1]] for e in need}
    keys = {e: list(strategy.strategy_key(d, opts, defn)) for e, d in dirs.items()}
    timed = set(which) - {"edit", "warp"}
    spans = (
        {
            e: metrics.pair_spans(metrics.timed_modes(d, opts, defn))
            for e, d in dirs.items()
        }
        if timed
        else {}
    )
    cache: dict[Any, float] = {}

    def distance(metric: str, x: Any, y: Any) -> float:
        if metric in ("edit", "warp"):
            key = (metric, tuple(keys[x]), tuple(keys[y]))
            if key not in cache:
                cache[key] = strategy.key_distance(keys[x], keys[y], metric)
            return cache[key]
        table = metrics.pair_table(spans[x], spans[y])
        if metric == "overlap":
            return metrics.overlap_distance(table)
        if metric == "duty":
            return metrics.duty_distance(table)
        return strategy.jaccard(frozenset(spans[x]), frozenset(spans[y]))

    out: dict[str, Any] = {
        "runs": len(runs),
        "episodes": sum(len(r["episodes"]) for r in runs),
        "instances": sorted({k for r in runs for k in r["episodes"]}),
        "within_pairs": len(within),
        "between_pairs": len(between),
        "metrics": {},
    }
    for metric in which:
        w = np.array([distance(metric, *pair) for pair in within])
        b = np.array([distance(metric, *pair) for pair in between])
        best = None
        if len(w) and len(b):
            # the threshold that best separates within from between (Youden)
            grid = np.round(np.arange(0.0, 1.0001, 0.01), 2)
            gain = [(w <= t).mean() - (b <= t).mean() for t in grid]
            best = float(grid[int(np.argmax(gain))])
        value = auc(w, b)
        out["metrics"][metric] = {
            "within": [round(float(v), 3) for v in w],
            "between": [round(float(v), 3) for v in b],
            "quantiles": {
                # rounded up, so that the share at this threshold is at least q
                f"q{int(q * 100)}": (
                    math.ceil(float(np.quantile(w, q)) * 100 - 1e-9) / 100
                    if len(w)
                    else None
                )
                for q in QUANTILES
            },
            "separation": best,
            "auc": None if value is None else round(value, 3),
        }
    return out
