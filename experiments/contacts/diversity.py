"""Diversity and optimality of the final policies of a scenario.

The population is the **final policy of every run** that solves the instance (one
episode each). Three numbers describe how diverse a group of them is; they answer
different questions and are shown together:

* ``richness``: how many strategies the group has (classes under the strategy
  definition and grouping, see ``strategy.Definition`` and
  ``histories.group_strategies``).
* ``effective``: the effective number of strategies, ``2 ** H`` with ``H`` the
  Shannon entropy (bits) of the class frequencies. It is the number of equally
  common strategies that gives the same entropy: 5 runs with strategies
  ``A A A A B`` have richness 2 but an effective number of 1.65. With ``n`` runs it
  cannot exceed ``n``, so a method with 5 runs saturates at 5.
* ``distance``: the mean distance between two runs of the group (edit distance
  between their strategies). It needs no classes and no threshold, and it says how
  *different* the strategies are, which the two counts above ignore.

For a scenario, ``within`` and ``between`` are the mean distance between two runs of
the same method and of different methods; ``p_value`` is a permutation test of
``between - within`` (method labels shuffled): a small value says that the method
explains part of the strategy.

Optimality: on these scenarios the reward is -1 per step, so the total reward is
minus the number of steps and "time to complete" is steps times the control period.
``policy_time_s`` is the compute time of the policy during the episode.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from experiments.contacts import histories, strategy
from experiments.contacts.commits import versions_path
from experiments.contacts.metrics import distance_matrices
from experiments.contacts.trace import load_meta

PERMUTATIONS = 2000


def group_metrics(
    labels: list[Any], dist: np.ndarray, index: list[int]
) -> dict[str, Any]:
    """Richness, effective number and mean pairwise distance of ``index``."""
    n = len(index)
    if n == 0:
        return {
            "n": 0,
            "richness": 0,
            "entropy": None,
            "effective": None,
            "distance": None,
        }
    counts = Counter(labels[i] for i in index)
    entropy = -sum(c / n * math.log2(c / n) for c in counts.values())
    sub = dist[np.ix_(index, index)]
    mean = float(sub[np.triu_indices(n, 1)].mean()) if n > 1 else None
    return {
        "n": n,
        "richness": len(counts),
        "entropy": round(entropy, 3),
        "effective": round(2**entropy, 2),
        "distance": None if mean is None else round(mean, 3),
    }


def within_between(dist: np.ndarray, methods: list[str]) -> dict[str, Any]:
    """Mean distance within and between methods, with a permutation test."""
    n = len(methods)
    out: dict[str, Any] = {"within": None, "between": None, "p_value": None}
    if n < 3:
        return out
    upper = np.triu_indices(n, 1)
    values = dist[upper]
    codes = np.unique(methods, return_inverse=True)[1]

    def split(code: np.ndarray) -> tuple[float, float] | None:
        same = (code[:, None] == code[None, :])[upper]
        if not same.any() or same.all():
            return None
        return float(values[same].mean()), float(values[~same].mean())

    observed = split(codes)
    if observed is None:
        return out
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(PERMUTATIONS):
        got = split(rng.permutation(codes))
        if got is not None:
            gaps.append(got[1] - got[0])
    gap = observed[1] - observed[0]
    out.update(
        within=round(observed[0], 3),
        between=round(observed[1], 3),
        p_value=round((1 + sum(g >= gap for g in gaps)) / (1 + len(gaps)), 4),
    )
    return out


def mds(dist: np.ndarray) -> np.ndarray:
    """Classical multidimensional scaling to two dimensions."""
    n = len(dist)
    if n < 3:
        return np.zeros((n, 2))
    centre = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centre @ (dist**2) @ centre
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))


def _median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def scenario_diversity(
    traces: Path,
    run_ids: list[str],
    instance: str,
    opts: strategy.Options,
    load_versions: Callable[[str], list[dict[str, Any]]],
    defn: strategy.Definition = strategy.Definition(),
    group: str = "similar",
    tau: float = 0.35,
    reported: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Diversity and optimality of the final policies of ``run_ids``."""
    reported = reported or {}
    runs: list[dict[str, Any]] = []
    for run_id in run_ids:
        if not versions_path(traces, run_id).exists():
            continue
        rows = histories.run_history(traces, run_id, load_versions(run_id), instance)
        last = rows[-1] if rows else {"state": "none", "dir": None, "trace": None}
        solved = [r for r in rows if r["state"] == "solved" and r["version"]]
        run: dict[str, Any] = {
            "run_id": run_id,
            "method": run_id.split("/")[0],
            "seed": run_id.rsplit("/", 1)[-1],
            "state": last["state"],
            "trace": last["trace"],
            "dir": last["dir"],
            "first_steps": solved[0]["num_steps"] if solved else None,
            "best_steps": min((r["num_steps"] for r in solved), default=None),
            "solved_versions": len(solved),
            **reported.get(run_id, {}),
        }
        if last["dir"] is not None:
            meta = load_meta(last["dir"])
            metrics = meta["metrics"]
            steps = metrics.get("num_steps") or 0
            run.update(
                steps=steps,
                reward=metrics.get("total_reward"),
                seconds=round(meta["num_ticks"] * meta["dt"], 2),
                policy_time_s=metrics.get("policy_time_s"),
            )
        runs.append(run)

    solved_runs = [r for r in runs if r["state"] == "solved"]
    n = len(solved_runs)
    keys, matrices = distance_matrices(
        [r["dir"] for r in solved_runs], opts, defn, (defn.metric,)
    )
    dist = matrices[defn.metric]
    labels: list[Any]
    if group == "exact" or n < 2:
        labels = list(keys)
    elif group == "similar":
        # complete linkage: no two members of one strategy are farther apart than tau
        tree = linkage(squareform(dist, checks=False), "complete")
        labels = [int(v) for v in fcluster(tree, tau + 1e-9, "distance")]
    else:
        raise ValueError(f"unknown grouping {group!r}")
    ranked = [label for label, _ in Counter(labels).most_common()]
    ids = {label: f"S{i}" for i, label in enumerate(ranked, 1)}
    coords = mds(dist)
    for i, run in enumerate(solved_runs):
        run["sid"] = ids[labels[i]]
        run["modes"] = len(keys[i])
        run["xy"] = [round(float(c), 4) for c in coords[i]]
    for run in runs:
        run.pop("dir")

    strategies = []
    for label in ranked:
        members = [i for i, lab in enumerate(labels) if lab == label]
        common = Counter(keys[i] for i in members).most_common(1)[0][0]
        steps = [solved_runs[i]["steps"] for i in members]
        strategies.append(
            {
                "id": ids[label],
                "modes": [sorted(m) for m in common],
                "runs": len(members),
                "methods": len({solved_runs[i]["method"] for i in members}),
                "variants": len({keys[i] for i in members}),
                "median_steps": _median(steps),
                "best_steps": min(steps),
            }
        )

    method_names = list(dict.fromkeys(r["method"] for r in runs))
    best = min((r["steps"] for r in solved_runs), default=None)
    methods = []
    for name in method_names:
        own = [r for r in runs if r["method"] == name]
        index = [i for i, r in enumerate(solved_runs) if r["method"] == name]
        final = [solved_runs[i] for i in index]
        steps = _median([r["steps"] for r in final])
        methods.append(
            {
                "method": name,
                "runs": len(own),
                **group_metrics(labels, dist, index),
                "median_steps": steps,
                "best_steps": min((r["steps"] for r in final), default=None),
                "relative_steps": (
                    None if steps is None or not best else round(steps / best, 3)
                ),
                "first_steps": _median([r["first_steps"] for r in final]),
                "policy_time_s": _median([r.get("policy_time_s") for r in final]),
                "solve_rate": _median([r.get("solve_rate") for r in own]),
                "finals": dict(Counter(r["sid"] for r in final)),
            }
        )
    between = np.full((len(method_names), len(method_names)), np.nan)
    for a, name_a in enumerate(method_names):
        for b, name_b in enumerate(method_names):
            ia = [i for i, r in enumerate(solved_runs) if r["method"] == name_a]
            ib = [i for i, r in enumerate(solved_runs) if r["method"] == name_b]
            block = dist[np.ix_(ia, ib)]
            if a == b:
                block = block[np.triu_indices(len(ia), 1)]
            if block.size:
                between[a, b] = block.mean()

    step_dt = next(
        (round(r["seconds"] / r["steps"], 4) for r in solved_runs if r["steps"]), None
    )
    return {
        "instance": instance,
        "level": opts.level,
        "group": group,
        "tau": tau,
        "definition": dataclasses.asdict(defn),
        "definition_text": defn.text(group, tau),
        "step_seconds": step_dt,
        "runs": runs,
        "solved": [r["run_id"] for r in solved_runs],
        "distance": [[round(float(v), 3) for v in row] for row in dist],
        "overall": {
            **group_metrics(labels, dist, list(range(n))),
            "runs": len(runs),
            "median_steps": _median([r["steps"] for r in solved_runs]),
            "best_steps": best,
            **within_between(dist, [r["method"] for r in solved_runs]),
        },
        "methods": methods,
        "method_distance": [
            [None if math.isnan(v) else round(float(v), 3) for v in row]
            for row in between
        ],
        "strategies": strategies,
    }


def summary(detail: dict[str, Any]) -> dict[str, Any]:
    """The part of ``scenario_diversity`` that the overview of all scenarios needs."""
    return {
        "overall": detail["overall"],
        "methods": {
            m["method"]: {k: v for k, v in m.items() if k not in ("method", "finals")}
            for m in detail["methods"]
        },
        "step_seconds": detail["step_seconds"],
    }
