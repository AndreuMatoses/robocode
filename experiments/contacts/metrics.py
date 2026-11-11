"""Several distances between two contact strategies, to compare and to explain them.

All distances are in [0, 1] and are computed on the same input: the contact modes
of an episode up to the robot's last contact, with the contact pairs chosen and
renamed by the strategy definition (``strategy.Definition.scope`` and
``.identity``). Time is the fraction of that span, so a slow and a fast episode
of the same strategy compare as equal.

``METRICS`` lists them with what each one sees and ignores. ``edit`` is the
distance that the strategy classes of the other pages use.

* ``set``: Jaccard distance between the sets of contact pairs.
* ``edit``: edit distance between the mode sequences; replacing a mode costs the
  Jaccard distance between the two modes (``strategy.edit_distance``). This one
  also follows ``Definition.order``.
* ``duty``: each pair has a duty factor, the fraction of the time that it is in
  contact. The distance is ``1 - sum(min) / sum(max)`` over the pairs (weighted
  Jaccard). The duty factor per contact is a common behaviour descriptor in the
  quality-diversity literature for legged robots.
* ``overlap``: ``1 - (time both in contact) / (time either in contact)``, summed over
  the pairs, with the two timelines laid on each other (temporal IoU).
* ``dtw``: dynamic time warping between the two timelines sampled at ``FRAMES``
  instants; the cost of matching two instants is the Jaccard distance between their
  modes, and the distance is the mean cost along the best warping path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from experiments.contacts import histories, strategy
from experiments.contacts.commits import versions_path
from experiments.contacts.trace import load_meta

FRAMES = 64  # instants per episode for the time-warped distance
TAUS = [round(0.01 * i, 2) for i in range(101)]

METRICS: dict[str, dict[str, str]] = {
    "edit": {
        "name": "mode edit distance",
        "text": "Fewest edits that turn one sequence of contact modes into the "
        "other, divided by the length of the longer sequence. To add or remove "
        "a mode costs 1; to replace a mode costs the share of contact pairs that "
        "the two modes do not have in common.",
        "sees": "which contacts, in which combinations, in which order",
        "ignores": "how long each mode lasts",
    },
    "warp": {
        "name": "elastic mode distance",
        "text": "The two sequences of contact modes are aligned in order, and one "
        "mode can match more than one mode of the other sequence. A matched pair "
        "of modes costs the share of contact pairs that the two modes do not have "
        "in common. The distance is the mean cost along the best alignment.",
        "sees": "which contacts, in which combinations, in which order",
        "ignores": "how long each mode lasts; a short mode that only adds or "
        "removes one contact pair costs little",
    },
    "set": {
        "name": "pair set (Jaccard)",
        "text": "Share of the contact pairs used by one of the two episodes that "
        "the other episode does not use.",
        "sees": "which contacts occur",
        "ignores": "order, combinations, repeats and time",
    },
    "duty": {
        "name": "contact duty",
        "text": "Each contact pair has a duty factor: the fraction of the episode "
        "that it is in contact. The distance is 1 − (sum of the smaller duty "
        "factors) ÷ (sum of the larger ones).",
        "sees": "which contacts occur and for what share of the time",
        "ignores": "order and when in the episode",
    },
    "overlap": {
        "name": "timeline overlap",
        "text": "The two timelines are laid on each other (time as a fraction of "
        "the episode). The distance is 1 − (time that a pair is in contact in "
        "both) ÷ (time that it is in contact in one or both).",
        "sees": "which contacts occur and when",
        "ignores": "nothing in the timeline; a small shift in time counts",
    },
    "dtw": {
        "name": "time-warped mode distance",
        "text": "Dynamic time warping of the two timelines: time is stretched "
        "locally so that similar contact modes meet. The distance is the mean "
        "share of contact pairs not in common between the matched instants.",
        "sees": "which contacts, in which order, and roughly for how long",
        "ignores": "shifts and changes of speed",
    },
}

Timed = list[tuple[strategy.Mode, float, float]]


def timed_modes(
    directory: Path, opts: strategy.Options, defn: strategy.Definition
) -> Timed:
    """The strategy modes of a trace with their span as fractions of the strategy.

    Scope and identity of ``defn`` are applied; equal neighbours are merged.
    """
    opts = defn.options(opts)
    count = strategy.strategy_length(directory, opts)
    modes = strategy.mode_sequence(directory, opts)[:count]
    if not modes:
        return []
    names = dataclasses.replace(defn, order="sequence")
    kinds = strategy.entity_kinds(directory)
    total = modes[-1][2] or 1.0
    out: list[list[Any]] = []
    for mode, start, end in modes:
        new = strategy.abstract((mode,), names, kinds)[0]
        if out and out[-1][0] == new:
            out[-1][2] = end / total
        else:
            out.append([new, start / total, end / total])
    return [(m, s, e) for m, s, e in out]


def pair_spans(timed: Timed) -> dict[str, list[list[float]]]:
    """The contact intervals of each pair, as fractions of the strategy."""
    spans: dict[str, list[list[float]]] = {}
    for mode, start, end in timed:
        for pair in mode:
            rows = spans.setdefault(pair, [])
            if rows and abs(rows[-1][1] - start) < 1e-9:
                rows[-1][1] = end
            else:
                rows.append([start, end])
    return spans


def _length(rows: list[list[float]]) -> float:
    return sum(end - start for start, end in rows)


def _common(a: list[list[float]], b: list[list[float]]) -> float:
    return sum(
        max(0.0, min(e_a, e_b) - max(s_a, s_b)) for s_a, e_a in a for s_b, e_b in b
    )


def pair_table(
    spans_a: dict[str, Any], spans_b: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per contact pair: both interval lists, duty factors and shared time."""
    rows = []
    for pair in sorted(set(spans_a) | set(spans_b)):
        a, b = spans_a.get(pair, []), spans_b.get(pair, [])
        both = _common(a, b)
        rows.append(
            {
                "pair": pair,
                "a": a,
                "b": b,
                "duty_a": _length(a),
                "duty_b": _length(b),
                "both": both,
                "either": _length(a) + _length(b) - both,
            }
        )
    return rows


def _ratio(shared: float, total: float) -> float:
    return 1.0 - shared / total if total > 1e-12 else 0.0


def duty_distance(table: list[dict[str, Any]]) -> float:
    """Weighted Jaccard distance between the duty factors of the pairs."""
    return _ratio(
        sum(min(r["duty_a"], r["duty_b"]) for r in table),
        sum(max(r["duty_a"], r["duty_b"]) for r in table),
    )


def overlap_distance(table: list[dict[str, Any]]) -> float:
    """One minus the temporal intersection over union, summed over the pairs."""
    return _ratio(sum(r["both"] for r in table), sum(r["either"] for r in table))


def frames(timed: Timed) -> list[strategy.Mode]:
    """The mode at ``FRAMES`` evenly spaced instants."""
    out, i = [], 0
    for k in range(FRAMES):
        time = (k + 0.5) / FRAMES
        while i < len(timed) - 1 and timed[i][2] <= time:
            i += 1
        out.append(timed[i][0])
    return out


def dtw_distance(cost: list[list[float]]) -> float:
    """Mean cost along the best warping path through ``cost`` (steps →, ↓, ↘)."""
    rows, cols = len(cost), len(cost[0])
    inf = float("inf")
    total = [[inf] * (cols + 1) for _ in range(rows + 1)]
    steps = [[0] * (cols + 1) for _ in range(rows + 1)]
    total[0][0] = 0.0
    for i in range(1, rows + 1):
        row, above, count, count_above, line = (
            total[i],
            total[i - 1],
            steps[i],
            steps[i - 1],
            cost[i - 1],
        )
        for k in range(1, cols + 1):
            best, length = above[k - 1], count_above[k - 1]
            if above[k] < best:
                best, length = above[k], count_above[k]
            if row[k - 1] < best:
                best, length = row[k - 1], count[k - 1]
            row[k] = best + line[k - 1]
            count[k] = length + 1
    return total[rows][cols] / steps[rows][cols]


def alignment(a: list[strategy.Mode], b: list[strategy.Mode]) -> list[dict[str, Any]]:
    """The edits behind ``strategy.edit_distance``: one row per aligned column.

    ``op`` is ``same``, ``replace``, ``remove`` (only in a) or ``add`` (only in b);
    ``i`` and ``j`` are the positions in the two sequences.
    """
    table = np.zeros((len(a) + 1, len(b) + 1))
    table[:, 0], table[0, :] = np.arange(len(a) + 1), np.arange(len(b) + 1)
    for i, mode_a in enumerate(a, 1):
        for j, mode_b in enumerate(b, 1):
            table[i, j] = min(
                table[i - 1, j] + 1,
                table[i, j - 1] + 1,
                table[i - 1, j - 1] + strategy.jaccard(mode_a, mode_b),
            )
    out, i, j = [], len(a), len(b)
    while i or j:
        if i and j:
            cost = strategy.jaccard(a[i - 1], b[j - 1])
            if abs(table[i, j] - table[i - 1, j - 1] - cost) < 1e-9:
                op = "same" if cost == 0 else "replace"
                out.append({"op": op, "i": i - 1, "j": j - 1, "cost": round(cost, 3)})
                i, j = i - 1, j - 1
                continue
        if i and abs(table[i, j] - table[i - 1, j] - 1) < 1e-9:
            out.append({"op": "remove", "i": i - 1, "j": None, "cost": 1.0})
            i -= 1
        else:
            out.append({"op": "add", "i": None, "j": j - 1, "cost": 1.0})
            j -= 1
    return out[::-1]


def _final_solved(
    traces: Path,
    run_ids: list[str],
    instance: str,
    load_versions: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """The final policy of every run that solves the instance (as ``diversity``)."""
    runs = []
    for run_id in run_ids:
        if not versions_path(traces, run_id).exists():
            continue
        rows = histories.run_history(traces, run_id, load_versions(run_id), instance)
        if not rows or rows[-1]["state"] != "solved":
            continue
        last = rows[-1]
        runs.append(
            {
                "run_id": run_id,
                "method": run_id.split("/")[0],
                "seed": run_id.rsplit("/", 1)[-1],
                "trace": last["trace"],
                "dir": last["dir"],
                "steps": load_meta(last["dir"])["metrics"].get("num_steps"),
            }
        )
    return runs


def distance_matrices(
    dirs: list[Path],
    opts: strategy.Options,
    defn: strategy.Definition = strategy.Definition(),
    which: tuple[str, ...] | None = None,
) -> tuple[list[tuple[strategy.Mode, ...]], dict[str, np.ndarray]]:
    """The strategy key of each trace and the distance matrix of each metric."""
    which = tuple(METRICS) if which is None else which
    n = len(dirs)
    keys = [strategy.strategy_key(d, opts, defn) for d in dirs]
    timed = (
        [timed_modes(d, opts, defn) for d in dirs]
        if set(which) - {"edit", "warp"}
        else []
    )
    spans = [pair_spans(t) for t in timed]
    used = [frozenset(s) for s in spans]
    sampled = [frames(t) if t else [] for t in timed] if "dtw" in which else []
    # the Jaccard distance between every two modes that occur at a sampled instant
    modes = list(dict.fromkeys(m for row in sampled for m in row))
    index = {m: i for i, m in enumerate(modes)}
    between = [[strategy.jaccard(a, b) for b in modes] for a in modes]
    ids = [[index[m] for m in row] for row in sampled] or [[] for _ in dirs]

    dist = {key: np.zeros((n, n)) for key in which}
    # the two distances on the keys: every two distinct keys at one time
    distinct = list(dict.fromkeys(keys))
    place = np.array([distinct.index(key) for key in keys], dtype=np.int64)
    for key in which:
        if key in ("edit", "warp") and n:
            dist[key] = strategy.key_distance_matrix(distinct, key)[
                np.ix_(place, place)
            ]
    timed_keys = [key for key in which if key not in ("edit", "warp")]
    for i in range(n if timed_keys else 0):
        for k in range(i + 1, n):
            table = pair_table(spans[i], spans[k]) if spans else []
            for key in timed_keys:
                if key == "set":
                    value = strategy.jaccard(used[i], used[k])
                elif key == "duty":
                    value = duty_distance(table)
                elif key == "overlap":
                    value = overlap_distance(table)
                elif ids[i] and ids[k]:
                    value = dtw_distance(
                        [[between[a][b] for b in ids[k]] for a in ids[i]]
                    )
                else:
                    value = 0.0 if not ids[i] and not ids[k] else 1.0
                dist[key][i, k] = dist[key][k, i] = value
    return keys, dist


def scenario_metrics(
    traces: Path,
    run_ids: list[str],
    instance: str,
    opts: strategy.Options,
    load_versions: Callable[[str], list[dict[str, Any]]],
    defn: strategy.Definition = strategy.Definition(),
) -> dict[str, Any]:
    """Every distance between every two final policies that solve the instance."""
    runs = _final_solved(traces, run_ids, instance, load_versions)
    n = len(runs)
    keys, dist = distance_matrices([r["dir"] for r in runs], opts, defn)

    upper = np.triu_indices(n, 1)
    rank: dict[str, dict[str, float | None]] = {a: {} for a in METRICS}
    for a in METRICS:
        for b in METRICS:
            x, y = dist[a][upper], dist[b][upper]
            if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
                rank[a][b] = None
            else:
                rank[a][b] = round(float(spearmanr(x, y)[0]), 3)
    classes: dict[str, list[list[int]]] = {}
    for key in METRICS:
        if n < 2:
            classes[key] = [[n, n] for _ in TAUS]
            continue
        tree = linkage(squareform(dist[key], checks=False), "complete")
        classes[key] = []
        for tau in TAUS:
            labels = fcluster(tree, tau + 1e-9, "distance")
            classes[key].append([int(labels.max()), int(np.bincount(labels).max())])
    for run, key in zip(runs, keys):
        run.pop("dir")
        run["modes"] = len(key)
    return {
        "instance": instance,
        "definition_text": defn.text("exact").rsplit(" Two episodes", 1)[0],
        "metrics": [{"key": key, **value} for key, value in METRICS.items()],
        "runs": runs,
        "distance": {
            key: [[round(float(v), 3) for v in row] for row in matrix]
            for key, matrix in dist.items()
        },
        "mean": {
            key: round(float(matrix[upper].mean()), 3) if n > 1 else None
            for key, matrix in dist.items()
        },
        "spearman": rank,
        "taus": TAUS,
        "classes": classes,
    }


def pair_detail(
    dir_a: Path,
    dir_b: Path,
    opts: strategy.Options,
    defn: strategy.Definition = strategy.Definition(),
) -> dict[str, Any]:
    """What is behind each distance between two traces."""
    opts = defn.options(opts)
    key_a = list(strategy.strategy_key(dir_a, opts, defn))
    key_b = list(strategy.strategy_key(dir_b, opts, defn))
    timed_a, timed_b = timed_modes(dir_a, opts, defn), timed_modes(dir_b, opts, defn)
    table = pair_table(pair_spans(timed_a), pair_spans(timed_b))
    edits = alignment(key_a, key_b)
    if timed_a and timed_b:
        frames_a, frames_b = frames(timed_a), frames(timed_b)
        warped = dtw_distance(
            [[strategy.jaccard(a, b) for b in frames_b] for a in frames_a]
        )
    else:
        warped = 0.0 if not timed_a and not timed_b else 1.0

    def side(directory: Path, key: list[strategy.Mode], timed: Timed) -> dict[str, Any]:
        meta = load_meta(directory)
        seconds = strategy.mode_sequence(directory, opts)
        count = strategy.strategy_length(directory, opts)
        return {
            "run_id": meta["run_id"],
            "steps": meta["metrics"].get("num_steps"),
            "seconds": seconds[count - 1][2] if count else 0.0,
            "key": [sorted(m) for m in key],
            "timeline": [
                {"pairs": sorted(m), "start": round(s, 4), "end": round(e, 4)}
                for m, s, e in timed
            ],
        }

    duty, overlap = duty_distance(table), overlap_distance(table)
    for row in table:
        for name in ("duty_a", "duty_b", "both", "either"):
            row[name] = round(row[name], 4)
        row["a"] = [[round(s, 4), round(e, 4)] for s, e in row["a"]]
        row["b"] = [[round(s, 4), round(e, 4)] for s, e in row["b"]]
    cost = sum(e["cost"] for e in edits)
    warp_cost, warp_steps = (
        strategy.warp_path(key_a, key_b) if key_a and key_b else (0, 0)
    )
    return {
        "a": side(dir_a, key_a, timed_a),
        "b": side(dir_b, key_b, timed_b),
        "pairs": table,
        "edits": edits,
        "edit_cost": round(cost, 3),
        "edit_length": max(len(key_a), len(key_b)),
        "warp_cost": round(warp_cost, 3),
        "warp_steps": warp_steps,
        "values": {
            "edit": round(strategy.edit_distance(key_a, key_b), 3),
            "warp": round(strategy.warp_distance(key_a, key_b), 3),
            "set": round(
                strategy.jaccard(
                    frozenset(r["pair"] for r in table if r["a"]),
                    frozenset(r["pair"] for r in table if r["b"]),
                ),
                3,
            ),
            "duty": round(duty, 3),
            "overlap": round(overlap, 3),
            "dtw": round(warped, 3),
        },
    }
