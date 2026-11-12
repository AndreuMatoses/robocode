"""Three views of the final policies of a scenario, side by side.

* **contact**: what touches what (``strategy.py``, ``metrics.py``);
* **motion**: where the bodies go (``motion.py``);
* **semantic**: what a reader says about the solution (``semantic.py``).

``scenario_views`` puts the distances of the three views on the same runs, so that
they can be compared: a 2-D map of each distance, the agreement of each distance
with each tag group, and the paths of the bodies.

Command line: ``python -m experiments.contacts.views <scenario> ...`` prints the
agreement table of each scenario.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from experiments.contacts import catalog, metrics, motion, semantic, strategy
from experiments.contacts.diversity import mds

PATH_POINTS = 100  # states of a path that the viewer draws
CONTACT = ("warp", "edit")  # the contact distances that are always shown


def effective(labels: list[Any]) -> float:
    """2 to the power of the entropy of the label frequencies."""
    total = len(labels)
    shares = [count / total for count in Counter(labels).values()]
    return float(2 ** -sum(p * math.log2(p) for p in shares)) if total else 0.0


def clusters(dist: np.ndarray, tau: float) -> list[str]:
    """Strategy ids S1, S2, ... (largest first): complete linkage at ``tau``."""
    count = len(dist)
    if count < 2:
        return ["S1"] * count
    tree = linkage(squareform(dist, checks=False), "complete")
    labels = [int(v) for v in fcluster(tree, tau + 1e-9, "distance")]
    ranked = [label for label, _ in Counter(labels).most_common()]
    return [f"S{ranked.index(label) + 1}" for label in labels]


def _paths(motions: list[motion.Motion]) -> dict[str, Any]:
    names = sorted(set.intersection(*(set(m.channels) for m in motions)))
    order = [n for n in ("ee", "base") if n in names]
    order += [n for n in names if n not in order]
    runs = []
    for item in motions:
        index = np.unique(
            np.linspace(0, item.steps - 1, PATH_POINTS).round().astype(int)
        )
        runs.append(
            {
                "t": [round(float(i * item.step_seconds), 3) for i in index],
                "ch": {n: np.round(item.channels[n][index], 3).tolist() for n in order},
            }
        )
    return {
        "channels": order,
        "dim": int(motions[0].channels[order[0]].shape[1]),
        "runs": runs,
    }


def _motion(name: str, traces: Path, dirs: list[Path], bodies: str) -> dict[str, Any]:
    """``motion.distance_matrices`` with a cache file for the matrices.

    The file is used again while the traces and ``motion.py`` do not change.
    """
    stamp = [(str(d), (d / "trace.npz").stat().st_mtime_ns) for d in dirs]
    stamp.append((motion.__file__, Path(motion.__file__).stat().st_mtime_ns))
    digest = hashlib.sha1(json.dumps([stamp, bodies]).encode()).hexdigest()[:16]
    path = traces / catalog.SCENARIO_DIR / name / f"motion_{bodies}.npz"
    if path.exists():
        with np.load(path) as data:
            if str(data["digest"]) == digest:
                motion.ensure(dirs)
                return {
                    "metres": {key: data[key] for key in motion.METRICS},
                    "scale": float(data["scale"]),
                    "motions": [motion.load(d) for d in dirs],
                }
    found = motion.distance_matrices(dirs, bodies)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, digest=digest, scale=found["scale"], **found["metres"])
    return found


def _classes(matrix: np.ndarray) -> dict[str, Any]:
    """Number of strategies and the largest one at 101 thresholds (complete linkage)."""
    top = float(matrix.max()) if len(matrix) else 0.0
    taus = [round(top * i / 100, 5) for i in range(101)]
    if len(matrix) < 2 or top == 0.0:
        return {"taus": taus, "classes": [[1, len(matrix)] for _ in taus]}
    tree = linkage(squareform(matrix, checks=False), "complete")
    classes = []
    for tau in taus:
        labels = fcluster(tree, tau + 1e-9, "distance")
        classes.append([int(labels.max()), int(np.bincount(labels).max())])
    return {"taus": taus, "classes": classes}


def _phases(
    directory: Path, opts: strategy.Options, defn: strategy.Definition
) -> tuple[list[strategy.Mode], list[tuple[float, float]]]:
    """The contact modes of a full episode and their time spans in seconds.

    The names follow the definition (scope, identity); equal neighbours are merged.
    The modes after the last robot contact are kept (the flight of a thrown cube).
    """
    names = dataclasses.replace(defn, order="sequence")
    kinds = strategy.entity_kinds(directory)
    modes: list[strategy.Mode] = []
    spans: list[tuple[float, float]] = []
    for mode, start, end in strategy.mode_sequence(directory, defn.options(opts)):
        new = strategy.abstract((mode,), names, kinds)[0]
        if modes and modes[-1] == new:
            spans[-1] = (spans[-1][0], end)
        else:
            modes.append(new)
            spans.append((start, end))
    return modes, spans


def events_matrix(
    dirs: list[Path],
    motions: list[motion.Motion],
    opts: strategy.Options,
    defn: strategy.Definition,
    cache: Path | None = None,
) -> np.ndarray:
    """The event-aligned motion distance between every two traces, in metres.

    ``cache`` is a file for the matrix; it is used again while the traces, the code
    and the definition do not change.
    """
    stamp = [(str(d), (d / "trace.npz").stat().st_mtime_ns) for d in dirs]
    stamp += [
        (f, Path(f).stat().st_mtime_ns) for f in (motion.__file__, strategy.__file__)
    ]
    stamp += [repr(defn), repr(opts), sorted(motions[0].channels) if motions else []]
    digest = hashlib.sha1(json.dumps(stamp).encode()).hexdigest()[:16]
    if cache is not None and cache.exists():
        with np.load(cache) as data:
            if str(data["digest"]) == digest:
                return data["matrix"]
    out = _events_matrix(dirs, motions, opts, defn)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, digest=digest, matrix=out)
    return out


def _events_matrix(
    dirs: list[Path],
    motions: list[motion.Motion],
    opts: strategy.Options,
    defn: strategy.Definition,
) -> np.ndarray:
    phases = [_phases(d, opts, defn) for d in dirs]
    count = len(dirs)
    out = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            links = strategy.warp_links(phases[i][0], phases[j][0])
            out[i, j] = out[j, i] = motion.events_distance(
                motions[i], motions[j], phases[i][1], phases[j][1], links
            )
    return out


def _entry(key: str, view: str, info: dict[str, str], matrix: np.ndarray, unit: str):
    upper = np.triu_indices(len(matrix), 1)
    return {
        "key": key,
        "view": view,
        "unit": unit,
        **info,
        "matrix": [[round(float(v), 4) for v in row] for row in matrix],
        "mean": round(float(matrix[upper].mean()), 4) if len(matrix) > 1 else None,
        "xy": [[round(float(c), 4) for c in row] for row in mds(matrix)],
        **_classes(matrix),
    }


def scenario_views(
    name: str,
    traces: Path,
    run_ids: list[str],
    instance: str,
    opts: strategy.Options,
    load_versions: Callable[[str], list[dict[str, Any]]],
    defn: strategy.Definition = strategy.Definition(),
    tau: float = 0.25,
    bodies: str = "all",
) -> dict[str, Any]:
    """The distances of the three views between the final policies that solve."""
    # pylint: disable=too-many-locals
    runs = metrics._final_solved(  # pylint: disable=protected-access
        traces, run_ids, instance, load_versions
    )
    dirs = [run.pop("dir") for run in runs]
    ids = [run["run_id"] for run in runs]
    count = len(runs)
    out: dict[str, Any] = {
        "name": name,
        "instance": instance,
        "runs": runs,
        "bodies": bodies,
        "bodies_text": motion.BODIES,
        "distances": [],
    }
    if count < 3:
        return out

    # contact view
    which = tuple(dict.fromkeys((*CONTACT, defn.metric)))
    keys, contact = metrics.distance_matrices(dirs, opts, defn, which)
    matrices: dict[str, np.ndarray] = {}
    for key in which:
        matrices["contact:" + key] = contact[key]
        info = {k: metrics.METRICS[key][k] for k in ("name", "text", "sees", "ignores")}
        out["distances"].append(
            _entry("contact:" + key, "contact", info, contact[key], "")
        )
    sids = clusters(contact[defn.metric], tau)
    for run, key, sid in zip(runs, keys, sids):
        run["modes"], run["sid"] = len(key), sid
    out["contact"] = {
        "metric": defn.metric,
        "tau": tau,
        "strategies": len(set(sids)),
        "effective": round(effective(sids), 2),
    }

    # motion view
    try:
        found = _motion(name, traces, dirs, bodies)
    except NotImplementedError as exc:
        out["motion_note"] = str(exc)
    else:
        out["scale"] = round(found["scale"], 3)
        out["options"] = motion.OPTIONS
        out["paths"] = _paths(found["motions"])
        channels = out["paths"]["channels"]
        out["bodies_available"] = [
            b
            for b in motion.BODIES
            if any(
                motion._in_bodies(c, b) for c in channels
            )  # pylint: disable=protected-access
        ]
        part = [m.select(bodies) for m in found["motions"]]
        mark = hashlib.sha1(repr((defn, opts)).encode()).hexdigest()[:10]
        cache = traces / catalog.SCENARIO_DIR / name / f"events_{bodies}_{mark}.npz"
        metres = {
            **found["metres"],
            "events": events_matrix(dirs, part, opts, defn, cache),
        }
        infos = {**motion.METRICS, "events": motion.EVENTS}
        for key, matrix in metres.items():
            matrices["motion:" + key] = matrix
            out["distances"].append(
                _entry("motion:" + key, "motion", infos[key], matrix, "m")
            )

    # semantic view
    tags = semantic.load(name)
    if tags is not None:
        labelled = [run_id for run_id in ids if run_id in tags["labels"]]
        for run in runs:
            run["tags"] = tags["labels"].get(run["run_id"])
            run["evidence"] = tags.get("evidence", {}).get(run["run_id"])
        out["semantic"] = {
            "codebook": tags["codebook"],
            "provenance": tags["provenance"],
            "insights": tags.get("insights", []),
            "labelled": len(labelled),
        }
        if len(labelled) == count:
            matrix = semantic.tag_distance(tags, ids)
            info = {
                "name": "tag distance",
                "text": "the share of the tag groups in which the two runs have "
                "different values",
                "sees": "what the tag groups ask",
                "ignores": "all other differences",
            }
            matrices["semantic:tags"] = matrix
            out["distances"].append(
                _entry("semantic:tags", "semantic", info, matrix, "")
            )
        out["separation"] = {
            group["key"]: {
                key: semantic.separation(
                    matrix, semantic.values(tags, ids, group["key"])
                )
                for key, matrix in matrices.items()
                if key != "semantic:tags"
            }
            for group in tags["codebook"]
        }

    upper = np.triu_indices(count, 1)
    rank: dict[str, dict[str, float | None]] = {}
    for a, x in matrices.items():
        rank[a] = {}
        for b, y in matrices.items():
            flat = np.ptp(x[upper]) == 0 or np.ptp(y[upper]) == 0
            rank[a][b] = (
                None if flat else round(float(spearmanr(x[upper], y[upper])[0]), 3)
            )
    out["spearman"] = rank
    return out


def summary(detail: dict[str, Any]) -> dict[str, Any]:
    """The part of ``scenario_views`` that the overview of all scenarios needs."""
    runs = detail["runs"]
    out: dict[str, Any] = {"name": detail["name"], "runs": len(runs)}
    if "contact" not in detail:
        return out
    out["contact"] = detail["contact"]
    means = {d["key"]: d["mean"] for d in detail["distances"]}
    if "scale" in detail:
        out["motion_share"] = round(means["motion:warp"] / detail["scale"], 4)
    if "semantic" not in detail:
        return out
    book = detail["semantic"]["codebook"]
    main = book[0]["key"]
    counts = Counter(run["tags"][main] for run in runs if run.get("tags"))
    both = [
        " / ".join(run["tags"][g["key"]] for g in book[:2])
        for run in runs
        if run.get("tags")
    ]
    out["main"] = {
        "key": main,
        "question": book[0]["question"],
        "values": {v: counts.get(v, 0) for v in book[0]["values"]},
        "effective": round(effective([run["tags"][main] for run in runs]), 2),
    }
    out["semantic"] = {
        "groups": [g["key"] for g in book[:2]],
        "strategies": len(set(both)),
        "effective": round(effective(both), 2),
    }
    views = {
        "contact": "contact:warp",
        "motion": "motion:warp",
        "events": "motion:events",
    }
    seen: dict[str, Any] = {}
    for view, key in views.items():
        if key not in means:
            continue
        cells = {g["key"]: detail["separation"][g["key"]][key]["auc"] for g in book}
        known = [v for v in cells.values() if v is not None]
        seen[view] = {
            "main": cells[main],
            "mean": round(sum(known) / len(known), 3) if known else None,
            "groups": cells,
        }
    out["sees"] = seen
    out["sources"] = {g["key"]: g["source"] for g in book}
    return out


def main() -> None:
    """Print the agreement of each distance with each tag group."""
    # pylint: disable=import-outside-toplevel
    import sys

    from experiments.contacts import viewer

    for name in sys.argv[1:]:
        detail = viewer.views(name)
        runs = detail["runs"]
        print(f"\n{name}: {len(runs)} final policies solve {detail['instance']}")
        if "contact" in detail:
            c = detail["contact"]
            print(
                f"  contact strategies ({c['metric']} at {c['tau']}): "
                f"{c['strategies']}, effective number {c['effective']}"
            )
        if "motion_note" in detail:
            print(f"  motion: {detail['motion_note']}")
        dists = [d for d in detail["distances"] if d["key"] != "semantic:tags"]
        for d in dists:
            unit = f" {d['unit']}" if d["unit"] else ""
            print(f"  mean {d['key']:16s} {d['mean']:.3f}{unit}")
        if "separation" not in detail:
            print("  no semantic tags")
            continue
        print(f"  {'tag group':12s} " + " ".join(f"{d['key'][-9:]:>9s}" for d in dists))
        for group in detail["semantic"]["codebook"]:
            row = detail["separation"][group["key"]]
            cells = [row[d["key"]]["auc"] for d in dists]
            print(
                f"  {group['key']:12s} "
                + " ".join(
                    f"{'–' if c is None else format(c, '.2f'):>9s}" for c in cells
                )
            )


if __name__ == "__main__":
    main()
