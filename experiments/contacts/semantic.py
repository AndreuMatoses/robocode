"""Semantic tags of the final policies, and how well a distance agrees with them.

A file ``semantic/<scenario>.json`` holds a **codebook** and the **labels** of the
runs of one scenario. The codebook is a list of tag groups. A group is one question
about the solution ("how does the arm swing?") with a closed set of values, and the
rule that assigns a value. A run has one value in each group.

A group has a **role**. A *plan* group says what the robot does (the tool, the
transfer type, the order, the route). A *style* group says how it does it (pace,
heights, distances, side effects). A **semantic strategy** is the combination of the
values of the plan groups (``plan_keys``, ``strategy_of``): two runs with the same
plan and another style have one semantic strategy. A scenario with no plan group
has one semantic strategy.

The tags come from a reader (the file says who and when) that looked at the motion,
at the contacts and at the program, and then wrote a rule for each group. The rules
are code: ``tag_rules/<scenario>.py``. ``build`` runs them and writes the file. The
evidence of each run is stored next to its labels, so that a person can check a value.

Command line: ``python -m experiments.contacts.semantic <scenario> ...`` builds the
files and prints the number of runs with each value.

``separation`` measures how well a distance agrees with one tag group: the chance
that two runs with different values are farther apart than two runs with the same
value (0.5 = no relation, 1 = the distance separates the values perfectly).
"""

from __future__ import annotations

import datetime
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

LABELS = Path(__file__).resolve().parent / "semantic"


def load(scenario: str) -> dict[str, Any] | None:
    """The codebook and the labels of a scenario, if the file exists."""
    path = LABELS / f"{scenario}.json"
    return json.loads(path.read_text()) if path.exists() else None


def values(data: dict[str, Any], run_ids: list[str], group: str) -> list[str | None]:
    """The value of each run in a tag group (None for a run without labels)."""
    return [data["labels"].get(run, {}).get(group) for run in run_ids]


ROLES = ("plan", "style", "program")


def plan_keys(groups: list[dict[str, Any]]) -> list[str]:
    """The keys of the tag groups that define a semantic strategy."""
    return [group["key"] for group in groups if group.get("role") == "plan"]


def strategy_of(tags: dict[str, str], keys: list[str]) -> str:
    """The semantic strategy of a run: its values in the plan groups."""
    return " / ".join(tags[key] for key in keys) or "one plan"


def separation(distance: np.ndarray, tags: list[str | None]) -> dict[str, Any]:
    """Chance that a pair with different tags is farther apart than a pair with equal tags."""
    index = [i for i, tag in enumerate(tags) if tag is not None]
    same, other = [], []
    for a, i in enumerate(index):
        for j in index[a + 1 :]:
            (same if tags[i] == tags[j] else other).append(distance[i, j])
    if not same or not other:
        return {"auc": None, "same": len(same), "different": len(other)}
    ranks = rankdata(np.concatenate([other, same]))
    auc = (ranks[: len(other)].sum() - len(other) * (len(other) + 1) / 2) / (
        len(other) * len(same)
    )
    return {
        "auc": round(float(auc), 3),
        "same": len(same),
        "different": len(other),
        "mean_same": round(float(np.mean(same)), 4),
        "mean_different": round(float(np.mean(other)), 4),
    }


def tag_distance(data: dict[str, Any], run_ids: list[str]) -> np.ndarray:
    """The share of tag groups in which two runs have different values."""
    groups = [group["key"] for group in data["codebook"]]
    table = np.array([[data["labels"][run][g] for g in groups] for run in run_ids])
    return (table[:, None, :] != table[None, :, :]).mean(axis=-1)


READER = "Claude (claude-fable-5-2)"


def build(scenario: str, instance: str = "inst0") -> dict[str, Any]:
    """Run the rules of a scenario on its final policies and write the JSON file."""
    # pylint: disable=import-outside-toplevel
    from experiments.contacts import features

    rules = importlib.import_module(f"experiments.contacts.tag_rules.{scenario}")
    labels, evidence = {}, {}
    for run in features.load_runs(scenario, instance):
        labels[run.run_id], evidence[run.run_id] = rules.label(run)
    for group in rules.CODEBOOK:
        if group.get("role") not in ROLES:
            raise ValueError(
                f"{scenario}/{group['key']}: no role (plan, style, program)"
            )
        used = {tags[group["key"]] for tags in labels.values()}
        if not used <= set(group["values"]):
            raise ValueError(f"{scenario}/{group['key']}: unknown values {used}")
    data = {
        "scenario": scenario,
        "instance": instance,
        "population": "final policy of each run that solves the instance",
        "provenance": {
            "reader": READER,
            "date": datetime.date.today().isoformat(),
            "status": "not checked by a person",
            "method": rules.METHOD,
            "rules": f"experiments/contacts/tag_rules/{scenario}.py",
        },
        "insights": list(getattr(rules, "INSIGHTS", [])),
        "codebook": rules.CODEBOOK,
        "labels": labels,
        "evidence": evidence,
    }
    (LABELS / f"{scenario}.json").write_text(json.dumps(data, indent=1) + "\n")
    return data


HISTORY = LABELS / "history"


def build_history(scenario: str, instance: str = "inst0") -> dict[str, Any]:
    """Run the rules on every solved commit of every run; write the history file.

    A row is one commit: ``commit``, ``elapsed`` (seconds), ``state``, ``version``
    (present if the commit changes the policy), ``tags`` (the values in the order of
    ``groups``; only for a commit that solves the instance) and ``trace``. A rule
    reads the episode only, so the tag ``aim`` of tossing3d (from the program text)
    is the value of the final program in every commit. A rule that fails on an
    episode gives no tags and the text of the error.
    """
    # pylint: disable=import-outside-toplevel
    from experiments.contacts import features

    rules = importlib.import_module(f"experiments.contacts.tag_rules.{scenario}")
    keys = [group["key"] for group in rules.CODEBOOK]
    runs: dict[str, list[dict[str, Any]]] = {}
    done: dict[int, dict[str, Any]] = {}
    for run_id, rows in features.load_history(scenario, instance).items():
        runs[run_id] = []
        for row in rows:
            entry: dict[str, Any] = {
                "commit": row["commit"],
                "elapsed": row["elapsed"],
                "state": row["state"],
            }
            run = row["run"]
            if row["version"]:
                entry["version"] = True
            if run is not None:
                if id(run) not in done:
                    try:
                        tags = rules.label(run)[0]
                        done[id(run)] = {"tags": [tags[k] for k in keys]}
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        done[id(run)] = {"error": f"{type(exc).__name__}: {exc}"}
                entry.update(done[id(run)])
                if row["version"]:
                    entry["trace"] = row["trace"]
            runs[run_id].append(entry)
    data = {
        "scenario": scenario,
        "instance": instance,
        "provenance": {
            "reader": READER,
            "date": datetime.date.today().isoformat(),
            "status": "not checked by a person",
            "rules": f"experiments/contacts/tag_rules/{scenario}.py",
        },
        "groups": [
            {
                "key": g["key"],
                "question": g["question"],
                "values": list(g["values"]),
                "source": g["source"],
                "role": g["role"],
            }
            for g in rules.CODEBOOK
        ],
        "runs": runs,
    }
    HISTORY.mkdir(exist_ok=True)
    (HISTORY / f"{scenario}.json").write_text(json.dumps(data, indent=None) + "\n")
    return data


def load_history(scenario: str) -> dict[str, Any] | None:
    """The tags over the commits of a scenario, if the file exists."""
    path = HISTORY / f"{scenario}.json"
    return json.loads(path.read_text()) if path.exists() else None


def history_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    """For each tag group: how often its value changes over the commits of a run.

    Only the commits that solve the instance and have tags count. ``runs`` is the
    number of runs with two such policy versions or more; ``changed`` the runs with
    one change of the value or more; ``moved`` the runs whose first value differs
    from their last value; ``paths`` counts the pairs first value -> last value.
    """
    out = []
    for group in data["groups"]:
        key = group["key"]
        index = [g["key"] for g in data["groups"]].index(key)
        if group.get("source") == "program":  # read in the final program only
            out.append({"key": key, "runs": None})
            continue
        runs = changed = moved = switches = 0
        paths: Counter[str] = Counter()
        for rows in data["runs"].values():
            seq = [r["tags"][index] for r in rows if r.get("tags") and r.get("version")]
            if len(seq) < 2:
                continue
            count = sum(a != b for a, b in zip(seq, seq[1:]))
            runs += 1
            changed += count > 0
            moved += seq[0] != seq[-1]
            switches += count
            paths[f"{seq[0]} -> {seq[-1]}"] += 1
        out.append(
            {
                "key": key,
                "runs": runs,
                "changed": changed,
                "moved": moved,
                "switches": switches,
                "paths": dict(paths.most_common()),
            }
        )
    return out


def history_strategy(data: dict[str, Any]) -> dict[str, int]:
    """Runs whose semantic strategy at the first solved version is that of the last.

    ``runs``: runs with two solved policy versions or more; ``kept``: the first and
    the last strategy are equal; ``constant``: the strategy never changes.
    """
    index = [k for k, g in enumerate(data["groups"]) if g.get("role") == "plan"]
    runs = kept = constant = 0
    for rows in data["runs"].values():
        seq = [
            tuple(r["tags"][k] for k in index)
            for r in rows
            if r.get("tags") and r.get("version")
        ]
        if len(seq) < 2:
            continue
        runs += 1
        kept += seq[0] == seq[-1]
        constant += len(set(seq)) == 1
    return {"runs": runs, "kept": kept, "constant": constant}


SCENES = LABELS / "scenes"
STABLE = 0.9  # a tag group is a property of the policy from this repeatability on


def _variant(scenario: str) -> int | None:
    """The object count of start scene 0: the variant of the scenario that we study."""
    # pylint: disable=import-outside-toplevel
    from experiments.contacts import features
    from experiments.contacts.trace import load_meta

    runs = features.load_runs(scenario, "inst0")
    if not runs:
        return None
    return load_meta(runs[0].directory)["metrics"].get("object_count")


def build_scenes(scenario: str, instances: tuple[int, ...] = tuple(range(20))) -> dict:
    """Run the rules on the final policies on several start scenes; write the file.

    A start scene is one layout of the variant of start scene 0 (the same number of
    objects). In some scenarios the seed also changes the number of objects; there
    the layout was recorded with a fixed count (``record --count``), and that trace
    is used. ``scenes[inst<k>][<run id>]`` has the tags (in the order of ``groups``),
    the episode length and the number of movable objects of each final policy that
    solves the layout. A rule that fails on a layout gives no row.
    """
    # pylint: disable=import-outside-toplevel
    from experiments.contacts import features

    rules = importlib.import_module(f"experiments.contacts.tag_rules.{scenario}")
    keys = [group["key"] for group in rules.CODEBOOK]
    variant = _variant(scenario)
    scenes: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for k in instances:
        # the layout with the fixed count of the variant, if it was recorded; the
        # trace without a fixed count if it has the same objects and more runs
        names = [f"inst{k}"]
        if variant is not None:
            names.insert(0, f"inst{k}_count{variant}")
        found: list[tuple[str, dict[str, Any]]] = []
        for name in names:
            rows = {}
            try:
                runs = features.load_runs(scenario, name)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            for run in runs:
                try:
                    tags = rules.label(run)[0]
                except Exception:  # pylint: disable=broad-exception-caught
                    continue
                rows[run.run_id] = {
                    "tags": [tags[key] for key in keys],
                    "steps": run.steps,
                    "objects": len(run.objects),
                }
            if rows:
                found.append((name, rows))
        if not found:
            continue
        name, rows = found[0]
        if len(found) == 2:
            count = lambda r: Counter(v["objects"] for v in r.values()).most_common(1)[
                0
            ][0]
            if count(found[1][1]) == count(rows) and len(found[1][1]) > len(rows):
                name, rows = found[1]
        scenes[f"inst{k}"] = rows
        sources[f"inst{k}"] = name
    data = {
        "scenario": scenario,
        "variant": variant,
        "provenance": {
            "reader": READER,
            "date": datetime.date.today().isoformat(),
            "status": "not checked by a person; the rules were checked on inst0 only",
            "rules": f"experiments/contacts/tag_rules/{scenario}.py",
        },
        "groups": [
            {"key": g["key"], "role": g["role"], "source": g["source"]}
            for g in rules.CODEBOOK
        ],
        "traces": sources,
        "scenes": scenes,
    }
    SCENES.mkdir(exist_ok=True)
    (SCENES / f"{scenario}.json").write_text(json.dumps(data, indent=None) + "\n")
    return data


def load_scenes(scenario: str) -> dict[str, Any] | None:
    """The tags on several start scenes, if the file exists."""
    path = SCENES / f"{scenario}.json"
    return json.loads(path.read_text()) if path.exists() else None


def comparable_scenes(data: dict[str, Any]) -> list[str]:
    """The start scenes with the number of objects of the first one.

    Another number of objects is another task, so those scenes are left out.
    """
    order = sorted(data["scenes"], key=lambda scene: int(scene.removeprefix("inst")))
    if not order:
        return []

    def count(scene: str) -> int:
        values = Counter(row["objects"] for row in data["scenes"][scene].values())
        return values.most_common(1)[0][0]

    return [scene for scene in order if count(scene) == count(order[0])]


def stability(data: dict[str, Any]) -> list[dict[str, Any]]:
    """For each tag group: is its value a property of the policy, or of the scene?

    * ``repeat``: the chance that ONE policy has the same value on two start scenes;
    * ``agree``: the chance that TWO policies have the same value on one start scene.

    ``kind`` is "scene" (``repeat`` below ``agree``: the start scene sets the value
    and the policies follow it), "policy" (else, with ``repeat`` of ``STABLE`` or
    more: the value holds over the start scenes) or "mixed". A group that is read in the program text
    has no rule for an episode, so it gets no numbers.
    """
    scenes = comparable_scenes(data)
    out = []
    for index, group in enumerate(data["groups"]):
        row: dict[str, Any] = {"key": group["key"], "scenes": len(scenes)}
        out.append(row)
        if group.get("source") == "program" or len(scenes) < 2:
            row.update(repeat=None, agree=None, pairs=0, kind=None)
            continue
        same = pairs = 0
        for i, first in enumerate(scenes):
            for second in scenes[i + 1 :]:
                a, b = data["scenes"][first], data["scenes"][second]
                for run_id in set(a) & set(b):
                    pairs += 1
                    same += a[run_id]["tags"][index] == b[run_id]["tags"][index]
        equal = total = 0
        for scene in scenes:
            counts = Counter(r["tags"][index] for r in data["scenes"][scene].values())
            runs = sum(counts.values())
            equal += sum(v * (v - 1) for v in counts.values())
            total += runs * (runs - 1)
        repeat = same / pairs if pairs else None
        agree = equal / total if total else None
        kind = None
        if repeat is not None and agree is not None:
            # "scene" comes first: a value that all policies share on each layout is
            # not a choice of a policy, also if it changes little between the layouts
            kind = (
                "scene" if repeat < agree else "policy" if repeat >= STABLE else "mixed"
            )
        row.update(
            repeat=None if repeat is None else round(repeat, 3),
            agree=None if agree is None else round(agree, 3),
            pairs=pairs,
            kind=kind,
        )
    return out


def main() -> None:
    """Build the tag files of the given scenarios.

    ``--history``: the tags over the commits. ``--scenes``: the tags of the final
    policies on start scenes 0 to 19, and the repeatability of each tag group.
    """
    import sys  # pylint: disable=import-outside-toplevel

    if "--scenes" in sys.argv:
        for name in [a for a in sys.argv[1:] if a != "--scenes"]:
            data = build_scenes(name)
            counts = {scene: len(rows) for scene, rows in data["scenes"].items()}
            print(f"{name}: {counts}; comparable {comparable_scenes(data)}")
            for item, group in zip(stability(data), data["groups"]):
                print(
                    f"  {item['key']:14s} {group['role']:7s} repeat {item['repeat']}  "
                    f"agree {item['agree']}  {item['kind']}"
                )
        return
    if "--history" in sys.argv:
        for name in [a for a in sys.argv[1:] if a != "--history"]:
            data = build_history(name)
            rows = [r for rows in data["runs"].values() for r in rows]
            tagged = sum("tags" in r for r in rows)
            errors = sum("error" in r for r in rows)
            print(
                f"{name}: {len(data['runs'])} runs, {len(rows)} commits, {tagged} with tags, {errors} rule errors"
            )
            for item in history_summary(data):
                if item["runs"] is None:
                    continue
                print(
                    f"  {item['key']:14s} runs {item['runs']:2d}  changed {item['changed']:2d}  "
                    f"first != last {item['moved']:2d}  switches {item['switches']:3d}"
                )
        return
    for name in sys.argv[1:]:
        data = build(name)
        print(f"{name}: {len(data['labels'])} runs")
        for group in data["codebook"]:
            counts = Counter(tags[group["key"]] for tags in data["labels"].values())
            row = ", ".join(f"{v} {counts.get(v, 0)}" for v in group["values"])
            print(f"  {group['key']:14s} {group['role']:7s} {row}")


if __name__ == "__main__":
    main()
