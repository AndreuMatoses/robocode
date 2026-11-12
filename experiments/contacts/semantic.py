"""Semantic tags of the final policies, and how well a distance agrees with them.

A file ``semantic/<scenario>.json`` holds a **codebook** and the **labels** of the
runs of one scenario. The codebook is a list of tag groups. A group is one question
about the solution ("how does the arm swing?") with a closed set of values, and the
rule that assigns a value. A run has one value in each group, so a strategy is a
combination of values, and runs can be grouped on any subset of the groups.

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


def main() -> None:
    """Build the tag files of the given scenarios."""
    import sys  # pylint: disable=import-outside-toplevel

    for name in sys.argv[1:]:
        data = build(name)
        print(f"{name}: {len(data['labels'])} runs")
        for group in data["codebook"]:
            counts = Counter(tags[group["key"]] for tags in data["labels"].values())
            row = ", ".join(f"{v} {counts.get(v, 0)}" for v in group["values"])
            print(f"  {group['key']:14s} {row}")


if __name__ == "__main__":
    main()
