"""What a policy does when the layout changes: a profile, and the rule behind it.

A **layout** is one set of start poses of a scenario (the same objects). The semantic
tags are made on each layout (``semantic.build_scenes``). ``semantic.stability`` says
which tag groups are a property of the policy. This module looks at the other groups:
the value of a policy changes with the layout, so the value is not a description of
the policy. Three descriptions are invariant to the layout:

* the **profile** of a policy in a tag group: its **logic**, one of

  - ``always <value>``: the same value on 90 % of its layouts or more;
  - ``follows the layout``: its value changes, and on each layout it is the value
    that most other policies have there (85 % of its layouts or more);
  - ``own rule``: its value changes, and not as the value of the other policies;

* the **rule class**: two policies with ``own rule`` are in one class if they have
  the same value on 85 % of their common layouts or more (complete linkage);

* for a tag group that asks which object the robot moves first, a **choice rule**:
  the simple rule on the start poses that predicts the object ("the nearest to the
  robot", "the one with the smallest x", "always cube1", ...). A policy has the rule
  if the rule is correct on 85 % of its layouts or more.

``python -m experiments.contacts.layouts <scenario ...>`` writes
``semantic/rules/<scenario>.json`` and prints a summary. It needs the tag files of
the layouts (``python -m experiments.contacts.semantic --scenes``).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from experiments.contacts import semantic

RULES = semantic.LABELS / "rules"
ALWAYS = 0.9  # share of the layouts with one value: a constant choice
FOLLOWS = 0.85  # share of the layouts with the value of the other policies
SAME_RULE = 0.85  # agreement of two policies that share a rule
LEAST = 10  # layouts that a policy must solve to get a profile
MOVED = 0.03  # [m] an object "moves" from this shift on
# tag groups that ask which object moves first: scenario -> (group, the objects)
CHOICES: dict[str, tuple[str, list[str]]] = {
    "sortclutteredblocks3d": ("first", ["cube1", "cube2", "cube3", "cube4"]),
    "table3d": ("cube", ["cube0", "cube1", "cube2", "cube3"]),
    "obstruction2d": ("order", ["obstruction0", "obstruction1"]),
    "obstruction3d": ("order", ["obstruction0", "obstruction1"]),
    "pr2packed": ("order", ["block0", "block1", "block2"]),
    "clutteredstorage2d": ("far_order", ["block3", "block4"]),
    "constrainedcupboard3d": ("order", ["cuboid_0", "cuboid_1", "cuboid_2"]),
    "packing3d": ("order", ["part0", "part1"]),
    "rearrange3d": ("order", ["boxed_drink_0", "can_0"]),
    "dynamicshelf3d": ("order", ["cube1", "cube2"]),
}


def matrix(data: dict[str, Any], index: int) -> tuple[list[str], list[str], list[list]]:
    """The value of each policy (rows) on each layout (columns); None if not solved."""
    scenes = semantic.comparable_scenes(data)
    runs = sorted({run for scene in scenes for run in data["scenes"][scene]})
    runs = [r for r in runs if sum(r in data["scenes"][s] for s in scenes) >= LEAST]
    values = [
        [
            data["scenes"][s][run]["tags"][index] if run in data["scenes"][s] else None
            for s in scenes
        ]
        for run in runs
    ]
    return runs, scenes, values


def agreement(values: list[list]) -> np.ndarray:
    """[policy, policy] share of their common layouts with the same value."""
    count = len(values)
    out = np.eye(count)
    for i in range(count):
        for j in range(i + 1, count):
            both = [
                a == b
                for a, b in zip(values[i], values[j])
                if a is not None and b is not None
            ]
            out[i, j] = out[j, i] = float(np.mean(both)) if len(both) >= LEAST else 0.0
    return out


def profiles(data: dict[str, Any], index: int) -> dict[str, Any]:
    """The logic of each policy in one tag group (see the module text)."""
    runs, scenes, values = matrix(data, index)
    out: dict[str, Any] = {}
    for p, run in enumerate(runs):
        own = [v for v in values[p] if v is not None]
        mix = Counter(own)
        main, hits = mix.most_common(1)[0]
        # on each layout: the most common value of the OTHER policies
        match = 0
        for column, value in enumerate(values[p]):
            if value is None:
                continue
            others = Counter(
                values[q][column]
                for q in range(len(runs))
                if q != p and values[q][column] is not None
            )
            if others and others[value] == max(others.values()):
                match += 1
        constant, conform = hits / len(own), match / len(own)
        if constant >= ALWAYS:
            logic = f"always {main}"
        elif conform >= FOLLOWS:
            logic = "follows the layout"
        else:
            logic = "own rule"
        out[run] = {
            "logic": logic,
            "main": main,
            "constant": round(constant, 3),
            "conform": round(conform, 3),
            "mix": dict(mix.most_common()),
            "layouts": len(own),
        }
    # the policies with their own rule: do some of them share it?
    alone = [p for p, run in enumerate(runs) if out[run]["logic"] == "own rule"]
    if len(alone) >= 2:
        agree = agreement([values[p] for p in alone])
        distance = 1.0 - agree
        np.fill_diagonal(distance, 0.0)
        labels = fcluster(
            linkage(squareform(distance, checks=False), "complete"),
            1.0 - SAME_RULE + 1e-9,
            "distance",
        )
        sizes = Counter(labels)
        order = {c: k + 1 for k, (c, _) in enumerate(sizes.most_common())}
        for p, label in zip(alone, labels):
            out[runs[p]]["rule_class"] = int(order[label])
            out[runs[p]]["rule_class_size"] = int(sizes[label])
    elif alone:
        out[runs[alone[0]]].update(rule_class=1, rule_class_size=1)
    # how much the layout decides: the share of the most common value on a layout
    shares = []
    for column in range(len(scenes)):
        column_values = Counter(
            row[column] for row in values if row[column] is not None
        )
        if column_values:
            shares.append(max(column_values.values()) / sum(column_values.values()))
    return {
        "policies": out,
        "layouts": len(scenes),
        "majority_share": round(float(np.mean(shares)), 3) if shares else None,
    }


# ------------------------------------------------------------------ choice rules
def episode(run: Any, objects: list[str]) -> dict[str, Any] | None:
    """The start poses of an episode and the object of ``objects`` that moves first."""
    channels = run.motion.channels
    if any(f"obj:{name}" not in channels for name in objects):
        return None
    first, chosen = None, None
    for name in objects:
        path = channels[f"obj:{name}"]
        moved = np.flatnonzero(np.linalg.norm(path - path[0], axis=1) >= MOVED)
        if len(moved) and (first is None or moved[0] < first):
            first, chosen = int(moved[0]), name
    start = {name: channels[name][0] for name in channels}
    return {"chosen": chosen, "start": start}


def candidates(start: dict[str, np.ndarray], objects: list[str]) -> dict[str, str]:
    """The object that each simple rule selects on one layout."""
    place = {name: np.asarray(start[f"obj:{name}"], dtype=float) for name in objects}
    out: dict[str, str] = {f"always {name}": name for name in objects}
    for axis, letter in enumerate("xyz"[: len(next(iter(place.values())))]):
        out[f"the smallest {letter}"] = min(objects, key=lambda n: place[n][axis])
        out[f"the largest {letter}"] = max(objects, key=lambda n: place[n][axis])
    for name, point in start.items():
        if name.startswith("obj:") and name.removeprefix("obj:") in objects:
            continue
        point = np.asarray(point, dtype=float)
        plain = {"base": "the robot base", "ee": "the gripper"}.get(
            name, name.removeprefix("obj:")
        )

        def far(n: str, origin: np.ndarray = point) -> float:
            return float(np.linalg.norm(place[n][:2] - origin[:2]))

        out[f"the nearest to {plain}"] = min(objects, key=far)
        out[f"the farthest from {plain}"] = max(objects, key=far)
    return out


def choice_rules(scenario: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """For each policy: the simple rule that predicts the object that it moves first."""
    if scenario not in CHOICES:
        return None
    # pylint: disable=import-outside-toplevel
    from experiments.contacts import features

    key, objects = CHOICES[scenario]
    picks: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for scene in semantic.comparable_scenes(data):
        for run in features.load_runs(scenario, data["traces"][scene]):
            item = episode(run, objects)
            if item is None or item["chosen"] is None:
                continue
            picks.setdefault(run.run_id, []).append(
                (item["chosen"], candidates(item["start"], objects))
            )
    out: dict[str, Any] = {}
    names: list[str] = []
    for run_id, rows in picks.items():
        if len(rows) < LEAST:
            continue
        names = list(rows[0][1])
        scores = {
            name: float(np.mean([chosen == rule[name] for chosen, rule in rows]))
            for name in names
        }
        # a constant choice is the simplest rule: it wins a tie
        best = max(names, key=lambda n: (round(scores[n], 6), n.startswith("always")))
        out[run_id] = {
            "rule": best if scores[best] >= FOLLOWS else "no simple rule",
            "best": best,
            "score": round(scores[best], 3),
            "layouts": len(rows),
        }
    # what a rule gives for a policy that selects an object at random
    return {
        "group": key,
        "objects": objects,
        "rules_tested": len(names),
        "chance": round(1 / len(objects), 3),
        "policies": out,
    }


def build(scenario: str) -> dict[str, Any]:
    """The profiles of all tag groups of a scenario, and its choice rules."""
    data = semantic.load_scenes(scenario)
    if data is None:
        raise FileNotFoundError(f"no tags on the layouts of {scenario}")
    holds = semantic.stability(data)
    groups = {}
    for index, (group, hold) in enumerate(zip(data["groups"], holds)):
        if hold["kind"] is None:
            continue
        item = profiles(data, index)
        logic = Counter(p["logic"] for p in item["policies"].values())
        groups[group["key"]] = {
            "role": group["role"],
            "kind": hold["kind"],
            "repeat": hold["repeat"],
            "agree": hold["agree"],
            "majority_share": item["majority_share"],
            "logic": dict(logic.most_common()),
            "policies": item["policies"],
        }
    out = {
        "scenario": scenario,
        "layouts": len(semantic.comparable_scenes(data)),
        "thresholds": {"always": ALWAYS, "follows": FOLLOWS, "same_rule": SAME_RULE},
        "groups": groups,
        "choice": choice_rules(scenario, data),
    }
    RULES.mkdir(exist_ok=True)
    (RULES / f"{scenario}.json").write_text(json.dumps(out, indent=None) + "\n")
    return out


def load(scenario: str) -> dict[str, Any] | None:
    """The profiles of a scenario, if the file exists."""
    path = RULES / f"{scenario}.json"
    return json.loads(path.read_text()) if path.exists() else None


def kinds_of_logic(group: dict[str, Any]) -> dict[str, int]:
    """Policies of a tag group by the kind of their logic."""
    out = {"always": 0, "follows the layout": 0, "own rule": 0}
    for policy in group["policies"].values():
        key = "always" if policy["logic"].startswith("always") else policy["logic"]
        out[key] += 1
    return out


def invariant_tag(
    group: dict[str, Any], choice: dict[str, Any] | None, run: str
) -> str | None:
    """The description of one policy in one tag group that does not change with the layout.

    A property of the policy: its value. Else its logic: ``always <value>``, the
    choice rule if the group asks which object moves first and a simple rule
    predicts it, ``follows the layout``, or ``own rule`` (with the number of its
    rule class if two policies or more share it).
    """
    policy = group["policies"].get(run)
    if policy is None:
        return None
    if group["kind"] == "policy" or policy["logic"].startswith("always"):
        return policy["main"]
    named = (choice or {}).get("policies", {}).get(run)
    if named and named["rule"] != "no simple rule":
        return f"rule: {named['rule']}"
    if policy["logic"] == "own rule" and policy.get("rule_class_size", 0) >= 2:
        return f"own rule {policy['rule_class']}"
    return policy["logic"]


def invariant_strategies(data: dict[str, Any]) -> dict[str, str]:
    """The semantic strategy of each policy from its invariant tags (plan groups)."""
    plan = [(key, g) for key, g in data["groups"].items() if g["role"] == "plan"]
    runs = sorted({run for g in data["groups"].values() for run in g["policies"]})
    out = {}
    for run in runs:
        parts = []
        for key, group in plan:
            choice = (
                data["choice"]
                if data["choice"] and data["choice"]["group"] == key
                else None
            )
            tag = invariant_tag(group, choice, run)
            parts.append("?" if tag is None else tag)
        out[run] = " / ".join(parts) or "one plan"
    return out


def main() -> None:
    """Build the files of the given scenarios and print a summary."""
    for name in sys.argv[1:]:
        out = build(name)
        print(f"{name}: {out['layouts']} layouts")
        for key, group in out["groups"].items():
            if group["kind"] == "policy":
                continue
            counts = kinds_of_logic(group)
            shared = sum(
                p.get("rule_class_size", 0) >= 2 for p in group["policies"].values()
            )
            print(
                f"  {key:13s} {group['role']:5s} {group['kind']:5s} "
                f"always {counts['always']:2d}  follows the layout "
                f"{counts['follows the layout']:2d}  own rule {counts['own rule']:2d} "
                f"(in a shared class {shared:2d})  majority on a layout "
                f"{group['majority_share']}"
            )
        if out["choice"]:
            rules = Counter(p["rule"] for p in out["choice"]["policies"].values())
            print(
                f"  choice rules for {out['choice']['group']}: {dict(rules.most_common())}"
            )


if __name__ == "__main__":
    main()
