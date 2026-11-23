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

* for every tag group, a **layout rule**: one measure of the start poses (a
  coordinate, a size, a difference or a distance of two bodies) and one limit that
  predict the value of the policy ("from_right if x of the robot base − x of the
  button > 0.2, else from_left"). A policy has the rule if the rule is correct on
  85 % of its layouts or more, and if shuffled values get such a rule in 5 % of the
  cases or less. This is the rule for a side, a count or a method.

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
SHUFFLES = 400  # shuffles of the values of a policy for the test of a layout rule
CHANCE = 0.05  # largest share of the shuffles that get a rule as good
SIDE = 3  # layouts on each side of the limit of a layout rule, at least
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
    # how much the policies agree: the share of the most common value on a layout
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


def episodes(scenario: str, data: dict[str, Any]) -> dict[str, list]:
    """The solved episodes of each layout of a scenario."""
    # pylint: disable=import-outside-toplevel
    from experiments.contacts import features

    return {
        scene: features.load_runs(scenario, data["traces"][scene])
        for scene in semantic.comparable_scenes(data)
    }


def choice_rules(scenario: str, loaded: dict[str, list]) -> dict[str, Any] | None:
    """For each policy: the simple rule that predicts the object that it moves first."""
    if scenario not in CHOICES:
        return None
    key, objects = CHOICES[scenario]
    picks: dict[str, list[tuple[str, dict[str, str]]]] = {}
    for runs in loaded.values():
        for run in runs:
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


# ------------------------------------------------------------------ layout rules
def start_poses(run: Any) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The position of each body at the start of an episode, and its size (2D only).

    A 2D trace has the outline of every body, so it gives the bodies that can not
    move and the sizes too. A 3D trace gives the robot and the movable objects.
    """
    plain = {"base": "the robot base", "ee": "the gripper"}
    place = {
        plain.get(name, name.removeprefix("obj:")): np.asarray(path[0], dtype=float)
        for name, path in run.motion.channels.items()
    }
    size: dict[str, np.ndarray] = {}
    with np.load(run.directory / "trace.npz") as data:
        if "verts" not in data.files:
            return place, size
        for entity in run.meta["entities"]:
            if not entity["name"].startswith("env."):
                continue
            geoms = [g["id"] for g in run.meta["geoms"] if g["entity"] == entity["id"]]
            points = data["verts"][0, geoms].reshape(-1, 2)
            points = points[~np.isnan(points[:, 0])]
            if len(points):
                name = entity["name"].removeprefix("env.")
                place[name] = points.mean(axis=0)
                size[name] = points.max(axis=0) - points.min(axis=0)
    return place, size


def measures(place: dict[str, np.ndarray], size: dict[str, np.ndarray]) -> dict:
    """The simple measures of one layout: coordinates, sizes, differences, distances."""
    names = sorted(place)
    out: dict[str, float] = {}
    for name in names:
        for axis, letter in enumerate("xyz"[: len(place[name])]):
            out[f"{letter} of {name}"] = float(place[name][axis])
    for name in sorted(size):
        out[f"width of {name}"] = float(size[name][0])
        out[f"height of {name}"] = float(size[name][1])
    for i, one in enumerate(names):
        for two in names[i + 1 :]:
            common = min(len(place[one]), len(place[two]))
            for axis, letter in enumerate("xyz"[:common]):
                out[f"{letter} of {one} − {letter} of {two}"] = float(
                    place[one][axis] - place[two][axis]
                )
            out[f"distance from {one} to {two}"] = float(
                np.linalg.norm(place[one][:2] - place[two][:2])
            )
    return out


def splits(table: np.ndarray, codes: np.ndarray, values: int) -> tuple:
    """The best limit of every measure for each row of ``codes``.

    ``table`` is [layout, measure], ``codes`` is [case, layout] (the value of the
    policy as a number). A limit puts ``SIDE`` layouts or more on each side, and the
    most common value must differ between the sides. Returns [case, measure] arrays:
    the share of the layouts that the rule gets right, the position of the limit,
    and the value below and above the limit.
    """
    order = np.argsort(table, axis=0, kind="stable")  # [layout, measure]
    ranked = np.take_along_axis(table, order, axis=0)
    hot = np.eye(values)[codes][:, order]  # [case, layout, measure, value]
    below = np.cumsum(hot, axis=1)[:, :-1]
    above = below[:, -1:] + hot[:, -1:] - below
    right = below.max(axis=-1) + above.max(axis=-1)
    count = len(table)
    cut = np.arange(1, count)[:, None]
    valid = (ranked[:-1] < ranked[1:] - 1e-9) & (cut >= SIDE) & (cut <= count - SIDE)
    valid = valid[None] & (below.argmax(axis=-1) != above.argmax(axis=-1))
    right = np.where(valid, right, 0.0) / count
    best = right.argmax(axis=1)  # [case, measure]
    take = best[:, None, :]
    low = np.take_along_axis(below.argmax(axis=-1), take, axis=1)[:, 0]
    high = np.take_along_axis(above.argmax(axis=-1), take, axis=1)[:, 0]
    return right.max(axis=1), best, low, high


def _fits(
    own: list[str], rows: list[dict[str, float]], names: list[str], generator: Any
) -> tuple[dict[tuple, dict[str, float]], int]:
    """The layout rules that give the values ``own`` on the layouts ``rows``.

    A rule is (measure, value below the limit, value above the limit). It must be
    correct on ``FOLLOWS`` of the layouts, and the best rule of a shuffle of ``own``
    is that good in ``CHANCE`` of the shuffles or less. Also returns the number of measures in the test.
    """
    kinds = sorted(set(own))
    table = np.array([[row[name] for name in names] for row in rows])
    keep = np.flatnonzero(table.std(axis=0) > 1e-6)
    if not len(keep):  # no measure changes with the layout (see ``start_poses``)
        return {}, 0
    table = table[:, keep]
    codes = np.array([kinds.index(value) for value in own])
    shuffled = np.stack([generator.permutation(codes) for _ in range(SHUFFLES)])
    right, best, low, high = splits(table, np.vstack([codes, shuffled]), len(kinds))
    # the share of the shuffles whose best rule is as good as this rule, or better
    best_shuffled = right[1:].max(axis=1)
    chance = (1 + (best_shuffled[:, None] >= right[0] - 1e-9).sum(axis=0)) / (
        1 + SHUFFLES
    )
    ranked = np.sort(table, axis=0)
    out = {}
    for column in np.flatnonzero((right[0] >= FOLLOWS - 1e-9) & (chance <= CHANCE)):
        limit = ranked[best[0, column] : best[0, column] + 2, column].mean()
        key = (names[keep[column]], kinds[low[0, column]], kinds[high[0, column]])
        out[key] = {"score": float(right[0, column]), "limit": float(limit)}
    return out, len(keep)


def _simple(key: tuple) -> tuple:
    """Sort key: a rule on one body before a rule on two, a short name first."""
    return (-key[0].count(" of ") - key[0].count(" to "), -len(key[0]), key)


def _rule(key: tuple, fit: dict[str, float]) -> dict[str, Any]:
    return {
        "measure": key[0],
        "below": key[1],
        "above": key[2],
        "score": round(fit["score"], 3),
        "limit": round(fit["limit"], 3),
    }


def layout_rules(
    data: dict[str, Any], index: int, layouts: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """For each policy: the measure of the layout and the limit that give its value.

    Only for a policy whose value changes (not ``always``). The policies of a tag
    group share a rule where they can: the rule that is valid for the most policies
    is given first. ``majority`` is the rule for the most common value of each
    layout: the rule that the policies that follow the layout share.
    """
    runs, scenes, values = matrix(data, index)
    names = sorted(set.intersection(*(set(layouts[s]) for s in scenes if s in layouts)))
    generator = np.random.default_rng(0)
    found: dict[str, dict[tuple, dict[str, float]]] = {}
    tested = 0
    for run, row in zip(runs, values):
        use = [k for k, s in enumerate(scenes) if row[k] is not None and s in layouts]
        own = [row[k] for k in use]
        if len(use) < LEAST or Counter(own).most_common(1)[0][1] / len(own) >= ALWAYS:
            continue
        found[run], count = _fits(
            own, [layouts[scenes[k]] for k in use], names, generator
        )
        tested = max(tested, count)
    # the policies share a rule where they can: the most common rule first
    out: dict[str, Any] = {run: None for run in found}
    while True:
        count = Counter(key for run in found if out[run] is None for key in found[run])
        if not count:
            break
        chosen = max(count, key=lambda k: (count[k], *_simple(k)))
        for run in found:
            if out[run] is None and chosen in found[run]:
                out[run] = _rule(chosen, found[run][chosen])
    # the most common value of each layout
    use = [k for k, s in enumerate(scenes) if s in layouts]
    common = [
        Counter(row[k] for row in values if row[k] is not None).most_common(1)[0][0]
        for k in use
    ]
    majority = None
    if len(use) >= LEAST and Counter(common).most_common(1)[0][1] / len(use) < ALWAYS:
        fits, _ = _fits(common, [layouts[scenes[k]] for k in use], names, generator)
        if fits:
            key = max(fits, key=lambda k: (round(fits[k]["score"], 6), *_simple(k)))
            majority = _rule(key, fits[key])
            majority["text"] = rule_text(majority)
    return {"measures_tested": tested, "policies": out, "majority": majority}


def rule_text(rule: dict[str, Any], limit: float | None = None) -> str:
    """A layout rule in words; ``limit`` replaces the limit of the policy."""
    limit = rule["limit"] if limit is None else limit
    return f"{rule['above']} if {rule['measure']} > {limit:.2g}, else {rule['below']}"


def build(scenario: str) -> dict[str, Any]:
    """The profiles of all tag groups of a scenario, and its choice rules."""
    data = semantic.load_scenes(scenario)
    if data is None:
        raise FileNotFoundError(f"no tags on the layouts of {scenario}")
    holds = semantic.stability(data)
    loaded = episodes(scenario, data)
    layouts = {
        scene: measures(*start_poses(runs[0])) for scene, runs in loaded.items() if runs
    }
    groups = {}
    for index, (group, hold) in enumerate(zip(data["groups"], holds)):
        if hold["kind"] is None:
            continue
        item = profiles(data, index)
        logic = Counter(p["logic"] for p in item["policies"].values())
        rules = layout_rules(data, index, layouts)
        # one text for the policies that share a rule: the median of their limits
        limits: dict[tuple, list[float]] = {}
        for rule in rules["policies"].values():
            if rule:
                key = (rule["measure"], rule["below"], rule["above"])
                limits.setdefault(key, []).append(rule["limit"])
        for run, rule in rules["policies"].items():
            if rule:
                key = (rule["measure"], rule["below"], rule["above"])
                rule["text"] = rule_text(rule, float(np.median(limits[key])))
            item["policies"][run]["layout_rule"] = rule
        groups[group["key"]] = {
            "measures_tested": rules["measures_tested"],
            "majority_rule": rules["majority"],
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
        "thresholds": {
            "always": ALWAYS,
            "follows": FOLLOWS,
            "same_rule": SAME_RULE,
            "chance": CHANCE,
        },
        "groups": groups,
        "choice": choice_rules(scenario, loaded),
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
    predicts it, the layout rule if one measure of the start poses predicts its
    value, ``follows the layout``, or ``own rule`` (with the number of its rule
    class if two policies or more share it).
    """
    policy = group["policies"].get(run)
    if policy is None:
        return None
    if group["kind"] == "policy" or policy["logic"].startswith("always"):
        return policy["main"]
    named = (choice or {}).get("policies", {}).get(run)
    if named and named["rule"] != "no simple rule":
        return f"rule: {named['rule']}"
    if policy.get("layout_rule"):
        return f"rule: {policy['layout_rule']['text']}"
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
            named = Counter(
                p["layout_rule"]["text"]
                for p in group["policies"].values()
                if p.get("layout_rule")
            )
            for text, count in named.most_common(3):
                print(f"      {count:2d}  {text}")
            if group["majority_rule"]:
                print(f"      shared rule: {group['majority_rule']['text']}")
        if out["choice"]:
            rules = Counter(p["rule"] for p in out["choice"]["policies"].values())
            print(
                f"  choice rules for {out['choice']['group']}: {dict(rules.most_common())}"
            )


if __name__ == "__main__":
    main()
