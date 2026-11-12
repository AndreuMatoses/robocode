"""Tag rules of dynscooppour2d: move small objects over a middle wall."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 30 runs (when each "
    "small object passes the wall, how high it goes, motion of the hook, contact "
    "pairs). The scene is a side view: y is up; the wall is at x = 1.75 m and is "
    "1.5 m high. No pictures and no program text were read."
)
WALL = 1.75
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "29 of 30 runs use the hook as a scoop. One run (astra seed42) moves the objects with "
    "the arm and the gripper.",
    "The objects pass the wall in one trip in 9 runs, in two trips in 16 runs, and in "
    "three trips or more in 5 runs.",
    "11 runs stop near the necessary 50 % (60 % of the objects or less on the right). 19 "
    "runs move more.",
]
CODEBOOK = [
    {
        "key": "tool",
        "question": "Does the robot use the hook as a scoop?",
        "source": "motion and contacts",
        "values": {
            "hook": "The gripper holds the hook, and the hook moves the objects.",
            "no_hook": "The hook stays in its place: the arm and the gripper move "
            "the objects.",
        },
        "rule": "hook if the hook moves 10 cm or more",
    },
    {
        "key": "trips",
        "question": "In how many trips do the objects pass the wall?",
        "source": "motion",
        "values": {
            "one": "All objects that pass the wall do so in one group.",
            "two": "In two groups.",
            "three_or_more": "In three groups or more.",
        },
        "rule": "number of groups of the env steps at which an object passes "
        "x = 1.75 m (a gap of more than 20 env steps splits two groups)",
    },
    {
        "key": "share",
        "question": "How many objects are on the right at the end?",
        "source": "motion",
        "values": {
            "just_enough": "60 % of the objects or less (the task needs 50 %).",
            "more": "More than 60 % of the objects.",
        },
        "rule": "share of the small objects with x above 1.75 m in the last state",
    },
    {
        "key": "height",
        "question": "How high do the objects go?",
        "source": "motion",
        "values": {
            "low": "No object goes above y = 2.4 m: just over the wall.",
            "high": "An object goes above y = 2.4 m.",
        },
        "rule": "largest height of a small object",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 639 env steps.",
            "slow": "More than 639 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    smalls = [p for n, p in run.objects.items() if n.startswith("small")]
    right = sum(bool(p[-1][0] > WALL) for p in smalls)
    passes = sorted(
        int(np.argmax(p[:, 0] > WALL)) for p in smalls if (p[:, 0] > WALL).any()
    )
    groups = int(1 + (np.diff(passes) > 20).sum()) if passes else 0
    top = max(float(p[:, 1].max()) for p in smalls)
    hook = summary(run.ch("obj:hook"))
    share = right / len(smalls)
    tags = {
        "tool": "hook" if hook["shift"] >= 0.10 or hook["path"] >= 0.5 else "no_hook",
        "trips": "one" if groups <= 1 else "two" if groups == 2 else "three_or_more",
        "share": "just_enough" if share <= 0.6 else "more",
        "height": "high" if top >= 2.4 else "low",
        "pace": "quick" if run.steps <= 639 else "slow",
    }
    evidence = {
        "objects_on_the_right": f"{right} of {len(smalls)}",
        "groups_that_pass_the_wall": groups,
        "largest_height_m": round(top, 2),
        "hook_path_m": round(hook["path"], 2),
        "base_path_m": round(path_length(run.ch("base")), 2),
        "steps": run.steps,
    }
    return tags, evidence
