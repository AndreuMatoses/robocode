"""Tag rules of constrainedcupboard3d: put long rods into the rows of a cupboard."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 18 runs (order, lift, "
    "path and contact pairs). The end position of each rod is given by the task, so "
    "it is not a tag group. Film strips of 4 runs were inspected; they agree with the "
    "order of the rods; the contacts are too small to see. No program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "Only 18 final policies solve the first start scene.",
    "16 of 18 runs start with the rod that is farthest from the cupboard.",
    "The arm or the gripper touches the cupboard in 14 of 18 runs when a rod goes in. 4 "
    "runs put the rods in with no contact.",
    "The motion view separates the order fully (1.00). The contact view does not see it "
    "(0.45).",
]
CODEBOOK = [
    {
        "key": "order",
        "question": "In which order does the robot move the rods?",
        "source": "motion",
        "values": {
            "0_1_2": "cuboid_0 (the rod farthest from the cupboard), then "
            "cuboid_1, then cuboid_2.",
            "0_2_1": "cuboid_0, then cuboid_2, then cuboid_1.",
            "1_2_0": "cuboid_1, then cuboid_2, then cuboid_0: the far rod is last.",
            "other": "Another order.",
        },
        "rule": "order of the first motion of each rod",
    },
    {
        "key": "carry",
        "question": "How high does the robot carry a rod?",
        "source": "motion",
        "values": {
            "low": "No rod goes 30 cm or more above its start: the rod goes in at "
            "about the height of the row.",
            "high": "A rod goes 30 cm or more above its start, then comes down to "
            "the row.",
        },
        "rule": "largest height of a rod above its start",
    },
    {
        "key": "insertion",
        "question": "Does the robot touch the cupboard when it puts a rod in?",
        "source": "contacts",
        "values": {
            "free": "No contact of the arm or the gripper with the cupboard.",
            "gripper": "The gripper touches a cupboard panel, the arm does not.",
            "arm": "An arm link touches the cupboard.",
        },
        "rule": "contact pairs of a robot part with a part of the cupboard",
    },
    {
        "key": "pick",
        "question": "Does the gripper touch the floor at the grasp?",
        "source": "contacts",
        "values": {
            "floor_touch": "A finger pad touches the floor at a grasp.",
            "above_floor": "No pad contact with the floor.",
        },
        "rule": "contact pair of a finger pad with the floor",
    },
    {
        "key": "base",
        "question": "How far does the base drive?",
        "source": "motion",
        "values": {
            "short": "The base path is below 5 m.",
            "long": "The base path is 5 m or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 400 env steps (40 s).",
            "slow": "More than 400 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    order = "_".join(name.split("_")[1] for name in moved_order(run))
    lift = max(summary(pos)["lift"] for pos in run.objects.values())
    base_path = path_length(run.ch("base")[:, :2])
    arm = run.touches(r"^arm\.\w+~cupboard")
    gripper = run.touches(r"^gripper\.\w+~cupboard")
    tags = {
        "order": order if order in ("0_1_2", "0_2_1", "1_2_0") else "other",
        "carry": "high" if lift >= 0.30 else "low",
        "insertion": "arm" if arm else "gripper" if gripper else "free",
        "pick": "floor_touch" if run.touches(r"gripper\.pad~floor") else "above_floor",
        "base": "long" if base_path >= 5 else "short",
        "pace": "quick" if run.steps <= 400 else "slow",
    }
    evidence = {
        "order": order,
        "largest_lift_m": round(lift, 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
        "robot_contacts": [
            p for p in run.pairs if p.split(".")[0] in ("arm", "gripper", "base")
        ],
    }
    return tags, evidence
