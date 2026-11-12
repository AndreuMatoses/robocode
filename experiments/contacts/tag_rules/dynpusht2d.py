"""Tag rules of dynpusht2d: push a T-shaped block to a goal pose with a dot robot."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, bouts, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 34 runs (path of the "
    "block and of the robot, number of contact modes, contact pairs). The traces "
    "give the position of the block; its angle was not read. No pictures and no "
    "program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "9 runs reach the goal in one or two pushes, 19 runs in three to six pushes, and 6 "
    "runs with seven or more small corrections. The contact view sees this (0.89); the "
    "motion view does not (0.48).",
    "10 of 34 runs push the block along the top wall or against it.",
    "The motion view has the position of the block only. It does not have the angle, "
    "which the task also sets.",
]
CODEBOOK = [
    {
        "key": "pushes",
        "question": "In how many pushes does the block reach the goal?",
        "source": "motion",
        "values": {
            "one_or_two": "The block moves one or two times.",
            "several": "The block moves three to six times.",
            "many": "The block moves seven times or more: many small corrections.",
        },
        "rule": "number of separate motions of the block (a pause of more than 3 "
        "env steps splits two motions)",
    },
    {
        "key": "wall",
        "question": "Does the block touch the top wall?",
        "source": "contacts",
        "values": {
            "free": "The block does not touch a wall.",
            "uses_wall": "The block touches the top wall: the robot pushes it "
            "along the wall or against it.",
        },
        "rule": "contact pair of the block with the top wall",
    },
    {
        "key": "block_path",
        "question": "Does the block go straight to the goal?",
        "source": "motion",
        "values": {
            "straight": "The path of the block is less than 1.1 times the straight "
            "line.",
            "detour": "1.1 times the straight line or more.",
        },
        "rule": "path length of the block divided by its shift",
    },
    {
        "key": "robot_travel",
        "question": "How much does the robot move around the block?",
        "source": "motion",
        "values": {
            "short": "The robot path is less than 3.2 times the block path.",
            "long": "3.2 times the block path or more: the robot goes around the "
            "block again and again.",
        },
        "rule": "path length of the robot divided by the path length of the block",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 192 env steps.",
            "slow": "More than 192 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    block = summary(run.ch("obj:tblock"))
    count = bouts(run.ch("obj:tblock"), gap=3)
    ratio = block["path"] / block["shift"]
    travel = path_length(run.ch("ee")) / block["path"]
    tags = {
        "pushes": "one_or_two" if count <= 2 else "several" if count <= 6 else "many",
        "wall": "uses_wall" if run.touches(r"tblock~top_wall") else "free",
        "block_path": "straight" if ratio < 1.1 else "detour",
        "robot_travel": "short" if travel < 3.2 else "long",
        "pace": "quick" if run.steps <= 192 else "slow",
    }
    evidence = {
        "block_motions": count,
        "block_path_ratio": round(ratio, 2),
        "robot_path_over_block_path": round(travel, 2),
        "contact_modes": len(run.modes),
        "steps": run.steps,
    }
    return tags, evidence
