"""Tag rules of dynobstruction2d: put the target block on a surface that is blocked."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (order and "
    "shift of the two obstructions and of the target block, contact pairs). No "
    "pictures and no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "30 of 35 runs do not clear the surface first: the robot pushes the target block onto "
    "the surface, and the block pushes the obstructions away. 5 runs clear the surface "
    "first.",
    "The fingers push the target block in 26 runs, and the base pushes it in 8 runs.",
    "In 2 runs an obstruction leaves the table.",
]
CODEBOOK = [
    {
        "key": "plan",
        "question": "How does the robot clear the surface?",
        "source": "motion",
        "values": {
            "bulldoze": "The target block moves first. The robot pushes it onto "
            "the surface, and the block pushes the obstructions away.",
            "clear_first": "An obstruction moves before the target block: the "
            "robot clears the surface, then moves the target block.",
        },
        "rule": "the first object that moves 5 cm or more",
    },
    {
        "key": "pusher",
        "question": "Which robot part moves the target block?",
        "source": "contacts",
        "values": {
            "fingers": "The fingers touch the target block; the base does not.",
            "palm": "The palm touches the target block; the fingers and the base "
            "do not.",
            "base": "The base touches the target block.",
            "none": "No robot part touches the target block.",
        },
        "rule": "contact pairs of a robot part with the target block",
    },
    {
        "key": "obstructions",
        "question": "Where do the obstructions end?",
        "source": "motion",
        "values": {
            "shifted": "Each obstruction ends less than 1 m from its start.",
            "thrown_off": "An obstruction ends 1 m or more from its start: it "
            "leaves the table.",
        },
        "rule": "largest shift of an obstruction",
    },
    {
        "key": "target_path",
        "question": "Does the target block go straight to the surface?",
        "source": "motion",
        "values": {
            "straight": "The path of the target block is less than 1.2 times the "
            "straight line.",
            "detour": "1.2 times the straight line or more.",
        },
        "rule": "path length of the target block divided by its shift",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 57 env steps.",
            "slow": "More than 57 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    order = moved_order(run, shift=0.05, path=0.05)
    target = summary(run.ch("obj:target_block"))
    far = max(summary(run.ch(f"obj:obstruction{k}"))["shift"] for k in (0, 1))
    ratio = target["path"] / target["shift"]
    if run.touches(r"^base~target_block"):
        pusher = "base"
    elif run.touches(r"gripper\.finger~target_block"):
        pusher = "fingers"
    elif run.touches(r"gripper\.palm~target_block"):
        pusher = "palm"
    else:
        pusher = "none"
    tags = {
        "plan": "bulldoze" if order and order[0] == "target_block" else "clear_first",
        "pusher": pusher,
        "obstructions": "thrown_off" if far >= 1.0 else "shifted",
        "target_path": "straight" if ratio < 1.2 else "detour",
        "pace": "quick" if run.steps <= 57 else "slow",
    }
    evidence = {
        "order": order,
        "largest_obstruction_shift_m": round(far, 2),
        "target_path_ratio": round(ratio, 2),
        "base_path_m": round(path_length(run.ch("base")), 2),
        "steps": run.steps,
        "contacts": run.pairs,
    }
    return tags, evidence
