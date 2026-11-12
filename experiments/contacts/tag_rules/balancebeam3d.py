"""Tag rules of balancebeam3d: put a large block and two small blocks on a seesaw."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, bouts, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 34 runs (order, end "
    "position, lift, path and contact pairs of each block). The end layout is the "
    "same in all runs (the task gives the target of each block), so it is not a tag "
    "group. Film strips of 2 runs were made, but the blocks are too small in the "
    "pictures to check the tags. No program text was read."
)
BLOCKS = ("large_block", "small_block_1", "small_block_2")
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "28 of 34 runs put the large block on the beam first. It goes to the centre, so the "
    "beam stays level. 6 runs start with a small block.",
    "The end layout is the same in all runs, because the task gives the target of each "
    "block. The differences are in the order and in the style of the motion.",
    "The carry height depends on the method: all opus55 and opus55-source runs carry low, "
    "and all astra-source runs carry high.",
    "The contact view finds 3 strategies, and it separates the order fully (1.00).",
]
CODEBOOK = [
    {
        "key": "order",
        "question": "Which block goes on the beam first?",
        "source": "motion",
        "values": {
            "large_first": "The large block moves first. It goes to the centre of "
            "the beam, so the beam stays level.",
            "small_first": "A small block moves first, and the large block moves "
            "last.",
        },
        "rule": "the block with the first motion",
    },
    {
        "key": "carry",
        "question": "How high does the robot carry a block?",
        "source": "motion",
        "values": {
            "low": "No block goes 15 cm or more above its start.",
            "high": "A block goes 15 cm or more above its start.",
        },
        "rule": "largest height of a block above its start",
    },
    {
        "key": "transfer",
        "question": "Does each block move in one motion?",
        "source": "motion",
        "values": {
            "continuous": "Each block moves in one motion without a pause.",
            "interrupted": "A block has a pause of more than 1 s in its transfer, "
            "or it moves again later (a correction or a second grasp).",
        },
        "rule": "number of separate motions of each block (a pause of more than 10 "
        "env steps splits a motion)",
    },
    {
        "key": "base",
        "question": "How far does the base drive?",
        "source": "motion",
        "values": {
            "direct": "The base path is below 5 m.",
            "detour": "The base path is 5 m or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "beam",
        "question": "Does the robot disturb the beam?",
        "source": "contacts",
        "values": {
            "steady": "The gripper does not touch the beam, and the beam does not "
            "touch the floor.",
            "disturbed": "The gripper touches the beam, or the beam tips to the "
            "floor.",
        },
        "rule": "contact pairs of the beam with the gripper and with the floor",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 300 env steps (30 s).",
            "slow": "More than 300 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    info = {name: summary(run.ch("obj:" + name)) for name in BLOCKS}
    order = moved_order(run)
    lift = max(i["lift"] for i in info.values())
    motions = {name: bouts(run.ch("obj:" + name)) for name in BLOCKS}
    base_path = path_length(run.ch("base")[:, :2])
    pad_beam = run.touches(r"gripper\.\w+~seesaw")
    beam_floor = run.touches(r"seesaw\.beam~floor")
    tags = {
        "order": "large_first" if order[0] == "large_block" else "small_first",
        "carry": "high" if lift >= 0.15 else "low",
        "transfer": "interrupted" if max(motions.values()) >= 2 else "continuous",
        "base": "detour" if base_path >= 5 else "direct",
        "beam": "disturbed" if pad_beam or beam_floor else "steady",
        "pace": "quick" if run.steps <= 300 else "slow",
    }
    evidence = {
        "order": order,
        "largest_lift_m": round(lift, 2),
        "motions_of_each_block": motions,
        "base_path_m": round(base_path, 2),
        "gripper_touches_beam": pad_beam,
        "beam_touches_floor": beam_floor,
        "steps": run.steps,
    }
    return tags, evidence
