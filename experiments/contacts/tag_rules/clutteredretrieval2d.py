"""Tag rules of clutteredretrieval2d: get a target block out of clutter into a region."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 30 runs (which blocks "
    "move and in which order, path of the target block, contact pairs). No pictures "
    "and no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "24 of 30 runs move an obstruction away before they take the target block. 6 runs "
    "take the target block first.",
    "Each run that moves an obstruction holds it with the suction pad. No run pushes an "
    "obstruction away.",
    "The base touches an obstruction in 13 runs.",
]
CODEBOOK = [
    {
        "key": "clearing",
        "question": "Does the robot move an obstruction before it takes the target?",
        "source": "motion",
        "values": {
            "clears_one": "One obstruction moves before the target block.",
            "clears_more": "Two or more obstructions move before the target block.",
            "direct": "The target block moves first. No obstruction moves before "
            "it.",
        },
        "rule": "the blocks that move 10 cm or more before the first motion of the "
        "target block",
    },
    {
        "key": "removal",
        "question": "How does an obstruction move?",
        "source": "contacts",
        "values": {
            "suction": "The suction pad holds the obstruction: pick and place.",
            "push": "An obstruction moves with no suction contact: a robot part or "
            "the target block pushes it.",
            "none": "No obstruction moves.",
        },
        "rule": "contact pair of the suction pad with an obstruction, for the runs "
        "in which an obstruction moves",
    },
    {
        "key": "carry",
        "question": "Does the target block go straight to the region?",
        "source": "motion",
        "values": {
            "straight": "The path of the target block is less than 1.15 times the "
            "straight line.",
            "detour": "1.15 times the straight line or more.",
        },
        "rule": "path length of the target block divided by its shift",
    },
    {
        "key": "base",
        "question": "Does the base touch an obstruction?",
        "source": "contacts",
        "values": {
            "clear": "No contact of the base with an obstruction.",
            "bumps": "The base touches an obstruction.",
        },
        "rule": "contact pair of the base with an obstruction",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 70 env steps.",
            "slow": "More than 70 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    order = moved_order(run, shift=0.10, path=0.10)
    before = order[: order.index("target_block")] if "target_block" in order else order
    moved = [name for name in order if name != "target_block"]
    target = summary(run.ch("obj:target_block"))
    ratio = target["path"] / target["shift"]
    suction = run.touches(r"gripper\.suction~obstruction")
    tags = {
        "clearing": (
            "direct"
            if not before
            else "clears_one" if len(before) == 1 else "clears_more"
        ),
        "removal": "none" if not moved else "suction" if suction else "push",
        "carry": "straight" if ratio < 1.15 else "detour",
        "base": "bumps" if run.touches(r"^base~obstruction") else "clear",
        "pace": "quick" if run.steps <= 70 else "slow",
    }
    evidence = {
        "order": order,
        "target_path_ratio": round(ratio, 2),
        "base_path_m": round(path_length(run.ch("base")), 2),
        "steps": run.steps,
        "contacts": run.pairs,
    }
    return tags, evidence
