"""Tag rules of dynpushpullhook2d: bring a target block to the middle wall with a hook."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, bouts, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 33 runs (shift of the "
    "target block and of the three obstructions, path of the hook, contact pairs). "
    "All runs take the hook. Film strips of 2 runs were inspected; they agree with "
    "the tags. No program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "All 33 runs take the hook. 24 runs sweep the target block sideways across the room, "
    "and 9 runs pull it straight to the wall.",
    "22 runs rake the full group of obstructions together with the target block. In 8 "
    "runs the hook does not touch the target block: an obstruction pushes it.",
    "The contact view finds 2 strategies, and no distance sees these tag groups (all at "
    "0.61 or less).",
]
CODEBOOK = [
    {
        "key": "direction",
        "question": "Which way does the target block go to the wall?",
        "source": "motion",
        "values": {
            "straight_down": "The target block ends less than 50 cm to the side of "
            "its start: a pull straight to the wall.",
            "sideways": "The target block ends 50 cm or more to the side: the hook "
            "sweeps it across the room.",
        },
        "rule": "x shift of the target block",
    },
    {
        "key": "obstructions",
        "question": "What happens to the obstructions?",
        "source": "motion",
        "values": {
            "rakes_all": "Each of the three obstructions ends 30 cm or more from "
            "its start: the hook rakes the full group.",
            "leaves_some": "One obstruction or more ends less than 30 cm from its "
            "start.",
        },
        "rule": "number of obstructions with a shift below 30 cm",
    },
    {
        "key": "hook_contact",
        "question": "Does the hook touch the target block?",
        "source": "contacts",
        "values": {
            "direct": "The hook touches the target block.",
            "through_obstruction": "The hook touches obstructions only, and an "
            "obstruction pushes the target block.",
        },
        "rule": "contact pair of the hook with the target block",
    },
    {
        "key": "strokes",
        "question": "Does the target block move in one stroke?",
        "source": "motion",
        "values": {
            "one": "The target block moves in one motion.",
            "several": "The target block moves two times or more (pauses of more "
            "than 5 env steps between).",
        },
        "rule": "number of separate motions of the target block",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 189 env steps.",
            "slow": "More than 189 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    target = summary(run.ch("obj:target_block"))
    side = abs(target["end"][0] - target["start"][0])
    shifts = [summary(run.ch(f"obj:obstruction{k}"))["shift"] for k in (0, 1, 2)]
    strokes = bouts(run.ch("obj:target_block"), gap=5)
    tags = {
        "direction": "straight_down" if side < 0.5 else "sideways",
        "obstructions": "leaves_some" if min(shifts) < 0.3 else "rakes_all",
        "hook_contact": (
            "direct" if run.touches(r"hook~target_block") else "through_obstruction"
        ),
        "strokes": "one" if strokes <= 1 else "several",
        "pace": "quick" if run.steps <= 189 else "slow",
    }
    evidence = {
        "target_start": target["start"],
        "target_end": target["end"],
        "obstruction_shifts_m": [round(v, 2) for v in shifts],
        "target_motions": strokes,
        "hook_path_m": round(path_length(run.ch("obj:hook")), 2),
        "steps": run.steps,
    }
    return tags, evidence
