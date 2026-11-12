"""Tag rules of obstruction3d: clear a target region and put the target block on it."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 32 runs (order, shift "
    "and lift of the two obstructions and of the target block, base path, contact "
    "pairs). No pictures and no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "All runs do the same three steps: an obstruction, the other obstruction, the target "
    "block. The contact view finds 2 strategies.",
    "The differences are in the places: 21 runs put the obstructions just beside the "
    "region, and 11 runs put them 25 cm or more away. All opus55 and opus55-source runs "
    "put them beside.",
    "The base does not move in 17 of 32 runs: the arm does all the work.",
]
CODEBOOK = [
    {
        "key": "order",
        "question": "Which obstruction moves first?",
        "source": "motion",
        "values": {
            "obstruction1_first": "obstruction1 moves first.",
            "obstruction0_first": "obstruction0 moves first.",
        },
        "rule": "first motion of each obstruction",
    },
    {
        "key": "drop_zone",
        "question": "How far from the region does the robot put the obstructions?",
        "source": "motion",
        "values": {
            "beside": "No obstruction ends 25 cm or more from its start: just off "
            "the region.",
            "far": "An obstruction ends 25 cm or more from its start.",
        },
        "rule": "largest shift of an obstruction",
    },
    {
        "key": "base",
        "question": "Does the base move?",
        "source": "motion",
        "values": {
            "fixed": "The base path is below 20 cm: the arm does all the work.",
            "moves": "The base path is 20 cm or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "carry",
        "question": "How high does the robot carry a block?",
        "source": "motion",
        "values": {
            "low": "No block goes 10 cm or more above its start.",
            "high": "A block goes 10 cm or more above its start.",
        },
        "rule": "largest height of a block above its start",
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
    info = {name: summary(pos) for name, pos in run.objects.items()}
    order = moved_order(run, shift=0.02, path=0.05)
    first = next((n for n in order if n.startswith("obstruction")), "obstruction0")
    far = max(info[f"obstruction{k}"]["shift"] for k in (0, 1))
    lift = max(i["lift"] for i in info.values())
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "order": first + "_first",
        "drop_zone": "far" if far >= 0.25 else "beside",
        "base": "moves" if base_path >= 0.2 else "fixed",
        "carry": "high" if lift >= 0.10 else "low",
        "pace": "quick" if run.steps <= 70 else "slow",
    }
    evidence = {
        "order": order,
        "largest_obstruction_shift_m": round(far, 2),
        "largest_lift_m": round(lift, 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
