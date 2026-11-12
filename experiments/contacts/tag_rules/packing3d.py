"""Tag rules of packing3d: put two parts into a rack."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 33 runs (order, end "
    "position and lift of the two parts, base path, contact pairs). part0 starts on "
    "the +y side of the rack and part1 on the -y side. Film strips of 3 runs were "
    "inspected; they agree with the tags. No program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "23 of 33 runs put each part on the other side of the rack from its start. 10 runs "
    "keep each part on its own side. The motion view sees this (0.81); the contact view "
    "does not (0.53).",
    "The base stays in all opus55-source and opus5-source runs. It moves in all opus55, "
    "astra, astra-source and sol runs.",
]
CODEBOOK = [
    {
        "key": "placement",
        "question": "On which side of the rack does each part end?",
        "source": "motion",
        "values": {
            "own_side": "Each part ends on the side of the rack where it starts "
            "(part0 at the larger y).",
            "crossed": "The parts change sides: part0 ends at the smaller y.",
        },
        "rule": "y of part0 against y of part1 in the last state",
    },
    {
        "key": "contact",
        "question": "Do the two parts touch?",
        "source": "contacts",
        "values": {
            "apart": "No contact between the two parts.",
            "touch": "The two parts touch: they end side by side or one on the "
            "other.",
        },
        "rule": "contact pair part with part",
    },
    {
        "key": "order",
        "question": "Which part goes first?",
        "source": "motion",
        "values": {
            "part0_first": "part0 moves first.",
            "part1_first": "part1 moves first.",
        },
        "rule": "first motion of each part",
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
        "question": "How high does the robot carry a part?",
        "source": "motion",
        "values": {
            "low": "No part goes 12 cm or more above its start.",
            "high": "A part goes 12 cm or more above its start.",
        },
        "rule": "largest height of a part above its start",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 54 env steps.",
            "slow": "More than 54 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    zero, one = summary(run.ch("obj:part0")), summary(run.ch("obj:part1"))
    order = moved_order(run, shift=0.02, path=0.05)
    lift = max(zero["lift"], one["lift"])
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "placement": "own_side" if zero["end"][1] > one["end"][1] else "crossed",
        "contact": "touch" if run.touches(r"^part~part$") else "apart",
        "order": order[0] + "_first",
        "base": "moves" if base_path >= 0.2 else "fixed",
        "carry": "high" if lift >= 0.12 else "low",
        "pace": "quick" if run.steps <= 54 else "slow",
    }
    evidence = {
        "end_part0": zero["end"],
        "end_part1": one["end"],
        "largest_lift_m": round(lift, 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
