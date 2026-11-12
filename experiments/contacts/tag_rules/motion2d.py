"""Tag rules of motion2d: drive through the gaps of three walls to a target."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length, stops

METHOD = (
    "Tag groups proposed after a look at the base path of all 35 runs (numbers only). "
    "Each value is assigned by the rule of its group. No pictures and no program text "
    "were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "All 35 runs take the same route through the three gaps. The scene has no other way.",
    "The contact view finds 5 strategies. One cause is a touch of an obstacle on the way: "
    "28 runs touch an obstacle, and 7 runs do not.",
    "All sol runs take wide paths. All opus55-source and astra runs take tight paths.",
]
CODEBOOK = [
    {
        "key": "route",
        "question": "Which way does the robot go to the target?",
        "source": "motion",
        "values": {
            "three_gaps": "Through the gap of each of the three walls. The scene "
            "has no other way.",
        },
        "rule": "the base path goes through the gap of each wall",
    },
    {
        "key": "clearance",
        "question": "Does the robot touch an obstacle on the way?",
        "source": "contacts",
        "values": {
            "touches": "The base, the arm or the gripper touches an obstacle "
            "(within 1 cm).",
            "clear": "No contact with an obstacle.",
        },
        "rule": "contact pairs of a robot part with an obstacle",
    },
    {
        "key": "pace",
        "question": "Does the robot go at full speed without a stop?",
        "source": "motion",
        "values": {
            "direct": "At most 53 env steps.",
            "slow": "More than 53 env steps: stops or slow parts.",
        },
        "rule": "number of env steps",
    },
    {
        "key": "path",
        "question": "How long is the path against the straight line?",
        "source": "motion",
        "values": {
            "tight": "Less than 1.25 times the straight line.",
            "wide": "1.25 times or more: wide turns or detours.",
        },
        "rule": "length of the base path divided by the distance from start to end",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    base = run.ch("base")
    length = path_length(base)
    ratio = length / float(np.linalg.norm(base[-1] - base[0]))
    tags = {
        "route": "three_gaps",
        "clearance": "touches" if run.touches(r"~obstacle$") else "clear",
        "pace": "direct" if run.steps <= 53 else "slow",
        "path": "tight" if ratio < 1.25 else "wide",
    }
    evidence = {
        "steps": run.steps,
        "path_m": round(length, 2),
        "path_ratio": round(ratio, 2),
        "waypoints": [
            base[int(f * (len(base) - 1))].round(2).tolist() for f in (0.25, 0.5, 0.75)
        ],
        "stops": stops(base, 1e-4),
        "contacts": run.pairs,
    }
    return tags, evidence
