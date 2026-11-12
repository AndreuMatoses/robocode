"""Tag rules of table3d: take one cube from the table and lift it 10 cm."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (which cube "
    "moves, its lift, base path, number of env steps). No pictures and no program "
    "text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "The task needs one cube only. 24 of 35 runs take cube1, the nearest cube; 7 runs "
    "take cube0 and 4 runs take cube2. No run takes cube3.",
    "19 runs end in at most 4 env steps.",
    "The contact view finds 2 strategies and does not see which cube the robot takes "
    "(0.43). The motion view sees it (0.95).",
]
CODEBOOK = [
    {
        "key": "cube",
        "question": "Which cube does the robot take?",
        "source": "motion",
        "values": {
            "cube1": "cube1, the cube nearest to the robot.",
            "cube0": "cube0.",
            "cube2": "cube2.",
            "cube3": "cube3.",
        },
        "rule": "the cube that moves",
    },
    {
        "key": "base",
        "question": "Does the base move to the cube?",
        "source": "motion",
        "values": {
            "fixed": "The base path is below 10 cm: the arm reaches the cube.",
            "moves": "The base path is 10 cm or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "lift",
        "question": "How high does the cube go?",
        "source": "motion",
        "values": {
            "just_enough": "Less than 15 cm (the task needs 10 cm).",
            "high": "15 cm or more.",
        },
        "rule": "largest height of the cube above its start",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "minimal": "At most 4 env steps.",
            "longer": "More than 4 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    order = moved_order(run, shift=0.03, path=0.05)
    lift = max(summary(pos)["lift"] for pos in run.objects.values())
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "cube": order[0],
        "base": "moves" if base_path >= 0.10 else "fixed",
        "lift": "high" if lift >= 0.15 else "just_enough",
        "pace": "minimal" if run.steps <= 4 else "longer",
    }
    evidence = {
        "cubes_that_move": order,
        "largest_lift_m": round(lift, 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
