"""Tag rules of dynamo3d: drive to a goal region among chairs that can move."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 34 runs (base path, "
    "motion of each chair, contact pairs). No pictures and no program text were "
    "read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "The usual solution is one dash: the base drives straight to the goal and pushes the "
    "chairs away (a straight path in 29 of 34 runs; at most 60 env steps in 29 runs).",
    "23 runs push chairs 1 and 3, and 7 runs push chair 1 only. A chair falls over in 10 "
    "runs, and a chair hits a wall in 12 runs.",
    "The 5 runs with a detour are the 5 opus5 runs.",
]
CODEBOOK = [
    {
        "key": "pushes",
        "question": "Which chairs does the base push away?",
        "source": "motion",
        "values": {
            "chairs_1_3": "Chair 1 and chair 3 move 10 cm or more, chair 2 stays.",
            "chair_1": "Only chair 1 moves 10 cm or more.",
            "all": "The three chairs move.",
            "other": "Another set of chairs moves.",
        },
        "rule": "the chairs with a shift of 10 cm or more",
    },
    {
        "key": "drive",
        "question": "Does the base drive straight to the goal?",
        "source": "motion",
        "values": {
            "straight": "The base path is less than 1.15 times the straight line "
            "from start to end.",
            "detour": "The base path is 1.15 times the straight line or more: the "
            "base goes around a chair, goes back, or corrects.",
        },
        "rule": "length of the base path divided by the distance from start to end",
    },
    {
        "key": "chairs",
        "question": "Do the chairs stay upright?",
        "source": "motion",
        "values": {
            "upright": "No chair goes down 20 cm or more.",
            "tipped": "A chair goes down 20 cm or more: it falls over.",
        },
        "rule": "largest drop of the centre of a chair",
    },
    {
        "key": "wall",
        "question": "Does a chair hit a wall?",
        "source": "contacts",
        "values": {
            "free": "No contact of a chair with a wall.",
            "to_wall": "The base pushes a chair against a wall.",
        },
        "rule": "contact pair of a chair with a wall",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "one_dash": "At most 60 env steps: one drive at full speed.",
            "slow": "More than 60 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    info = {name[-1]: summary(pos) for name, pos in run.objects.items()}
    moved = "".join(sorted(k for k, i in info.items() if i["shift"] >= 0.10))
    base = run.ch("base")[:, :2]
    ratio = path_length(base) / float(np.linalg.norm(base[-1] - base[0]))
    drop = max(i["drop"] for i in info.values())
    tags = {
        "pushes": {"13": "chairs_1_3", "1": "chair_1", "123": "all"}.get(
            moved, "other"
        ),
        "drive": "straight" if ratio < 1.15 else "detour",
        "chairs": "tipped" if drop >= 0.20 else "upright",
        "wall": "to_wall" if run.touches(r"obstacle_chair~wall") else "free",
        "pace": "one_dash" if run.steps <= 60 else "slow",
    }
    evidence = {
        "shift_of_each_chair_m": {k: round(i["shift"], 2) for k, i in info.items()},
        "largest_drop_m": round(drop, 2),
        "base_path_m": round(path_length(base), 2),
        "path_ratio": round(ratio, 2),
        "base_end": base[-1].round(2).tolist(),
        "steps": run.steps,
        "contacts": run.pairs,
    }
    return tags, evidence
