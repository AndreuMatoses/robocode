"""Tag rules of sweepintodrawer3d: open a drawer and sweep five cubes into it."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 22 runs (lift, drop and "
    "first motion of each cube, motion of the wiper, base path, contact pairs). No "
    "pictures and no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "21 of 22 runs do not use the wiper: the gripper moves the cubes. 1 run sweeps with "
    "the wiper.",
    "14 runs sweep the cubes to the edge, and the cubes fall into the drawer. 8 runs pick "
    "each cube and drop it. All astra runs pick; all opus55-source and astra-source runs "
    "sweep.",
    "The contact view finds 17 strategies for 22 runs: almost each run is a strategy of "
    "its own.",
]
CODEBOOK = [
    {
        "key": "tool",
        "question": "What moves the cubes?",
        "source": "motion",
        "values": {
            "gripper": "The gripper moves the cubes. The wiper stays in its place.",
            "wiper": "The robot takes the wiper and sweeps with it.",
        },
        "rule": "wiper if the wiper moves 5 cm or more",
    },
    {
        "key": "transfer",
        "question": "Do the cubes slide or does the robot lift them?",
        "source": "motion",
        "values": {
            "sweep": "No cube goes 4 cm or more above the counter: the cubes slide "
            "to the edge and fall into the drawer.",
            "pick": "Each cube goes 4 cm or more above the counter: pick and drop.",
            "mixed": "Some cubes slide and some are lifted.",
        },
        "rule": "number of cubes with a lift of 4 cm or more",
    },
    {
        "key": "batches",
        "question": "How many cubes move together?",
        "source": "motion",
        "values": {
            "one_by_one": "The five cubes start to move at five separate times.",
            "groups": "The cubes start to move in two to four groups.",
            "one_sweep": "All cubes start to move within 1.5 s: one sweep.",
        },
        "rule": "number of groups of the first-motion steps of the cubes (a gap of "
        "more than 15 env steps splits two groups)",
    },
    {
        "key": "drawer",
        "question": "How does the robot open the drawer?",
        "source": "contacts",
        "values": {
            "handle": "A finger pad touches the drawer handle.",
            "other": "No pad contact with the handle.",
        },
        "rule": "contact pair of a finger pad with the drawer handle",
    },
    {
        "key": "base",
        "question": "Does the base drive during the task?",
        "source": "motion",
        "values": {
            "stays": "The base path is below 2 m.",
            "drives": "The base path is 2 m or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 400 env steps (40 s).",
            "slow": "More than 400 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    cubes = [summary(p) for n, p in run.objects.items() if n.startswith("cube")]
    wiper = summary(run.ch("obj:wiper_0"))
    lifted = sum(c["lift"] >= 0.04 for c in cubes)
    firsts = sorted(c["first"] for c in cubes if c["first"] is not None)
    groups = int(1 + (np.diff(firsts) > 15).sum())
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "tool": "wiper" if wiper["shift"] >= 0.05 else "gripper",
        "transfer": (
            "sweep" if lifted == 0 else "pick" if lifted == len(cubes) else "mixed"
        ),
        "batches": (
            "one_sweep"
            if groups == 1
            else "one_by_one" if groups >= len(cubes) else "groups"
        ),
        "drawer": "handle" if run.touches(r"gripper\.pad~.*handle") else "other",
        "base": "drives" if base_path >= 2 else "stays",
        "pace": "quick" if run.steps <= 400 else "slow",
    }
    evidence = {
        "cubes_lifted": int(lifted),
        "largest_lift_m": round(max(c["lift"] for c in cubes), 2),
        "first_motion_steps": firsts,
        "groups": groups,
        "wiper_shift_m": round(wiper["shift"], 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
