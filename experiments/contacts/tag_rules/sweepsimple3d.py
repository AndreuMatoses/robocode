"""Tag rules of sweepsimple3d: move the cubes on the floor into a region."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, moves_with, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 15 runs (end position and "
    "lift of each cube, motion of the wiper, base path, contact pairs). All runs put "
    "the cubes in the same region. Film strips of 3 runs were inspected. They showed "
    "that the run with a contact between the wiper and a cube does not sweep with the "
    "wiper, so the rule now needs the wiper and the cubes to move together. No "
    "program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "Only 15 final policies solve the first start scene.",
    "The name of the task says sweep, but no run sweeps with the wiper. 14 of 15 runs carry each cube with the gripper, and 1 run (astra-source seed424) slides each cube on the floor with the gripper.",
    "For all runs the wiper is an obstacle: 6 runs move it 50 cm or more away, and 9 runs leave it near its place.",
    "A first rule used the contact of the wiper with a cube, and it gave one run that sweeps. The film strip showed that this contact is a touch in passing.",
]
CODEBOOK = [
    {
        "key": "transfer",
        "question": "Do the cubes slide or does the robot carry them?",
        "source": "motion",
        "values": {
            "carry": "Each cube goes 2 cm or more above the floor: the gripper "
            "takes it and carries it to the region.",
            "drag": "The cubes stay on the floor: the gripper slides each cube to "
            "the region.",
        },
        "rule": "smallest lift of a cube (2 cm or more: carry)",
    },
    {
        "key": "wiper",
        "question": "What does the robot do with the wiper?",
        "source": "motion and contacts",
        "values": {
            "used": "The wiper moves together with the cubes: the robot sweeps "
            "with it.",
            "removed": "The robot moves the wiper 50 cm or more away and does not "
            "sweep with it.",
            "left": "The wiper moves less than 50 cm (it falls over or the robot "
            "pushes it a little).",
        },
        "rule": "used if the wiper moves in half or more of the env steps in "
        "which a cube moves; else the shift of the wiper. A contact of the wiper "
        "with a cube is not sufficient: it can be a touch in passing",
    },
    {
        "key": "layout",
        "question": "How do the cubes lie in the region at the end?",
        "source": "motion",
        "values": {
            "row": "The cubes end in one straight row (within 2 cm of a line along "
            "x or along y).",
            "heap": "The cubes end in a loose group.",
        },
        "rule": "spread of the end positions of the cubes along x and along y",
    },
    {
        "key": "carry_height",
        "question": "How high does a cube go?",
        "source": "motion",
        "values": {
            "low": "No cube goes 8 cm or more above the floor.",
            "high": "A cube goes 8 cm or more above the floor.",
        },
        "rule": "largest lift of a cube",
    },
    {
        "key": "base",
        "question": "How far does the base drive?",
        "source": "motion",
        "values": {
            "short": "The base path is below 11 m.",
            "long": "The base path is 11 m or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 350 env steps (35 s).",
            "slow": "More than 350 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    cubes = [summary(p) for n, p in run.objects.items() if n.startswith("cube")]
    wiper = summary(run.ch("obj:wiper_0"))
    lifts = [c["lift"] for c in cubes]
    ends = np.array([c["end"][:2] for c in cubes])
    spread = ends.std(axis=0)
    base_path = path_length(run.ch("base")[:, :2])
    together = float(
        np.mean(
            [
                moves_with(run, n, "obj:wiper_0")
                for n in run.motion.channels
                if n.startswith("obj:cube")
            ]
        )
    )
    sweeps = together >= 0.5
    tags = {
        "transfer": "carry" if min(lifts) >= 0.02 else "drag",
        "wiper": "used" if sweeps else "removed" if wiper["shift"] >= 0.5 else "left",
        "layout": "row" if spread.min() <= 0.02 and spread.max() >= 0.05 else "heap",
        "carry_height": "high" if max(lifts) >= 0.08 else "low",
        "base": "long" if base_path >= 11 else "short",
        "pace": "quick" if run.steps <= 350 else "slow",
    }
    evidence = {
        "lift_of_each_cube_m": [round(v, 2) for v in lifts],
        "end_of_each_cube": ends.round(2).tolist(),
        "wiper_shift_m": round(wiper["shift"], 2),
        "cube_steps_with_wiper_motion": round(together, 2),
        "wiper_touches_a_cube": run.touches(r"wiper~cube"),
        "gripper_touches_wiper": run.touches(r"gripper\.\w+~wiper"),
        "base_touches_wiper": run.touches(r"base~wiper"),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
