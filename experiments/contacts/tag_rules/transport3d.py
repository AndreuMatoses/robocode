"""Tag rules of transport3d: put two cubes and a box on a table."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import MOVE, Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 30 runs (order, end "
    "position and lift of the box and of the two cubes, base path, contact pairs). In "
    "some runs the cubes end at the place of the box, so one tag group asks if a cube "
    "travels in the box. Film strips of 3 runs were inspected; they agree with the "
    "tags (the cubes are in the box when it goes to the table in the tray run). No "
    "program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "5 of 30 runs use the box as a tray: a cube travels in the box to the table. 25 runs "
    "carry each object by itself.",
    "The box goes first in 18 runs, and the two cubes go first in 10 runs.",
    "The base touches a cube or the box in 22 runs.",
]
CODEBOOK = [
    {
        "key": "box_use",
        "question": "Does the robot use the box to carry a cube?",
        "source": "motion",
        "values": {
            "tray": "A cube stays within 10 cm of the box for 5 env steps or more "
            "while the box moves: the cube travels in the box.",
            "plain": "No cube travels with the box: each object goes to the table "
            "by itself.",
        },
        "rule": "env steps in which the box moves and a cube is within 10 cm of it",
    },
    {
        "key": "order",
        "question": "Does the box move before the cubes?",
        "source": "motion",
        "values": {
            "box_first": "The box moves first.",
            "cubes_first": "The two cubes move before the box.",
            "box_between": "The box moves between the two cubes.",
        },
        "rule": "order of the first motion of the box and of the cubes",
    },
    {
        "key": "base",
        "question": "Does the base touch an object?",
        "source": "contacts",
        "values": {
            "clear": "No contact of the base with a cube or with the box.",
            "bumps": "The base touches a cube or the box.",
        },
        "rule": "contact pairs of the base with a cube or with the box",
    },
    {
        "key": "carry",
        "question": "How high does the robot carry an object?",
        "source": "motion",
        "values": {
            "low": "No object goes 65 cm or more above its start.",
            "high": "An object goes 65 cm or more above its start.",
        },
        "rule": "largest height of an object above its start",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 115 env steps.",
            "slow": "More than 115 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    box = run.ch("obj:box0")
    box_moves = np.linalg.norm(np.diff(box, axis=0), axis=1) > MOVE
    rides = {}
    for name in ("cube0", "cube1"):
        near = np.linalg.norm((run.ch("obj:" + name) - box)[1:, :2], axis=1) < 0.10
        rides[name] = int((near & box_moves).sum())
    order = moved_order(run, shift=0.05, path=0.10)
    place = order.index("box0")
    lift = max(summary(pos)["lift"] for pos in run.objects.values())
    tags = {
        "box_use": "tray" if max(rides.values()) >= 5 else "plain",
        "order": ("box_first", "box_between", "cubes_first")[place],
        "base": "bumps" if run.touches(r"^base~(cube|box)") else "clear",
        "carry": "high" if lift >= 0.65 else "low",
        "pace": "quick" if run.steps <= 115 else "slow",
    }
    evidence = {
        "order": order,
        "steps_a_cube_travels_with_the_box": rides,
        "largest_lift_m": round(lift, 2),
        "base_path_m": round(path_length(run.ch("base")[:, :2]), 2),
        "steps": run.steps,
    }
    return tags, evidence
