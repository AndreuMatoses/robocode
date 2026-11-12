"""Tag rules of scooppour3d: move a pile of cubes from the yellow bin to the green bin."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 22 runs (motion of the "
    "two bins and of the cubes, contact pairs). The scoop does not move in any run, "
    "so no tag group asks for it. No pictures and no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "No run uses the scoop: it does not move in any of the 22 runs.",
    "14 runs lift the yellow bin and pour the cubes into the green bin. 5 runs take the "
    "cubes out one at a time.",
    "2 runs (astra-source seed24 and seed444) move no cube: they lift the green bin and "
    "bring it to the yellow bin. 1 run puts the yellow bin, with the cubes in it, into "
    "the green bin.",
    "The two views see the transfer type (0.93).",
]
CODEBOOK = [
    {
        "key": "transfer",
        "question": "How do the cubes get into the green bin?",
        "source": "motion",
        "values": {
            "pour": "The robot lifts the yellow bin, carries it over the green bin "
            "and pours the cubes out.",
            "pick_each": "The yellow bin stays. The gripper takes the cubes out one "
            "at a time.",
            "nest_bin": "The robot puts the yellow bin, with the cubes in it, into "
            "the green bin. The cubes are not poured out.",
            "move_target": "No cube moves. The robot lifts the green bin and brings "
            "it to the yellow bin.",
        },
        "rule": "shift of the yellow bin (20 cm or more: pour or nest_bin; nest_bin "
        "if the cubes end 5 cm or more above the floor of the green bin), else "
        "move_target if no cube moves, else pick_each",
    },
    {
        "key": "bins",
        "question": "Do the two bins touch?",
        "source": "contacts",
        "values": {
            "apart": "No contact between the yellow bin and the green bin.",
            "touch": "The yellow bin touches the green bin.",
        },
        "rule": "contact pair of the yellow bin with the green bin",
    },
    {
        "key": "base",
        "question": "Does the base drive during the task?",
        "source": "motion",
        "values": {
            "stays": "The base path is below 2.5 m.",
            "drives": "The base path is 2.5 m or more: the base moves for each "
            "grasp.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "island",
        "question": "Does the base touch the kitchen island?",
        "source": "contacts",
        "values": {
            "clear": "No contact of the base with the island.",
            "bumps": "The base touches a part of the island.",
        },
        "rule": "contact pairs of the robot base with the kitchen island",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 300 env steps (30 s).",
            "slow": "More than 300 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    yellow = summary(run.ch("obj:bin_yellow_0"))
    green = summary(run.ch("obj:bin_green_0"))
    cubes = [summary(p) for n, p in run.objects.items() if n.startswith("cube")]
    moved = [c for c in cubes if c["shift"] > 0.05]
    height = float(
        np.mean([c["end"][2] for c in cubes]) - np.mean([c["start"][2] for c in cubes])
    )
    if yellow["shift"] >= 0.2:
        transfer = "nest_bin" if height >= 0.05 else "pour"
    elif not moved:
        transfer = "move_target"
    else:
        transfer = "pick_each"
    base_path = path_length(run.ch("base")[:, :2])
    firsts = sorted(c["first"] for c in moved)
    tags = {
        "transfer": transfer,
        "bins": "touch" if run.touches(r"bin_yellow~bin_green") else "apart",
        "base": "drives" if base_path >= 2.5 else "stays",
        "island": "bumps" if run.touches(r"^base~kitchen_island") else "clear",
        "pace": "quick" if run.steps <= 300 else "slow",
    }
    evidence = {
        "yellow_bin_shift_m": round(yellow["shift"], 2),
        "yellow_bin_lift_m": round(yellow["lift"], 2),
        "green_bin_shift_m": round(green["shift"], 2),
        "green_bin_lift_m": round(green["lift"], 2),
        "cubes_that_move": len(moved),
        "first_motion_of_the_cubes_steps": [firsts[0], firsts[-1]] if firsts else None,
        "cubes_end_height_change_m": round(height, 2),
        "gripper_touches_a_cube": run.touches(r"gripper\.pad~cube"),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
