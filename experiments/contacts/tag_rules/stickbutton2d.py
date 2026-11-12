"""Tag rules of stickbutton2d: touch all buttons; two are on a table, out of reach."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, nearest_gap, path_length

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (path and end of "
    "the stick, end of the base, contact pairs) and at the place of the three buttons "
    "in the scene: button0 and button1 are on the table, button2 is on the floor. All "
    "runs take the stick. Film strips of 3 runs (one for each plan) were inspected. "
    "They showed that the base, not the stick, is at the floor button in the plan "
    "table_then_drive, so the rule of floor_button now uses distances. No program "
    "text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "All 35 runs take the stick. Two buttons are on the table, out of reach.",
    "The stick is necessary for the two table buttons only. 26 of 35 runs press the "
    "floor button with the robot base; 9 runs press all three buttons with the stick.",
    "There are three plans. 18 runs press the table buttons with the stick, then drive "
    "the base onto the floor button. 9 runs press the table buttons, then reach to the "
    "floor button with the stick from a distance. 8 runs press the floor button with "
    "the base first, then the table buttons.",
    "The motion view sees the plan (0.91). The contact view does not (0.55), with 11 "
    "contact strategies.",
]
FLOOR_BUTTON = np.array([2.82, 0.06])  # centre of button2 in the first start scene
PRESS = 0.06  # [m] the button radius plus 1 cm
CODEBOOK = [
    {
        "key": "plan",
        "question": "In which order does the robot press the buttons, and from "
        "where?",
        "source": "motion",
        "values": {
            "table_then_reach": "The table buttons first. Then the robot reaches "
            "with the stick to the floor button from a distance (the base stays on "
            "the left of x = 2.2 m).",
            "table_then_drive": "The table buttons first. Then the base drives to "
            "the floor button with the stick in the gripper.",
            "floor_then_table": "The floor button first. The episode ends with the "
            "stick at the table buttons.",
        },
        "rule": "height of the stick in the last state (1.2 m or more: at the "
        "table), and x of the base in the last state",
    },
    {
        "key": "floor_button",
        "question": "What presses the floor button?",
        "source": "motion",
        "values": {
            "stick": "The stick comes within 6 cm of the centre of the floor "
            "button, and no robot part does.",
            "robot": "The base, the arm or the gripper head comes within 6 cm of "
            "the centre of the floor button: the robot presses it directly.",
        },
        "rule": "smallest distance over the episode from the outline of the stick "
        "and of each robot part to the centre of the floor button (the button has a "
        "radius of 5 cm). The contact pairs are not used, because the press can be "
        "the last instant of the episode",
    },
    {
        "key": "walls",
        "question": "Does the base touch a wall?",
        "source": "contacts",
        "values": {
            "clear": "No contact of the base with a wall.",
            "touches": "The base touches a wall.",
        },
        "rule": "contact pair of the base with a wall",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 99 env steps.",
            "slow": "More than 99 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    stick, base = run.ch("obj:stick"), run.ch("base")
    if stick[-1][1] >= 1.2:
        plan = "floor_then_table"
    else:
        plan = "table_then_reach" if base[-1][0] < 2.2 else "table_then_drive"
    gaps = {
        part.split(".")[-1]: round(nearest_gap(run, part, FLOOR_BUTTON), 3)
        for part in ("env.stick", "robot.base", "robot.arm", "robot.gripper.head")
    }
    robot = min(gaps["base"], gaps["arm"], gaps["head"]) <= PRESS
    tags = {
        "plan": plan,
        "floor_button": "robot" if robot else "stick",
        "walls": "touches" if run.touches(r"^base~\w+_wall") else "clear",
        "pace": "quick" if run.steps <= 99 else "slow",
    }
    evidence = {
        "stick_end": np.asarray(stick[-1], dtype=float).round(2).tolist(),
        "base_end": np.asarray(base[-1], dtype=float).round(2).tolist(),
        "base_path_m": round(path_length(base), 2),
        "steps": run.steps,
        "nearest_to_floor_button_m": gaps,
        "contacts": run.pairs,
    }
    return tags, evidence
