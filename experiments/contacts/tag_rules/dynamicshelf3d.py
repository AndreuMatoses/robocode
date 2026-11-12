"""Tag rules of dynamicshelf3d: take cubes from the floor to a shelf of a cupboard."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length

METHOD = (
    "Tag groups proposed after a look at the numbers of all 29 runs (order, end "
    "position of each cube on the shelf, base path, contact pairs). Film strips of 3 "
    "runs were inspected; the camera shows the cupboard only, and the pictures agree "
    "with the layout tags. No program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "The cubes end close together near the middle of the shelf in 18 runs, and one on "
    "each side in 11 runs. No distance sees this difference (0.54 and 0.52).",
    "27 of 29 runs take the nearer cube first.",
    "No opus55, opus55-source or astra run touches the cupboard. The 9 runs that touch it "
    "are from the other methods.",
]
CODEBOOK = [
    {
        "key": "layout",
        "question": "Where do the two cubes go on the shelf?",
        "source": "motion",
        "values": {
            "close": "The cubes end less than 20 cm apart, near the middle of the "
            "shelf.",
            "apart": "The cubes end 20 cm or more apart: one on each side of the "
            "shelf.",
        },
        "rule": "distance between the two cubes along the shelf in the last state",
    },
    {
        "key": "depth",
        "question": "How deep on the shelf do the cubes go?",
        "source": "motion",
        "values": {
            "front": "The cubes end at the front part of the shelf (x below 1.52 m).",
            "deep": "The cubes end at x = 1.52 m or more.",
        },
        "rule": "mean x of the two cubes in the last state",
    },
    {
        "key": "order",
        "question": "Which cube goes first?",
        "source": "motion",
        "values": {
            "near_first": "cube1, the cube nearer to the robot at the start.",
            "far_first": "cube2, the cube on the other side of the room.",
        },
        "rule": "the cube with the first motion",
    },
    {
        "key": "insertion",
        "question": "Does the robot touch the cupboard when it puts a cube in?",
        "source": "contacts",
        "values": {
            "free": "No contact of the arm or the gripper with the cupboard.",
            "touches": "The gripper or an arm link touches the cupboard.",
        },
        "rule": "contact pairs of a robot part with a part of the cupboard",
    },
    {
        "key": "pick",
        "question": "Does the gripper touch the floor at the grasp?",
        "source": "contacts",
        "values": {
            "floor_touch": "A finger pad touches the floor at a grasp.",
            "above_floor": "No pad contact with the floor.",
        },
        "rule": "contact pair of a finger pad with the floor",
    },
    {
        "key": "base",
        "question": "How far does the base drive?",
        "source": "motion",
        "values": {
            "short": "The base path is below 4.6 m.",
            "long": "The base path is 4.6 m or more.",
        },
        "rule": "length of the base path",
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
    one, two = run.ch("obj:cube1")[-1], run.ch("obj:cube2")[-1]
    gap = float(abs(one[1] - two[1]))
    depth = float((one[0] + two[0]) / 2)
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "layout": "close" if gap < 0.2 else "apart",
        "depth": "deep" if depth >= 1.52 else "front",
        "order": "near_first" if moved_order(run)[0] == "cube1" else "far_first",
        "insertion": (
            "touches" if run.touches(r"^(arm|gripper)\.\w+~cupboard") else "free"
        ),
        "pick": "floor_touch" if run.touches(r"gripper\.pad~floor") else "above_floor",
        "base": "long" if base_path >= 4.6 else "short",
        "pace": "quick" if run.steps <= 300 else "slow",
    }
    evidence = {
        "end_cube1": one.round(2).tolist(),
        "end_cube2": two.round(2).tolist(),
        "gap_on_shelf_m": round(gap, 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
        "robot_contacts": [
            p for p in run.pairs if p.split(".")[0] in ("arm", "gripper")
        ],
    }
    return tags, evidence
