"""Tag rules of sortclutteredblocks3d: sort a pile of cubes into bins by colour."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, bouts, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 27 runs (order, lift and "
    "path of each cube, base path, contact pairs). The bin of each cube is given by "
    "the task. Film strips of 5 runs were inspected; they agree with the tags. No "
    "program text was read."
)
CUBES = ("cube1", "cube2", "cube3", "cube4")
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "The task gives the bin of each cube, so the differences are small: which cube goes "
    "first (cube1 in 15 runs, cube4 in 8 runs) and the style of the grasp.",
    "In 21 of 27 runs a grasp disturbs another cube of the pile.",
    "The contact view finds 12 strategies, but no distance sees a tag group well (all "
    "below 0.75).",
]
CODEBOOK = [
    {
        "key": "first",
        "question": "Which cube does the robot take first?",
        "source": "motion",
        "values": {
            "cube1": "cube1 (on the far side of the pile).",
            "cube4": "cube4 (on the far side of the pile).",
            "cube2": "cube2 (on the near side of the pile).",
            "cube3": "cube3 (on the near side of the pile).",
        },
        "rule": "the cube with the first motion",
    },
    {
        "key": "pile",
        "question": "Does a grasp disturb the other cubes of the pile?",
        "source": "contacts and motion",
        "values": {
            "clean": "No contact between two cubes, and each cube moves one time.",
            "disturbs": "Two cubes touch, or a cube moves two times or more: a "
            "grasp pushes a neighbour.",
        },
        "rule": "contact pair cube with cube, and the number of separate motions "
        "of each cube",
    },
    {
        "key": "carry",
        "question": "How high does the robot carry a cube?",
        "source": "motion",
        "values": {
            "low": "No cube goes 20 cm or more above its start.",
            "high": "A cube goes 20 cm or more above its start.",
        },
        "rule": "largest height of a cube above its start",
    },
    {
        "key": "pick",
        "question": "Does the gripper touch the table at a grasp?",
        "source": "contacts",
        "values": {
            "table_touch": "A finger pad touches the table top.",
            "above_table": "No pad contact with the table top.",
        },
        "rule": "contact pair of a finger pad with the table top",
    },
    {
        "key": "base",
        "question": "Does the base move between the cubes?",
        "source": "motion",
        "values": {
            "stays": "The base path is below 1 m.",
            "moves": "The base path is 1 m or more.",
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
    order = moved_order(run)
    lift = max(summary(run.ch("obj:" + c))["lift"] for c in CUBES)
    motions = {c: bouts(run.ch("obj:" + c)) for c in CUBES}
    base_path = path_length(run.ch("base")[:, :2])
    touch = run.touches(r"^cube~cube$")
    tags = {
        "first": order[0],
        "pile": "disturbs" if touch or max(motions.values()) >= 2 else "clean",
        "carry": "high" if lift >= 0.20 else "low",
        "pick": "table_touch" if run.touches(r"gripper\.pad~table") else "above_table",
        "base": "moves" if base_path >= 1.0 else "stays",
        "pace": "quick" if run.steps <= 300 else "slow",
    }
    evidence = {
        "order": order,
        "largest_lift_m": round(lift, 2),
        "motions_of_each_cube": motions,
        "cubes_touch": touch,
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
