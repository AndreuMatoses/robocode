"""Tag rules of pr2blocked: get one green block onto the plate; a red block is in the way."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the task text (it names two solutions: move "
    "the blocker, or get a spare block from the far table) and at the numbers of all "
    "33 runs (which blocks move, where the blocker ends, base path). Film strips of 5 "
    "runs were inspected; the blocks are small in the pictures, and the pictures "
    "agree with the plan of each run. The program text of the 3 pen_direct runs was "
    "read to find how they take the block."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "The task text names two solutions. 24 of 33 runs move the red blocker and take the block from the pen. 6 runs drive to the far table and bring a spare block.",
    "3 runs take the block from the pen and do not move the blocker, in 6 to 8 env steps. The task text says that only a horizontal grasp clears the walls. The three programs (all from methods that see the simulator code) search grasps in a private copy of the simulator; one of them names the solution: a side grasp tilted by about 20 degrees.",
    "The blocker ends on the floor in 12 runs, on the plate in 6 runs and on the table in 6 runs.",
    "The two views see the plan (contact 0.91, motion 0.82).",
]
CODEBOOK = [
    {
        "key": "plan",
        "question": "Which green block goes to the plate, and how?",
        "source": "motion",
        "values": {
            "move_blocker": "The robot moves the red blocker, then takes the green "
            "block from the pen.",
            "fetch_spare": "The robot drives to the far table and brings a spare "
            "green block.",
            "pen_direct": "The robot takes the green block from the pen, and the "
            "blocker does not move.",
        },
        "rule": "the green block that moves, and if the blocker moves 5 cm or more",
    },
    {
        "key": "blocker",
        "question": "Where does the blocker end?",
        "source": "motion and contacts",
        "values": {
            "stays": "The blocker does not move.",
            "on_table": "The blocker ends at table height, on the near table.",
            "on_plate": "The blocker touches the plate.",
            "on_floor": "The blocker ends 50 cm or more below its start: on the "
            "floor.",
        },
        "rule": "shift and drop of the blocker, and its contact with the plate",
    },
    {
        "key": "drive",
        "question": "How far does the base drive?",
        "source": "motion",
        "values": {
            "short": "The base path is below 3 m.",
            "long": "The base path is 3 m or more.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 37 env steps.",
            "slow": "More than 37 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    blocker = summary(run.ch("obj:blocker"))
    order = moved_order(run, shift=0.05, path=0.10)
    greens = [name for name in order if name.startswith("green")]
    moved = blocker["shift"] >= 0.05
    if greens and greens[-1] != "green0":
        plan = "fetch_spare"
    else:
        plan = "move_blocker" if moved else "pen_direct"
    if not moved:
        fate = "stays"
    elif blocker["start"][2] - blocker["end"][2] >= 0.5:
        fate = "on_floor"
    elif run.touches(r"blocker~plate"):
        fate = "on_plate"
    else:
        fate = "on_table"
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "plan": plan,
        "blocker": fate,
        "drive": "long" if base_path >= 3 else "short",
        "pace": "quick" if run.steps <= 37 else "slow",
    }
    evidence = {
        "blocks_that_move": order,
        "blocker_shift_m": round(blocker["shift"], 2),
        "blocker_end": blocker["end"],
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
