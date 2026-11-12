"""Tag rules of rearrange3d: move a boxed drink and a can on a shelf."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the object motion and the robot contacts of "
    "all 32 runs. Each value is assigned by the rule of its group from numbers of the "
    "recorded episode. Film strips of 5 runs were inspected; they agree with the tags "
    "(objects that lie on their side in the tipped runs). No program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "20 runs lift each object (pick and place), and 11 runs slide the objects on the "
    "shelf. No distance sees this difference well (0.60 and 0.61).",
    "All opus5-source, astra and astra-source runs lift.",
    "25 of 32 runs move the boxed drink first.",
    "The contact view sees the hold, the end pose and the side effects (about 0.9). Two "
    "of these three tag groups come from the contacts.",
]
CODEBOOK = [
    {
        "key": "transport",
        "question": "How does the robot move an object on the shelf?",
        "source": "motion",
        "values": {
            "lift": "Each moved object goes 8 cm or more above its start: pick and "
            "place.",
            "slide": "No moved object goes more than 5 cm above its start: the "
            "object slides on the shelf.",
            "mixed": "One object is lifted and one slides, or the lift is between 5 "
            "and 8 cm.",
        },
        "rule": "largest height of the boxed drink and of the can above their start",
    },
    {
        "key": "order",
        "question": "Which object moves first?",
        "source": "motion",
        "values": {
            "boxed_first": "The boxed drink moves before the can.",
            "can_first": "The can moves before the boxed drink (or the boxed drink "
            "does not move).",
        },
        "rule": "first env step in which each object moves",
    },
    {
        "key": "hold",
        "question": "Does the gripper hold each moved object with its pads?",
        "source": "contacts",
        "values": {
            "pinch": "A finger pad touches each object that moves.",
            "push": "An object moves with no pad contact: the fingers or the palm "
            "push it.",
        },
        "rule": "contact pairs of the gripper pads with the boxed drink and the can",
    },
    {
        "key": "base",
        "question": "How much does the base move?",
        "source": "motion",
        "values": {
            "fixed": "The base does not move (path below 5 cm).",
            "one_approach": "The base path is below 1.2 m: one approach.",
            "repositions": "The base path is 1.2 m or more: it moves again between "
            "the objects.",
        },
        "rule": "length of the base path",
    },
    {
        "key": "end_pose",
        "question": "Do the objects stay upright?",
        "source": "motion",
        "values": {
            "upright": "Each object ends at the height of its start.",
            "tipped": "The boxed drink or the can ends 3 cm or more below its "
            "start: it lies on its side.",
        },
        "rule": "height of each object in the last state against its start",
    },
    {
        "key": "side_effect",
        "question": "Does the robot disturb the scene?",
        "source": "motion and contacts",
        "values": {
            "clean": "No contact of the base with the furniture, and the bowl stays "
            "in place.",
            "bumps": "The base touches the furniture, or the bowl moves 2 cm or "
            "more.",
        },
        "rule": "contact pairs of the robot base, and the shift of the bowl",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    boxed, can = summary(run.ch("obj:boxed_drink_0")), summary(run.ch("obj:can_0"))
    bowl = summary(run.ch("obj:bowl_0"))
    base_path = path_length(run.ch("base")[:, :2])
    moved = {
        name: info
        for name, info in (("boxed_drink", boxed), ("can", can))
        if info["shift"] > 0.03
    }
    lifts = [info["lift"] for info in moved.values()]

    def first(info: dict[str, Any]) -> int:
        return (
            info["first"]
            if info["shift"] > 0.03 and info["first"] is not None
            else 10**9
        )

    pads = {p.split("~")[1] for p in run.pairs if p.startswith("gripper.pad~")}
    sunk = any(i["start"][2] - i["end"][2] >= 0.03 for i in (boxed, can))
    tags = {
        "transport": (
            "lift" if min(lifts) >= 0.08 else "slide" if max(lifts) <= 0.05 else "mixed"
        ),
        "order": "boxed_first" if first(boxed) < first(can) else "can_first",
        "hold": "pinch" if set(moved) <= pads else "push",
        "base": (
            "fixed"
            if base_path < 0.05
            else "one_approach" if base_path < 1.2 else "repositions"
        ),
        "end_pose": "tipped" if sunk else "upright",
        "side_effect": (
            "bumps" if run.touches(r"^base~") or bowl["shift"] >= 0.02 else "clean"
        ),
    }
    evidence = {
        "lift_boxed_m": round(boxed["lift"], 2),
        "lift_can_m": round(can["lift"], 2),
        "first_step_boxed": boxed["first"],
        "first_step_can": can["first"],
        "shift_boxed_m": round(boxed["shift"], 2),
        "shift_can_m": round(can["shift"], 2),
        "shift_bowl_m": round(bowl["shift"], 2),
        "base_path_m": round(base_path, 2),
        "end_boxed": boxed["end"],
        "end_can": can["end"],
        "robot_contacts": [
            p
            for p in run.pairs
            if p.split("~")[0].split(".")[0] in ("gripper", "arm", "base")
        ],
    }
    return tags, evidence
