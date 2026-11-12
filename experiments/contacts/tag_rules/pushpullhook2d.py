"""Tag rules of pushpullhook2d: move a button onto the target button with a hook."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (place of the "
    "base at the push, place of the grasp on the hook, path of the hook, contact "
    "pairs). All runs take the hook and push the button with it. Film strips of 2 "
    "runs were inspected; they agree with the tags. No program text was read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "All 35 runs take the hook and push the button with it.",
    "The runs split in two halves by the side from which the robot works: 18 from the "
    "left of the button, 17 from below or from the right.",
    "The side depends on the method: all opus55-source and astra-source runs work from "
    "the left, and all opus5 and sol runs work from the right.",
]
CODEBOOK = [
    {
        "key": "side",
        "question": "From which side of the button does the robot work?",
        "source": "motion",
        "values": {
            "from_left": "The base is on the left of the button at the push "
            "(x below 1.7 m).",
            "from_right": "The base is below or on the right of the button "
            "(x of 1.7 m or more).",
        },
        "rule": "x of the base at the first motion of the button (the button "
        "starts at x = 1.92 m)",
    },
    {
        "key": "grip",
        "question": "Where does the robot hold the hook?",
        "source": "motion",
        "values": {
            "end": "70 cm or more from the centre of the hook: at the end of the "
            "long bar.",
            "middle": "Less than 70 cm from the centre of the hook.",
        },
        "rule": "distance from the end effector to the centre of the hook in the "
        "last state",
    },
    {
        "key": "hook_path",
        "question": "Does the hook go straight to the button?",
        "source": "motion",
        "values": {
            "direct": "The path of the hook is less than 1.3 times the straight "
            "line.",
            "detour": "1.3 times the straight line or more.",
        },
        "rule": "path length of the hook divided by its shift",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 55 env steps.",
            "slow": "More than 55 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    button, hook = run.ch("obj:movable_button"), run.ch("obj:hook")
    base, ee = run.ch("base"), run.ch("ee")
    first = summary(button)["first"] or 0
    grip = float(np.linalg.norm(ee[-1] - hook[-1]))
    info = summary(hook)
    ratio = info["path"] / info["shift"]
    tags = {
        "side": "from_left" if base[first][0] < 1.7 else "from_right",
        "grip": "end" if grip >= 0.7 else "middle",
        "hook_path": "direct" if ratio < 1.3 else "detour",
        "pace": "quick" if run.steps <= 55 else "slow",
    }
    evidence = {
        "base_at_the_push": np.asarray(base[first], dtype=float).round(2).tolist(),
        "grip_to_hook_centre_m": round(grip, 2),
        "hook_path_ratio": round(ratio, 2),
        "button_shift": np.asarray(button[-1] - button[0], dtype=float)
        .round(2)
        .tolist(),
        "base_path_m": round(path_length(base), 2),
        "steps": run.steps,
    }
    return tags, evidence
