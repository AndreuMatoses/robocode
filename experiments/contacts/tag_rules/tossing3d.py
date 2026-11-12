"""Tag rules of tossing3d: throw a cube over a barrier into a bin."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run

METHOD = (
    "Tag groups proposed after a look at the release numbers of all 30 runs and at "
    "film strips of 18 runs. Each value is assigned by the rule of its group from "
    "numbers of the recorded episode; 'aim' is read in the program text."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "There are two swing types: 18 runs throw overhand (the cube starts behind the base), "
    "and 12 runs swing forward from the front. All opus55 and opus55-source runs throw "
    "overhand; all astra-source runs swing forward.",
    "The contact view does not see the swing type (0.49); the motion view sees it (0.83). "
    "No distance sees the direction of the cube at the release.",
    "8 programs simulate candidate throws before they act. All 8 are from methods that "
    "see the simulator code.",
    "1 run (opus55 seed222) throws at an angle of 29 degrees to the room axis, so its "
    "base drives less.",
]
CODEBOOK = [
    {
        "key": "swing",
        "question": "From where does the arm swing?",
        "source": "motion",
        "values": {
            "overhand": "The cube is behind the base before the swing. The arm "
            "swings up and over the top.",
            "underarm": "The cube is in front of the base, low or at middle height. "
            "The arm swings forward and up.",
        },
        "rule": "overhand if the end effector is behind the base (against the throw "
        "direction) in the last slow state before the swing",
    },
    {
        "key": "arc",
        "question": "In which direction does the cube leave the gripper?",
        "source": "motion",
        "values": {
            "lob": "The cube leaves at 25 degrees or more above the horizontal: a "
            "high arc.",
            "flat": "The cube leaves between -12 and 25 degrees.",
            "downward": "The cube leaves at less than -12 degrees: thrown down from "
            "a high release point.",
        },
        "rule": "elevation of the cube velocity in the first step after the release",
    },
    {
        "key": "alignment",
        "question": "Does the robot throw along the room axis?",
        "source": "motion",
        "values": {
            "straight": "The base drives to the line through the bin and throws "
            "straight along the room axis.",
            "diagonal": "The base stops before that line and throws at an angle, so "
            "it drives less.",
        },
        "rule": "diagonal if the direction from the base to the bin at the release "
        "is 3 degrees or more from the room axis",
    },
    {
        "key": "standoff",
        "question": "How far from the bin does the robot throw?",
        "source": "motion",
        "values": {
            "near": "Less than 2.25 m: close to the barrier.",
            "middle": "2.25 m to 3 m.",
            "far": "More than 3 m.",
        },
        "rule": "horizontal distance from the base to the bin at the release",
    },
    {
        "key": "base",
        "question": "Does the base move during the throw?",
        "source": "motion",
        "values": {
            "standing": "The base is at rest at the release.",
            "moving": "The base moves at 0.2 m/s or more at the release.",
        },
        "rule": "speed of the base at the release",
    },
    {
        "key": "aim",
        "question": "How does the program select the throw?",
        "source": "program",
        "values": {
            "calibrated": "Fixed motion or a table of constants that the agent "
            "tuned in its tests (open loop).",
            "rollout": "The program has a simulator model of its own and simulates "
            "candidate throws before it acts.",
        },
        "rule": "rollout if the program text has a rollout function, or a shadow "
        "or twin copy of the simulator that it steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    ee, base = run.ch("ee"), run.ch("base")
    cube, target = run.ch("obj:cube_0"), run.ch("obj:bin_0")
    held = np.flatnonzero(np.linalg.norm(cube - ee, axis=1) < 0.08)
    release = int(held[-1])  # last state with the cube at the gripper
    to_bin = target[0, :2] - base[release, :2]
    standoff = float(np.linalg.norm(to_bin))
    forward = to_bin / standoff
    heading = float(np.degrees(np.arctan2(to_bin[1], to_bin[0])))
    last = len(cube) - 1
    base_speed = float(
        np.linalg.norm((base[min(release + 1, last)] - base[release - 1])[:2])
        / (2 * run.dt)
    )
    velocity = (cube[min(release + 2, last)] - cube[min(release + 1, last)]) / run.dt
    elevation = float(np.degrees(np.arctan2(velocity[2], velocity[:2] @ forward)))
    # the wind-up: the last slow state in the 12 steps before the release
    low = max(release - 12, 1)
    speeds = np.linalg.norm(np.diff(ee[low - 1 : release + 1], axis=0), axis=1)
    windup = low + int(np.argmax(speeds > 0.25 * speeds.max())) - 1
    ahead = float((ee[windup, :2] - base[windup, :2]) @ forward)
    simulates = run.program_has(r"def \w*rollout|shadow|twin\.step")
    tags = {
        "swing": "overhand" if ahead < 0 else "underarm",
        "arc": "lob" if elevation >= 25 else "flat" if elevation >= -12 else "downward",
        "alignment": "diagonal" if abs(heading) >= 3 else "straight",
        "standoff": "near" if standoff < 2.25 else "far" if standoff > 3 else "middle",
        "base": "moving" if base_speed >= 0.2 else "standing",
        "aim": "rollout" if simulates else "calibrated",
    }
    evidence = {
        "windup_ahead_m": round(ahead, 2),
        "windup_height_m": round(float(ee[windup, 2]), 2),
        "release_height_m": round(float(ee[release, 2]), 2),
        "release_elevation_deg": round(elevation),
        "cube_speed_m_s": round(float(np.linalg.norm(velocity)), 1),
        "apex_m": round(float(cube[:, 2].max()), 2),
        "heading_deg": round(heading),
        "standoff_m": round(standoff, 2),
        "base_speed_m_s": round(base_speed, 1),
        "release_step": release,
        "rollout_words_in_program": simulates,
    }
    return tags, evidence
