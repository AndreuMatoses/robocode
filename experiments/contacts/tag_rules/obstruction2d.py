"""Tag rules of obstruction2d: clear a surface and put the target block on it."""

from __future__ import annotations

from typing import Any

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (order, end "
    "position and path of the two obstructions and of the target block, contact "
    "pairs). The scene is a side view: y is up. No pictures and no program text "
    "were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "29 of 35 runs move obstruction1 first, the one nearer to the target block.",
    "24 runs put one obstruction on each side, and 8 runs put the two on the right.",
    "25 runs carry the target block up and over. 10 runs move it low and direct.",
    "The contact view finds 6 strategies, but it sees none of the tag groups (all near "
    "0.5).",
]
CODEBOOK = [
    {
        "key": "order",
        "question": "Which obstruction moves first?",
        "source": "motion",
        "values": {
            "obstruction1_first": "obstruction1 (the one nearer to the target "
            "block) moves first.",
            "obstruction0_first": "obstruction0 moves first.",
        },
        "rule": "first motion of each obstruction",
    },
    {
        "key": "sides",
        "question": "To which side does each obstruction go?",
        "source": "motion",
        "values": {
            "split": "One obstruction ends on the left of its start and one on the "
            "right.",
            "both_right": "Both end on the right of their start (away from the "
            "target block).",
            "both_left": "Both end on the left of their start.",
            "one_only": "Only one obstruction ends 3 cm or more from its start.",
        },
        "rule": "sign of the x shift of each obstruction (3 cm or more)",
    },
    {
        "key": "distance",
        "question": "How far do the obstructions go?",
        "source": "motion",
        "values": {
            "just_off": "No obstruction ends 20 cm or more from its start.",
            "far": "An obstruction ends 20 cm or more from its start.",
        },
        "rule": "largest shift of an obstruction",
    },
    {
        "key": "carry",
        "question": "How does the target block go to the surface?",
        "source": "motion",
        "values": {
            "direct": "The path of the target block is less than 1.5 times the "
            "straight line: a low and direct move.",
            "up_and_over": "1.5 times or more: the block goes up, across and down.",
        },
        "rule": "path length of the target block divided by its shift",
    },
    {
        "key": "base",
        "question": "Does the base or the arm touch a block?",
        "source": "contacts",
        "values": {
            "clear": "Only the gripper touches the blocks.",
            "bumps": "The base or the arm touches a block.",
        },
        "rule": "contact pairs of the base or the arm with a block",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 59 env steps.",
            "slow": "More than 59 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    zero, one = summary(run.ch("obj:obstruction0")), summary(run.ch("obj:obstruction1"))
    target = summary(run.ch("obj:target_block"))
    order = moved_order(run, shift=0.02, path=0.04)
    shifts = [info["end"][0] - info["start"][0] for info in (zero, one)]
    signs = [0 if abs(dx) < 0.03 else (1 if dx > 0 else -1) for dx in shifts]
    if 0 in signs:
        sides = "one_only"
    elif signs[0] != signs[1]:
        sides = "split"
    else:
        sides = "both_right" if signs[0] > 0 else "both_left"
    ratio = target["path"] / target["shift"]
    first = next((n for n in order if n.startswith("obstruction")), "obstruction0")
    tags = {
        "order": first + "_first",
        "sides": sides,
        "distance": "far" if max(zero["shift"], one["shift"]) >= 0.20 else "just_off",
        "carry": "direct" if ratio < 1.5 else "up_and_over",
        "base": (
            "bumps"
            if run.touches(r"^(base|arm)~(obstruction|target_block)")
            else "clear"
        ),
        "pace": "quick" if run.steps <= 59 else "slow",
    }
    evidence = {
        "order": order,
        "x_shift_obstruction0_m": round(shifts[0], 2),
        "x_shift_obstruction1_m": round(shifts[1], 2),
        "target_path_ratio": round(ratio, 2),
        "base_path_m": round(path_length(run.ch("base")), 2),
        "steps": run.steps,
    }
    return tags, evidence
