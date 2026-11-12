"""Tag rules of clutteredstorage2d: put all blocks inside a shelf."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 31 runs (which blocks "
    "move and in which order, end position of each block, contact pairs). Blocks 0 "
    "to 2 start at the mouth of the shelf; blocks 3 to 6 start in the room. No "
    "pictures and no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "Three blocks start at the mouth of the shelf. 21 of 31 runs move the three deeper "
    "first: 12 from right to left and 9 from left to right. 9 runs move only one or two "
    "of them.",
    "The room blocks end in one column of the shelf in 10 runs, and in three columns or "
    "more in 15 runs.",
    "The contact view finds 3 strategies and sees none of these differences (all near "
    "0.55). The motion view sees the order of the two farthest blocks (0.96).",
]
CODEBOOK = [
    {
        "key": "shelf_blocks",
        "question": "What does the robot do with the three blocks at the shelf?",
        "source": "motion",
        "values": {
            "right_to_left": "It moves the three blocks deeper, block2 first, then "
            "block1, then block0.",
            "left_to_right": "It moves the three blocks deeper, block0 first, then "
            "block1, then block2.",
            "partial": "It moves one or two of the three blocks only.",
            "other": "It moves the three blocks in another order, or between the "
            "room blocks.",
        },
        "rule": "the blocks 0 to 2 that move 5 cm or more, and their order",
    },
    {
        "key": "columns",
        "question": "In how many columns of the shelf do the room blocks end?",
        "source": "motion",
        "values": {
            "one": "The four room blocks end in one column (the same x within "
            "15 cm).",
            "two": "They end in two columns.",
            "three_or_more": "They end in three columns or more.",
        },
        "rule": "number of groups of the end x of the blocks 3 to 6 (a gap of more "
        "than 15 cm splits two groups)",
    },
    {
        "key": "far_order",
        "question": "Which of the two farthest blocks goes first?",
        "source": "motion",
        "values": {
            "block3_first": "block3 moves before block4.",
            "block4_first": "block4 moves before block3.",
        },
        "rule": "first motion of block3 and of block4",
    },
    {
        "key": "base",
        "question": "Does the base touch a block?",
        "source": "contacts",
        "values": {
            "clear": "No contact of the base with a block.",
            "bumps": "The base touches a block.",
        },
        "rule": "contact pair of the base with a block",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 420 env steps.",
            "slow": "More than 420 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    order = [name[-1] for name in moved_order(run, shift=0.05, path=0.05)]
    near = "".join(k for k in order if k in "012")
    lead = "".join(order[: len(near)])
    if len(near) < 3:
        shelf = "partial"
    elif lead == "210":
        shelf = "right_to_left"
    elif lead == "012":
        shelf = "left_to_right"
    else:
        shelf = "other"
    ends = sorted(summary(run.ch(f"obj:block{k}"))["end"][0] for k in "3456")
    columns = int(1 + (np.diff(ends) > 0.15).sum())
    tags = {
        "shelf_blocks": shelf,
        "columns": (
            "one" if columns == 1 else "two" if columns == 2 else "three_or_more"
        ),
        "far_order": (
            "block3_first" if order.index("3") < order.index("4") else "block4_first"
        ),
        "base": "bumps" if run.touches(r"^base~block") else "clear",
        "pace": "quick" if run.steps <= 420 else "slow",
    }
    evidence = {
        "order": "".join(order),
        "end_x_of_blocks_3_to_6": [round(v, 2) for v in ends],
        "base_path_m": round(path_length(run.ch("base")), 2),
        "steps": run.steps,
    }
    return tags, evidence
