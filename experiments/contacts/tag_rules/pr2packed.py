"""Tag rules of pr2packed: put every block on a small plate, packed together."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, moved_order, path_length, summary

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (order, end "
    "position and lift of the three blocks, base path). No pictures and no program "
    "text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "The blocks end in a triangle or an L shape in 26 runs, and in a straight row in 9 "
    "runs. No distance sees the layout (0.49 and 0.53).",
    "31 of 35 runs start with block1. Then block0 goes before block2 in 16 runs, and "
    "block2 before block0 in 15 runs.",
    "5 runs move the blocks at table height, with no lift.",
]
CODEBOOK = [
    {
        "key": "layout",
        "question": "How do the blocks lie on the plate?",
        "source": "motion",
        "values": {
            "row": "The three blocks end in one straight row.",
            "cluster": "The three blocks end in a triangle or an L shape.",
        },
        "rule": "the end positions of the blocks: row if the spread across the "
        "main direction is less than 20 % of the spread along it",
    },
    {
        "key": "order",
        "question": "In which order do the blocks go to the plate?",
        "source": "motion",
        "values": {
            "1_0_2": "block1, then block0, then block2.",
            "1_2_0": "block1, then block2, then block0.",
            "other": "Another order.",
        },
        "rule": "order of the first motion of each block",
    },
    {
        "key": "carry",
        "question": "How high does the robot lift a block?",
        "source": "motion",
        "values": {
            "level": "No block goes 3 cm or more above the table: the block moves "
            "at table height.",
            "low": "A block goes 3 cm to 15 cm above the table.",
            "high": "A block goes 15 cm or more above the table.",
        },
        "rule": "largest height of a block above its start",
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
            "quick": "At most 45 env steps.",
            "slow": "More than 45 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    info = {name: summary(pos) for name, pos in run.objects.items()}
    ends = np.array([i["end"][:2] for i in info.values()])
    spread = np.linalg.svd(ends - ends.mean(axis=0), compute_uv=False)
    flat = float(spread[1] / max(spread[0], 1e-9))
    order = "_".join(name[-1] for name in moved_order(run, shift=0.05, path=0.10))
    lift = max(i["lift"] for i in info.values())
    base_path = path_length(run.ch("base")[:, :2])
    tags = {
        "layout": "row" if flat < 0.2 else "cluster",
        "order": order if order in ("1_0_2", "1_2_0") else "other",
        "carry": "level" if lift < 0.03 else "low" if lift < 0.15 else "high",
        "drive": "long" if base_path >= 3 else "short",
        "pace": "quick" if run.steps <= 45 else "slow",
    }
    evidence = {
        "order": order,
        "end_of_each_block": {k: i["end"][:2] for k, i in info.items()},
        "spread_across_over_along": round(flat, 2),
        "largest_lift_m": round(lift, 2),
        "base_path_m": round(base_path, 2),
        "steps": run.steps,
    }
    return tags, evidence
