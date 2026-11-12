"""Tag rules of basemotion3d: drive the base to a target region."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length

METHOD = (
    "Tag groups proposed after a look at the numbers of all 35 runs (base path and "
    "end position). All runs drive to the region in 3 env steps and touch nothing, "
    "so the scenario has almost no room for different solutions. No pictures and "
    "no program text were read."
)
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "This scenario has almost no room for different solutions: all 35 runs drive to the "
    "region in 3 env steps and touch nothing.",
    "The contact view finds 1 strategy. The motion view sees only small differences of "
    "the path (a bend, the stop point).",
]
CODEBOOK = [
    {
        "key": "route",
        "question": "Does the base drive in a straight line?",
        "source": "motion",
        "values": {
            "straight": "The base path is less than 1.02 times the straight line.",
            "bent": "1.02 times the straight line or more: the path has a bend.",
        },
        "rule": "length of the base path divided by the distance from start to end",
    },
    {
        "key": "stop",
        "question": "Where in the target region does the base stop?",
        "source": "motion",
        "values": {
            "edge": "Less than 1.05 m from the start: at the near edge of the "
            "region.",
            "inside": "1.05 m or more from the start: deeper in the region.",
        },
        "rule": "distance from the start to the end of the base",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    base = run.ch("base")[:, :2]
    shift = float(np.linalg.norm(base[-1] - base[0]))
    ratio = path_length(base) / shift
    tags = {
        "route": "straight" if ratio < 1.02 else "bent",
        "stop": "edge" if shift < 1.05 else "inside",
    }
    evidence = {
        "base_end": base[-1].round(3).tolist(),
        "base_shift_m": round(shift, 3),
        "path_ratio": round(ratio, 3),
        "steps": run.steps,
    }
    return tags, evidence
