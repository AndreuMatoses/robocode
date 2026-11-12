"""Tag rules of rovers: two rovers take samples and pictures and send them to a lander."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.contacts.features import Run, path_length

METHOD = (
    "Tag groups proposed after a look at the task text and at the numbers of all 34 "
    "runs (the samples that each rover drives to, the path of each rover, the order "
    "of the contact events). A sample counts as taken by a rover that comes within "
    "27 cm of it. No pictures and no program text were read."
)
NEAR = 0.27
# The main observations of this scenario, for its page in the viewer. Check them
# against the tag counts when a rule changes.
INSIGHTS = [
    "32 of 34 runs give all the sample work to rover1; rover0 takes pictures only. 2 runs "
    "split the samples between the rovers.",
    "28 runs take the same two samples, rock0 and soil0.",
    "The contact view finds 9 strategies (effective number 7.9), but the plan is almost "
    "the same in all runs. One difference is the order of the events: the first event is "
    "a sample in 22 runs and a picture in 12 runs.",
]
CODEBOOK = [
    {
        "key": "sampling",
        "question": "Which rover takes the samples?",
        "source": "motion",
        "values": {
            "rover1_both": "rover1 takes the stone sample and the soil sample. "
            "rover0 takes no sample.",
            "split": "Each rover takes one sample.",
            "rover0_both": "rover0 takes the two samples.",
        },
        "rule": "the rovers that come within 27 cm of a stone or a soil sample",
    },
    {
        "key": "samples",
        "question": "Which samples do the rovers take?",
        "source": "motion",
        "values": {
            "rock0_soil0": "rock0 and soil0 only: the two samples nearest to each "
            "other.",
            "other": "Another stone or soil sample, or more than two samples.",
        },
        "rule": "the samples that a rover comes within 27 cm of",
    },
    {
        "key": "first_event",
        "question": "What is the first event of the episode?",
        "source": "contacts",
        "values": {
            "picture": "A rover is at an objective before a rover is at a sample.",
            "sample": "A rover is at a sample before a rover is at an objective.",
            "together": "The two events start in the same contact mode.",
        },
        "rule": "first contact mode with an objective against the first contact "
        "mode with a stone or a soil sample",
    },
    {
        "key": "travel",
        "question": "How far do the two rovers drive?",
        "source": "motion",
        "values": {
            "short": "The two paths add up to less than 11 m.",
            "long": "11 m or more.",
        },
        "rule": "sum of the path lengths of the two rovers",
    },
    {
        "key": "pace",
        "question": "How long is the episode?",
        "source": "motion",
        "values": {
            "quick": "At most 26 env steps.",
            "slow": "More than 26 env steps.",
        },
        "rule": "number of env steps",
    },
]


def label(run: Run) -> tuple[dict[str, str], dict[str, Any]]:
    """Tags and evidence of one run."""
    taken: dict[str, list[str]] = {}
    for rover in ("rover0", "rover1"):
        path = run.ch("base:" + rover)[:, :2]
        taken[rover] = [
            name
            for name, pos in run.objects.items()
            if name[:4] in ("rock", "soil")
            and np.linalg.norm(path - pos[0, :2], axis=1).min() < NEAR
        ]
    if taken["rover0"] and taken["rover1"]:
        sampling = "split"
    else:
        sampling = "rover0_both" if taken["rover0"] else "rover1_both"
    every = sorted(taken["rover0"] + taken["rover1"])
    picture = run.first_contact(r"rover~objective")
    sample = run.first_contact(r"rover~(rock|soil)")
    if picture == sample:
        first = "together"
    else:
        first = "picture" if (picture or 0) < (sample or 0) else "sample"
    total = sum(path_length(run.ch("base:" + r)) for r in ("rover0", "rover1"))
    tags = {
        "sampling": sampling,
        "samples": "rock0_soil0" if every == ["rock0", "soil0"] else "other",
        "first_event": first,
        "travel": "short" if total < 11 else "long",
        "pace": "quick" if run.steps <= 26 else "slow",
    }
    evidence = {
        "samples_of_each_rover": taken,
        "path_rover0_m": round(path_length(run.ch("base:rover0")), 2),
        "path_rover1_m": round(path_length(run.ch("base:rover1")), 2),
        "contact_modes": len(run.modes),
        "steps": run.steps,
    }
    return tags, evidence
