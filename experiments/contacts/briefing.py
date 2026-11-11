"""What the agent was told: the task and environment description and the prompts.

Every run folder of the dataset has ``env_description.md`` (the description of the
task, the observation, the actions and the reward) and ``run_experiment.log``, which
starts with the system prompt and the prompt that the agent got. The description is a
part of that prompt. The text depends on the method: a black-box agent gets only the
description, a ``-source`` agent also gets the path of the source code, and the two
agent programs (Claude Code, Codex) have different system prompts.

Runs with the same three texts are one *variant*, so a scenario has about four.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

LOG_HEAD = 400_000  # [bytes] the prompts are at the start of the log
_STAMP = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+\]\[[^\]]+\]\[\w+\] - (.*)$")
_BLOCKS = {"System prompt:": "system", "Prompt:": "prompt"}


def prompts_of_log(log: Path) -> dict[str, str]:
    """The ``system`` prompt and the ``prompt`` of a run, as far as the log has them.

    A log record starts with a time stamp; the lines without one continue the record
    before. The log of a run that was only evaluated again has no prompt.
    """
    out: dict[str, list[str]] = {}
    current: str | None = None
    with log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle.read(LOG_HEAD).splitlines():
            record = _STAMP.match(line)
            if record is None:
                if current is not None:
                    out[current].append(line)
                continue
            if current == "prompt":
                break  # the record after the prompt
            current = _BLOCKS.get(record.group(1).strip())
            if current is not None:
                out[current] = []
    return {key: "\n".join(lines).strip() for key, lines in out.items()}


def _facts(run_dir: Path) -> dict[str, Any]:
    """The limits of a run, from its experiment configuration."""
    path = run_dir / ".hydra" / "config.yaml"
    if not path.exists():
        return {}
    config = yaml.safe_load(path.read_text()) or {}
    approach = config.get("approach") or {}
    environment = config.get("environment") or {}
    facts = {
        "budget_usd": approach.get("max_budget_usd"),
        "design_counts": environment.get("design_counts"),
        "eval_counts": environment.get("eval_counts"),
        "max_steps": config.get("max_steps"),
        "eval_timeout_s": config.get("eval_timeout"),
        "num_eval_tasks": config.get("num_eval_tasks"),
        "tools": config.get("mcp_tools"),
    }
    return {key: value for key, value in facts.items() if value is not None}


def _label(methods: dict[str, list[str]], split: set[str]) -> str:
    """Method names; with the seeds for a method that is in more than one variant."""
    return ", ".join(
        f"{method} ({', '.join(seeds)})" if method in split else method
        for method, seeds in methods.items()
    )


def scenario_briefing(
    data: Path, scenario: str, method_order: list[str]
) -> dict[str, Any]:
    """The variants of the information given to the agents of one scenario."""
    order = {method: i for i, method in enumerate(method_order)}
    run_dirs = sorted(
        (d for d in (data / "runs").glob(f"*/{scenario}/seed*") if d.is_dir()),
        key=lambda d: (
            order.get(d.parts[-3], len(order)),
            d.parts[-3],
            int(d.name.removeprefix("seed")),
        ),
    )
    variants: dict[tuple[str, str, str], dict[str, Any]] = {}
    silent: list[tuple[Path, str]] = []  # runs whose log has no prompt
    for run_dir in run_dirs:
        path = run_dir / "env_description.md"
        description = path.read_text(encoding="utf-8") if path.exists() else ""
        log = run_dir / "run_experiment.log"
        texts = prompts_of_log(log) if log.exists() else {}
        if "prompt" not in texts:
            silent.append((run_dir, description))
            continue
        _add(variants, (description, texts.get("system", ""), texts["prompt"]), run_dir)
    # A run without a prompt in its log: if other runs of its method have the same
    # description, it is only named; if not, its description is a variant of its own.
    without_prompt = []
    for run_dir, description in silent:
        method = run_dir.parts[-3]
        if any(
            k[0] == description and method in v["methods"]
            for k, v in variants.items()
            if k[2]
        ):
            without_prompt.append(f"{method}/{run_dir.name}")
        elif description:
            _add(variants, (description, "", ""), run_dir)
    found = list(variants.values())
    seen: dict[str, int] = {}
    for variant in found:
        for method in variant["methods"]:
            seen[method] = seen.get(method, 0) + 1
    split = {method for method, count in seen.items() if count > 1}
    for variant in found:
        variant["label"] = _label(variant["methods"], split)
    return {"scenario": scenario, "variants": found, "without_prompt": without_prompt}


def _add(
    variants: dict[tuple[str, str, str], dict[str, Any]],
    key: tuple[str, str, str],
    run_dir: Path,
) -> None:
    """Count a run in the variant of its three texts."""
    facts = _facts(run_dir)
    variant = variants.setdefault(
        key,
        {
            "methods": {},
            "runs": 0,
            "description": key[0],
            "system": key[1],
            "prompt": key[2],
            "facts": facts,
        },
    )
    # Only the limits that all runs of the variant share.
    variant["facts"] = {k: v for k, v in variant["facts"].items() if facts.get(k) == v}
    variant["methods"].setdefault(run_dir.parts[-3], []).append(run_dir.name)
    variant["runs"] += 1
