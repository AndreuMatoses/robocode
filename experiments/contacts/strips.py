"""Film strips of final policies: a few pictures of each episode, side by side.

A film strip is the evidence that a reader looks at to check a semantic tag. The
pictures of a run are at equal steps of path length (the mean distance that the
bodies move), so a long wait gives one picture and a fast motion gives several.

``python -m experiments.contacts.strips <scenario> ...`` selects a few runs that
cover the values of the first tag groups, and writes for each scenario

* ``<traces>/_scenarios/<scenario>/strips/sheet.png``: all selected runs;
* one ``<method>_<seed>.png`` for each selected run;
* ``strips.json``: the runs, their files, the env steps of the pictures.

The viewer shows the strips on the Semantic tab. Options: ``--runs N`` (default 5),
``--frames K`` (default 6), ``--all`` (every run, not a selection).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from experiments.contacts import catalog, features, semantic

WIDTH, HEIGHT = 320, 240
LABEL = 18  # [px] text line above a strip


def key_steps(run: features.Run, count: int) -> list[int]:
    """``count`` env steps at equal steps of path length (first and last kept)."""
    names = sorted(run.motion.channels)
    steps = run.motion.steps
    if steps <= count:
        return list(range(steps))
    moved = sum(
        np.linalg.norm(np.diff(run.motion.channels[k], axis=0), axis=1) for k in names
    ) / len(names)
    length = np.concatenate([[0.0], np.cumsum(moved)])
    if length[-1] <= 0:
        at = np.linspace(0, steps - 1, count)
    else:
        at = np.interp(np.linspace(0, length[-1], count), length, np.arange(steps))
    picked = sorted({int(round(v)) for v in at} | {0, steps - 1})
    return picked


def select(tags: dict[str, Any], run_ids: list[str], limit: int) -> list[str]:
    """Runs that cover the values of the tag groups, the first groups first."""
    labels = tags["labels"]
    need = [
        (group["key"], value)
        for group in tags["codebook"]
        for value in group["values"]
        if any(labels[r][group["key"]] == value for r in run_ids)
    ]
    chosen: list[str] = []
    while need and len(chosen) < limit:
        goal = need[0]
        pool = [r for r in run_ids if r not in chosen and labels[r][goal[0]] == goal[1]]
        if not pool:
            need.pop(0)
            continue
        # the run that also covers the most other open values
        best = max(pool, key=lambda r: sum(labels[r][g] == v for g, v in need))
        chosen.append(best)
        need = [(g, v) for g, v in need if labels[best][g] != v]
    return chosen


def _fit(picture: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(picture)[..., :3].astype(np.uint8))
    image.thumbnail((WIDTH, HEIGHT))
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    canvas.paste(image, ((WIDTH - image.width) // 2, (HEIGHT - image.height) // 2))
    return canvas


def render_runs(runs: list[features.Run], steps: list[list[int]]) -> list[list[Any]]:
    """The pictures of each run at its steps (one scene for all runs)."""
    # pylint: disable=import-outside-toplevel
    backend = runs[0].meta.get("backend", "mujoco")
    out: list[list[Any]] = []
    if backend in ("kin2d", "pymunk"):
        from experiments.contacts import render2d

        for run, picked in zip(runs, steps):
            meta, poses = render2d.load(run.directory)
            frames = render2d.episode_frames(meta, poses[picked], WIDTH * 2, HEIGHT * 2)
            out.append([_fit(f) for f in frames])
        return out
    if backend == "bullet":
        from experiments.contacts import replay_bullet

        env, scene, _ = replay_bullet.open_scene(runs[0].directory)
        try:
            for run, picked in zip(runs, steps):
                states = replay_bullet.load_states(run.directory)[picked]
                out.append([_fit(f) for f in replay_bullet.frames(scene, states)])
        finally:
            env.close()
        return out
    from experiments.contacts import replay

    env, _, _ = replay.open_trace_env(runs[0].directory)
    try:
        for run, picked in zip(runs, steps):
            qpos = replay.load_qpos(run.directory)[picked]
            frames = replay.replay_frames(env, qpos, WIDTH * 2, HEIGHT * 2)
            out.append([_fit(f) for f in frames])
    finally:
        env.close()
    return out


def _strip(run: features.Run, picked: list[int], frames: list[Any], text: str):
    strip = Image.new("RGB", (WIDTH * len(frames), HEIGHT + LABEL), "white")
    draw = ImageDraw.Draw(strip)
    draw.text((4, 3), text, fill=(0, 0, 0))
    for column, (step, frame) in enumerate(zip(picked, frames)):
        strip.paste(frame, (column * WIDTH, LABEL))
        draw.rectangle(
            [column * WIDTH, LABEL, column * WIDTH + 62, LABEL + 13],
            fill=(255, 255, 255),
        )
        draw.text((column * WIDTH + 3, LABEL + 1), f"step {step}", fill=(190, 30, 30))
    return strip


def build(scenario: str, traces: Path, limit: int, count: int, every: bool) -> Path:
    """Write the film strips of a scenario; the folder of the files."""
    runs = features.load_runs(scenario)
    tags = semantic.load(scenario)
    by_id = {run.run_id: run for run in runs}
    if every or tags is None:
        chosen = list(by_id)
    else:
        chosen = select(tags, list(by_id), limit)
    picked = [key_steps(by_id[r], count) for r in chosen]
    pictures = render_runs([by_id[r] for r in chosen], picked)
    folder = traces / catalog.SCENARIO_DIR / scenario / "strips"
    folder.mkdir(parents=True, exist_ok=True)
    index, strips = [], []
    for run_id, steps, frames in zip(chosen, picked, pictures):
        run = by_id[run_id]
        labels = tags["labels"][run_id] if tags else {}
        text = f"{run.method} {run.seed} ({run.steps} steps)   " + "  ".join(
            f"{k}: {v}" for k, v in labels.items()
        )
        strip = _strip(run, steps, frames, text)
        name = f"{run.method}_{run.seed}.png"
        strip.save(folder / name)
        strips.append(strip)
        index.append({"run_id": run_id, "file": name, "steps": steps})
    width = max(s.width for s in strips)
    sheet = Image.new("RGB", (width, sum(s.height for s in strips)), "white")
    top = 0
    for strip in strips:
        sheet.paste(strip, (0, top))
        top += strip.height
    sheet.save(folder / "sheet.png")
    (folder / "strips.json").write_text(json.dumps(index, indent=1) + "\n")
    return folder


def main() -> None:
    """Write the film strips of the given scenarios."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("scenarios", nargs="+")
    parser.add_argument("--traces", type=Path, default=catalog.DEFAULT_OUT)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    for name in args.scenarios:
        folder = build(name, args.traces.resolve(), args.runs, args.frames, args.all)
        print(f"{name}: {folder / 'sheet.png'}")


if __name__ == "__main__":
    main()
