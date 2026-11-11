"""Record the contact trace of a dataset policy on chosen instances.

For each instance this rolls the policy out once (no rendering), logging every
MuJoCo contact at substep resolution together with the simulator configuration at
each env step, then renders the video from the recorded configurations. Output:

    outputs/contact_traces/<method>/<env>/seed<N>/commit<k>/inst<seed>[_count<c>]/
        trace.npz   contacts (see recorder.py) + qpos, actions, rewards
        meta.json   run, commit, instance, outcome, entity table
        video.mp4

Every run uses the same instance seeds, so traces are comparable across runs. They
are not the scored episodes (the dataset withholds the eval seed); each trace
carries its own outcome. The policy is model-written code and runs in this process.

    python -m experiments.contacts.record opus55/tossing3d/seed24
    python -m experiments.contacts.record 'opus55/tossing3d/*' --instances 0 1 2
    python -m experiments.contacts.record '*/tossing3d/*' --commit 0 --count 2
    python -m experiments.contacts.record '*/tossing3d/*' --commits versions \
        --instances 0 --no-video --jobs 16

``--commits`` records the history of a run: ``versions`` = every commit that changes
the policy (see ``commits.py``) plus the last one, ``sample`` = those plus one
unchanged commit per streak (their traces must be identical), ``all`` = every commit.

Several (run, commit) tasks are recorded one subprocess each, so one policy's modules
and crashes cannot affect the next; ``--jobs`` runs that many at a time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.contacts import render2d, replay_bullet
from experiments.contacts.backend import backend_of_scenario, make_recorder, recording
from experiments.contacts.commits import load_versions, select_commits
from experiments.contacts.entities import robot_env
from experiments.contacts.recorder import STEP_COLUMNS
from experiments.contacts.replay import (
    render_camera_name,
    replay_frames,
    set_qpos,
    write_diagram,
    write_video,
)
from experiments.render_run import DEFAULT_DATA, REPO, max_steps_for, open_rollout
from experiments.results_viewer import _ActionRecorder
from robocode.environments.mujoco_gl import configure_gl_backend
from robocode.utils.episode import reset_for_episode, run_episode

DEFAULT_OUT = REPO / "outputs" / "contact_traces"
FPS = 10  # one frame per env step at 10 Hz control: videos play in real time


def instance_dir(
    out: Path, run_id: str, commit: int, seed: int, count: int | None
) -> Path:
    """Where the trace of one (run, commit, instance) lives."""
    name = f"inst{seed}" + (f"_count{count}" if count is not None else "")
    return out / run_id / f"commit{commit}" / name


def diagram_dir(out: Path, scenario: str, seed: int, count: int | None) -> Path:
    """Where the labelled diagram of one scenario instance lives."""
    name = f"inst{seed}" + (f"_count{count}" if count is not None else "")
    return out / "_scenarios" / scenario / name


def matching_runs(data: Path, patterns: list[str]) -> list[str]:
    """Run ids of the agentic methods (those with commits) matching the globs."""
    found: list[str] = []
    for table in sorted((data / "data" / "commits").glob("*.parquet")):
        runs = pd.read_parquet(table, columns=["run_id"]).run_id.unique()
        found += [r for r in runs if any(fnmatch.fnmatch(r, p) for p in patterns)]
    return sorted(set(found))


def record_instance(
    ro: Any, slot: list, seed: int, count: int | None, max_steps: int, backend: str
) -> tuple[dict[str, np.ndarray], dict[str, Any], Any]:
    """Roll the policy out on one instance; return (arrays, metrics, recorder)."""
    state, info = reset_for_episode(ro.env, seed, count)
    recorder = make_recorder(backend, ro.env)
    tape = _ActionRecorder(ro.approach)
    slot[0] = recorder
    started = time.monotonic()
    try:
        metrics, _, _ = run_episode(
            ro.env, tape, seed, max_steps, initial=(state, info)  # type: ignore[arg-type]
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        traceback.print_exc()
        metrics = {"crashed": f"{type(exc).__name__}: {exc}", "solved": False}
    finally:
        slot[0] = None
    metrics["wall_time_s"] = round(time.monotonic() - started, 1)
    if "object_count" not in metrics and "object_count" in info:
        metrics["object_count"] = info["object_count"]
    arrays = recorder.finish()
    metrics["num_steps"] = len(arrays["qpos"]) - 1
    try:
        arrays["actions"] = np.asarray(tape.actions, dtype=np.float32)
    except ValueError:  # ragged: some steps used a control schedule
        pass
    return arrays, metrics, recorder


def record_run(args: argparse.Namespace, run_id: str) -> None:
    """Record every requested instance of one run at one commit."""
    configure_gl_backend()
    backend = backend_of_scenario(run_id.split("/")[1])
    with open_rollout(args.data, run_id, args.commit) as ro, recording(backend) as slot:
        commit = ro.commits.iloc[ro.pos]
        print(f"{run_id} commit {ro.pos}/{len(ro.commits) - 1}: {commit.subject}")
        max_steps = max_steps_for(ro, args.count)
        for seed in args.instances:
            out = instance_dir(args.out, run_id, ro.pos, seed, args.count)
            if (out / "meta.json").exists() and not args.force:
                print(f"  instance {seed}: already recorded")
                continue
            arrays, metrics, recorder = record_instance(
                ro, slot, seed, args.count, max_steps, backend
            )
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "trace.npz", **arrays)
            camera = render_camera_name(ro.env) if backend == "mujoco" else backend
            reeval = commit.reeval_solve_rate
            meta = {
                "backend": backend,
                "run_id": run_id,
                "method": run_id.split("/")[0],
                "environment": run_id.split("/")[1],
                "model": str(ro.row.model),
                "commit": ro.pos,
                "num_commits": len(ro.commits),
                "sha": str(commit.sha),
                "subject": str(commit.subject),
                "commit_solve_rate": None if pd.isna(reeval) else float(reeval),
                "run_solve_rate": float(ro.row.solve_rate),
                "instance_seed": seed,
                "pinned_count": args.count,
                "metrics": metrics,
                "fps": FPS,
                "camera": camera,
                "dt": recorder.dt,
                "num_ticks": recorder.tick,
                "step_columns": STEP_COLUMNS,
                **recorder.emap.to_json(),
                **recorder.extra_meta(),
            }
            diagram = diagram_dir(args.out, run_id.split("/")[1], seed, args.count)
            if backend in render2d.BACKENDS:
                outlines = render2d.frames_of(meta, arrays)
                frames = render2d.episode_frames(
                    meta, outlines, args.width, args.height
                )
                if not (diagram / "diagram.json").exists():
                    render2d.write_diagram(
                        meta, outlines[0], diagram, args.width, args.height
                    )
            elif backend == "bullet":
                frames = replay_bullet.frames(recorder.scene, arrays["qpos"])
                if not (diagram / "diagram.json").exists():
                    replay_bullet.write_diagram(
                        recorder.scene, meta, arrays["qpos"][0], diagram
                    )
            else:
                frames = replay_frames(
                    ro.env, arrays["qpos"], args.width, args.height, camera
                )
            if not args.no_video:
                write_video(frames, out / "video.mp4", FPS)
            if backend == "mujoco" and not (diagram / "diagram.json").exists():
                set_qpos(robot_env(ro.env), arrays["qpos"][0])
                write_diagram(ro.env, recorder.emap, diagram, args.width, args.height)
            (out / "meta.json").write_text(json.dumps(meta))
            print(
                f"  instance {seed}: {metrics}; {len(arrays['keys'])} contact keys, "
                f"{len(arrays['intervals'])} intervals -> {out}"
            )


def _tasks(args: argparse.Namespace, runs: list[str]) -> list[tuple[str, int]]:
    """The (run, commit) pairs to record; commit -1 is resolved by the worker."""
    if args.commits is None:
        return [(run_id, args.commit) for run_id in runs]
    tasks = []
    for run_id in runs:
        versions = load_versions(args.data, args.out, run_id)
        for commit in select_commits(versions, args.commits):
            done = all(
                (
                    instance_dir(args.out, run_id, commit, seed, args.count)
                    / "meta.json"
                ).exists()
                for seed in args.instances
            )
            if args.force or not done:
                tasks.append((run_id, commit))
    return tasks


def main() -> None:
    """Record one run in this process, or several in one subprocess each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("runs", nargs="+", help="run ids or globs, '*/tossing3d/*'")
    parser.add_argument("--commit", type=int, default=-1, help="-1 = last (default)")
    parser.add_argument(
        "--commits",
        choices=["versions", "sample", "all"],
        help="record the history of each run instead of one commit",
    )
    parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--count", type=int, help="pin the object count")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--force", action="store_true", help="re-record existing")
    parser.add_argument(
        "--no-video", action="store_true", help="trace only; render on demand later"
    )
    parser.add_argument("--jobs", type=int, default=1, help="subprocesses at a time")
    parser.add_argument(
        "--timeout", type=float, default=1800, help="seconds per subprocess"
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    runs = matching_runs(args.data, args.runs)
    if not runs:
        raise SystemExit(f"No run with commits matches {args.runs}")
    if len(runs) == 1 and args.commits is None:
        record_run(args, runs[0])
        return
    tasks = _tasks(args, runs)
    passthrough = ["--instances", *map(str, args.instances)]
    passthrough += ["--width", str(args.width), "--height", str(args.height)]
    passthrough += ["--data", str(args.data), "--out", str(args.out)]
    passthrough += ["--count", str(args.count)] if args.count is not None else []
    passthrough += ["--force"] if args.force else []
    passthrough += ["--no-video"] if args.no_video else []

    def work(task: tuple[str, int]) -> tuple[tuple[str, int], int, str]:
        run_id, commit = task
        cmd = [sys.executable, "-m", "experiments.contacts.record", run_id]
        cmd += ["--commit", str(commit), *passthrough]
        try:
            done = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=args.timeout
            )
        except subprocess.TimeoutExpired:
            return task, 1, f"  timed out after {args.timeout:.0f} s"
        lines = [
            line
            for line in (done.stdout + done.stderr).splitlines()
            if line.startswith(("  instance", run_id))
        ]
        return task, done.returncode, "\n".join(lines) or done.stderr[-2000:]

    failed = []
    print(f"{len(tasks)} recording(s) to make, {args.jobs} at a time", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as pool:
        for i, (task, code, text) in enumerate(pool.map(work, tasks), 1):
            print(f"[{i}/{len(tasks)}] {task[0]} commit {task[1]}", flush=True)
            print(text, flush=True)
            if code:
                failed.append(f"{task[0]}@{task[1]}")
    if failed:
        print(f"{len(failed)} recordings failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
