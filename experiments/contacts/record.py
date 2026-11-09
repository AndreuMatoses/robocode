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

Several matching runs are recorded one subprocess each, so one policy's modules and
crashes cannot affect the next.
"""

from __future__ import annotations

import argparse
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

from experiments.contacts.entities import robot_env
from experiments.contacts.recorder import STEP_COLUMNS, ContactRecorder, recording
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
    ro: Any, slot: list, seed: int, count: int | None, max_steps: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], Any]:
    """Roll the policy out on one instance; return (arrays, metrics, recorder)."""
    state, info = reset_for_episode(ro.env, seed, count)
    recorder = ContactRecorder(robot_env(ro.env))
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
    with open_rollout(args.data, run_id, args.commit) as ro, recording() as slot:
        commit = ro.commits.iloc[ro.pos]
        print(f"{run_id} commit {ro.pos}/{len(ro.commits) - 1}: {commit.subject}")
        max_steps = max_steps_for(ro, args.count)
        for seed in args.instances:
            out = instance_dir(args.out, run_id, ro.pos, seed, args.count)
            if (out / "meta.json").exists() and not args.force:
                print(f"  instance {seed}: already recorded")
                continue
            arrays, metrics, recorder = record_instance(
                ro, slot, seed, args.count, max_steps
            )
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "trace.npz", **arrays)
            camera = render_camera_name(ro.env)
            write_video(
                replay_frames(ro.env, arrays["qpos"], args.width, args.height, camera),
                out / "video.mp4",
                FPS,
            )
            diagram = diagram_dir(args.out, run_id.split("/")[1], seed, args.count)
            if not (diagram / "diagram.json").exists():
                set_qpos(robot_env(ro.env), arrays["qpos"][0])
                write_diagram(ro.env, recorder.emap, diagram, args.width, args.height)
            reeval = commit.reeval_solve_rate
            meta = {
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
            }
            (out / "meta.json").write_text(json.dumps(meta))
            print(
                f"  instance {seed}: {metrics}; {len(arrays['keys'])} contact keys, "
                f"{len(arrays['intervals'])} intervals -> {out}"
            )


def main() -> None:
    """Record one run in this process, or several in one subprocess each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("runs", nargs="+", help="run ids or globs, '*/tossing3d/*'")
    parser.add_argument("--commit", type=int, default=-1, help="-1 = last (default)")
    parser.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--count", type=int, help="pin the object count")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--force", action="store_true", help="re-record existing")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    runs = matching_runs(args.data, args.runs)
    if not runs:
        raise SystemExit(f"No run with commits matches {args.runs}")
    if len(runs) == 1:
        record_run(args, runs[0])
        return
    passthrough = [a for a in sys.argv[1:] if a not in args.runs]
    failed = []
    for i, run_id in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {run_id}", flush=True)
        cmd = [sys.executable, "-m", "experiments.contacts.record", run_id]
        if subprocess.run(cmd + passthrough, check=False).returncode:
            failed.append(run_id)
    if failed:
        print(f"{len(failed)} runs failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
