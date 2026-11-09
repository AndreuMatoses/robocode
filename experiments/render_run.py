"""Render a policy from the agentic-gentamp-runs dataset as an MP4 (or GIF) video.

The repo already replays frozen programs (``run_experiment.py`` with
``approach.load_dir`` and the results viewer). This script adds only what the
dataset needs on top of that:

* it addresses a run's program by dataset ``run_id`` and commit index, using the
  ``runs`` and ``commits`` tables (which also carry each commit's held-out score);
* it renders instances you choose. The dataset withholds the eval seed, so the
  scored episodes cannot be replayed (the viewer would fall back to the
  replicate seed and label those instances as the scored episodes).

The program comes from the run's archive, unpacked once into
``runs/<method>/<env>/seed<N>/`` with ``sandbox/.git`` restored from its bundle (so
the results viewer can browse it too). The last commit uses the scored ``sandbox/``
tree; earlier commits are exported from git with the viewer's ``_export_snapshot``.
It is then loaded and rolled out with the repo's own code: ``AgenticApproach``
with ``load_dir`` (as in ``run_experiment.py`` and the viewer), and ``run_episode``
from ``robocode.utils.episode``.

Videos are H.264 MP4 by default, encoded from the raw frames (``open_mp4_writer``),
since the repo's ``open_video_writer`` only writes GIFs, whose 256-colour palette
bands shaded 3D scenes. ``--format gif`` uses the repo's writer instead.

Commits: ``-1`` = last (default), ``-2`` = the one before, ``0..`` = index.

The program is model-written code and runs in this process, as in the repo's own
scoring. Needs pyarrow<18 in the venv (the project pins NumPy 1.x). From the repo root:

    python experiments/render_run.py opus55/motion2d/seed42 --list
    python experiments/render_run.py opus55/motion2d/seed42 --instances 0 1
    python experiments/render_run.py astra-source/balancebeam3d/seed24 --commit 2 --count 2
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from robocode.environments.mujoco_gl import configure_gl_backend
from robocode.primitives import build_primitives
from robocode.utils.approach_history import _export_snapshot
from robocode.utils.episode import open_video_writer, run_episode

REPO = Path(__file__).resolve().parent.parent
CONF_ROOT = REPO / "experiments" / "conf"
DEFAULT_DATA = REPO / "viewer_runs" / "agentic-gentamp-runs"
DEFAULT_OUT = REPO / "outputs" / "renders"
RUN_ID = re.compile(r"^[a-z0-9-]+/[a-z0-9]+/seed\d+$")


# ------------------------------------------------------------- dataset lookup


def load_tables(data: Path, run_id: str) -> tuple[pd.Series, pd.DataFrame]:
    """The run's row in ``runs`` and its commits, oldest first."""
    method = run_id.split("/")[0]
    runs = pd.read_parquet(data / "data" / "runs" / f"{method}.parquet")
    row = runs[runs.run_id == run_id]
    commits_file = data / "data" / "commits" / f"{method}.parquet"
    if row.empty or not commits_file.exists():
        raise SystemExit(f"{run_id}: no such run, or a method without commits")
    commits = pd.read_parquet(commits_file)
    commits = commits[commits.run_id == run_id].sort_values("commit_index")
    return row.iloc[0], commits.reset_index(drop=True)


def env_config_name(short: str) -> str:
    """Dataset env names drop ``_generalized``; map back to the Hydra config."""
    for name in (f"{short}_generalized", short):
        if (CONF_ROOT / "environment" / f"{name}.yaml").exists():
            return name
    raise SystemExit(f"No environment config for {short!r}")


# ------------------------------------------------- program at a given commit


def unpack_archive(archive: Path) -> Path | None:
    """Unpack ``runs/<method>/<env>/seed<N>.tar.gz`` once next to itself and restore
    ``sandbox/.git`` from the bundle; None if the archive is absent."""
    if not archive.exists():
        return None
    run_dir = archive.with_name(archive.name.removesuffix(".tar.gz"))
    if not run_dir.exists():
        with tarfile.open(archive) as tar:
            # Members are stored as <method>/<env>/seed<N>/..., rooted at runs/.
            tar.extractall(archive.parents[2], filter="data")
    sandbox, bundle = run_dir / "sandbox", run_dir / "sandbox.bundle"
    if not (sandbox / ".git").exists() and bundle.exists():
        with tempfile.TemporaryDirectory(dir=run_dir) as tmp:
            subprocess.run(
                ["git", "clone", "-q", "--no-checkout", str(bundle), tmp], check=True
            )
            (Path(tmp) / ".git").rename(sandbox / ".git")
    return run_dir


def export_commit(sandbox: Path, sha: str, dest: Path) -> list[str]:
    """Export commit ``sha`` with the viewer's ``_export_snapshot``.

    If git cannot read a file of that tree, export the readable files only and return
    the skipped paths. This is for the dataset's one damaged history,
    ``opus5/rovers/seed444``, whose probe dump ``feats.pkl`` is a corrupt blob that
    ``approach.py`` never reads.
    """
    try:
        _export_snapshot(sandbox, sha, dest)
        return []
    except subprocess.CalledProcessError:
        pass

    def git(*args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", "-C", str(sandbox), *args], capture_output=True)

    readable, skipped = [], []
    for entry in filter(
        None, git("ls-tree", "-r", "-z", sha).stdout.decode().split("\0")
    ):
        meta, path = entry.split("\t", 1)
        ok = git("cat-file", "blob", meta.split()[2]).returncode == 0
        (readable if ok else skipped).append(path)
    archive = git("archive", sha, "--", *readable)
    archive.check_returncode()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(dest, filter="data")
    return skipped


def materialize(
    data: Path, row: pd.Series, commits: pd.DataFrame, pos: int, tmp: Path
) -> tuple[Path, str]:
    """Return (a ``load_dir`` holding ``sandbox/approach.py``, its provenance)."""
    run_dir = unpack_archive(data / row.archive_path)
    if run_dir is None:
        raise SystemExit(f"Archive not found: {data / row.archive_path}")
    if pos == len(commits) - 1:
        return run_dir, "the scored sandbox"
    sha = commits.iloc[pos].sha
    skipped = export_commit(run_dir / "sandbox", sha, tmp / "sandbox")
    note = f" (unreadable in git, skipped: {', '.join(skipped)})" if skipped else ""
    return tmp, f"git commit {sha[:8]}{note}"


# ----------------------------------------------------------------------- video


@contextlib.contextmanager
def open_mp4_writer(path: Path, fps: int) -> Iterator[Callable[[np.ndarray], None]]:
    """Like ``robocode.utils.episode.open_video_writer``, but H.264 MP4.

    Frames stream to ffmpeg through imageio-ffmpeg's writer (a repo dependency) as
    they are captured, so memory stays at one frame. ``quality=8`` maps to CRF 10,
    visually lossless; ``macro_block_size=2`` pads odd sizes by at most one pixel.
    """
    writer: Any = None

    def append(frame: np.ndarray) -> None:
        nonlocal writer
        rgb = np.ascontiguousarray(frame[..., :3], dtype=np.uint8)
        if writer is None:
            writer = imageio_ffmpeg.write_frames(
                str(path),
                (rgb.shape[1], rgb.shape[0]),
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=2,
                ffmpeg_log_level="error",
                # Explicit format: the target is a temporary .part name.
                output_params=["-movflags", "+faststart", "-f", "mp4"],
            )
            writer.send(None)  # start the generator
        writer.send(rgb)

    try:
        yield append
    finally:
        if writer is not None:
            writer.close()


def scale_mujoco_cameras(scale: float) -> Callable[[], int]:
    """Render the MuJoCo (dynamic 3D) environments' cameras ``scale`` times larger.

    Wraps kinder's ``MujocoEnv.__init__``: every MuJoCo robot env passes it
    ``camera_width``/``camera_height`` by keyword, it sizes ``MjSim``'s offscreen
    buffer from them, and the render context grows ``offwidth``/``offheight`` to fit.
    Those cameras only feed ``render()`` (and an optional debug window), never the
    observation, so the scene, physics and policy are unchanged; scaling both sides
    keeps the framing. Returns a counter of the envs it scaled.
    """
    configure_gl_backend()  # importing mujoco locks the GL backend; let robocode pick it
    # pylint: disable-next=import-outside-toplevel
    from kinder.envs.dynamic3d.mujoco_utils import MujocoEnv

    original = MujocoEnv.__init__
    scaled = [0]

    @functools.wraps(original)
    def init(
        self: Any,
        *args: Any,
        camera_width: int = 640,
        camera_height: int = 480,
        **kwargs: Any,
    ) -> None:
        scaled[0] += 1
        original(
            self,
            *args,
            camera_width=round(camera_width * scale),
            camera_height=round(camera_height * scale),
            **kwargs,
        )

    MujocoEnv.__init__ = init  # type: ignore[method-assign]
    return lambda: scaled[0]


# ------------------------------------------------------------------------ main


def main() -> None:
    """Find the program at the chosen commit and render it on the chosen instances."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("run_id", help="e.g. opus55/motion2d/seed42")
    parser.add_argument("--commit", type=int, default=-1, help="-1 = last (default)")
    parser.add_argument("--list", action="store_true", help="list commits and exit")
    parser.add_argument("--instances", type=int, nargs="+", default=[0])
    parser.add_argument("--count", type=int, help="pin the object count")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--fps", type=int, default=20, help="playback frame rate")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="render resolution multiplier, e.g. 2 = 1280x960 (MuJoCo 3D envs only)",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("run_id must look like <method>/<env>/seed<N>")
    row, commits = load_tables(args.data, args.run_id)

    if args.list:
        cols = ["commit_index", "time", "subject", "reeval_solve_rate"]
        with pd.option_context("display.width", 200, "display.max_colwidth", 90):
            print(commits[cols].to_string(index=False))
        print(f"\nFinal solve rate: {row.solve_rate}")
        return

    pos = args.commit if args.commit >= 0 else len(commits) + args.commit
    if not 0 <= pos < len(commits):
        raise SystemExit(f"This run has commits 0..{len(commits) - 1}")
    commit = commits.iloc[pos]
    method, env_short, run_seed = args.run_id.split("/")
    exp_id = str(row.experiment_id)
    blackbox, strict = "__blackbox__" in exp_id, "__strict" in exp_id
    print(f"{args.run_id}  ({row.model}; blackbox={blackbox}, strict={strict})")
    print(f"commit {pos}/{len(commits) - 1}: {commit.subject}")
    print(f"solve rate: this commit {commit.reeval_solve_rate}, final {row.solve_rate}")

    out_dir = args.out / method / env_short / run_seed
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        load_dir, source = materialize(args.data, row, commits, pos, Path(tmp))
        print(f"program from {source}")
        # Rebuild the approach the way run_experiment.py and the results viewer do.
        with initialize_config_dir(version_base=None, config_dir=str(CONF_ROOT)):
            cfg = compose(
                config_name="config",
                overrides=[
                    "approach=agentic",
                    f"++approach.load_dir={load_dir}",
                    f"++approach.output_dir={tmp}/out",
                    f"approach.blackbox={blackbox}",
                    f"approach.blackbox_strict={strict}",
                    "primitive_level=none",
                    "mcp_tools=[]",
                    f"environment={env_config_name(env_short)}",
                    f"replicate_seed={run_seed.removeprefix('seed')}",
                ],
            )
        scaled = scale_mujoco_cameras(args.scale) if args.scale != 1 else None
        env = instantiate(cfg.environment)
        if scaled is not None and not scaled():
            print(
                f"note: --scale only applies to the MuJoCo (dynamic 3D) envs; "
                f"{env_short} renders at its native size"
            )
        approach = instantiate(
            cfg.approach,
            action_space=env.action_space,
            observation_space=env.observation_space,
            seed=cfg.replicate_seed,
            env_description_path=None,
            mcp_tools=(),
            env=env,
            env_cfg=json.dumps({}),
            max_steps=cfg.max_steps,
            eval_timeout=cfg.eval_timeout,
            _partial_=True,
        )(primitives=build_primitives(env, cfg.primitives))
        approach.train()  # loads load_dir/sandbox/approach.py; no agent is run

        max_steps = cfg.max_steps
        if args.count is not None and hasattr(env, "max_steps_for_count"):
            max_steps = env.max_steps_for_count(args.count)
        writer = open_mp4_writer if args.format == "mp4" else open_video_writer
        for seed in args.instances:
            part = out_dir / f"commit{pos}_inst{seed}.{args.format}.part"
            try:
                with writer(part, fps=args.fps) as append_frame:
                    metrics, _, _ = run_episode(
                        env,
                        approach,
                        seed,
                        max_steps,
                        render=True,
                        count=args.count,
                        frame_sink=append_frame,
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                part.unlink(missing_ok=True)
                print(f"  instance {seed}: crashed: {type(exc).__name__}: {exc}")
                continue
            count = metrics.get("object_count", args.count)
            suffix = f"_count{count}" if count is not None else ""
            video = part.with_name(f"commit{pos}_inst{seed}{suffix}.{args.format}")
            part.replace(video)
            print(f"  instance {seed}: {metrics}\n    -> {video}")
        env.close()


if __name__ == "__main__":
    main()
