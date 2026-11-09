"""Unpack agentic-gentamp-runs archives so the results viewer can show them.

The viewer reads run folders, not the dataset's ``.tar.gz`` archives. This unpacks
the matching ``runs/<method>/<env>/seed<N>.tar.gz`` next to themselves and restores
each ``sandbox/.git`` from its bundle (via ``render_run.unpack_archive``), so the
viewer can also show the commit history. Runs whose scores came from a later
re-scoring pass get the viewer's ``env_variant.json`` marker so they are listed
(see ``make_viewer_visible``). Nothing is executed.

Patterns are globs over ``<method>/<env>/seed<N>``. Unpacked runs take several
times their archive size (the agents' probe files), so unpack what you need:

    uv run python experiments/unpack_runs.py 'astra/motion2d/*'
    uv run python experiments/unpack_runs.py '*/tossing3d/seed42' 'opus55-source/*/seed24'
    uv run python experiments/unpack_runs.py --dry-run '*/motion2d/*'
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "experiments")]
# pylint: disable=wrong-import-position
from render_run import DEFAULT_DATA, unpack_archive  # noqa: E402

from experiments.results_viewer import _parse_overrides  # noqa: E402


def dir_size(path: Path) -> int:
    """Total bytes of the files under ``path``."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def make_viewer_visible(run_dir: Path) -> str | None:
    """Let the viewer list a run whose final scores came from a re-scoring pass.

    The viewer hides any results.json whose .hydra records ``approach.load_dir``
    (it treats those as re-evaluation passes) unless an ``env_variant.json`` marker
    sits next to it. Some dataset runs were re-scored that way, either in place or
    into a ``rescore/`` subfolder; their score is still the run's score.
    """
    results = run_dir / "results.json"
    rescored = run_dir / "rescore" / "results.json"
    note = None
    if not results.exists() and rescored.exists():
        shutil.copy2(rescored, results)
        note = "copied rescore/results.json"
    overrides = _parse_overrides(run_dir / ".hydra")
    marker = run_dir / "env_variant.json"
    if results.exists() and "approach.load_dir" in overrides and not marker.exists():
        marker.write_text(
            json.dumps({"approach": overrides.get("approach", "agentic")})
        )
        note = f"{note + '; ' if note else ''}added env_variant.json"
    return note


def main() -> None:
    """Unpack every archive matching the given patterns."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("patterns", nargs="+", help="e.g. 'astra/motion2d/*'")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--dry-run", action="store_true", help="only list matches")
    args = parser.parse_args()

    runs = args.data / "runs"
    archives = sorted(
        {a for pattern in args.patterns for a in runs.glob(f"{pattern}.tar.gz")}
    )
    if not archives:
        raise SystemExit(f"No archives under {runs} match {args.patterns}")
    packed = sum(a.stat().st_size for a in archives)
    print(f"{len(archives)} archives, {packed / 1e6:.0f} MB packed")
    if args.dry_run:
        for a in archives:
            print(" ", a.relative_to(runs))
        return

    added = 0
    for i, archive in enumerate(archives, 1):
        run_dir = archive.with_name(archive.name.removesuffix(".tar.gz"))
        existed = run_dir.exists()
        unpack_archive(archive)
        fix = make_viewer_visible(run_dir)
        size = dir_size(run_dir)
        added += 0 if existed else size
        note = "already unpacked" if existed else f"{size / 1e6:.0f} MB"
        if fix:
            note += f"; {fix}"
        print(f"[{i}/{len(archives)}] {run_dir.relative_to(runs)}  ({note})")
    free = shutil.disk_usage(runs).free
    print(f"Added {added / 1e9:.2f} GB; {free / 1e9:.1f} GB free on this disk")
    print("Reload the viewer page (or restart it) to see the new runs.")


if __name__ == "__main__":
    main()
