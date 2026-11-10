"""Which commits of a run change the policy.

Agents commit often, and many commits only add logs, notes or scratch scripts. The
rollout of a commit depends on ``approach.py`` and on what it loads from the
sandbox, so two commits with the same *policy key* give the same trace on the same
instance and only one of them needs recording.

The policy key of a commit is a hash of:

* ``approach.py``;
* every sandbox module it imports, followed transitively (``import kin``,
  ``from toss import x``, ``from pkg.mod import y``);
* every other sandbox file whose name appears in the text of those sources (a
  heuristic for parameters loaded from ``.json`` / ``.npy`` files).

Imports built at run time and files read by a computed name are not seen. Record a
sample of "unchanged" commits (``record.py --commits sample``) to check that their
traces are identical.

    python -m experiments.contacts.commits 'opus55/tossing3d/*'
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from experiments.render_run import DEFAULT_DATA, load_tables, unpack_archive

IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+([A-Za-z_][\w.]*)[ \t]+import[ \t]+([^\n#]+)"
    r"|import[ \t]+([^\n#]+))",
    re.M,
)


def _git(sandbox: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(sandbox), *args], capture_output=True, check=False
    )
    return done.stdout.decode(errors="replace") if done.returncode == 0 else ""


def _tree(sandbox: Path, sha: str) -> dict[str, str]:
    """Path -> blob hash of every file of a commit."""
    files = {}
    for entry in filter(None, _git(sandbox, "ls-tree", "-r", "-z", sha).split("\0")):
        meta, path = entry.split("\t", 1)
        files[path] = meta.split()[2]
    return files


@functools.lru_cache(maxsize=4096)
def _blob(sandbox: Path, blob: str) -> str:
    return _git(sandbox, "cat-file", "blob", blob)


def _imported(source: str) -> set[str]:
    """Dotted module names a source may import (candidates, not all exist)."""
    names: set[str] = set()
    for module, items, plain in IMPORT.findall(source):
        if module:
            names.add(module)
            for item in items.replace("(", "").replace(")", "").split(","):
                item = item.split(" as ")[0].strip()
                if item.isidentifier():
                    names.add(f"{module}.{item}")
        for item in plain.split(","):
            item = item.split(" as ")[0].strip()
            if item:
                names.add(item)
    return names


def policy_files(sandbox: Path, files: dict[str, str]) -> dict[str, str]:
    """The files of a commit that the policy depends on (path -> blob hash)."""
    if "approach.py" not in files:
        return {}
    used = {"approach.py": files["approach.py"]}
    queue = ["approach.py"]
    text = ""
    while queue:
        source = _blob(sandbox, files[queue.pop()])
        text += source
        for name in _imported(source):
            base = name.replace(".", "/")
            for path in (f"{base}.py", f"{base}/__init__.py"):
                if path in files and path not in used:
                    used[path] = files[path]
                    queue.append(path)
    for path, blob in files.items():
        name = path.rsplit("/", 1)[-1]
        if path not in used and not path.endswith(".py") and "." in name:
            if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", text):
                used[path] = blob
    return used


def policy_versions(data: Path, run_id: str) -> list[dict[str, Any]]:
    """One row per commit: its policy key and the first commit with the same policy.

    ``same_as`` is the index of the earliest commit of the current streak with an
    identical policy (the commit itself when the policy changed), or ``None`` when
    the commit has no ``approach.py``.
    """
    row, commits = load_tables(data, run_id)
    run_dir = unpack_archive(data / row.archive_path)
    if run_dir is None:
        raise SystemExit(f"Archive not found: {data / row.archive_path}")
    sandbox = run_dir / "sandbox"
    out: list[dict[str, Any]] = []
    last_key, last_start = None, None
    for pos, commit in enumerate(commits.itertuples()):
        used = policy_files(sandbox, _tree(sandbox, commit.sha))
        key = (
            hashlib.sha1(json.dumps(sorted(used.items())).encode()).hexdigest()[:12]
            if used
            else None
        )
        if key is None:
            same_as = None
        elif key == last_key:
            same_as = last_start
        else:
            same_as = pos
        last_key, last_start = key, same_as
        out.append(
            {
                "commit": pos,
                "sha": str(commit.sha),
                "subject": str(commit.subject),
                "time": str(commit.time),
                "key": key,
                "same_as": same_as,
                "files": sorted(used),
                "reeval_status": (
                    None if commit.reeval_status is None else str(commit.reeval_status)
                ),
                "reeval_solve_rate": (
                    None
                    if commit.reeval_solve_rate != commit.reeval_solve_rate
                    else float(commit.reeval_solve_rate)
                ),
            }
        )
    return out


def versions_path(out: Path, run_id: str) -> Path:
    """Where the commit table of a run is cached next to its traces."""
    return out / run_id / "commits.json"


def load_versions(data: Path, out: Path, run_id: str) -> list[dict[str, Any]]:
    """``policy_versions`` cached as ``<out>/<run_id>/commits.json``."""
    path = versions_path(out, run_id)
    if path.exists():
        return json.loads(path.read_text())
    versions = policy_versions(data, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(versions))
    return versions


def select_commits(versions: list[dict[str, Any]], mode: str) -> list[int]:
    """Commit indices to record.

    ``versions``: the first commit of every policy change, plus the last commit.
    ``sample``: as ``versions``, plus the last commit of every streak of two or more
    commits with one policy (their traces must equal the first of the streak).
    ``all``: every commit with an ``approach.py``.
    """
    with_policy = [v for v in versions if v["key"] is not None]
    if mode == "all":
        return [v["commit"] for v in with_policy]
    chosen = {v["commit"] for v in with_policy if v["same_as"] == v["commit"]}
    if versions:
        chosen.add(versions[-1]["commit"])
    if mode == "sample":
        for prev, cur in zip(with_policy, with_policy[1:] + [None]):  # type: ignore[list-item]
            ends = cur is None or cur["same_as"] != prev["same_as"]
            if ends and prev["same_as"] != prev["commit"]:
                chosen.add(prev["commit"])
    return sorted(chosen)


def main() -> None:
    """Print how many commits of each run change the policy."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.record import DEFAULT_OUT, matching_runs

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("runs", nargs="+", help="run ids or globs")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    total = [0, 0, 0]
    for run_id in matching_runs(args.data, args.runs):
        versions = load_versions(args.data, args.out, run_id)
        counts = [
            len(versions),
            len(select_commits(versions, "versions")),
            len(select_commits(versions, "sample")),
        ]
        total = [a + b for a, b in zip(total, counts)]
        print(
            f"{run_id}: {counts[0]} commits, {counts[1]} versions, {counts[2]} sample"
        )
    print(f"total: {total[0]} commits, {total[1]} versions, {total[2]} with sample")


if __name__ == "__main__":
    main()
