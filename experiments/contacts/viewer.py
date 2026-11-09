"""Contact-trace viewer: browse scenarios, compare the contact strategies of runs.

A small stdlib HTTP server (the frontend is the plain files in ``static/``). It
reads the dataset tables for the list of runs and ``outputs/contact_traces`` for
what has been recorded, and serves entity-pair summaries of the traces
(``trace.summarise``). Recording, contact-highlight videos and ghost overlays run
as queued subprocesses of ``record.py`` / ``replay.py``; only recording executes
policy code.

    uv run python -m experiments.contacts.viewer            # http://127.0.0.1:8010
    uv run python -m experiments.contacts.viewer --port 8020 --traces <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import mimetypes
import queue
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from experiments.contacts import catalog
from experiments.contacts.trace import load_meta, summarise, trace_dirs, trace_id
from experiments.render_run import DEFAULT_DATA, REPO
from experiments.results_viewer import Handler as _BaseHandler

STATIC = Path(__file__).resolve().parent / "static"
METHOD_ORDER = [
    "opus55",
    "opus55-source",
    "opus5",
    "opus5-source",
    "astra",
    "astra-source",
    "sol",
]


@dataclasses.dataclass
class Config:
    """Where the viewer reads from."""

    data: Path = DEFAULT_DATA
    traces: Path = catalog.DEFAULT_OUT


CONFIG = Config()

# ------------------------------------------------------------------------ jobs


@dataclasses.dataclass
class Job:
    """A queued subprocess (a recording or a render)."""

    job_id: str
    kind: str
    label: str
    command: list[str]
    output: str | None = None  # URL of the result, if it is a file
    state: str = "queued"  # queued, running, done, error
    log: str = ""
    started_at: float | None = None
    finished_at: float | None = None


JOBS: dict[str, Job] = {}
_QUEUE: queue.Queue[str] = queue.Queue()
_LOCK = threading.Lock()


def _enqueue(kind: str, label: str, command: list[str], output: str | None) -> Job:
    with _LOCK:
        job = Job(f"job{len(JOBS) + 1}", kind, label, command, output)
        JOBS[job.job_id] = job
    _QUEUE.put(job.job_id)
    return job


def _worker() -> None:
    while True:
        job = JOBS[_QUEUE.get()]
        job.state, job.started_at = "running", time.time()
        try:
            with subprocess.Popen(
                job.command,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as proc:
                assert proc.stdout is not None
                for line in proc.stdout:
                    job.log = (job.log + line)[-4000:]
            job.state = "done" if proc.returncode == 0 else "error"
        except OSError as exc:
            job.state, job.log = "error", f"{job.log}\n{exc}"
        job.finished_at = time.time()
        summarise.cache_clear()
        load_meta.cache_clear()


def _python(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", f"experiments.contacts.{module}", *args]


def start_job(body: dict[str, Any]) -> Job:
    """Queue a recording, a contact-highlight video or a ghost overlay."""
    kind, traces = body.get("kind"), CONFIG.traces
    if kind == "record":
        args = [str(r) for r in body["runs"]]
        args += ["--commit", str(int(body.get("commit", -1)))]
        args += ["--instances", *[str(int(i)) for i in body.get("instances", [0])]]
        if body.get("count") is not None:
            args += ["--count", str(int(body["count"]))]
        args += ["--data", str(CONFIG.data), "--out", str(traces)]
        label = f"record {len(body['runs'])} run(s)"
        return _enqueue("record", label, _python("record", *args), None)
    if kind == "contact":
        directory = _trace_dir(body["trace"])
        out = directory / "contact.mp4"
        cmd = _python("replay", "contact", str(directory), "-o", str(out))
        return _enqueue("contact", "contact video", cmd, _file_url(out))
    if kind == "ghost":
        dirs = [_trace_dir(t) for t in body["traces"]]
        colours = [str(c) for c in body.get("colours", [])]
        all_bodies = bool(body.get("all_bodies"))
        key = json.dumps([body["traces"], colours, all_bodies])
        digest = hashlib.sha1(key.encode()).hexdigest()[:12]
        meta = load_meta(dirs[0])
        out = (
            traces
            / catalog.SCENARIO_DIR
            / meta["environment"]
            / "ghosts"
            / f"{digest}.mp4"
        )
        cmd = _python("replay", "ghost", *map(str, dirs), "-o", str(out))
        cmd += ["--colours", *colours] + (["--all-bodies"] if all_bodies else [])
        return _enqueue("ghost", f"ghost video of {len(dirs)}", cmd, _file_url(out))
    raise ValueError(f"unknown job kind {kind!r}")


# ------------------------------------------------------------------- payloads


def _file_url(path: Path) -> str:
    return "/files/" + path.relative_to(CONFIG.traces).as_posix()


def _trace_dir(tid: str) -> Path:
    directory = (CONFIG.traces / tid).resolve()
    if (
        not directory.is_relative_to(CONFIG.traces.resolve())
        or not (directory / "meta.json").exists()
    ):
        raise FileNotFoundError(tid)
    return directory


def _trace_row(directory: Path) -> dict[str, Any]:
    meta = load_meta(directory)
    metrics = meta["metrics"]
    return {
        "id": trace_id(CONFIG.traces, directory),
        "run_id": meta["run_id"],
        "method": meta["method"],
        "run_seed": meta["run_id"].rsplit("/", 1)[-1],
        "commit": meta["commit"],
        "num_commits": meta["num_commits"],
        "subject": meta["subject"],
        "commit_solve_rate": meta["commit_solve_rate"],
        "instance": directory.name,
        "solved": bool(metrics.get("solved")),
        "crashed": metrics.get("crashed"),
        "num_steps": metrics.get("num_steps"),
        "object_count": metrics.get("object_count"),
        "has_contact_video": (directory / "contact.mp4").exists(),
    }


def scenario_list() -> list[dict[str, Any]]:
    """Every scenario with tags and how much has been recorded."""
    counts: dict[str, int] = {}
    for directory in trace_dirs(CONFIG.traces):
        scenario = directory.parents[2].name
        counts[scenario] = counts.get(scenario, 0) + 1
    table = catalog.runs_table(CONFIG.data)
    runs = table[table.num_commits > 0].groupby("environment").size().to_dict()
    out = catalog.scenarios(CONFIG.data)
    for scenario in out:
        name = scenario["name"]
        scenario["num_runs"] = int(runs.get(name, 0))
        scenario["num_traces"] = counts.get(name, 0)
        scenario["has_thumb"] = catalog.thumbnail_path(CONFIG.traces, name).exists()
    # Scenarios only present as traces (development data) are listed too.
    for name in sorted(set(counts) - {s["name"] for s in out}):
        out.append({"name": name, "num_runs": 0, "num_traces": counts[name]})
    return out


def scenario_detail(name: str) -> dict[str, Any]:
    """Runs and recorded traces of one scenario."""
    table = catalog.runs_table(CONFIG.data)
    rows = table[(table.environment == name) & (table.num_commits > 0)]
    runs = [
        {
            "run_id": r.run_id,
            "method": r.run_id.split("/")[0],
            "run_seed": r.run_id.rsplit("/", 1)[-1],
            "model": r.model,
            "label": r.method,
            "solve_rate": float(r.solve_rate),
            "num_commits": int(r.num_commits),
        }
        for r in rows.itertuples()
    ]
    traces = [
        _trace_row(d) for d in trace_dirs(CONFIG.traces) if d.parents[2].name == name
    ]
    instances = {}
    for instance in sorted({t["instance"] for t in traces}):
        directory = CONFIG.traces / catalog.SCENARIO_DIR / name / instance
        index = directory / "diagram.json"
        instances[instance] = {
            "diagram": json.loads(index.read_text()) if index.exists() else None,
            "base": _file_url(directory),
        }
    try:
        tags = catalog.scenario_tags(name)
    except SystemExit:
        tags = {}
    methods = METHOD_ORDER + sorted(
        {t["method"] for t in traces} | {r["method"] for r in runs} - set(METHOD_ORDER)
    )
    return {
        "name": name,
        "tags": tags,
        "runs": runs,
        "traces": traces,
        "instances": instances,
        "methods": [m for m in dict.fromkeys(methods)],
    }


# --------------------------------------------------------------------- server


class Handler(_BaseHandler):
    """Routes of the contact viewer (HTTP helpers come from the results viewer)."""

    def do_GET(self) -> None:  # noqa: N802  pylint: disable=invalid-name
        """Static files, JSON APIs and recorded files."""
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        path = url.path
        try:
            if path == "/":
                self._static("index.html")
            elif path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
            elif path == "/api/scenarios":
                self._json(scenario_list())
            elif path == "/api/scenario":
                self._json(scenario_detail(query["name"]))
            elif path == "/api/trace":
                self._json(summarise(_trace_dir(query["id"])))
            elif path == "/api/job":
                self._json(dataclasses.asdict(JOBS[query["id"]]))
            elif path == "/api/jobs":
                self._json([dataclasses.asdict(j) for j in JOBS.values()][-20:])
            elif path == "/thumb":
                self._file(catalog.thumbnail_path(CONFIG.traces, query["name"]))
            elif path.startswith("/files/"):
                self._file(CONFIG.traces / path.removeprefix("/files/"))
            else:
                self._err(404, "not found")
        except (KeyError, FileNotFoundError) as exc:
            self._err(404, f"not found: {exc}")

    def do_POST(self) -> None:  # noqa: N802  pylint: disable=invalid-name
        """Start a job."""
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if urlparse(self.path).path != "/api/job":
                self._err(404, "not found")
                return
            self._json(dataclasses.asdict(start_job(body)))
        except (KeyError, ValueError, FileNotFoundError) as exc:
            self._err(400, f"bad request: {exc}")

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if not path.is_relative_to(STATIC) or not path.is_file():
            raise FileNotFoundError(name)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def _file(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_relative_to(CONFIG.traces.resolve()) or not path.is_file():
            raise FileNotFoundError(path.name)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_file(path, ctype)


def main() -> None:
    """Serve the viewer."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--traces", type=Path, default=catalog.DEFAULT_OUT)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    CONFIG.data, CONFIG.traces = args.data, args.traces.resolve()
    threading.Thread(target=_worker, daemon=True).start()
    print(f"contact viewer: http://{args.host}:{args.port}  (traces: {CONFIG.traces})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
