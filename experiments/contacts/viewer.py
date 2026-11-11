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
import functools
import hashlib
import json
import mimetypes
import queue
import re
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from experiments.contacts import (
    backend,
    catalog,
    diversity,
    histories,
    metrics,
    repeat,
    strategy,
)
from experiments.contacts.commits import load_versions
from experiments.contacts.trace import load_meta, summarise, trace_dirs, trace_id
from experiments.render_run import DEFAULT_DATA, REPO, RUN_ID
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
        strategy.pair_intervals.cache_clear()
        strategy.mode_sequence.cache_clear()
        strategy.strategy_key.cache_clear()
        _diversity.cache_clear()
        _metrics.cache_clear()
        _repeat.cache_clear()


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
    if kind == "video":
        directory = _trace_dir(body["trace"])
        out = directory / "video.mp4"
        cmd = _python("replay", "video", str(directory), "-o", str(out))
        return _enqueue("video", "episode video", cmd, _file_url(out))
    if kind == "contact":
        directory = _trace_dir(body["trace"])
        out = directory / "contact.mp4"
        cmd = _python("replay", "contact", str(directory), "-o", str(out))
        return _enqueue("contact", "contact video", cmd, _file_url(out))
    if kind == "ghost":
        dirs = [_trace_dir(t) for t in body["traces"]]
        colours = [str(c) for c in body.get("colours", [])]
        all_bodies, equal = bool(body.get("all_bodies")), bool(body.get("equal"))
        key = json.dumps([body["traces"], colours, all_bodies, equal])
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
        cmd += ["--equal"] if equal else []
        label = f"ghost video of {len(dirs)}"
        # A video that is newer than its traces and than the renderer is reused.
        sources = [d / "trace.npz" for d in dirs] + [
            Path(__file__).with_name("replay.py")
        ]
        if out.exists() and all(
            s.exists() and s.stat().st_mtime < out.stat().st_mtime for s in sources
        ):
            with _LOCK:
                job = Job(f"job{len(JOBS) + 1}", "ghost", label, cmd, _file_url(out))
                job.state, job.log = "done", "reused the video of an earlier render"
                JOBS[job.job_id] = job
            return job
        return _enqueue("ghost", label, cmd, _file_url(out))
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
        "has_video": (directory / "video.mp4").exists(),
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


def evolution_detail(run_id: str, instance: str, level: str) -> dict[str, Any]:
    """The contact strategy of one run at each of its recorded commits.

    One row per commit of the run. A commit whose policy equals an earlier one
    (``same_as``) reuses that commit's trace unless it was recorded itself, in which
    case ``identical`` says whether the two recordings agree exactly.
    """
    if level not in ("class", "entity", "surface"):
        raise ValueError(f"unknown level {level!r}")
    if not RUN_ID.fullmatch(run_id) or not re.fullmatch(
        r"inst\d+(_count\d+)?", instance
    ):
        raise ValueError("run must look like <method>/<env>/seed<N>")
    opts = strategy.Options(level=level)
    versions = load_versions(CONFIG.data, CONFIG.traces, run_id)
    root = CONFIG.traces / run_id

    def directory(commit: int | None) -> Path | None:
        if commit is None:
            return None
        path = root / f"commit{commit}" / instance
        return path if (path / "meta.json").exists() else None

    own = {v["commit"]: directory(v["commit"]) for v in versions}
    recorded = [c for c, d in own.items() if d is not None]
    final = own[max(recorded)] if recorded else None
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    prev: Path | None = None  # trace of the previous policy version
    for v in versions:
        mine, first = own[v["commit"]], directory(v["same_as"])
        new_version = v["same_as"] == v["commit"]
        shown = mine or first
        row: dict[str, Any] = {
            **{k: v[k] for k in ("commit", "sha", "subject", "time", "key")},
            "same_as": v["same_as"],
            "files": v["files"],
            "solve_rate": v["reeval_solve_rate"],
            "status": v["reeval_status"],
            "recorded": mine is not None,
            "trace": trace_id(CONFIG.traces, shown) if shown else None,
        }
        if mine is not None and first is not None and mine != first:
            row["identical"] = strategy.identical_raw(mine, first)
            row["d_same"] = round(strategy.sequence_distance(mine, first, opts), 4)
        if shown is not None:
            metrics = load_meta(shown)["metrics"]
            pairs = strategy.pairs_used(shown, opts)
            row.update(
                solved=bool(metrics.get("solved")),
                crashed=metrics.get("crashed"),
                num_steps=metrics.get("num_steps"),
                modes=strategy.describe(shown, opts)["modes"],
                num_pairs=len(pairs),
                new_pairs=sorted(pairs - seen),
            )
            if final is not None:
                row["d_final"] = round(
                    strategy.sequence_distance(shown, final, opts), 4
                )
                row["p_final"] = round(strategy.pair_distance(shown, final, opts), 4)
            if (new_version or mine is not None) and prev is not None and prev != shown:
                row["d_prev"] = round(strategy.sequence_distance(prev, shown, opts), 4)
                row["p_prev"] = round(strategy.pair_distance(prev, shown, opts), 4)
            seen |= pairs
            prev = shown
        rows.append(row)
    meta = load_meta(final) if final else {}
    return {
        "run_id": run_id,
        "environment": run_id.split("/")[1],
        "method": run_id.split("/")[0],
        "instance": instance,
        "level": level,
        "duration": max(
            [r["modes"][-1]["end"] for r in rows if r.get("modes")], default=0.0
        ),
        "run_solve_rate": meta.get("run_solve_rate"),
        "options": dataclasses.asdict(opts),
        "commits": rows,
    }


def _check(level: str, instance: str, *run_ids: str) -> strategy.Options:
    if level not in ("class", "entity", "surface"):
        raise ValueError(f"unknown level {level!r}")
    if not re.fullmatch(r"inst\d+(_count\d+)?", instance):
        raise ValueError(f"bad instance {instance!r}")
    if not all(RUN_ID.fullmatch(r) for r in run_ids):
        raise ValueError("run must look like <method>/<env>/seed<N>")
    return strategy.Options(level=level)


def _versions(run_id: str) -> list[dict[str, Any]]:
    return load_versions(CONFIG.data, CONFIG.traces, run_id)


def _scenario_runs(name: str) -> list[str]:
    """Run ids of a scenario, ordered by method then seed."""
    table = catalog.runs_table(CONFIG.data)
    rows = table[(table.environment == name) & (table.num_commits > 0)]
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    return sorted(
        rows.run_id.tolist(),
        key=lambda r: (
            order.get(r.split("/")[0], 99),
            int(r.rsplit("seed", 1)[-1]),
        ),
    )


Grouping = tuple[strategy.Definition, str, float]


def _definition(query: dict[str, str]) -> Grouping:
    """The strategy definition, grouping and threshold of a request."""
    times = {
        field: float(query[field])
        for field in ("gap", "min_duration", "min_mode")
        if query.get(field) not in (None, "")
    }
    defn = strategy.Definition(
        query.get("scope", "all"),
        query.get("identity", "keep"),
        query.get("order", "sequence"),
        query.get("fingers", "all"),
        query.get("metric", "edit"),
        **times,
    )
    group, tau = query.get("group", "similar"), float(query.get("tau", 0.35))
    if group not in ("exact", "similar", "pairs") or not 0.0 < tau <= 1.0:
        raise ValueError("group must be exact or similar, tau in (0, 1]")
    return defn, group, tau


def definitions() -> dict[str, Any]:
    """The choices of a strategy definition, with their descriptions."""
    return {
        "scope": strategy.SCOPES,
        "identity": strategy.IDENTITIES,
        "order": strategy.ORDERS,
        "fingers": strategy.FINGERS,
        "metric": strategy.DISTANCES,
        "times": {
            field: getattr(strategy.Options(), field)
            for field in ("gap", "min_duration", "min_mode")
        },
        "presets": strategy.PRESETS,
    }


def scenario_histories(
    name: str, instance: str, level: str, grouping: Grouping
) -> dict[str, Any]:
    """Strategy histories of every run of a scenario (see ``histories.py``)."""
    opts = _check(level, instance)
    defn, group, tau = grouping
    out = histories.scenario_histories(
        CONFIG.traces, _scenario_runs(name), instance, opts, _versions, group, tau, defn
    )
    out["name"] = name
    out["preset"] = _preset(name)
    out["definition_text"] += _time_note(name)
    out["methods"] = [
        m for m in METHOD_ORDER if any(r["method"] == m for r in out["runs"])
    ]
    return out


@functools.lru_cache(maxsize=256)
def _diversity(
    name: str, instance: str, level: str, grouping: Grouping
) -> dict[str, Any]:
    opts = _check(level, instance)
    defn, group, tau = grouping
    table = catalog.runs_table(CONFIG.data)
    rows = table[table.environment == name]
    reported = {
        row.run_id: {
            "solve_rate": (
                None if row.solve_rate != row.solve_rate else float(row.solve_rate)
            ),
            "mean_eval_steps": (
                None
                if row.mean_eval_steps != row.mean_eval_steps
                else float(row.mean_eval_steps)
            ),
        }
        for row in rows.itertuples()
    }
    out = diversity.scenario_diversity(
        CONFIG.traces,
        _scenario_runs(name),
        instance,
        opts,
        _versions,
        defn,
        group,
        tau,
        reported,
    )
    out["name"] = name
    out["preset"] = _preset(name)
    out["definition_text"] += _time_note(name)
    out["method_order"] = [
        m for m in METHOD_ORDER if any(r["method"] == m for r in out["runs"])
    ]
    return out


def _backend(name: str) -> str:
    simulator = catalog.scenario_tags(name)["simulator"]
    return backend.BACKEND_OF_SIMULATOR.get(simulator, "mujoco")


def _time_note(name: str) -> str:
    """What is different about the debounce times of this scenario's simulator."""
    return strategy.time_note(_backend(name))


def _preset(name: str) -> str:
    """The tuned preset of this scenario's simulator."""
    return strategy.TUNED_PRESET.get(_backend(name), "tuned")


def scenario_diversity(
    name: str, instance: str, level: str, grouping: Grouping
) -> dict[str, Any]:
    """Diversity and optimality of the final policies (see ``diversity.py``)."""
    return _diversity(name, instance, level, grouping)


@functools.lru_cache(maxsize=64)
def _metrics(
    name: str, instance: str, level: str, defn: strategy.Definition
) -> dict[str, Any]:
    out = metrics.scenario_metrics(
        CONFIG.traces,
        _scenario_runs(name),
        instance,
        _check(level, instance),
        _versions,
        defn,
    )
    out["name"] = name
    out["preset"] = _preset(name)
    out["method_order"] = [
        m for m in METHOD_ORDER if any(r["method"] == m for r in out["runs"])
    ]
    return out


@functools.lru_cache(maxsize=64)
def _repeat(name: str, level: str, defn: strategy.Definition) -> dict[str, Any]:
    """Repeatability of the final policies over the recorded instances."""
    out = repeat.scenario_repeatability(
        CONFIG.traces, _scenario_runs(name), _check(level, "inst0"), _versions, defn
    )
    out["name"] = name
    return out


def diversity_overview(instance: str, level: str, grouping: Grouping) -> dict[str, Any]:
    """``diversity.summary`` of every scenario with recorded commit histories."""
    names = sorted(
        {p.parents[1].name for p in CONFIG.traces.glob("*/*/*/commits.json")}
    )
    scenarios = []
    for name in names:
        detail = _diversity(name, instance, level, grouping)
        if detail["runs"]:
            scenarios.append({"name": name, **diversity.summary(detail)})
    defn, group, tau = grouping
    return {
        "instance": instance,
        "level": level,
        "group": group,
        "tau": tau,
        "definition": dataclasses.asdict(defn),
        "definition_text": defn.text(group, tau),
        "methods": METHOD_ORDER,
        "scenarios": scenarios,
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
            elif path == "/api/evolution":
                self._json(
                    evolution_detail(
                        query["run"],
                        query.get("instance", "inst0"),
                        query.get("level", "class"),
                    )
                )
            elif path == "/api/histories":
                self._json(
                    scenario_histories(
                        query["name"],
                        query.get("instance", "inst0"),
                        query.get("level", "class"),
                        _definition(query),
                    )
                )
            elif path == "/api/diversity":
                self._json(
                    scenario_diversity(
                        query["name"],
                        query.get("instance", "inst0"),
                        query.get("level", "class"),
                        _definition(query),
                    )
                )
            elif path == "/api/diversity_overview":
                self._json(
                    diversity_overview(
                        query.get("instance", "inst0"),
                        query.get("level", "class"),
                        _definition(query),
                    )
                )
            elif path == "/api/definitions":
                self._json(definitions())
            elif path == "/api/metrics":
                self._json(
                    _metrics(
                        query["name"],
                        query.get("instance", "inst0"),
                        query.get("level", "class"),
                        _definition(query)[0],
                    )
                )
            elif path == "/api/repeatability":
                self._json(
                    _repeat(
                        query["name"],
                        query.get("level", "class"),
                        _definition(query)[0],
                    )
                )
            elif path == "/api/metric_pair":
                opts = _check(query.get("level", "class"), "inst0")
                self._json(
                    metrics.pair_detail(
                        _trace_dir(query["a"]),
                        _trace_dir(query["b"]),
                        opts,
                        _definition(query)[0],
                    )
                )
            elif path == "/api/history_pair":
                instance, level = query.get("instance", "inst0"), query.get(
                    "level", "class"
                )
                opts = _definition(query)[0].options(
                    _check(level, instance, query["a"], query["b"])
                )
                self._json(
                    histories.history_pair(
                        CONFIG.traces, query["a"], query["b"], instance, opts, _versions
                    )
                )
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
        except ValueError as exc:
            self._err(400, f"bad request: {exc}")

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
