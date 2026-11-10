"""Symbolic contact strategies and distances between them.

A trace becomes a **contact-mode sequence**: at each instant the *mode* is the set
of contact pairs that are active, and the sequence lists the distinct modes in
order. This file builds that sequence and compares two of them.

Steps, each with a parameter in ``Options``:

1. **Label** every contact by its pair at a level of abstraction: ``class``
   (``robot.gripper.pad <> env.cube``), ``entity`` (``... <> env.cube_0``) or
   ``surface`` (adds geom index and face). ``merge_sides`` folds ``left_``/``right_``
   robot parts into one, so mirrored grasps are the same strategy.
2. **Filter**: keep the chosen categories; drop robot self-contact and *resting*
   pairs (no robot part, active for nearly the whole episode).
3. **Debounce** each pair: merge gaps shorter than ``gap``; drop intervals shorter
   than ``min_duration`` unless their peak force reaches ``keep_force`` (impacts).
4. **Modes**: sweep the intervals; absorb modes shorter than ``min_mode`` into the
   mode before them; merge equal neighbours.
5. **Tail** (``trim_tail``): the modes after the robot's last contact are the passive
   outcome (a thrown cube landing in the bin or on the floor) and depend on when
   the episode stops. They are kept for display but are not part of the strategy:
   ``strategy_modes`` and the distances leave them out.

Distances, all in [0, 1]:

* ``pair_distance``: Jaccard distance between the sets of pairs ever used. Ignores
  order and time: "does one run use a contact the other never does?"
* ``sequence_distance``: edit distance between the mode sequences, where replacing
  one mode by another costs their Jaccard distance. Sensitive to order, not to
  duration.

    python -m experiments.contacts.strategy show <trace dir>
    python -m experiments.contacts.strategy evolution 'opus55/tossing3d/*'
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import re
from pathlib import Path
from typing import Any

import numpy as np

from experiments.contacts.entities import FACES
from experiments.contacts.trace import RESTING_FRACTION, load_meta

SEP = " <> "
DEFAULT_CATEGORIES = (
    "robot-object",
    "robot-fixture",
    "robot-world",
    "object-self",
    "object-fixture",
    "object-world",
)
_SIDE = re.compile(r"\b(left|right)_")
_PAD = re.compile(r"^robot\.gripper\.(left_|right_)?pad$")
_FINGER = re.compile(r"^robot\.gripper\.(left_|right_)?finger$")


@dataclasses.dataclass(frozen=True)
class Options:
    """How a trace is turned into a contact-mode sequence."""

    level: str = "class"  # class | entity | surface
    merge_sides: bool = True  # left/right robot parts count as one
    # [s] merge gaps of a pair up to this. Most gaps are contact chatter (86 % are
    # below 5 ms); few gaps are between 0.2 s and 1 s, and a longer gap is a new
    # contact. 0.3 s is in that valley and is the most repeatable value.
    gap: float = 0.3
    min_duration: float = 0.03  # [s] drop shorter intervals ...
    keep_force: float = 20.0  # [N] ... unless their peak force reaches this
    keep_impulse: float = float("inf")  # [N s] ... or their impulse reaches this
    min_mode: float = 0.05  # [s] absorb modes shorter than this
    categories: tuple[str, ...] = DEFAULT_CATEGORIES
    resting: bool = False  # keep resting pairs
    trim_tail: bool = True  # the strategy ends at the robot's last contact
    # A finger link on a thing counts only while no pad is on that same thing: the
    # linkage that touches an object held between the pads is part of the grasp.
    pad_first: bool = False


Mode = frozenset[str]

# What counts as "one strategy". Each choice has a one-line description that the
# viewer prints next to its controls, so a number is never shown without it.
SCOPES = {
    "all": "every contact pair",
    "robot": "only the contact pairs with a robot part",
}
IDENTITIES = {
    "keep": "each object and fixture keeps its name",
    "merge": "numbered and coloured copies are one thing (bin_red = bin_yellow, "
    "shelf2 = shelf3)",
    "kind": "only the kind of thing counts (object, fixture, world; gripper, arm "
    "or base of the robot)",
}
ORDERS = {
    "sequence": "the full sequence of contact modes",
    "first": "each contact mode once, in the order of its first occurrence "
    "(repeats for more objects are ignored)",
    "onset": "each contact pair once, in the order in which the pairs first occur "
    "(which contacts are simultaneous is ignored)",
    "set": "the set of contact pairs, in any order",
}
FINGERS = {
    "all": "a finger link counts as its own robot part at all times",
    "pad_first": "a finger link on a thing does not count while a pad is on that "
    "same thing",
}
# The distance between two episodes (``metrics.py`` computes them).
DISTANCES = {
    "edit": "the edit distance between the two mode sequences (the order counts, "
    "the time does not)",
    "warp": "the elastic distance between the two mode sequences (the order "
    "counts, the time does not; a mode can match more than one mode of the other "
    "sequence, so a short extra mode costs little)",
    "overlap": "1 - the share of the time in which the two contact timelines agree "
    "(the time counts; the order choice has no effect)",
    "set": "the Jaccard distance between the two sets of contact pairs",
    "duty": "the difference in the share of time of each contact pair",
    "dtw": "the time-warped distance between the two mode timelines",
}
# Each preset sets the parts that it names and leaves the other parts as they are.
# "tuned" is the result of the repeatability test (see ``repeat.py`` and the README):
# a final policy is the same strategy as itself on another start scene in about 9
# cases of 10, with one threshold for all scenarios. The names stay as given (a
# choice of the user): on one instance all runs see the same objects, so the names
# say which object a run handles first. The test prefers "merge" only where the
# object names change with the instance (sortclutteredblocks3d).
PRESETS: dict[str, dict[str, Any]] = {
    "tuned": {
        "scope": "all",
        "identity": "keep",
        "order": "sequence",
        "fingers": "pad_first",
        "metric": "warp",
        "group": "similar",
        "tau": 0.25,
    },
    "detailed": {"scope": "all", "identity": "keep", "order": "sequence"},
    "object-agnostic": {"scope": "all", "identity": "merge", "order": "first"},
    "robot skeleton": {"scope": "robot", "identity": "merge", "order": "onset"},
    "contact set": {"scope": "all", "identity": "merge", "order": "set"},
}
_COPY = re.compile(
    r"_(red|green|blue|yellow|orange|purple|pink|black|white)(?![A-Za-z0-9])"
    r"|_s\d+c\d+|_?\d+(?![A-Za-z0-9])"
)


@dataclasses.dataclass(frozen=True)
class Definition:
    """What counts as one strategy (see ``SCOPES``, ``IDENTITIES``, ``ORDERS``)."""

    scope: str = "all"
    identity: str = "keep"
    order: str = "sequence"
    fingers: str = "all"
    metric: str = "edit"
    # Debounce times [s]; ``None`` = the default of ``Options``.
    gap: float | None = None
    min_duration: float | None = None
    min_mode: float | None = None

    def __post_init__(self) -> None:
        for value, known in (
            (self.scope, SCOPES),
            (self.identity, IDENTITIES),
            (self.order, ORDERS),
            (self.fingers, FINGERS),
            (self.metric, DISTANCES),
        ):
            if value not in known:
                raise ValueError(f"unknown strategy definition part {value!r}")
        for value in (self.gap, self.min_duration, self.min_mode):
            if value is not None and not 0.0 <= value <= 5.0:
                raise ValueError("a debounce time must be between 0 and 5 s")

    @property
    def naming(self) -> tuple[str, str, str]:
        """The part that ``abstract`` applies to a mode sequence."""
        return self.scope, self.identity, self.order

    def text(self, group: str = "exact", tau: float = 0.35) -> str:
        """One sentence that says what one strategy is."""
        same = (
            "are equal"
            if group == "exact"
            else f"are within a distance of {tau:g} of each other, where the "
            f"distance is {DISTANCES[self.metric]}"
        )
        opts = self.options(Options())
        return (
            f"A strategy is {ORDERS[self.order]}, up to the robot's last contact. "
            f"It uses {SCOPES[self.scope]}, and {IDENTITIES[self.identity]}. "
            + ("" if self.fingers == "all" else f"Also, {FINGERS[self.fingers]}. ")
            + f"Two contacts of one pair with a gap of {opts.gap:g} s or less are one "
            f"contact, and a mode shorter than {opts.min_mode:g} s does not count. "
            f"Two episodes have the same strategy when these {same}."
        )

    def options(self, opts: Options) -> Options:
        """``opts`` with the choices of this definition that act on the intervals."""
        change: dict[str, Any] = (
            {"pad_first": True} if self.fingers == "pad_first" else {}
        )
        for field in ("gap", "min_duration", "min_mode"):
            if getattr(self, field) is not None:
                change[field] = getattr(self, field)
        return dataclasses.replace(opts, **change) if change else opts


def _category(kind_a: str, kind_b: str) -> str:
    return f"{kind_a}-{kind_b}" if kind_a != kind_b else f"{kind_a}-self"


def _subtract(rows: list[list[float]], mask: list[list[float]]) -> list[list[float]]:
    """The parts of the ``[start, end]`` rows that no ``mask`` interval covers."""
    out: list[list[float]] = []
    for start, end in rows:
        for m_start, m_end, *_ in mask:
            if m_end <= start or m_start >= end:
                continue
            if m_start > start:
                out.append([start, m_start])
            start = max(start, m_end)
            if start >= end:
                break
        if start < end:
            out.append([start, end])
    return out


def _union(rows: list[tuple[float, ...]], gap: float) -> list[list[float]]:
    """Union of ``(start, end, peak[, impulse])`` intervals, merging gaps <= ``gap``.

    The merged interval has the largest peak and the sum of the impulses.
    """
    merged: list[list[float]] = []
    for start, end, peak, *more in sorted(rows):
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2] = max(merged[-1][2], peak)
            if more:
                merged[-1][3] += more[0]
        else:
            merged.append([start, end, peak, *more])
    return merged


@functools.lru_cache(maxsize=4096)
def pair_intervals(
    directory: Path, opts: Options = Options()
) -> dict[str, list[list[float]]]:
    """Debounced ``[start, end, peak force]`` intervals (seconds) per pair label."""
    meta = load_meta(directory)
    with np.load(directory / "trace.npz") as data:
        keys, intervals, steps = data["keys"], data["intervals"], data["steps"]
        step_ticks = data["step_ticks"]
    dt, total = meta["dt"], max(int(meta["num_ticks"]), 1)
    geoms = {g["id"]: g for g in meta["geoms"]}
    entities = meta["entities"]

    def label(geom: int, face: int) -> tuple[str, str]:
        entity = entities[geoms[geom]["entity"]]
        name = entity["cls"] if opts.level == "class" else entity["name"]
        if opts.merge_sides and entity["kind"] == "robot":
            name = _SIDE.sub("", name)
        if opts.level == "surface":
            name += f"[{geoms[geom]['index']}:{FACES[face]}]"
        return name, entity["kind"]

    # The rows of each key (the files are not sorted by key).
    def by_key(rows: np.ndarray) -> list[np.ndarray]:
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        bounds = np.searchsorted(rows[:, 0], np.arange(len(keys) + 1))
        return [rows[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    key_intervals, key_steps = by_key(intervals), by_key(steps)
    ticks = step_ticks.astype(float)

    # Per key: the other entity when one side is a pad or a finger link.
    held: dict[int, list[tuple[float, float, float]]] = {}
    finger: dict[int, int] = {}
    if opts.pad_first:
        for key, (g_a, _, g_b, _) in enumerate(keys.tolist()):
            ids = [geoms[g_a]["entity"], geoms[g_b]["entity"]]
            for part, other in (ids, ids[::-1]):
                if _PAD.match(entities[part]["name"]):
                    own = key_intervals[key][:, 1:]
                    held.setdefault(other, []).extend(
                        (s * dt, e * dt, 0.0) for s, e in own.tolist()
                    )
                elif _FINGER.match(entities[part]["name"]):
                    finger[key] = other
    # The same debounce as the pad pairs below, so that no instant loses both.
    masks = {
        other: [
            iv for iv in _union(rows, opts.gap) if iv[1] - iv[0] >= opts.min_duration
        ]
        for other, rows in held.items()
    }

    raw: dict[str, list[tuple[float, ...]]] = {}
    robot: dict[str, bool] = {}
    for key, (g_a, f_a, g_b, f_b) in enumerate(keys.tolist()):
        (name_a, kind_a), (name_b, kind_b) = label(g_a, f_a), label(g_b, f_b)
        if _category(kind_a, kind_b) not in opts.categories:
            continue
        name = SEP.join((name_a, name_b))
        robot[name] = "robot" in (kind_a, kind_b)
        own = key_intervals[key][:, 1:]
        rows = key_steps[key]
        # Force is stored per env step: an interval gets the peak of its steps, and
        # as impulse the mean normal force of each step times its time in that step.
        fn_mean, fn_max = np.zeros(len(ticks)), np.zeros(len(ticks))
        at = rows[:, 1].astype(int).clip(0, len(ticks) - 1)
        fn_mean[at], fn_max[at] = rows[:, 3], rows[:, 4]
        spans = own.tolist()
        if finger.get(key) in masks:
            free = _subtract([[s * dt, e * dt] for s, e in spans], masks[finger[key]])
            spans = [[s / dt, e / dt] for s, e in free]
        if not spans:
            continue
        span = np.asarray(spans, dtype=float)
        # the first and the last env step that each interval is in
        low = (np.searchsorted(ticks, span[:, 0], side="right") - 1).clip(0)
        high = (np.searchsorted(ticks, span[:, 1], side="left") - 1).clip(0)
        high = np.maximum(high, low).clip(max=len(ticks) - 1)
        peaks = fn_max[low]
        impulses = fn_mean[low] * (span[:, 1] - span[:, 0])
        for i in np.flatnonzero(high > low):  # an interval over more than one step
            start, end = span[i]
            steps_in = np.arange(low[i], high[i] + 1)
            upper = ticks[(steps_in + 1).clip(max=len(ticks) - 1)]
            inside = np.minimum(upper, end) - np.maximum(ticks[steps_in], start)
            peaks[i] = fn_max[steps_in].max()
            impulses[i] = (fn_mean[steps_in] * inside.clip(0)).sum()
        raw.setdefault(name, []).extend(
            zip(
                (span[:, 0] * dt).tolist(),
                (span[:, 1] * dt).tolist(),
                peaks.tolist(),
                (impulses * dt).tolist(),
            )
        )

    out: dict[str, list[list[float]]] = {}
    for name, rows_ in raw.items():
        if not opts.resting and not robot[name]:
            active = sum(iv[1] - iv[0] for iv in _union(rows_, 0.0))
            if active >= RESTING_FRACTION * total * dt:
                continue
        kept: list[tuple[float, ...]] = []
        for start, end, peak, impulse in _union(rows_, opts.gap):
            if end - start >= opts.min_duration:
                kept.append((start, end, peak))
            elif peak >= opts.keep_force or impulse >= opts.keep_impulse:
                # An impact: short but strong. It counts as a contact of at least
                # ``min_mode``, so that the mode it makes is not absorbed.
                kept.append((start, max(end, start + opts.min_mode), peak))
        if kept:
            out[name] = [
                [round(s, 4), round(e, 4), round(p, 2)] for s, e, p in _union(kept, 0.0)
            ]
    return out


@functools.lru_cache(maxsize=4096)
def mode_sequence(
    directory: Path, opts: Options = Options()
) -> tuple[tuple[Mode, float, float], ...]:
    """The contact-mode sequence of a trace: ``(mode, start, end)`` in seconds."""
    meta = load_meta(directory)
    duration = max(int(meta["num_ticks"]), 1) * meta["dt"]
    events: list[tuple[float, int, str]] = []
    for name, rows in pair_intervals(directory, opts).items():
        for start, end, _ in rows:
            events += [(start, 1, name), (min(end, duration), -1, name)]
    events.sort(key=lambda e: (e[0], e[1]))
    modes: list[list[Any]] = []
    active: dict[str, int] = {}
    time = 0.0
    i = 0
    while True:
        nxt = events[i][0] if i < len(events) else duration
        if nxt > time:
            modes.append([frozenset(n for n, c in active.items() if c > 0), time, nxt])
            time = nxt
        if i >= len(events):
            break
        while i < len(events) and events[i][0] == nxt:
            active[events[i][2]] = active.get(events[i][2], 0) + events[i][1]
            i += 1
    # absorb short modes into the mode before them (the first into the next one)
    kept: list[list[Any]] = []
    for mode in modes:
        if mode[2] - mode[1] < opts.min_mode and kept:
            kept[-1][2] = mode[2]
        elif kept and kept[-1][0] == mode[0]:
            kept[-1][2] = mode[2]
        else:
            kept.append(mode)
    if len(kept) > 1 and kept[0][2] - kept[0][1] < opts.min_mode:
        kept[1][1] = kept[0][1]
        kept.pop(0)
    merged: list[list[Any]] = []
    for mode in kept:
        if merged and merged[-1][0] == mode[0]:
            merged[-1][2] = mode[2]
        else:
            merged.append(mode)
    return tuple((m, round(s, 4), round(e, 4)) for m, s, e in merged)


def has_robot(mode: Mode) -> bool:
    """Whether a robot part is in contact in this mode."""
    return any(part.startswith("robot.") for pair in mode for part in pair.split(SEP))


def strategy_length(directory: Path, opts: Options = Options()) -> int:
    """How many leading modes of ``mode_sequence`` make up the strategy."""
    modes = mode_sequence(directory, opts)
    if not opts.trim_tail:
        return len(modes)
    last = max((i for i, (m, _, _) in enumerate(modes) if has_robot(m)), default=-1)
    return last + 1


def strategy_modes(directory: Path, opts: Options = Options()) -> tuple[Mode, ...]:
    """The contact strategy of a trace: its modes up to the robot's last contact."""
    modes = mode_sequence(directory, opts)
    return tuple(m for m, _, _ in modes[: strategy_length(directory, opts)])


def _rename(part: str, identity: str, kinds: dict[str, str]) -> str:
    if identity == "keep":
        return part
    if part.startswith("robot."):
        return ".".join(part.split(".")[:2]) if identity == "kind" else part
    if identity == "kind":
        return kinds.get(part, part.split(".")[0])
    return _COPY.sub("", part)


def abstract(
    modes: tuple[Mode, ...], defn: Definition, kinds: dict[str, str]
) -> tuple[Mode, ...]:
    """Rewrite a mode sequence under a strategy definition.

    ``kinds`` maps an entity name to its kind (``object``, ``fixture``, ``world``).
    With ``order == "set"`` the result is one mode: the union of all pairs, so that
    ``edit_distance`` between two results is the Jaccard distance of the sets. With
    ``order == "onset"`` every mode is one pair.
    """
    out: list[Mode] = []
    for mode in modes:
        pairs = set()
        for pair in mode:
            parts = pair.split(SEP)
            if defn.scope == "robot" and not any(p.startswith("robot.") for p in parts):
                continue
            pairs.add(SEP.join(_rename(p, defn.identity, kinds) for p in parts))
        new = frozenset(pairs)
        if not out or out[-1] != new:
            out.append(new)
    if defn.order == "first":
        out = list(dict.fromkeys(out))
    elif defn.order == "onset":
        pairs_seen: dict[str, None] = {}
        for mode in out:
            pairs_seen.update(dict.fromkeys(sorted(mode)))
        out = [frozenset((pair,)) for pair in pairs_seen]
    elif defn.order == "set":
        out = [frozenset().union(*out)] if out else []
    return tuple(out)


@functools.lru_cache(maxsize=16384)
def strategy_key(
    directory: Path, opts: Options = Options(), defn: Definition = Definition()
) -> tuple[Mode, ...]:
    """The strategy of a trace under a definition (``abstract`` of its modes)."""
    modes = strategy_modes(directory, defn.options(opts))
    if defn.naming == Definition().naming:
        return modes
    return abstract(modes, defn, entity_kinds(directory))


def entity_kinds(directory: Path) -> dict[str, str]:
    """The kind (``object``, ``fixture``, ...) of each entity name and class."""
    kinds: dict[str, str] = {}
    for entity in load_meta(directory)["entities"]:
        kinds[entity["cls"]] = kinds[entity["name"]] = entity["kind"]
    return kinds


def pairs_used(directory: Path, opts: Options = Options()) -> Mode:
    """Every pair label that occurs in the strategy."""
    return frozenset().union(*strategy_modes(directory, opts))


def jaccard(a: Mode, b: Mode) -> float:
    """Jaccard distance between two sets (0 when both are empty)."""
    union = len(a | b)
    return 1.0 - len(a & b) / union if union else 0.0


def edit_distance(a: list[Mode], b: list[Mode]) -> float:
    """Edit distance between mode sequences, normalised by the longer one.

    Insertion and deletion cost 1; replacing a mode costs the Jaccard distance
    between the two modes, so a mode that gains one pair is a small change.
    """
    if not a and not b:
        return 0.0
    prev = np.arange(len(b) + 1, dtype=float)
    for i, mode_a in enumerate(a, 1):
        cur = np.empty(len(b) + 1)
        cur[0] = i
        for j, mode_b in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + jaccard(mode_a, mode_b)
            )
        prev = cur
    return float(prev[-1]) / max(len(a), len(b))


def warp_path(a: list[Mode], b: list[Mode]) -> tuple[float, int]:
    """Total cost and length of the best elastic alignment of two mode sequences.

    The alignment keeps the order and matches every mode of each sequence with one
    or more modes of the other (dynamic time warping over modes, not over time).
    A matched pair of modes costs their Jaccard distance. A mode that only adds a
    contact pair to its neighbour is thus cheap, where ``edit_distance`` charges a
    full edit for it.
    """
    inf = float("inf")
    total = [[inf] * (len(b) + 1) for _ in range(len(a) + 1)]
    steps = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    total[0][0] = 0.0
    for i, mode_a in enumerate(a, 1):
        for j, mode_b in enumerate(b, 1):
            best, length = total[i - 1][j - 1], steps[i - 1][j - 1]
            for other in ((i - 1, j), (i, j - 1)):
                if total[other[0]][other[1]] < best:
                    best, length = total[other[0]][other[1]], steps[other[0]][other[1]]
            total[i][j] = best + jaccard(mode_a, mode_b)
            steps[i][j] = length + 1
    return total[-1][-1], steps[-1][-1]


def warp_distance(a: list[Mode], b: list[Mode]) -> float:
    """Mean cost along ``warp_path``; 1 when only one sequence is empty."""
    if not a or not b:
        return 0.0 if not a and not b else 1.0
    cost, length = warp_path(a, b)
    return cost / length


def key_distance(a: list[Mode], b: list[Mode], metric: str = "edit") -> float:
    """A distance between two mode sequences that does not use time."""
    return warp_distance(a, b) if metric == "warp" else edit_distance(a, b)


def pair_distance(dir_a: Path, dir_b: Path, opts: Options = Options()) -> float:
    """Jaccard distance between the sets of pairs two traces use."""
    return jaccard(pairs_used(dir_a, opts), pairs_used(dir_b, opts))


def sequence_distance(dir_a: Path, dir_b: Path, opts: Options = Options()) -> float:
    """Edit distance between the mode sequences of two traces."""
    return edit_distance(
        list(strategy_modes(dir_a, opts)), list(strategy_modes(dir_b, opts))
    )


def identical_raw(dir_a: Path, dir_b: Path) -> bool:
    """Whether two traces have exactly the same contact keys and intervals."""
    with np.load(dir_a / "trace.npz") as a, np.load(dir_b / "trace.npz") as b:
        return all(
            a[k].shape == b[k].shape and bool(np.array_equal(a[k], b[k]))
            for k in ("keys", "intervals", "qpos")
        )


def describe(directory: Path, opts: Options = Options()) -> dict[str, Any]:
    """JSON-ready strategy of one trace."""
    length = strategy_length(directory, opts)
    return {
        "pairs": pair_intervals(directory, opts),
        "modes": [
            {"pairs": sorted(m), "start": s, "end": e, "tail": i >= length}
            for i, (m, s, e) in enumerate(mode_sequence(directory, opts))
        ],
    }


def short(mode: Mode) -> str:
    """A compact name of a mode for the terminal."""
    if not mode:
        return "(free)"
    trim = lambda n: n.replace("robot.gripper.", "").replace("robot.", "")  # noqa: E731
    trim2 = lambda n: trim(n).replace("env.", "").replace("world.", "")  # noqa: E731
    return " + ".join(sorted("~".join(map(trim2, p.split(SEP))) for p in mode))


def main() -> None:
    """Print strategies, or the evolution of runs over their recorded commits."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.commits import load_versions
    from experiments.contacts.record import (  # pylint: disable=import-outside-toplevel
        DEFAULT_OUT,
        instance_dir,
        matching_runs,
    )
    from experiments.render_run import (  # pylint: disable=import-outside-toplevel
        DEFAULT_DATA,
    )

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=["show", "evolution"])
    parser.add_argument("targets", nargs="+")
    parser.add_argument("--level", default="class")
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    opts = Options(level=args.level)
    if args.command == "show":
        for target in args.targets:
            directory = Path(target).resolve()
            print(target)
            length = strategy_length(directory, opts)
            for i, (mode, start, end) in enumerate(mode_sequence(directory, opts)):
                tail = "   (after the last robot contact)" if i >= length else ""
                print(f"  {start:7.2f} - {end:7.2f} s  {short(mode)}{tail}")
        return
    for run_id in matching_runs(args.data, args.targets):
        versions = load_versions(args.data, args.out, run_id)
        dirs = {
            v["commit"]: instance_dir(
                args.out, run_id, v["commit"], args.instance, None
            )
            for v in versions
        }
        dirs = {c: d for c, d in dirs.items() if (d / "meta.json").exists()}
        if not dirs:
            continue
        final = dirs[max(dirs)]
        print(f"\n{run_id}: {len(dirs)} of {len(versions)} commits recorded")
        prev = None
        for v in versions:
            directory = dirs.get(v["commit"])
            if directory is None:
                continue
            metrics = load_meta(directory)["metrics"]
            seq = strategy_modes(directory, opts)
            d_prev = (
                "    -"
                if prev is None
                else f"{sequence_distance(prev, directory, opts):5.2f}"
            )
            rate = v["reeval_solve_rate"]
            print(
                f"  c{v['commit']:<4d} "
                f"{'crash' if metrics.get('crashed') else 'ok   ' if metrics.get('solved') else 'fail '}"
                f" sr={'  - ' if rate is None else f'{rate:4.2f}'} modes={len(seq):3d}"
                f" d_prev={d_prev} d_final={sequence_distance(directory, final, opts):5.2f}"
                f" pairs_final={pair_distance(directory, final, opts):5.2f}"
                f"  {v['subject'][:60]}"
            )
            prev = directory


if __name__ == "__main__":
    main()
