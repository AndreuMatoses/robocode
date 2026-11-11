"""Draw recorded dynamic2d episodes from the trace alone.

A pymunk trace holds the shape of every entity (in the frame of its body) and the
pose ``x, y, theta`` of every entity at every env step. That is enough to draw the
episode, so no environment is built here: videos, contact-highlight videos, ghost
overlays and the labelled diagram all come from ``meta.json`` and ``trace.npz``.

The picture is not the one of ``env.render()``: it shows the collision shapes that
the contacts are computed on, in the colours of the state objects.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

Colour = tuple[float, float, float]
KIND_COLOUR: dict[str, Colour] = {
    "robot": (0.45, 0.35, 0.75),
    "object": (0.93, 0.62, 0.22),
    "fixture": (0.62, 0.64, 0.68),
    "world": (0.25, 0.26, 0.29),
}
BACKGROUND = (250, 250, 248)
ROBOT_TINT, OTHER_TINT = (1.0, 0.45, 0.05), (0.1, 0.75, 1.0)
SUPERSAMPLE = 2  # draw larger and scale down: smooth edges


def _rgb(colour: Iterable[float], alpha: float | None = None) -> tuple[int, ...]:
    values = [int(round(255 * min(max(c, 0.0), 1.0))) for c in colour]
    return tuple(values if alpha is None else values + [int(round(255 * alpha))])


class Scene:
    """World-to-pixel transform and the drawable parts of one trace."""

    def __init__(self, meta: dict[str, Any], width: int, height: int) -> None:
        self.meta, self.width, self.height = meta, width, height
        x0, x1, y0, y1 = meta["world"]
        big_w, big_h = width * SUPERSAMPLE, height * SUPERSAMPLE
        margin = 0.03 * max(x1 - x0, y1 - y0)
        self.scale = min(big_w / (x1 - x0 + 2 * margin), big_h / (y1 - y0 + 2 * margin))
        self.off_x = (big_w - self.scale * (x1 - x0)) / 2 - self.scale * x0
        self.off_y = (big_h + self.scale * (y1 - y0)) / 2 + self.scale * y0
        self.size = (big_w, big_h)
        self.entities = meta["entities"]
        self.shapes = meta["shapes"]
        z = meta.get("z_order", {})
        self.order = sorted(
            range(len(self.entities)), key=lambda e: (float(z.get(str(e), 1.0)), e)
        )
        colours = meta.get("colours", {})
        self.colours: list[Colour] = [
            (
                tuple(colours[str(e["id"])])
                if str(e["id"]) in colours
                else KIND_COLOUR[e["kind"]]
            )
            for e in self.entities
        ]  # type: ignore[misc]

    def pixel(self, points: np.ndarray) -> list[tuple[float, float]]:
        """World ``[n, 2]`` points as pixel coordinates (y points down)."""
        xs = self.off_x + self.scale * points[:, 0]
        ys = self.off_y - self.scale * points[:, 1]
        return list(zip(xs.tolist(), ys.tolist()))

    def outline(self, geom: int, pose: np.ndarray) -> list[tuple[float, float]]:
        """The outline of one shape at a body pose, in pixels."""
        shape = self.shapes[geom]
        if shape["kind"] == "circle":
            angles = np.linspace(0.0, 2 * math.pi, 40, endpoint=False)
            local = np.array(shape["offset"]) + shape["radius"] * np.column_stack(
                (np.cos(angles), np.sin(angles))
            )
        else:
            local = np.array(shape["points"])
        cos, sin = math.cos(pose[2]), math.sin(pose[2])
        world = local @ np.array([[cos, sin], [-sin, cos]]) + pose[:2]
        return self.pixel(world)

    def blank(self) -> Image.Image:
        """The background with the parts that have no body (goal regions)."""
        image = Image.new("RGB", self.size, BACKGROUND)
        draw = ImageDraw.Draw(image)
        for item in sorted(self.meta.get("decor", []), key=lambda d: d["z"]):
            draw.polygon(
                self.pixel(np.array(item["points"])), fill=_rgb(item["colour"])
            )
        return image

    def paint(
        self,
        image: Image.Image,
        poses: np.ndarray,
        entities: Iterable[int],
        colour_of: Callable[[int], Colour] | None = None,
        alpha: float | None = None,
    ) -> None:
        """Draw ``entities`` at ``poses`` on ``image`` (translucent with ``alpha``)."""
        draw = ImageDraw.Draw(image, "RGBA")
        wanted = set(entities)
        line = max(1, SUPERSAMPLE)
        for entity in self.order:
            if entity not in wanted:
                continue
            colour = (
                colour_of(entity) if colour_of is not None else self.colours[entity]
            )
            edge = _rgb(
                [c * 0.45 for c in colour],
                1.0 if alpha is None else min(1.0, alpha * 2.2),
            )
            for geom in self.entities[entity]["geoms"]:
                draw.polygon(
                    self.outline(geom, poses[entity]),
                    fill=_rgb(colour, 1.0 if alpha is None else alpha),
                    outline=edge,
                    width=line,
                )

    def finish(self, image: Image.Image) -> np.ndarray:
        """Scale the picture down to the requested size."""
        return np.asarray(
            image.resize((self.width, self.height), Image.Resampling.LANCZOS)
        )


def poses_of(qpos: np.ndarray) -> np.ndarray:
    """``[T, E, 3]`` poses from the ``qpos`` array of a trace."""
    return np.asarray(qpos, dtype=np.float64).reshape(len(qpos), -1, 3)


def load(directory: Path) -> tuple[dict[str, Any], np.ndarray]:
    """``meta.json`` and the poses of a pymunk trace."""
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("backend") != "pymunk":
        raise ValueError(f"{directory} is not a pymunk trace")
    with np.load(directory / "trace.npz") as data:
        return meta, poses_of(data["qpos"])


def episode_frames(
    meta: dict[str, Any],
    poses: np.ndarray,
    width: int,
    height: int,
    tint_for_step: Callable[[int], dict[int, Colour]] | None = None,
) -> Iterable[np.ndarray]:
    """One frame per recorded env step."""
    scene = Scene(meta, width, height)
    blank = scene.blank()
    everything = range(len(scene.entities))
    for step, pose in enumerate(poses):
        image = blank.copy()
        tint = tint_for_step(step) if tint_for_step is not None else {}
        scene.paint(image, pose, everything, lambda e: tint.get(e, scene.colours[e]))
        yield scene.finish(image)


def contact_frames(directory: Path, width: int, height: int) -> Iterable[np.ndarray]:
    """The episode with every entity in a (non-resting) contact lit up."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.trace import active_entities_per_step

    meta, poses = load(directory)
    active = active_entities_per_step(directory)
    kinds = [e["kind"] for e in meta["entities"]]

    def tint_for_step(step: int) -> dict[int, Colour]:
        if not active:
            return {}
        touching = active[max(step - 1, 0)] | active[min(step, len(active) - 1)]
        return {e: ROBOT_TINT if kinds[e] == "robot" else OTHER_TINT for e in touching}

    return episode_frames(meta, poses, width, height, tint_for_step)


def ghost_frames(
    metas: list[dict[str, Any]],
    trajectories: list[np.ndarray],
    colours: list[Colour],
    width: int,
    height: int,
    alpha: float | None = None,
    robot_only: bool = True,
    equal: bool = False,
) -> Iterable[np.ndarray]:
    """Overlay several episodes of one scene, one colour per episode.

    The same rules as ``replay.ghost_frames``: without ``equal`` the first episode
    is drawn solid and the others are ghosts; with ``equal`` every episode is a
    ghost. With ``robot_only`` the objects are those of the first episode; without
    it an object is ghosted too, but only while its pose differs between the
    episodes. An episode that ended holds its last pose.
    """
    for meta in metas[1:]:
        if meta["signature"] != metas[0]["signature"]:
            raise SystemExit("the traces are not of the same scene")
    scene = Scene(metas[0], width, height)
    blank = scene.blank()
    count = len(trajectories)
    if alpha is None:
        alpha = min(0.35, max(0.15, 1.4 / math.sqrt(max(count, 1))))
    kinds = [e["kind"] for e in scene.entities]
    robots = [e for e, kind in enumerate(kinds) if kind == "robot"]
    movable = [
        e for e, kind in enumerate(kinds) if kind in ("robot", "object", "fixture")
    ]
    fixed = [e for e, kind in enumerate(kinds) if kind == "world"]
    for step in range(max(len(t) for t in trajectories)):
        poses = [t[min(step, len(t) - 1)] for t in trajectories]
        image = blank.copy()
        scene.paint(image, poses[0], fixed)
        if robot_only:
            ghosted = set(robots)
        else:
            stack = np.stack(poses)
            apart = np.abs(stack - stack[0]).max(axis=(0, 2)) > 2e-3
            ghosted = {e for e in movable if kinds[e] == "robot" or apart[e]}
        scene.paint(image, poses[0], [e for e in movable if e not in ghosted])
        for k in range(count - 1, -1, -1):  # the first episode on top
            if k == 0 and not equal:
                scene.paint(image, poses[0], ghosted)
            else:
                scene.paint(image, poses[k], ghosted, lambda _e, k=k: colours[k], alpha)
        yield scene.finish(image)


def write_diagram(
    meta: dict[str, Any], pose: np.ndarray, out: Path, width: int, height: int
) -> None:
    """The labelled-diagram assets of one state (see ``replay.write_diagram``)."""
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene(meta, width, height)
    image = scene.blank()
    everything = range(len(scene.entities))
    scene.paint(image, pose, everything)
    Image.fromarray(scene.finish(image)).save(out / "scene.png")
    # Entity id of each pixel, as red + 256 * green; 65535 = nothing. No smoothing.
    plain = Scene(meta, width, height)
    ids = Image.new("RGB", plain.size, (255, 255, 0))
    draw = ImageDraw.Draw(ids)
    for entity in plain.order:
        for geom in plain.entities[entity]["geoms"]:
            draw.polygon(
                plain.outline(geom, pose[entity]), fill=(entity & 255, entity >> 8, 0)
            )
    small = np.asarray(ids.resize((width, height), Image.Resampling.NEAREST))
    Image.fromarray(small).save(out / "scene_ids.png")
    ent = small[..., 0].astype(np.int32) + (small[..., 1].astype(np.int32) << 8)
    visible = {}
    for entity in np.unique(ent[ent != 65535]).tolist():
        ys, xs = np.nonzero(ent == entity)
        visible[str(entity)] = {
            "pixels": int(xs.size),
            "x": round(float(xs.mean()), 1),
            "y": round(float(ys.mean()), 1),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        }
    index = {"width": width, "height": height, "views": {"scene": visible}}
    (out / "diagram.json").write_text(json.dumps(index))
