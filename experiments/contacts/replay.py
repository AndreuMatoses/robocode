"""Re-render recorded traces without running any policy.

A trace stores the simulator's ``qpos`` at every env step, and a reset with the same
seed and object count rebuilds the same scene. So videos, labelled diagrams and
ghost overlays are all produced here by setting ``qpos`` on a fresh environment and
rendering, at any resolution and from any camera.

kinder's render context updates the scene and draws it in one call; ``render``
below repeats those two MuJoCo calls with a hook in between, to recolour geoms, add
ghost geoms, or switch to a segmentation render.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mujoco
import numpy as np

from experiments.contacts.entities import EntityMap, build_entity_map, robot_env

SceneEdit = Callable[[Any], None]  # receives the mjvScene before it is drawn


def render_camera_name(env: Any) -> str:
    """The camera ``env.render()`` uses."""
    backend = getattr(env, "_current_backend", None) or env._kinder_env
    # pylint: disable-next=protected-access
    return backend._object_centric_env._render_camera_name


def render(
    renv: Any,
    camera: str | dict[str, Any],
    width: int,
    height: int,
    edit: SceneEdit | None = None,
    segment: bool = False,
) -> np.ndarray:
    """Render a kinder ``MujocoEnv`` at its current state.

    ``camera`` is a model camera name, or a free camera given as a dict with
    ``lookat``, ``distance``, ``azimuth`` and ``elevation``.

    With ``segment`` the result is an ``[h, w]`` int array of model geom ids
    (-1 = background or not a model geom).
    """
    # pylint: disable-next=import-outside-toplevel
    from kinder.envs.dynamic3d import mujoco_utils

    sim = renv.sim
    ctx = sim._render_context_offscreen  # pylint: disable=protected-access
    with mujoco_utils._MjSim_render_lock:  # pylint: disable=protected-access
        ctx.gl_ctx.make_current()
        if width > ctx.con.offWidth or height > ctx.con.offHeight:
            ctx.update_offscreen_size(
                max(width, ctx.con.offWidth), max(height, ctx.con.offHeight)
            )
        if isinstance(camera, str):
            ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            ctx.cam.fixedcamid = sim.model.camera_name2id(camera)
        else:  # free camera: lookat, distance, azimuth, elevation
            ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            ctx.cam.lookat[:] = camera["lookat"]
            ctx.cam.distance = camera["distance"]
            ctx.cam.azimuth = camera["azimuth"]
            ctx.cam.elevation = camera["elevation"]
        mujoco.mjv_updateScene(
            ctx.model,
            ctx.data,
            ctx.vopt,
            ctx.pert,
            ctx.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            ctx.scn,
        )
        if edit is not None:
            edit(ctx.scn)
        flags = (mujoco.mjtRndFlag.mjRND_SEGMENT, mujoco.mjtRndFlag.mjRND_IDCOLOR)
        for flag in flags:
            ctx.scn.flags[flag] = int(segment)
        try:
            mujoco.mjr_render(
                viewport=mujoco.MjrRect(0, 0, width, height), scn=ctx.scn, con=ctx.con
            )
            image = ctx.read_pixels(width, height)
        finally:
            for flag in flags:
                ctx.scn.flags[flag] = 0
        if not segment:
            return image
        seg = image.astype(np.int32)
        seg_id = seg[..., 0] + (seg[..., 1] << 8) + (seg[..., 2] << 16) - 1
        lookup = np.full(ctx.scn.ngeom + 1, -1, dtype=np.int32)
        for i in range(ctx.scn.ngeom):
            geom = ctx.scn.geoms[i]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                lookup[i] = geom.objid
        return lookup[np.clip(seg_id, -1, ctx.scn.ngeom - 1)]


def set_qpos(renv: Any, qpos: np.ndarray) -> None:
    """Put the simulator in a recorded configuration (kinematics only)."""
    data = renv.sim.data.mj_data
    data.qpos[:] = qpos
    mujoco.mj_kinematics(renv.sim.model.mj_model, data)
    mujoco.mj_camlight(renv.sim.model.mj_model, data)


def tint_entities(emap: EntityMap, colours: dict[int, tuple[float, float, float]]):
    """A scene edit that recolours the geoms of the given entities."""

    def edit(scn: Any) -> None:
        for i in range(scn.ngeom):
            geom = scn.geoms[i]
            if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
                continue
            colour = colours.get(int(emap.geom_entity[geom.objid]))
            if colour is not None:
                geom.rgba[:3] = colour
                geom.emission = 0.35
                geom.matid = -1  # drop the texture so the tint shows

    return edit


def replay_frames(
    env: Any,
    qpos: Iterable[np.ndarray],
    width: int,
    height: int,
    camera: str | None = None,
    edit_for_step: Callable[[int], SceneEdit | None] | None = None,
) -> Iterable[np.ndarray]:
    """Yield one frame per recorded configuration."""
    renv = robot_env(env)
    camera = camera or render_camera_name(env)
    for step, q in enumerate(qpos):
        set_qpos(renv, q)
        edit = edit_for_step(step) if edit_for_step is not None else None
        yield render(renv, camera, width, height, edit=edit)


def write_video(frames: Iterable[np.ndarray], path: Path, fps: int) -> None:
    """Encode frames as H.264 MP4 (``render_run.open_mp4_writer``)."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.render_run import open_mp4_writer

    part = path.with_name(path.name + ".part")
    with open_mp4_writer(part, fps=fps) as append:
        for frame in frames:
            append(frame)
    part.replace(path)


# ------------------------------------------------------------ scene of a trace


def open_trace_env(directory: Path) -> tuple[Any, EntityMap, dict[str, Any]]:
    """Rebuild the scene a trace was recorded in: (env, entity map, meta).

    Raises if the reset does not reproduce the recorded scene.
    """
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.catalog import make_env

    # pylint: disable-next=import-outside-toplevel
    from robocode.utils.episode import reset_for_episode

    meta = json.loads((directory / "meta.json").read_text())
    env = make_env(meta["environment"])
    reset_for_episode(env, meta["instance_seed"], meta["pinned_count"])
    emap = build_entity_map(robot_env(env).sim.model.mj_model)
    if emap.signature != meta["signature"]:
        env.close()
        raise RuntimeError(f"{directory}: the reset did not rebuild the recorded scene")
    return env, emap, meta


def load_qpos(directory: Path) -> np.ndarray:
    """The recorded configurations, one per env step (plus the initial one)."""
    with np.load(directory / "trace.npz") as data:
        return data["qpos"]


# --------------------------------------------------------------------- diagram


def diagram_views(renv: Any) -> dict[str, str | dict[str, Any]]:
    """Cameras for the labelled diagram: the task view and two robot close-ups."""
    model, data = renv.sim.model.mj_model, renv.sim.data.mj_data
    views: dict[str, str | dict[str, Any]] = {}
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot_pinch_site")
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_link")
    if base >= 0:
        lookat = np.array(data.xpos[base]) + [0.0, 0.0, 0.55]
        views["robot"] = {
            "lookat": lookat.tolist(),
            "distance": 2.1,
            "azimuth": 135.0,
            "elevation": -20.0,
        }
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
    if pinch >= 0 and palm >= 0:
        # Look at the fingers from in front of the gripper, a little to one side,
        # so the arm is behind them rather than in the way.
        tip = np.array(data.site_xpos[pinch])
        ahead = tip - np.array(data.xpos[palm])
        views["gripper"] = {
            "lookat": tip.tolist(),
            "distance": 0.38,
            "azimuth": float(np.degrees(np.arctan2(-ahead[1], -ahead[0]))) + 40.0,
            "elevation": -15.0,
        }
    return views


def write_diagram(
    env: Any, emap: EntityMap, out: Path, width: int, height: int
) -> None:
    """Write the labelled-diagram assets of the env's current state.

    Per view: ``<view>.png`` (colour) and ``<view>_ids.png`` (entity id of each
    pixel, as red + 256 * green; 65535 = nothing), plus ``diagram.json`` with each
    visible entity's pixel count, centroid and bounding box per view.
    """
    renv = robot_env(env)
    out.mkdir(parents=True, exist_ok=True)
    views: dict[str, str | dict[str, Any]] = {"scene": render_camera_name(env)}
    views.update(diagram_views(renv))
    index: dict[str, Any] = {"width": width, "height": height, "views": {}}
    for name, camera in views.items():
        colour = render(renv, camera, width, height)
        geom_ids = render(renv, camera, width, height, segment=True)
        ent = np.where(geom_ids >= 0, emap.geom_entity[np.clip(geom_ids, 0, None)], -1)
        ent = np.where(ent < 0, 65535, ent).astype(np.uint16)
        ids = np.zeros((height, width, 3), dtype=np.uint8)
        ids[..., 0], ids[..., 1] = ent & 255, ent >> 8
        iio.imwrite(out / f"{name}.png", colour)
        iio.imwrite(out / f"{name}_ids.png", ids)
        visible = {}
        for entity in np.unique(ent[ent != 65535]).tolist():
            ys, xs = np.nonzero(ent == entity)
            visible[str(entity)] = {
                "pixels": int(xs.size),
                "x": round(float(xs.mean()), 1),
                "y": round(float(ys.mean()), 1),
                "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            }
        index["views"][name] = visible
    (out / "diagram.json").write_text(json.dumps(index))


# ------------------------------------------------------------ contact video

ROBOT_TINT, OTHER_TINT = (1.0, 0.45, 0.05), (0.1, 0.75, 1.0)


def contact_video(directory: Path, out: Path, width: int, height: int) -> None:
    """The episode with every entity in a (non-resting) contact lit up."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.trace import active_entities_per_step

    env, emap, meta = open_trace_env(directory)
    try:
        active = active_entities_per_step(directory)
        kinds = [e.kind for e in emap.entities]

        def edit_for_step(step: int) -> SceneEdit:
            # A frame shows the state after a step: light what touched during it.
            touching = active[max(step - 1, 0)] | active[min(step, len(active) - 1)]
            return tint_entities(
                emap,
                {
                    e: ROBOT_TINT if kinds[e] == "robot" else OTHER_TINT
                    for e in touching
                },
            )

        frames = replay_frames(
            env, load_qpos(directory), width, height, meta["camera"], edit_for_step
        )
        write_video(frames, out, meta["fps"])
    finally:
        env.close()


# --------------------------------------------------------------- ghost video


def ghost_frames(
    env: Any,
    emap: EntityMap,
    trajectories: list[np.ndarray],
    colours: list[tuple[float, float, float]],
    width: int,
    height: int,
    camera: str,
    alpha: float = 0.35,
    robot_only: bool = True,
) -> Iterator[np.ndarray]:
    """Frames with several recorded trajectories of the same instance overlaid.

    The first trajectory is drawn solid; the others are added to the same MuJoCo
    scene as translucent copies tinted with their colour. With ``robot_only`` the
    ghosts are robots only (objects are shown for the first trajectory), otherwise
    every moving body is ghosted. A trajectory that ended holds its last pose.
    """
    renv = robot_env(env)
    model = renv.sim.model.mj_model
    ctx = renv.sim._render_context_offscreen  # pylint: disable=protected-access
    scratch = [mujoco.MjData(model) for _ in trajectories[1:]]
    is_robot = np.array([e.kind == "robot" for e in emap.entities])[emap.geom_entity]

    def edit(scn: Any) -> None:
        for i in range(scn.ngeom):  # tint the solid robot too, to tell it apart
            geom = scn.geoms[i]
            if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and is_robot[geom.objid]:
                geom.rgba[:3] = 0.5 * np.array(geom.rgba[:3]) + 0.5 * np.array(
                    colours[0]
                )
                geom.matid = -1
        for data, colour in zip(scratch, colours[1:]):
            start = scn.ngeom
            mujoco.mjv_addGeoms(
                model, data, ctx.vopt, ctx.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn
            )
            for i in range(start, scn.ngeom):
                geom = scn.geoms[i]
                keep = geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and (
                    not robot_only or is_robot[geom.objid]
                )
                geom.rgba[:3] = colour
                geom.rgba[3] = alpha if keep else 0.0
                geom.matid = -1
                geom.transparent = 1
                geom.segid = -1

    for step in range(max(len(t) for t in trajectories)):
        set_qpos(renv, trajectories[0][min(step, len(trajectories[0]) - 1)])
        for data, trajectory in zip(scratch, trajectories[1:]):
            data.qpos[:] = trajectory[min(step, len(trajectory) - 1)]
            mujoco.mj_kinematics(model, data)
        yield render(renv, camera, width, height, edit=edit)


def ghost_video(
    directories: list[Path],
    colours: list[tuple[float, float, float]],
    out: Path,
    width: int,
    height: int,
    robot_only: bool = True,
) -> None:
    """Overlay the traces in ``directories`` (same scenario and instance)."""
    env, emap, meta = open_trace_env(directories[0])
    try:
        for directory in directories[1:]:
            other = json.loads((directory / "meta.json").read_text())
            if other["signature"] != meta["signature"]:
                raise SystemExit(f"{directory} is a different scene than the first")
        frames = ghost_frames(
            env,
            emap,
            [load_qpos(d) for d in directories],
            colours,
            width,
            height,
            meta["camera"],
            robot_only=robot_only,
        )
        write_video(frames, out, meta["fps"])
    finally:
        env.close()


def _colour(text: str) -> tuple[float, float, float]:
    value = text.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore


def main() -> None:
    """Render a contact-highlight video or a ghost overlay from recorded traces."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("mode", choices=["contact", "ghost", "diagram"])
    parser.add_argument("traces", type=Path, nargs="+", help="trace directories")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--colours", nargs="*", default=[], help="hex, one per trace")
    parser.add_argument("--all-bodies", action="store_true", help="ghost objects too")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "contact":
        contact_video(args.traces[0], args.out, args.width, args.height)
    elif args.mode == "diagram":
        env, emap, _ = open_trace_env(args.traces[0])
        try:
            write_diagram(env, emap, args.out, args.width, args.height)
        finally:
            env.close()
    else:
        palette = ["#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7"]
        colours = [
            _colour(args.colours[i] if i < len(args.colours) else palette[i % 6])
            for i in range(len(args.traces))
        ]
        ghost_video(
            args.traces, colours, args.out, args.width, args.height, not args.all_bodies
        )
    print(args.out)


if __name__ == "__main__":
    main()
