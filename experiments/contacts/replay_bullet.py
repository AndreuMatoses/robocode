"""Re-render recorded traces of the kinematic PyBullet scenarios.

A trace stores the state of the environment at every env step (``qpos``), and a
reset with the same seed and object count rebuilds the same scene. A *scene*
adapter (``recorder_bullet.KinderScene``, ``scenes_pddl``) restores a state and
renders it with the camera of ``env.render()``.

* episode video: one picture per recorded state;
* contact video: the links in contact are recoloured before each picture;
* ghost video: PyBullet has no translucent copies of a robot, so the episodes are
  rendered one after the other and the pictures are blended. A pixel of an episode
  counts where it differs from the picture of the start scene;
* labelled diagram: the picture plus PyBullet's segmentation mask.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pybullet as p

Colour = tuple[float, float, float]
ROBOT_TINT, OTHER_TINT = (1.0, 0.45, 0.05), (0.1, 0.75, 1.0)
DIFFERENT = 22  # [0-255] a pixel differs from the start picture by more than this


def open_scene(directory: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Rebuild the scene a trace was recorded in: (env, scene adapter, meta)."""
    # pylint: disable=import-outside-toplevel
    from experiments.contacts.catalog import make_env
    from experiments.contacts.recorder_bullet import scene_of
    from robocode.utils.episode import reset_for_episode

    meta = json.loads((directory / "meta.json").read_text())
    env = make_env(meta["environment"])
    reset_for_episode(env, meta["instance_seed"], meta["pinned_count"])
    return env, scene_of(env), meta


def load_states(directory: Path) -> np.ndarray:
    """The recorded states, one per env step (plus the initial one)."""
    with np.load(directory / "trace.npz") as data:
        return data["qpos"]


def frames(scene: Any, states: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
    """One picture per recorded state."""
    for state in states:
        scene.restore(state)
        yield scene.render()


def contact_frames(
    scene: Any, directory: Path, meta: dict[str, Any]
) -> Iterable[np.ndarray]:
    """The episode with every entity in a (non-resting) contact recoloured."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.trace import active_entities_per_step

    cid = scene.client
    active = active_entities_per_step(directory)
    kinds = [e["kind"] for e in meta["entities"]]
    links: dict[int, list[tuple[int, int]]] = {e["id"]: [] for e in meta["entities"]}
    owner = {g["id"]: g["entity"] for g in meta["geoms"]}
    for geom, (body, link) in enumerate(meta["bodies"]):
        links[owner[geom]].append((body, link))
    original: dict[tuple[int, int], list[float]] = {}
    for body in {b for b, _ in meta["bodies"]}:
        for shape in p.getVisualShapeData(body, physicsClientId=cid):
            original.setdefault((shape[0], shape[1]), list(shape[7]))
    for step, state in enumerate(load_states(directory)):
        scene.restore(state)
        touching: set[int] = set()
        if active:
            touching = active[max(step - 1, 0)] | active[min(step, len(active) - 1)]
        changed = []
        for entity in touching:
            tint = ROBOT_TINT if kinds[entity] == "robot" else OTHER_TINT
            for body, link in links[entity]:
                if (body, link) in original:
                    p.changeVisualShape(
                        body, link, rgbaColor=[*tint, 1.0], physicsClientId=cid
                    )
                    changed.append((body, link))
        yield scene.render()
        for body, link in changed:
            p.changeVisualShape(
                body, link, rgbaColor=original[body, link], physicsClientId=cid
            )


RENDER_BUDGET = 300  # pictures that one ghost video may render (0.1 to 0.5 s each)


def ghost_stride(trajectories: list[np.ndarray]) -> int:
    """Show every n-th env step, so that a ghost video of many runs stays fast.

    Every episode is one picture per shown step, and PyBullet's software renderer
    needs about 0.1 s for each, so 30 episodes of 100 steps would need 5 minutes.
    """
    steps = max(len(t) for t in trajectories)
    return max(1, math.ceil(len(trajectories) * steps / RENDER_BUDGET))


def ghost_frames(
    scene: Any,
    trajectories: list[np.ndarray],
    colours: list[Colour],
    alpha: float | None = None,
    equal: bool = False,
    stride: int = 1,
) -> Iterable[np.ndarray]:
    """Blend several episodes of one scene, one colour per episode.

    Without ``equal`` the first episode is the picture and the others are blended
    onto it; with ``equal`` all are blended onto the start picture. An episode that
    ended holds its last state. A moved object leaves its start place visible as a
    faint mark, because the blend works on pictures, not on bodies.
    """
    count = len(trajectories)
    if alpha is None:
        alpha = min(0.6, max(0.3, 2.0 / math.sqrt(max(count, 1))))
    scene.restore(trajectories[0][0])
    start = scene.render().astype(np.float32)
    last = max(len(t) for t in trajectories) - 1
    shown = list(range(0, last + 1, stride))
    if shown[-1] != last:
        shown.append(last)  # always the end of the longest episode
    for step in shown:
        pictures = []
        for states in trajectories:
            scene.restore(states[min(step, len(states) - 1)])
            pictures.append(scene.render().astype(np.float32))
        out = start.copy() if equal else pictures[0].copy()
        for k in range(count - 1, -1, -1):  # the first episode on top
            if k == 0 and not equal:
                continue
            mask = np.abs(pictures[k] - start).max(axis=-1) > DIFFERENT
            tint = 0.55 * pictures[k][mask] + 0.45 * 255.0 * np.array(
                colours[k], dtype=np.float32
            )
            out[mask] = (1.0 - alpha) * out[mask] + alpha * tint
        yield out.clip(0, 255).astype(np.uint8)


def _matrices(cid: int, camera: dict[str, Any]) -> tuple[Any, Any, int, int]:
    """View and projection matrices as ``pybullet_helpers.camera.capture_image``."""
    width, height = int(camera.get("image_width", 1674)), int(
        camera.get("image_height", 900)
    )
    if camera.get("specify_position"):
        # pylint: disable-next=import-outside-toplevel
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_quat(
            camera.get("camera_orientation", (0, 0, 0, 1))
        ).as_matrix()
        eye = np.array(camera.get("camera_position", (0, 0, 0)), dtype=float)
        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=(eye - rot[:, 2]).tolist(),
            cameraUpVector=rot[:, 1].tolist(),
            physicsClientId=cid,
        )
    else:
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camera.get("camera_target", (0, 0, 0.5)),
            distance=camera.get("camera_distance", 1.5),
            yaw=camera.get("camera_yaw", 0),
            pitch=camera.get("camera_pitch", -15),
            roll=0,
            upAxisIndex=2,
            physicsClientId=cid,
        )
    proj = p.computeProjectionMatrixFOV(
        fov=camera.get("fov", 60.0),
        aspect=float(width / height),
        nearVal=camera.get("near_val", 0.1),
        farVal=camera.get("far_val", 100.0),
        physicsClientId=cid,
    )
    return view, proj, width, height


def write_diagram(
    scene: Any, meta: dict[str, Any], state: np.ndarray, out: Path
) -> bool:
    """The labelled-diagram assets of one state (see ``replay.write_diagram``).

    Returns False for a scene that does not give its camera (no mask then).
    """
    camera = scene.camera()
    if camera is None:
        return False
    cid = scene.client
    scene.restore(state)
    view, proj, width, height = _matrices(cid, camera)
    _, _, rgba, _, seg = p.getCameraImage(
        width,
        height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid,
    )
    colour = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[..., :3]
    seg = np.asarray(seg, dtype=np.int64).reshape(height, width)
    owner = {g["id"]: g["entity"] for g in meta["geoms"]}
    entity_of = {(b, link): owner[g] for g, (b, link) in enumerate(meta["bodies"])}
    ent = np.full((height, width), 65535, dtype=np.int64)
    for value in np.unique(seg[seg >= 0]).tolist():
        body, link = value & ((1 << 24) - 1), (value >> 24) - 1
        entity = entity_of.get((body, link))
        if entity is not None:
            ent[seg == value] = entity
    ids = np.zeros((height, width, 3), dtype=np.uint8)
    ids[..., 0], ids[..., 1] = ent & 255, ent >> 8
    out.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out / "scene.png", colour)
    iio.imwrite(out / "scene_ids.png", ids)
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
    return True
