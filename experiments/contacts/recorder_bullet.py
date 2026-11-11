"""Contacts of the kinematic PyBullet scenarios, where nothing ever collides.

The kinematic3d environments (and the PR2 and rover environments) use PyBullet as a
collision checker only: joints are set, a motion into collision is rejected, and a
grasp attaches the object to the gripper. So there is no contact force to read.
This recorder builds the contacts of an env step from the geometry, with the three
sources that together are nearest to "two bodies touch at this instant" (see the
README, "Simulator backends"):

* **touch**: two link shapes are nearer than ``touch`` to each other (PyBullet's
  closest-point query). An object at rest is within some millimetres of its
  support, and a closed gripper is on the object that it holds.
* **hold**: an object is attached to the gripper. Used only when no link of the
  gripper touches the object in the geometry.
* **blocked**: the env rejects the motion of the step. The pairs that collide in
  the rejected configuration are read while the env is still in it.

There are no forces and no substeps: one tick is one env step, and every force is
zero. A *scene* object adapts one family of environments (``KinderScene`` here;
the PR2 and rover scenes are in ``scenes_pddl.py``).
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pybullet as p

from experiments.contacts.entities import (
    BOX,
    CAPSULE,
    CYLINDER,
    MESH,
    PLANE,
    SPHERE,
    Entity,
    EntityMap,
    hash_text,
    local_points,
)
from experiments.contacts.recorder import BaseRecorder
from experiments.contacts.recorder2d import object_centric_env

STEP_SECONDS = 0.1  # one env step, as in the MuJoCo scenarios (10 Hz control)
TOUCH = 0.005  # [m] the placement distance of the kinematic3d environments
_BULLET_TYPE = {
    p.GEOM_SPHERE: SPHERE,
    p.GEOM_BOX: BOX,
    p.GEOM_CYLINDER: CYLINDER,
    p.GEOM_MESH: MESH,
    p.GEOM_PLANE: PLANE,
    p.GEOM_CAPSULE: CAPSULE,
}
# Link names of the Kinova Gen3 with the Robotiq 2F-85, as the MuJoCo names.
_GEN3_LINKS = [
    (r"base_link", "arm.base"),
    (r"(shoulder|half_arm_1|half_arm_2|forearm)_link", r"arm.\1"),
    (r"(spherical_wrist_1|spherical_wrist_2|bracelet)_link", r"arm.\1"),
    (
        r"(end_effector_link|tool_frame|robotiq_arg2f_base_link|inertia_link|finger_tip)",
        "gripper.palm",
    ),
    (r"(right|left)_inner_finger_pad", r"gripper.\1_pad"),
    (
        r"(right|left)_(outer_knuckle|outer_finger|inner_finger|inner_knuckle)",
        r"gripper.\1_finger",
    ),
]


def gen3_part(link: str) -> str:
    """The robot part of a link of the Gen3 arm (``arm.forearm``, ``gripper.left_pad``)."""
    for pattern, repl in _GEN3_LINKS:
        if re.fullmatch(pattern, link):
            return re.sub(pattern, repl, link)
    return f"arm.{link}"


def _class_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "env":
        parts[1] = re.sub(r"_?\d+$", "", parts[1]) or parts[1]
    return ".".join(parts)


class KinderScene:
    """A kinder kinematic3d environment as the recorder sees it."""

    def __init__(self, env: Any) -> None:
        self.oc = object_centric_env(env)
        self.client: int = self.oc.physics_client_id
        self.step_owner = self.oc
        self.touch = max(
            TOUCH, float(getattr(self.oc.config, "min_placement_dist", TOUCH))
        )
        arm = self.oc.robot.arm.robot_id
        self.robots: dict[int, Callable[[str], str]] = {arm: gen3_part}
        for index in range(p.getNumBodies(physicsClientId=self.client)):
            body = p.getBodyUniqueId(index, physicsClientId=self.client)
            name = p.getBodyInfo(body, physicsClientId=self.client)[1].decode()
            if "base" in name and body != arm:
                self.robots[body] = lambda _link: "base"
        movable = {
            self.oc._object_name_to_pybullet_id(
                n
            ): n  # pylint: disable=protected-access
            for n in self.oc._get_movable_object_names()  # pylint: disable=protected-access
        }
        names: dict[int, str] = dict(movable)
        for obj in self.oc._get_obs():  # pylint: disable=protected-access
            try:
                body = self.oc._object_name_to_pybullet_id(
                    obj.name
                )  # pylint: disable=protected-access
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if isinstance(body, int):
                names.setdefault(body, obj.name)
        for attr, value in vars(self.oc).items():
            if (
                isinstance(value, int)
                and attr.endswith("_id")
                and attr != "physics_client_id"
            ):
                names.setdefault(value, attr.strip("_").removesuffix("_id"))
        collide = (
            self.oc._get_collision_object_ids()
        )  # pylint: disable=protected-access
        self.named: dict[int, tuple[str, str]] = {}
        for body in sorted(collide | set(movable)):
            name = names.get(body, f"body{body}")
            kind = "object" if body in movable else "fixture"
            self.named[body] = (f"env.{name}", kind)
        self.grasp_links = ("left_inner_finger_pad", "right_inner_finger_pad")
        self._arm = arm

    def held(self) -> list[int]:
        """The bodies that the gripper holds now."""
        body = self.oc._grasped_object_id  # pylint: disable=protected-access
        return [] if body is None else [body]

    def on_blocked(self, report: Callable[[], None]) -> Callable[[], None]:
        """Call ``report`` while the env is in a configuration that it rejects."""
        inner = (
            self.oc._robot_or_held_object_collision_exists
        )  # pylint: disable=protected-access

        def wrapped() -> bool:
            hit = inner()
            if hit:
                report()
            return bool(hit)

        self.oc._robot_or_held_object_collision_exists = (
            wrapped  # pylint: disable=protected-access
        )

        def undo() -> None:
            self.oc.__dict__.pop("_robot_or_held_object_collision_exists", None)

        return undo

    def state(self) -> np.ndarray:
        """The state of the env as one vector (see ``restore``)."""
        obs = self.oc._get_obs()  # pylint: disable=protected-access
        return np.concatenate(
            [
                np.asarray(obs.data[o], dtype=np.float64)
                for o in sorted(obs.data, key=lambda o: o.name)
            ]
        )

    def restore(self, vector: np.ndarray) -> None:
        """Put the env in a recorded state."""
        obs = self.oc._get_obs()  # pylint: disable=protected-access
        start = 0
        for obj in sorted(obs.data, key=lambda o: o.name):
            size = len(obs.data[obj])
            obs.data[obj] = np.asarray(
                vector[start : start + size], dtype=obs.data[obj].dtype
            )
            start += size
        self.oc._set_state(obs)  # pylint: disable=protected-access

    def tool_position(self) -> np.ndarray:
        """Where the gripper is."""
        return np.array(self.oc.robot.arm.get_end_effector_pose().position)

    def render(self) -> np.ndarray:
        """The picture of ``env.render()``."""
        return np.asarray(self.oc.render())[..., :3]

    def camera(self) -> dict[str, Any]:
        """The arguments of ``pybullet_helpers.camera.capture_image`` of this env."""
        config = self.oc.config
        return {
            "image_width": int(config.render_image_width),
            "image_height": int(config.render_image_height),
            **config.get_camera_kwargs(),
        }


def scene_of(env: Any) -> Any:
    """The scene adapter of a kinematic PyBullet environment."""
    inner = getattr(env, "_current_backend", None) or getattr(env, "_kinder_env", None)
    if inner is not None and hasattr(inner, "_object_centric_env"):
        return KinderScene(env)
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts import scenes_pddl

    return scenes_pddl.scene_of(env)


class BulletRecorder(BaseRecorder):
    """Builds the contacts of one episode of one kinematic PyBullet environment."""

    backend = "bullet"

    def __init__(self, scene: Any) -> None:
        self.scene = scene
        self.cid: int = scene.client
        self.touch: float = scene.touch
        self.dt = STEP_SECONDS
        self._pending: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._build()
        self._rank = self.emap.rank()
        self._undo_blocked = scene.on_blocked(self._note_blocked)
        self._env_step = scene.step_owner.step
        scene.step_owner.step = self._wrapped_step  # an instance attribute
        super().__init__()

    # ------------------------------------------------------------------ scene

    def _build(self) -> None:
        cid = self.cid
        entities: list[Entity] = []
        by_name: dict[str, Entity] = {}
        self._geom_of: dict[tuple[int, int], int] = {}
        self._body_link: list[tuple[int, int]] = []
        geom_entity: list[int] = []
        geom_index: list[int] = []
        geom_type: list[int] = []
        extent: list[list[float]] = []
        self._robot_bodies = set(self.scene.robots)
        self._grasp_geoms: list[int] = []
        class_of = getattr(self.scene, "cls", _class_name)

        def add(name: str, kind: str, body: int, link: int, shape: Any) -> None:
            entity = by_name.get(name)
            if entity is None:
                entity = Entity(len(entities), name, class_of(name), kind)
                entities.append(entity)
                by_name[name] = entity
            geom = len(geom_entity)
            self._geom_of[body, link] = geom
            self._body_link.append((body, link))
            geom_entity.append(entity.id)
            geom_index.append(len(entity.geoms))
            entity.geoms.append(geom)
            kind_code = _BULLET_TYPE.get(shape[2], MESH)
            size = list(shape[3])
            if kind_code == BOX:
                half = [abs(v) / 2 for v in size]
            elif kind_code == SPHERE:
                half = [size[0]] * 3
            elif kind_code in (CYLINDER, CAPSULE):
                half = [size[1], size[1], size[0] / 2]
            else:  # a mesh: the box around the link now
                low, high = p.getAABB(body, link, physicsClientId=cid)
                half = [(b - a) / 2 for a, b in zip(low, high)]
            geom_type.append(kind_code)
            extent.append([max(float(v), 1e-6) for v in half])

        bodies = list(self.scene.robots) + [
            b for b in self.scene.named if b not in self.scene.robots
        ]
        for body in bodies:
            links = {-1: p.getBodyInfo(body, physicsClientId=cid)[0].decode()}
            for joint in range(p.getNumJoints(body, physicsClientId=cid)):
                links[joint] = p.getJointInfo(body, joint, physicsClientId=cid)[
                    12
                ].decode()
            for link, link_name in links.items():
                shapes = p.getCollisionShapeData(body, link, physicsClientId=cid)
                if not shapes:
                    continue
                if body in self.scene.robots:
                    name, kind = "robot." + self.scene.robots[body](link_name), "robot"
                else:
                    name, kind = self.scene.named[body]
                add(name, kind, body, link, shapes[0])
                if body in self.scene.robots and link_name in self.scene.grasp_links:
                    self._grasp_geoms.append(self._geom_of[body, link])
        count = len(geom_entity)
        text = "|".join(f"{e.name}:{len(e.geoms)}" for e in entities)
        self.emap = EntityMap(
            entities=entities,
            geom_entity=np.array(geom_entity, dtype=np.int32),
            geom_index=np.array(geom_index, dtype=np.int32),
            geom_type=np.array(geom_type, dtype=np.int32),
            geom_center=np.zeros((count, 3)),
            geom_extent=np.array(extent, dtype=np.float64).reshape(count, 3),
            signature=f"{len(entities)}/{count}/{hash_text(text)}",
        )
        self._movers = [
            b for b, (_, kind) in self.scene.named.items() if kind == "object"
        ]

    def extra_meta(self) -> dict[str, Any]:
        return {
            "touch": self.touch,
            "step_seconds": self.dt,
            "bodies": [[int(b), int(link)] for b, link in self._body_link],
        }

    # ----------------------------------------------------------- per env step

    def _near(
        self, first: list[int], second: list[int], distance: float
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """The geom pairs of two groups of bodies that are within ``distance``."""
        found: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        best: dict[tuple[int, int], float] = {}
        done: set[tuple[int, int]] = set()
        for body_a in first:
            for body_b in second:
                if body_a == body_b or (body_b, body_a) in done:
                    continue
                if body_a in self._robot_bodies and body_b in self._robot_bodies:
                    continue
                done.add((body_a, body_b))
                points = p.getClosestPoints(
                    body_a, body_b, distance=distance, physicsClientId=self.cid
                )
                for point in points:
                    geom_a = self._geom_of.get((point[1], point[3]))
                    geom_b = self._geom_of.get((point[2], point[4]))
                    if geom_a is None or geom_b is None:
                        continue
                    pair = (geom_a, geom_b)
                    if pair not in best or point[8] < best[pair]:
                        best[pair] = point[8]
                        found[pair] = (np.array(point[5]), np.array(point[7]))
        return found

    def _note_blocked(self) -> None:
        """Called while the env is in a configuration that it will reject."""
        moving = list(self._robot_bodies) + self.scene.held()
        others = [b for b in self.scene.named if b not in moving]
        self._pending.update(self._near(moving, others, 1e-3))

    def _wrapped_step(self, action: Any) -> Any:
        self._pending = {}
        result = self._env_step(action)
        self.substep()
        self.end_step()
        return result

    def substep(self) -> None:
        """One env step is one tick: the contacts of the state after the step."""
        self.tick += 1
        everything = list(self.scene.named)
        found = self._near(
            list(self._robot_bodies) + self._movers, everything, self.touch
        )
        for pair, value in self._pending.items():
            found.setdefault(pair, value)
        self._pending = {}
        # the task events of a scene that has them (the rovers)
        for body_a, body_b in getattr(self.scene, "events", list)():
            geoms_a = [g for (b, _), g in self._geom_of.items() if b == body_a]
            geoms_b = [g for (b, _), g in self._geom_of.items() if b == body_b]
            if geoms_a and geoms_b:
                where = np.array(
                    p.getBasePositionAndOrientation(body_b, physicsClientId=self.cid)[0]
                )
                found.setdefault(
                    (min(geoms_a), min(geoms_b)), (where, np.array([0.0, 0.0, 1.0]))
                )
        # the hold: an attached object that no link of the robot touches
        for body in self.scene.held():
            geoms = {g for (b, _), g in self._geom_of.items() if b == body}
            robot_touch = any(
                (
                    a in geoms
                    and self.emap.entities[self.emap.geom_entity[b]].kind == "robot"
                )
                or (
                    b in geoms
                    and self.emap.entities[self.emap.geom_entity[a]].kind == "robot"
                )
                for a, b in found
            )
            if not robot_touch and geoms:
                where = np.array(
                    p.getBasePositionAndOrientation(body, physicsClientId=self.cid)[0]
                )
                for grasp in self._grasp_geoms:
                    found[grasp, min(geoms)] = (where, np.array([0.0, 0.0, 1.0]))
        if not found:
            return
        geoms_ = np.array(list(found), dtype=np.int64)
        points = np.array([v[0] for v in found.values()])
        normals = np.array([v[1] for v in found.values()])
        swap = self._rank[geoms_[:, 0]] > self._rank[geoms_[:, 1]]
        geoms_[swap] = geoms_[swap][:, ::-1]
        xpos, xmat = self._frames(np.unique(geoms_))
        sides = [
            local_points(self.emap, geoms_[:, k], points, normals, xpos, xmat)
            for k in (0, 1)
        ]
        zeros = np.zeros(len(geoms_))
        self._buffer.append(
            (
                np.full(len(geoms_), self.tick, dtype=np.int64),
                geoms_[:, 0],
                sides[0][1],
                geoms_[:, 1],
                sides[1][1],
                zeros,
                zeros,
                sides[0][0],
                sides[1][0],
            )
        )

    def _frames(self, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``geom_xpos`` / ``geom_xmat`` rows (as MuJoCo has them) of ``geoms``."""
        count = len(self._body_link)
        xpos, xmat = np.zeros((count, 3)), np.zeros((count, 9))
        for geom in geoms.tolist():
            body, link = self._body_link[geom]
            if link < 0:
                pos, orn = p.getBasePositionAndOrientation(
                    body, physicsClientId=self.cid
                )
            else:
                state = p.getLinkState(body, link, physicsClientId=self.cid)
                pos, orn = state[4], state[5]
            xpos[geom] = pos
            xmat[geom] = p.getMatrixFromQuaternion(orn)
        return xpos, xmat

    def _snapshot(self) -> None:
        self._qpos.append(np.asarray(self.scene.state(), dtype=np.float64))
        self._time.append(self.tick * self.dt)
        self._ee.append(np.asarray(self.scene.tool_position(), dtype=np.float64))

    def finish(self) -> dict[str, np.ndarray]:
        """Give the env its own methods back, then return the episode's arrays."""
        self.scene.step_owner.__dict__.pop("step", None)
        self._undo_blocked()
        return super().finish()


@contextlib.contextmanager
def recording() -> Iterator[list[BulletRecorder | None]]:
    """A slot for the active recorder (it needs no hook in a simulator)."""
    yield [None]
