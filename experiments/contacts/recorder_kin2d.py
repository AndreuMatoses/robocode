"""Contacts of the kinematic 2D scenarios, where nothing ever collides.

A kinematic2d environment has no physics: an action moves the robot, a motion that
would put two shapes into collision is rejected (the state stays as it was), and an
object is carried while it is in the suction zone of the gripper. So there is no
contact to read from a simulator. This recorder builds the contacts of an env step
from the geometry, with the three sources that together are nearest to "two bodies
touch at this instant" (see the README, "Simulator backends"):

* **touch**: two shapes that can collide (their z-orders say so) are nearer than
  ``TOUCH`` to each other. A shape that collides with nothing (a target region, a
  button) is touched by a shape that overlaps it: the object is in the region, the
  stick is on the button.
* **hold**: the object is in the suction zone while the vacuum is on. The pair is
  ``robot.gripper.suction`` - object.
* **blocked**: the robot did not move although the action commands a motion. The
  rejected state is built again, and the pairs that overlap in it are the contacts
  of the step ("the robot is stopped by the wall").

There are no forces and no substeps: one tick is one env step, and every force is
zero. The trace has the same arrays as the other backends, plus ``verts``: the
outline of every geom in world coordinates at every env step, which ``render2d``
draws.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from experiments.contacts.entities import (
    BOX,
    MESH,
    SPHERE,
    Entity,
    EntityMap,
    hash_text,
)
from experiments.contacts.recorder import BaseRecorder
from experiments.contacts.recorder2d import object_centric_env

TOUCH = 0.01  # [world units] two solid shapes nearer than this touch (a step: 0.05)
STEP_SECONDS = 0.1  # one env step, as in the MuJoCo scenarios (10 Hz control)
CIRCLE_POINTS = 40
_ROBOT_PARTS = {
    "base": "robot.base",
    "arm": "robot.arm",
    "gripper": "robot.gripper.head",
    "suction": "robot.gripper.suction",
}
_DRAW_RANK = {"FLOOR": 1.0, "NONE": 1.5, "SURFACE": 2.0, "ALL": 3.0}  # low = below


def _outline(geom: Any) -> np.ndarray:
    """The outline ``[n, 2]`` of a ``tomsgeoms2d`` geom in world coordinates."""
    if hasattr(geom, "radius"):
        angles = np.linspace(0.0, 2 * np.pi, CIRCLE_POINTS, endpoint=False)
        return np.column_stack(
            (
                geom.x + geom.radius * np.cos(angles),
                geom.y + geom.radius * np.sin(angles),
            )
        )
    return np.array([[float(x), float(y)] for x, y in geom.vertices])


def _solid(geom: Any) -> Any:
    """A shapely shape of a geom (an exact disc for a circle)."""
    if hasattr(geom, "radius"):
        return Point(float(geom.x), float(geom.y)).buffer(float(geom.radius), 32)
    shape = Polygon([(float(x), float(y)) for x, y in geom.vertices])
    return shape if shape.is_valid else shape.buffer(0)


def _class_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "env":
        parts[1] = re.sub(r"_?\d+$", "", parts[1]) or parts[1]
    return ".".join(parts)


class Kin2DRecorder(BaseRecorder):
    """Builds the contacts of one episode of one kinematic2d environment."""

    backend = "kin2d"

    def __init__(self, env: Any, touch: float = TOUCH) -> None:
        # pylint: disable=import-outside-toplevel
        from kinder.envs.kinematic2d.object_types import CRVRobotType
        from kinder.envs.kinematic2d.utils import (
            get_suctioned_objects,
            snap_suctioned_objects,
        )
        from kinder.envs.utils import object_to_multibody2d, z_orders_may_collide

        self._robot_type = CRVRobotType
        self._suctioned = get_suctioned_objects
        self._snap = snap_suctioned_objects
        self._multibody = object_to_multibody2d
        self._may_collide = z_orders_may_collide
        self.oc = object_centric_env(env)
        self.touch = touch
        self.dt = STEP_SECONDS
        self._verts: list[np.ndarray] = []
        self._pending: dict[tuple[int, int], tuple[float, float]] = {}
        self._z_cache: dict[tuple[str, str], bool] = {}
        self._build()
        self._rank = self.emap.rank()
        self._robot_geom = [
            self.emap.entities[e].kind == "robot" for e in self.emap.geom_entity
        ]
        self._env_step = self.oc.step
        self.oc.step = self._wrapped_step  # an instance attribute: this env only
        super().__init__()

    # ------------------------------------------------------------------ scene

    def _bodies(self, state: Any) -> dict[tuple[str, str], Any]:
        """Every body of a full state, by (object name, body name)."""
        out = {}
        for obj in state:
            for body in self._multibody(obj, state, {}).bodies:
                out[obj.name, body.name] = body
        return out

    def _build(self) -> None:
        # pylint: disable-next=import-outside-toplevel
        from matplotlib.colors import to_rgb

        state = self.oc.full_state
        self._robot = next(o for o in state if o.is_instance(self._robot_type))
        entities: list[Entity] = []
        self._geom_of: dict[tuple[str, str], int] = {}
        self._solid_z: list[str] = []  # per geom: the z-order name
        self._static: list[bool] = []  # per geom: of a static object
        geom_entity: list[int] = []
        geom_index: list[int] = []
        geom_type: list[int] = []
        colours: dict[int, list[float]] = {}
        z_order: dict[int, float] = {}
        faint: list[int] = []

        def add(
            name: str, kind: str, obj: str, body: str, z: str, static: bool
        ) -> Entity:
            entity = next((e for e in entities if e.name == name), None)
            if entity is None:
                entity = Entity(len(entities), name, _class_name(name), kind)
                entities.append(entity)
            geom = len(geom_entity)
            self._geom_of[obj, body] = geom
            self._solid_z.append(z)
            self._static.append(static)
            geom_entity.append(entity.id)
            geom_index.append(len(entity.geoms))
            entity.geoms.append(geom)
            return entity

        # the robot: the suction zone has a geom also while the vacuum is off
        for body, name in _ROBOT_PARTS.items():
            z = {"base": "ALL", "suction": "NONE"}.get(body, "SURFACE")
            entity = add(name, "robot", self._robot.name, body, z, False)
            geom_type.append(SPHERE if body == "base" else BOX)
            colours[entity.id] = [0.45, 0.35, 0.75]
            z_order[entity.id] = 10.0 + len(entities)
            if body == "suction":
                faint.append(entity.id)
        features = state.type_features
        for obj in sorted((o for o in state if o != self._robot), key=lambda o: o.name):
            static = "static" in features[obj.type] and state.get(obj, "static") > 0.5
            if not static:
                name, kind = f"env.{obj.name}", "object"
            elif "wall" in obj.name:
                name, kind = f"world.{obj.name}", "world"
            else:
                name, kind = f"env.{obj.name}", "fixture"
            for body in self._multibody(obj, state, {}).bodies:
                z = str(body.z_order).rsplit(".", maxsplit=1)[-1]
                entity = add(name, kind, obj.name, body.name, z, static)
                round_ = hasattr(body.geom, "radius")
                corners = 0 if round_ else len(body.geom.vertices)
                geom_type.append(SPHERE if round_ else BOX if corners == 4 else MESH)
                face = body.rendering_kwargs.get("facecolor")
                if face is not None and entity.id not in colours:
                    colours[entity.id] = [round(float(c), 3) for c in to_rgb(face)]
                z_order[entity.id] = min(
                    z_order.get(entity.id, 9.0), _DRAW_RANK.get(z, 2.0)
                )
        count = len(geom_entity)
        text = "|".join(f"{e.name}:{len(e.geoms)}" for e in entities)
        self.emap = EntityMap(
            entities=entities,
            geom_entity=np.array(geom_entity, dtype=np.int32),
            geom_index=np.array(geom_index, dtype=np.int32),
            geom_type=np.array(geom_type, dtype=np.int32),
            geom_center=np.zeros((count, 3)),
            geom_extent=np.ones((count, 3)),
            signature=f"{len(entities)}/{count}/{hash_text(text)}",
        )
        self._meta = {
            "colours": {str(k): v for k, v in colours.items()},
            "z_order": {str(k): v for k, v in z_order.items()},
            "faint": faint,
        }

    def extra_meta(self) -> dict[str, Any]:
        config = self.oc.config
        return {
            "world": [
                float(config.world_min_x),
                float(config.world_max_x),
                float(config.world_min_y),
                float(config.world_max_y),
            ],
            "touch": self.touch,
            "step_seconds": self.dt,
            **self._meta,
        }

    # ----------------------------------------------------------- per env step

    def _touching(self, state: Any) -> dict[tuple[int, int], tuple[float, float]]:
        """The pairs in contact in ``state`` and one point of each."""
        bodies = self._bodies(state)
        geoms = [
            (self._geom_of[key], body)
            for key, body in bodies.items()
            if key in self._geom_of
        ]
        solids = {geom: _solid(body.geom) for geom, body in geoms}
        boxes = {geom: np.array(shape.bounds) for geom, shape in solids.items()}
        owner = self.emap.geom_entity
        found: dict[tuple[int, int], tuple[float, float]] = {}
        suction = self._geom_of[self._robot.name, "suction"]
        order = sorted(solids)
        for i, geom_a in enumerate(order):
            for geom_b in order[i + 1 :]:
                if owner[geom_a] == owner[geom_b] or suction in (geom_a, geom_b):
                    continue
                if self._robot_geom[geom_a] and self._robot_geom[geom_b]:
                    continue  # two parts of the robot: always
                if self._static[geom_a] and self._static[geom_b]:
                    continue  # a wall on a wall: always, and of no interest
                box_a, box_b = boxes[geom_a], boxes[geom_b]
                if (
                    box_a[0] > box_b[2] + self.touch
                    or box_b[0] > box_a[2] + self.touch
                    or box_a[1] > box_b[3] + self.touch
                    or box_b[1] > box_a[3] + self.touch
                ):
                    continue
                z_a, z_b = self._solid_z[geom_a], self._solid_z[geom_b]
                shape_a, shape_b = solids[geom_a], solids[geom_b]
                if "NONE" in (z_a, z_b):  # a region: something is in it
                    hit = shape_a.intersects(shape_b)
                elif self._z_collide(z_a, z_b):
                    hit = shape_a.distance(shape_b) <= self.touch
                else:
                    hit = False
                if hit:
                    middle = (
                        shape_a.centroid
                        if shape_a.area < shape_b.area
                        else shape_b.centroid
                    )
                    found[geom_a, geom_b] = (float(middle.x), float(middle.y))
        # the hold: an object in the suction zone while the vacuum is on
        current = self.oc._current_state  # pylint: disable=protected-access
        for obj, _ in self._suctioned(current, self._robot):
            for (name, _), body in bodies.items():
                if name == obj.name:
                    geom = self._geom_of[name, body.name]
                    centre = solids[geom].centroid
                    pair = (min(suction, geom), max(suction, geom))
                    found[pair] = (float(centre.x), float(centre.y))
        return found

    def _z_collide(self, z_a: str, z_b: str) -> bool:
        known = self._z_cache.get((z_a, z_b))
        if known is None:
            # pylint: disable-next=import-outside-toplevel
            from kinder.envs.utils import ZOrder

            known = bool(self._may_collide(ZOrder[z_a], ZOrder[z_b]))
            self._z_cache[z_a, z_b] = known
        return known

    def _blocked(
        self, before: Any, action: Any
    ) -> dict[tuple[int, int], tuple[float, float]]:
        """The pairs that made the env reject the motion of this step, if it did."""
        robot = self._robot
        now = self.oc._current_state  # pylint: disable=protected-access
        fields = ("x", "y", "theta", "arm_joint")
        if any(abs(now.get(robot, f) - before.get(robot, f)) > 1e-9 for f in fields):
            return {}
        dx, dy, dtheta, darm = (float(v) for v in np.asarray(action)[:4])
        arm = float(
            np.clip(
                before.get(robot, "arm_joint") + darm,
                before.get(robot, "base_radius"),
                before.get(robot, "arm_length"),
            )
        )
        if (
            max(
                abs(dx), abs(dy), abs(dtheta), abs(arm - before.get(robot, "arm_joint"))
            )
            < 1e-9
        ):
            return {}  # no motion was commanded
        wanted = before.copy()
        wanted.set(robot, "x", before.get(robot, "x") + dx)
        wanted.set(robot, "y", before.get(robot, "y") + dy)
        wanted.set(robot, "theta", before.get(robot, "theta") + dtheta)
        wanted.set(robot, "arm_joint", arm)
        held = self._suctioned(before, robot)
        self._snap(wanted, robot, held)
        full = wanted.copy()
        full.data.update(self.oc.initial_constant_state.data)
        bodies = self._bodies(full)
        moving = {robot.name} | {obj.name for obj, _ in held}
        found: dict[tuple[int, int], tuple[float, float]] = {}
        for (name_a, body_a), one in bodies.items():
            if name_a not in moving or (name_a, body_a) not in self._geom_of:
                continue
            geom_a = self._geom_of[name_a, body_a]
            shape_a = _solid(one.geom)
            for (name_b, body_b), other in bodies.items():
                if name_b in moving or (name_b, body_b) not in self._geom_of:
                    continue
                geom_b = self._geom_of[name_b, body_b]
                if not self._z_collide(self._solid_z[geom_a], self._solid_z[geom_b]):
                    continue
                shape_b = _solid(other.geom)
                if shape_a.intersects(shape_b):
                    middle = shape_a.intersection(shape_b).centroid
                    point = (
                        (float(middle.x), float(middle.y))
                        if not middle.is_empty
                        else (0.0, 0.0)
                    )
                    found[min(geom_a, geom_b), max(geom_a, geom_b)] = point
        return found

    def _wrapped_step(self, action: Any) -> Any:
        before = self.oc._current_state.copy()  # pylint: disable=protected-access
        result = self._env_step(action)
        self._pending = self._blocked(before, action)
        self.substep()
        self.end_step()
        return result

    def substep(self) -> None:
        """One env step is one tick: the contacts of the state after the step."""
        self.tick += 1
        found = self._touching(self.oc.full_state)
        for pair, point in self._pending.items():
            found.setdefault(pair, point)
        self._pending = {}
        if not found:
            return
        geoms = np.array(list(found), dtype=np.int64)
        swap = self._rank[geoms[:, 0]] > self._rank[geoms[:, 1]]
        geoms[swap] = geoms[swap][:, ::-1]
        points = np.array([[x, y, 0.0] for x, y in found.values()], dtype=np.float32)
        zeros = np.zeros(len(geoms))
        face = np.zeros(len(geoms), dtype=np.int8)
        self._buffer.append(
            (
                np.full(len(geoms), self.tick, dtype=np.int64),
                geoms[:, 0],
                face,
                geoms[:, 1],
                face,
                zeros,
                zeros,
                points,
                points,
            )
        )

    def _snapshot(self) -> None:
        state = self.oc.full_state
        bodies = self._bodies(state)
        verts = np.full(
            (len(self.emap.geom_entity), CIRCLE_POINTS, 2), np.nan, dtype=np.float32
        )
        for key, body in bodies.items():
            geom = self._geom_of.get(key)
            if geom is not None:
                outline = _outline(body.geom)[:CIRCLE_POINTS]
                verts[geom, : len(outline)] = outline
        self._verts.append(verts)
        current = self.oc._current_state  # pylint: disable=protected-access
        self._qpos.append(
            np.concatenate(
                [
                    np.asarray(current.data[o], dtype=np.float64)
                    for o in sorted(current.data, key=lambda o: o.name)
                ]
            )
        )
        self._time.append(self.tick * self.dt)
        tip = bodies.get((self._robot.name, "gripper"))
        if tip is not None:
            centre = _outline(tip.geom).mean(axis=0)
            self._ee.append(np.array([centre[0], centre[1], 0.0]))

    def finish(self) -> dict[str, np.ndarray]:
        """Give the env its own ``step`` back, then return the episode's arrays."""
        if self.oc.__dict__.get("step") is not None:
            del self.oc.step
        arrays = super().finish()
        arrays["verts"] = np.stack(self._verts)
        return arrays


@contextlib.contextmanager
def recording() -> Iterator[list[Kin2DRecorder | None]]:
    """A slot for the active recorder (it needs no hook in a simulator)."""
    yield [None]
