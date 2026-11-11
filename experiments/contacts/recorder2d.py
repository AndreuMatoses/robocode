"""Log pymunk contacts at every simulation substep of a dynamic2d rollout.

An ``env.step`` of the dynamic2d environments runs 100 ``Space.step`` substeps
(10 ms each). ``recording()`` wraps ``pymunk.Space.step`` at class level, so a
``PymunkRecorder`` sees every substep without changing the simulation; the recorder
wraps the ``step`` of its own environment to know where an env step ends.

What pymunk gives and what it does not:

* Every pair of shapes that collide has an *arbiter* with the contact points, the
  normal and the impulse of the substep. The recorder reads the arbiters of the
  dynamic bodies after every substep: object on object, object on a wall or a
  table, robot part on object.
* The robot parts, the walls and the tables are kinematic bodies. When a robot part
  runs into a wall or a table, pymunk calls a handler of the env, which moves the
  robot back to its last pose; no impulse is applied and the pair is in no arbiter
  list. ``recording()`` wraps those handlers, so the recorder gets these pairs too,
  with a force of zero ("the robot is stopped by the wall").
* A grasp replaces the dynamic body of the object by a kinematic body that moves
  with the gripper. From then on there is no arbiter between the fingers and the
  object, so the recorder adds the pairs finger - held object itself, with a force
  of zero (the force of a grasp is not simulated).

The trace has the same arrays as a MuJoCo trace (see ``recorder.py``). A "geom" is
one pymunk shape; ``qpos`` holds the pose ``x, y, theta`` of every entity at every
env step, which is all that ``render2d`` needs to draw the episode again.
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pymunk

from experiments.contacts.entities import (
    BOX,
    MESH,
    SPHERE,
    Entity,
    EntityMap,
    hash_text,
    local_points,
)
from experiments.contacts.recorder import BaseRecorder

ROBOT_COLOUR = (0.45, 0.35, 0.75)
BUSY_BODIES, BUSY_STRIDE = 12, 5  # from this many movable bodies: every 5th substep
# The collision of the robot with a wall or a table: the env moves the robot back
# in one of these handlers (looked up in the module of each env at a reset).
BLOCK_MODULES = (
    "base_env",
    "dyn_obstruction2d",
    "dyn_pushpullhook2d",
    "dyn_pusht2d",
    "dyn_scooppour2d",
)
BLOCK_HANDLERS = ("on_collision_w_static", "on_dot_robot_collision_w_static")
_ROBOT_PARTS = {  # KinRobot attribute -> entity name, per shape attribute
    "_base_shape": "robot.base",
    "_arm_shape": "robot.arm",
    "_gripper_base_shape": "robot.gripper.palm",
    "_left_finger_shape": "robot.gripper.left_finger",
    "_right_finger_shape": "robot.gripper.right_finger",
}


def object_centric_env(env: Any) -> Any:
    """The kinder object-centric environment behind a robocode environment."""
    backend = getattr(env, "_current_backend", None) or getattr(
        env, "_kinder_env", None
    )
    return backend._object_centric_env  # pylint: disable=protected-access


def _local_shape(shape: pymunk.Shape) -> dict[str, Any]:
    """A shape in the frame of its body, for ``meta.json``."""
    if isinstance(shape, pymunk.Circle):
        return {
            "kind": "circle",
            "radius": round(float(shape.radius), 5),
            "offset": [round(float(v), 5) for v in shape.offset],
        }
    if isinstance(shape, pymunk.Poly):
        points = [
            [round(float(v.x), 5), round(float(v.y), 5)] for v in shape.get_vertices()
        ]
        return {"kind": "poly", "points": points}
    if isinstance(shape, pymunk.Segment):
        a, b, r = shape.a, shape.b, max(float(shape.radius), 1e-3)
        normal = (b - a).perpendicular_normal() * r
        corners = (a + normal, b + normal, b - normal, a - normal)
        return {
            "kind": "poly",
            "points": [[round(float(v.x), 5), round(float(v.y), 5)] for v in corners],
        }
    raise TypeError(f"unknown pymunk shape {type(shape).__name__}")


def _box_of(local: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Centre and half-extents of a local shape (z: 0 and 1)."""
    if local["kind"] == "circle":
        r = local["radius"]
        return [*local["offset"], 0.0], [r, r, 1.0]
    points = np.array(local["points"])
    low, high = points.min(axis=0), points.max(axis=0)
    centre, half = (low + high) / 2, np.maximum((high - low) / 2, 1e-6)
    return [float(centre[0]), float(centre[1]), 0.0], [
        float(half[0]),
        float(half[1]),
        1.0,
    ]


def _shape_order(shape: pymunk.Shape) -> tuple[float, float]:
    """A key that is the same for a shape and for a copy of it on another body."""
    centre = _box_of(_local_shape(shape))[0]
    return round(centre[0], 4), round(centre[1], 4)


class PymunkRecorder(BaseRecorder):
    """Accumulates the contacts of one episode of one dynamic2d environment."""

    backend = "pymunk"

    def __init__(self, env: Any, stride: int | None = None) -> None:
        self.oc = object_centric_env(env)
        self.space: pymunk.Space = self.oc.pymunk_space
        # To read every arbiter in Python costs much more than the simulation: a
        # scene with many small objects (scoop and pour: 30) is 35 times slower.
        # Such a scene is read at every ``stride``-th substep (50 ms, not 10 ms).
        dynamic = sum(b.body_type == pymunk.Body.DYNAMIC for b in self.space.bodies)
        self.stride = stride or (BUSY_STRIDE if dynamic >= BUSY_BODIES else 1)
        self._substeps = 0
        self._blocked: dict[tuple[int, int], tuple[float, ...]] = {}
        self.dt = self.stride / float(self.oc.config.sim_hz)
        self._robot = getattr(self.oc, "robot", None)  # KinRobot, or None
        self._geom_of: dict[pymunk.Shape, int] = {}
        self._entity_of_body: dict[pymunk.Body, int] = {}
        self._object_of_entity: dict[int, Any] = {}  # entity id -> state object
        self._body_of_entity: dict[int, pymunk.Body] = {}  # robot parts and the rest
        self._shapes: list[dict[str, Any]] = []  # per geom: the local shape
        self._order: list[list[tuple[float, float]]] = []  # per entity: shape keys
        self._colours: dict[int, tuple[float, float, float]] = {}
        self._z: dict[int, float] = {}
        self._build()
        self._rank = self.emap.rank()
        self._fingers = [
            e.geoms[0] for e in self.emap.entities if e.name.endswith("_finger")
        ]
        self._ee_entity = next(
            (e.id for e in self.emap.entities if e.name == "robot.gripper.palm"),
            next((e.id for e in self.emap.entities if e.kind == "robot"), None),
        )
        self._env_step = self.oc.step
        self.oc.step = self._wrapped_step  # an instance attribute: this env only
        super().__init__()

    # ------------------------------------------------------------------ scene

    def _add_entity(
        self, name: str, kind: str, shapes: list[pymunk.Shape], body: pymunk.Body
    ) -> Entity:
        entity = Entity(len(self._entities), name, _class_name(name), kind)
        self._entities.append(entity)
        self._order.append([])
        for shape in sorted(shapes, key=_shape_order):
            geom = len(self._shapes)
            self._shapes.append(_local_shape(shape))
            self._order[entity.id].append(_shape_order(shape))
            self._geom_of[shape] = geom
            self._geom_entity.append(entity.id)
            self._geom_index.append(len(entity.geoms))
            entity.geoms.append(geom)
        self._body_of_entity[entity.id] = body
        return entity

    def _build(self) -> None:
        self._entities: list[Entity] = []
        self._geom_entity: list[int] = []
        self._geom_index: list[int] = []
        state = self.oc.full_state
        bodies = dict(
            self.oc._state_obj_to_pymunk_body
        )  # pylint: disable=protected-access
        # Robot parts: the bodies of the space that belong to no state object.
        named: dict[pymunk.Shape, str] = {}
        for holder in (self._robot, getattr(self.oc, "dot_robot", None)):
            for attr, name in _ROBOT_PARTS.items():
                shape = getattr(holder, attr, None)
                if shape is not None:
                    named[shape] = name
            if getattr(holder, "_shape", None) is not None:  # DotRobot
                named[holder._shape] = "robot.dot"  # pylint: disable=protected-access
        taken = set(bodies.values())
        loose = [b for b in self.space.bodies if b not in taken]
        parts: dict[str, tuple[list[pymunk.Shape], pymunk.Body]] = {}
        for i, body in enumerate(sorted(loose, key=lambda b: b.id)):
            for shape in body.shapes:
                name = named.get(shape, f"robot.part{i}")
                parts.setdefault(name, ([], body))[0].append(shape)
        for name in sorted(parts):
            entity = self._add_entity(name, "robot", *parts[name])
            self._colours[entity.id] = ROBOT_COLOUR
            self._z[entity.id] = 100.0
        for body in loose:
            self._entity_of_body.setdefault(
                body,
                next(
                    e.id for e in self._entities if self._body_of_entity[e.id] is body
                ),
            )
        # State objects with a body: movable objects, fixtures and walls.
        for obj in sorted(bodies, key=lambda o: o.name):
            body = bodies[obj]
            if body.body_type == pymunk.Body.DYNAMIC:
                name, kind = f"env.{obj.name}", "object"
            elif "wall" in obj.name:
                name, kind = f"world.{obj.name}", "world"
            else:
                name, kind = f"env.{obj.name}", "fixture"
            entity = self._add_entity(name, kind, list(body.shapes), body)
            self._entity_of_body[body] = entity.id
            self._object_of_entity[entity.id] = obj
            features = state.type_features[obj.type]
            if "color_r" in features:
                self._colours[entity.id] = tuple(
                    float(state.get(obj, f"color_{c}")) for c in "rgb"
                )  # type: ignore[assignment]
            self._z[entity.id] = (
                float(state.get(obj, "z_order")) if "z_order" in features else 1.0
            )
        types, centres, extents = [], [], []
        for local in self._shapes:
            centre, half = _box_of(local)
            centres.append(centre)
            extents.append(half)
            if local["kind"] == "circle":
                types.append(SPHERE)
            else:
                types.append(BOX if len(local["points"]) == 4 else MESH)
        text = "|".join(
            f"{e.name}:{[self._shapes[g]['kind'] for g in e.geoms]}"
            for e in self._entities
        )
        self.emap = EntityMap(
            entities=self._entities,
            geom_entity=np.array(self._geom_entity, dtype=np.int32),
            geom_index=np.array(self._geom_index, dtype=np.int32),
            geom_type=np.array(types, dtype=np.int32),
            geom_center=np.array(centres, dtype=np.float64),
            geom_extent=np.array(extents, dtype=np.float64),
            signature=f"{len(self._entities)}/{len(self._shapes)}/{hash_text(text)}",
        )
        self._decor = _decor(self.oc, state, set(bodies))

    def extra_meta(self) -> dict[str, Any]:
        config = self.oc.config
        return {
            "world": [
                float(config.world_min_x),
                float(config.world_max_x),
                float(config.world_min_y),
                float(config.world_max_y),
            ],
            "shapes": self._shapes,
            "colours": {
                str(k): [round(c, 3) for c in v] for k, v in self._colours.items()
            },
            "z_order": {str(k): v for k, v in self._z.items()},
            "decor": self._decor,
            # One env step is 100 substeps = 1 s here (0.1 s in the MuJoCo scenarios).
            "step_seconds": self._step_seconds(),
            "substep_stride": self.stride,
        }

    def _step_seconds(self) -> float:
        steps = max(len(self._qpos) - 1, 1)
        return self.dt * round(self.tick / steps) if self.tick else 1.0

    # ------------------------------------------------------------ per substep

    def _geom(self, shape: pymunk.Shape) -> int | None:
        """The geom of a shape; a shape of a new body is matched to its entity."""
        geom = self._geom_of.get(shape)
        if geom is not None:
            return geom
        entity = self._entity_of_body.get(shape.body)
        if entity is None:
            entity = self._adopt(shape.body)
            if entity is None:
                return None
        keys = self._order[entity]
        key = _shape_order(shape)
        index = min(range(len(keys)), key=lambda i: math.dist(keys[i], key))
        geom = self._entities[entity].geoms[index]
        self._geom_of[shape] = geom
        return geom

    def _adopt(self, body: pymunk.Body) -> int | None:
        """The entity of a body that the env made during a grasp or a release.

        The env swaps the body of an object (dynamic <-> kinematic). After a
        release its own table already names the new body. After a grasp the table
        is updated only at the end of the env step; until then the new body belongs
        to the one object whose body is no longer in the space.
        """
        table = self.oc._state_obj_to_pymunk_body  # pylint: disable=protected-access
        chosen = None
        for entity, obj in self._object_of_entity.items():
            if table.get(obj) is body:
                chosen = entity
                break
        if chosen is None:
            for entity in self._object_of_entity:
                if self._body_of_entity[entity].space is not self.space:
                    chosen = entity
                    break
        if chosen is not None:
            self._entity_of_body[body] = chosen
            self._body_of_entity[chosen] = body
        return chosen

    def blocked(self, arbiter: pymunk.Arbiter) -> None:
        """A robot part runs into a wall or a table: the env will move it back.

        Called from the env's collision handler, during ``Space.step``. The pair
        joins the contacts of the next substep that is read, with a force of zero
        (the env cancels the collision: no impulse is applied).
        """
        shape_a, shape_b = arbiter.shapes
        geom_a, geom_b = self._geom(shape_a), self._geom(shape_b)
        if geom_a is None or geom_b is None:
            return
        pair = (geom_a, geom_b) if geom_a < geom_b else (geom_b, geom_a)
        points = arbiter.contact_point_set
        if pair in self._blocked or not points.points:
            return
        x = sum(p.point_a.x for p in points.points) / len(points.points)
        y = sum(p.point_a.y for p in points.points) / len(points.points)
        self._blocked[pair] = (0.0, 0.0, x, y, points.normal.x, points.normal.y)

    def raw_substep(self) -> None:
        """Called after every ``Space.step``; reads every ``stride``-th one."""
        self._substeps += 1
        if self._substeps % self.stride == 0:
            self.substep()

    def substep(self) -> None:
        """Read the arbiters of the substep that just finished."""
        self.tick += 1
        found: dict[tuple[int, int], tuple[float, ...]] = {}

        def visit(arbiter: pymunk.Arbiter) -> None:
            shape_a, shape_b = arbiter.shapes
            geom_a, geom_b = self._geom(shape_a), self._geom(shape_b)
            if geom_a is None or geom_b is None:
                return
            pair = (geom_a, geom_b) if geom_a < geom_b else (geom_b, geom_a)
            if pair in found:
                return
            points = arbiter.contact_point_set
            if not points.points:
                return
            normal, impulse = points.normal, arbiter.total_impulse
            x = sum(p.point_a.x for p in points.points) / len(points.points)
            y = sum(p.point_a.y for p in points.points) / len(points.points)
            found[pair] = (
                abs(impulse.dot(normal)) / self.dt,
                abs(impulse.cross(normal)) / self.dt,
                x,
                y,
                normal.x,
                normal.y,
            )

        for body in self.space.bodies:
            if body.body_type == pymunk.Body.DYNAMIC:
                body.each_arbiter(visit)
        # A held object is a kinematic body: add the finger - object pairs.
        for held, _, _ in getattr(self._robot, "held_objects", []):
            for shape in held[1]:
                geom = self._geom(shape)
                if geom is None:
                    continue
                x, y = held[0].position
                for finger in self._fingers:
                    found.setdefault((finger, geom), (0.0, 0.0, x, y, 1.0, 0.0))
        for pair, value in self._blocked.items():
            found.setdefault(pair, value)
        self._blocked.clear()
        if not found:
            return
        geoms = np.array(list(found), dtype=np.int64)
        values = np.array(list(found.values()))
        swap = self._rank[geoms[:, 0]] > self._rank[geoms[:, 1]]
        geoms[swap] = geoms[swap][:, ::-1]
        points = np.column_stack((values[:, 2:4], np.zeros(len(values))))
        normals = np.column_stack((values[:, 4:6], np.zeros(len(values))))
        xpos, xmat = self._geom_frames(np.unique(geoms))
        sides = [
            local_points(self.emap, geoms[:, k], points, normals, xpos, xmat)
            for k in (0, 1)
        ]
        self._buffer.append(
            (
                np.full(len(geoms), self.tick, dtype=np.int64),
                geoms[:, 0],
                sides[0][1],
                geoms[:, 1],
                sides[1][1],
                values[:, 0],
                values[:, 1],
                sides[0][0],
                sides[1][0],
            )
        )

    def _pose(self, entity: int) -> tuple[float, float, float]:
        obj = self._object_of_entity.get(entity)
        body = self._body_of_entity[entity]
        if obj is not None and body.space is not self.space:
            table = (
                self.oc._state_obj_to_pymunk_body
            )  # pylint: disable=protected-access
            current = table.get(obj)
            if current is not None and current.space is self.space:
                body = current
                self._body_of_entity[entity] = body
                self._entity_of_body[body] = entity
        return float(body.position.x), float(body.position.y), float(body.angle)

    def _geom_frames(self, geoms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``geom_xpos`` / ``geom_xmat`` rows (as MuJoCo has them) of ``geoms``."""
        xpos = np.zeros((len(self._shapes), 3))
        xmat = np.zeros((len(self._shapes), 9))
        for geom in geoms.tolist():
            x, y, theta = self._pose(self._geom_entity[geom])
            cos, sin = math.cos(theta), math.sin(theta)
            xpos[geom] = (x, y, 0.0)
            xmat[geom] = (cos, -sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0)
        return xpos, xmat

    # ----------------------------------------------------------- per env step

    def _wrapped_step(self, action: Any) -> Any:
        result = self._env_step(action)
        self.end_step()
        return result

    def _snapshot(self) -> None:
        poses = [self._pose(e.id) for e in self._entities]
        self._qpos.append(np.array(poses, dtype=np.float64).ravel())
        self._time.append(self.tick * self.dt)
        if self._ee_entity is not None:
            x, y, _ = poses[self._ee_entity]
            self._ee.append(np.array([x, y, 0.0]))

    def finish(self) -> dict[str, np.ndarray]:
        """Give the env its own ``step`` back, then return the episode's arrays."""
        if self.oc.__dict__.get("step") is not None:
            del self.oc.step
        return super().finish()


def _class_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "env":
        parts[1] = re.sub(r"_?\d+$", "", parts[1]) or parts[1]
    return ".".join(parts)


def _decor(oc: Any, state: Any, with_body: set[Any]) -> list[dict[str, Any]]:
    """State objects that are drawn but have no body (goal regions, floors)."""
    # pylint: disable-next=import-outside-toplevel
    from kinder.envs.utils import object_to_multibody2d

    out: list[dict[str, Any]] = []
    for obj in state.data:
        if obj in with_body or "robot" in obj.type.name:
            continue
        features = state.type_features[obj.type]
        try:
            multibody = object_to_multibody2d(obj, state, {})
        except Exception:  # pylint: disable=broad-exception-caught
            continue
        colour = (
            [float(state.get(obj, f"color_{c}")) for c in "rgb"]
            if "color_r" in features
            else [0.8, 0.8, 0.8]
        )
        z_order = float(state.get(obj, "z_order")) if "z_order" in features else 0.0
        for body in multibody.bodies:
            geom = body.geom
            if hasattr(geom, "vertices"):
                points = [[float(x), float(y)] for x, y in geom.vertices]
            elif hasattr(geom, "radius"):
                angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
                points = [
                    [
                        float(geom.x + geom.radius * math.cos(a)),
                        float(geom.y + geom.radius * math.sin(a)),
                    ]
                    for a in angles
                ]
            else:
                continue
            out.append(
                {"name": obj.name, "points": points, "colour": colour, "z": z_order}
            )
    return out


@contextlib.contextmanager
def recording() -> Iterator[list[PymunkRecorder | None]]:
    """Route every ``pymunk.Space.step`` to the active recorder.

    Yields a one-slot list: put a ``PymunkRecorder`` in ``slot[0]`` after a reset
    (the space is rebuilt on every reset) and set it back to None when the episode
    ends. Steps of other spaces, and the settle steps of a reset, are ignored.
    """
    # pylint: disable-next=import-outside-toplevel
    import importlib

    slot: list[PymunkRecorder | None] = [None]
    space_step = pymunk.Space.step

    def with_report(handler: Any) -> Any:
        @functools.wraps(handler)
        def patched(arbiter: pymunk.Arbiter, space: pymunk.Space, robot: Any) -> Any:
            recorder = slot[0]
            if recorder is not None and space is recorder.space:
                recorder.blocked(arbiter)
            return handler(arbiter, space, robot)

        return patched

    # The envs take these handlers from their module at every reset.
    swapped: list[tuple[Any, str, Any]] = []
    for name in BLOCK_MODULES:
        module = importlib.import_module(f"kinder.envs.dynamic2d.{name}")
        for attr in BLOCK_HANDLERS:
            if hasattr(module, attr):
                swapped.append((module, attr, getattr(module, attr)))
                setattr(module, attr, with_report(getattr(module, attr)))

    @functools.wraps(space_step)
    def patched_step(self: pymunk.Space, dt: float) -> None:
        space_step(self, dt)
        recorder = slot[0]
        if recorder is not None and self is recorder.space:
            recorder.raw_substep()

    pymunk.Space.step = patched_step  # type: ignore[method-assign]
    try:
        yield slot
    finally:
        pymunk.Space.step = space_step  # type: ignore[method-assign]
        for module, attr, handler in swapped:
            setattr(module, attr, handler)
