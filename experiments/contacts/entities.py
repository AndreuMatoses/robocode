"""Human-readable names for what touches what in a MuJoCo (dynamic3d) scene.

Three levels, built from the compiled model:

* **entity**: the thing that gets an ID in diagrams, with a dotted hierarchical name:
  ``robot.gripper.left_pad``, ``robot.arm.forearm``, ``env.cube_0``,
  ``env.table_1.top``, ``world.floor``. Its ``cls`` drops the object index
  (``env.cube``), to compare instances with different object counts.
* **geom**: a collision primitive of that entity (``index`` within the entity).
* **surface**: the face of the primitive a contact point lies on (``FACES``).

Most object geoms are unnamed in kinder's models, so names come from the bodies
(the task's object names) and, where present, from geom names (fixture parts).
A contact point is expressed in the geom's frame, relative to the geom's centre and
divided by its half-extents, so every coordinate is in [-1, 1] whatever the size.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np

PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7
GEOM_TYPES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
}
# Surface codes. Boxes use the six faces, cylinders/capsules +z, -z and side.
FACES = ["surface", "+x", "-x", "+y", "-y", "+z", "-z", "side"]
KINDS = ["robot", "object", "fixture", "world"]  # also the order within a pair

_ROBOT_BODIES = [
    (r"base_link", "base"),
    (r"gen3/base_link", "arm.base"),
    (r"(shoulder|half_arm_1|half_arm_2|forearm)_link", r"arm.\1"),
    (r"(spherical_wrist_1|spherical_wrist_2|bracelet)_link", r"arm.\1"),
    (r"base", "gripper.palm"),
    (r"(right|left)_(driver|coupler|spring_link|follower)", r"gripper.\1_finger"),
    (r"(right|left)_(silicone_)?pad", r"gripper.\1_pad"),
]
_WORLD_ROOTS = re.compile(r"(^scene$|_room$)")


def robot_env(env: Any) -> Any:
    """The kinder ``MujocoEnv`` behind a robocode environment (after a reset)."""
    backend = getattr(env, "_current_backend", None) or getattr(
        env, "_kinder_env", None
    )
    try:
        # pylint: disable-next=protected-access
        return backend._object_centric_env._robot_env
    except AttributeError as exc:
        raise TypeError(f"{type(env).__name__} is not a MuJoCo environment") from exc


@dataclasses.dataclass
class Entity:
    """One labelled thing in the scene."""

    id: int
    name: str
    cls: str
    kind: str
    geoms: list[int] = dataclasses.field(default_factory=list)  # collision geoms


@dataclasses.dataclass
class EntityMap:
    """Entities of a compiled model and the geom-level lookups for contacts."""

    entities: list[Entity]
    geom_entity: np.ndarray  # [ngeom] entity id of every geom (visual ones too)
    geom_index: np.ndarray  # [ngeom] index among its entity's collision geoms, or -1
    geom_type: np.ndarray  # [ngeom]
    geom_center: np.ndarray  # [ngeom, 3] centre of the shape in the geom frame
    geom_extent: np.ndarray  # [ngeom, 3] half-extents used to normalise
    signature: str  # changes if a reset does not rebuild the same scene

    def rank(self) -> np.ndarray:
        """Per-geom sort key (kind, then entity id) that orders the two sides."""
        kind = np.array([KINDS.index(e.kind) for e in self.entities])
        return kind[self.geom_entity] * 100_000 + self.geom_entity

    def to_json(self) -> dict[str, Any]:
        """The tables a viewer needs to name contacts."""
        return {
            "signature": self.signature,
            "faces": FACES,
            "entities": [dataclasses.asdict(e) for e in self.entities],
            "geoms": [
                {
                    "id": g,
                    "entity": int(self.geom_entity[g]),
                    "index": int(self.geom_index[g]),
                    "type": GEOM_TYPES.get(int(self.geom_type[g]), "?"),
                    "extent": [round(float(x), 5) for x in self.geom_extent[g]],
                }
                for g in np.flatnonzero(self.geom_index >= 0).tolist()
            ],
        }


def _strip(name: str, *prefixes: str) -> str:
    for prefix in sorted(prefixes, key=len, reverse=True):
        if prefix and name.startswith(prefix + "_"):
            return name[len(prefix) + 1 :]
    return name


def _name_and_kind(model: Any, geom: int, movable_roots: set[int]) -> tuple[str, str]:
    body = int(model.geom_bodyid[geom])
    root = int(model.body_rootid[body])
    bname, rname = model.body(body).name, model.body(root).name
    gname = model.geom(geom).name
    if body == 0:
        return f"world.{gname or 'ground'}", "world"
    if rname.startswith("robot_"):
        short = bname.removeprefix("robot_")
        for pattern, repl in _ROBOT_BODIES:
            if re.fullmatch(pattern, short):
                return "robot." + re.sub(pattern, repl, short), "robot"
        return f"robot.{short}", "robot"
    if _WORLD_ROOTS.search(rname) and root not in movable_roots:
        part = _strip(gname, rname).removesuffix("_visual") if gname else rname
        return f"world.{part}", "world"
    name = f"env.{rname}"
    if body != root:
        name += "." + _strip(bname, rname)
    if gname:
        part = _strip(gname, bname, rname).removesuffix("_geom")
        if part not in (gname, "geom"):  # a named part of this body: table_1_top
            name += f".{part}"
    return name, ("object" if root in movable_roots else "fixture")


def _class_name(name: str) -> str:
    parts = name.split(".")
    if parts[0] == "env":
        parts[1] = re.sub(r"_?\d+$", "", parts[1]) or parts[1]
    return ".".join(parts)


def build_entity_map(model: Any) -> EntityMap:
    """Group the model's geoms into named entities (``model`` is a raw MjModel)."""
    free_roots = {
        int(model.body_rootid[model.jnt_bodyid[j]])
        for j in range(model.njnt)
        if model.jnt_type[j] == 0  # mjJNT_FREE
    }
    collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    entities: list[Entity] = []
    by_name: dict[str, Entity] = {}
    geom_entity = np.full(model.ngeom, -1, dtype=np.int32)
    geom_index = np.full(model.ngeom, -1, dtype=np.int32)
    body_entity: dict[int, int] = {}
    # Collision geoms define the entities; visual-only geoms then join the entity of
    # their body (or of the nearest ancestor that has one), for segmentation renders.
    for geom in np.flatnonzero(collides).tolist():
        name, kind = _name_and_kind(model, geom, free_roots)
        entity = by_name.get(name)
        if entity is None:
            entity = Entity(len(entities), name, _class_name(name), kind)
            entities.append(entity)
            by_name[name] = entity
        geom_entity[geom] = entity.id
        geom_index[geom] = len(entity.geoms)
        entity.geoms.append(geom)
        body_entity.setdefault(int(model.geom_bodyid[geom]), entity.id)
    for geom in np.flatnonzero(~collides).tolist():
        name, kind = _name_and_kind(model, geom, free_roots)
        if name in by_name:
            geom_entity[geom] = by_name[name].id
            continue
        body = int(model.geom_bodyid[geom])
        while body and body not in body_entity:
            body = int(model.body_parentid[body])
        if body in body_entity:
            geom_entity[geom] = body_entity[body]
        else:  # scenery with no collision at all (e.g. background walls)
            entity = Entity(len(entities), name, _class_name(name), kind)
            entities.append(entity)
            by_name[name] = entity
            geom_entity[geom] = entity.id

    size = np.array(model.geom_size, dtype=np.float64)
    gtype = np.array(model.geom_type, dtype=np.int32)
    extent, center = size.copy(), np.zeros_like(size)
    round_ = np.isin(gtype, (SPHERE,))
    extent[round_] = size[round_, :1]
    tube = np.isin(gtype, (CYLINDER, CAPSULE))
    extent[tube] = size[tube][:, [0, 0, 1]]
    extent[gtype == CAPSULE, 2] += size[gtype == CAPSULE, 0]
    mesh = gtype == MESH
    aabb = np.array(model.geom_aabb, dtype=np.float64).reshape(-1, 6)
    center[mesh], extent[mesh] = aabb[mesh, :3], aabb[mesh, 3:]
    extent[gtype == PLANE] = 1.0  # planes are unbounded: keep metres
    extent[extent <= 0] = 1.0
    signature = "|".join(
        f"{model.geom_bodyid[g]}:{model.geom_type[g]}:{e.name}"
        for e in entities
        for g in e.geoms
    )
    return EntityMap(
        entities=entities,
        geom_entity=geom_entity,
        geom_index=geom_index,
        geom_type=gtype,
        geom_center=center,
        geom_extent=extent,
        signature=f"{model.nq}/{model.ngeom}/{hash_text(signature)}",
    )


def hash_text(text: str) -> str:
    """A short stable digest (``hash()`` is salted per process)."""
    import hashlib  # pylint: disable=import-outside-toplevel

    return hashlib.sha1(text.encode()).hexdigest()[:12]


def local_points(
    emap: EntityMap,
    geoms: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    xpos: np.ndarray,
    xmat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised local coordinates and surface codes of world ``points`` on ``geoms``.

    ``normals`` are the contact normals (world frame, either sign) and
    ``xpos``/``xmat`` the data's ``geom_xpos``/``geom_xmat``. The surface is the
    face the normal is most aligned with, on the side of the geom the point is on;
    going by position alone would be ambiguous at edges and corners, where a box
    resting on a plane has its contact points. Returns (``[n, 3]`` coordinates in
    the geom frame divided by its half-extents, ``[n]`` indices into ``FACES``).
    """
    rot = xmat[geoms].reshape(-1, 3, 3)
    local = np.einsum("nji,nj->ni", rot, points - xpos[geoms])
    normal = np.abs(np.einsum("nji,nj->ni", rot, normals))
    norm = (local - emap.geom_center[geoms]) / emap.geom_extent[geoms]
    gtype = emap.geom_type[geoms]
    face = np.zeros(len(geoms), dtype=np.int8)
    box = gtype == BOX
    if box.any():
        axis = normal[box].argmax(axis=1)
        negative = norm[box][np.arange(axis.size), axis] < 0
        face[box] = 1 + 2 * axis + negative
    tube = (gtype == CYLINDER) | (gtype == CAPSULE)
    if tube.any():
        cap = normal[tube, 2] >= np.hypot(normal[tube, 0], normal[tube, 1])
        face[tube] = np.where(cap, np.where(norm[tube, 2] < 0, 6, 5), 7)
    return norm.astype(np.float32), face
