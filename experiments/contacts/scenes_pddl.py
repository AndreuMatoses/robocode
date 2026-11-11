"""Scene adapters for the PR2 and rover environments (``recorder_bullet``).

Both families are robocode environments on top of PDDLStream scenes, with PyBullet
as a collision checker. An adapter tells the recorder which bodies there are, what
they are called, what the robot holds, when a motion is rejected, and how to save
and restore a state.

The rovers do not manipulate anything. Their "contacts" are the events of the task,
each with a pair of bodies: a rover takes a sample (rover - rock or soil), takes a
picture (rover - objective), or sends its results (rover - lander). Together with
"a rover is stopped by an obstacle" they give a sequence that the same analysis can
compare.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pybullet as p

TOUCH = 0.005  # [m] as the kinematic3d scenes; a settled block is 0.1 mm above
_NOT_A_BODY = ("num", "max", "obs", "client", "tool", "robot", "grasp", "np")
_NOT_A_BODY_WORDS = ("joint", "conf", "link", "limit", "dim", "extent", "circular")


def _inner(env: Any) -> Any:
    """The environment behind a variable-count wrapper."""
    return getattr(env, "_current", None) or env


def scene_of(env: Any) -> Any:
    """The adapter of a PR2 or rover environment."""
    inner = _inner(env)
    if hasattr(inner, "_rovers"):
        return RoverScene(inner)
    if hasattr(inner, "_movables") and hasattr(inner, "_robot"):
        return Pr2Scene(inner)
    raise TypeError(f"no contact recorder for {type(inner).__name__}")


def _names(inner: Any, skip: tuple[str, ...] = ()) -> dict[int, str]:
    """Body names from the attributes of an env: ``_plate``, ``_walls`` -> wall0 ..."""
    names: dict[int, str] = {}
    for attr, value in vars(inner).items():
        label = attr.strip("_")
        if label in skip or label.startswith(_NOT_A_BODY):
            continue
        if any(word in label for word in _NOT_A_BODY_WORDS):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            names.setdefault(value, label)
        elif (
            isinstance(value, (list, tuple))
            and value
            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        ):
            stem = label[:-1] if label.endswith("s") else label
            for i, body in enumerate(value):
                names.setdefault(body, f"{stem}{i}")
    return names


def pr2_part(link: str) -> str:
    """The robot part of a PR2 link, in the words of the other robots."""
    side = {"l": "left", "r": "right"}
    finger = re.fullmatch(r"[lr]_gripper_([lr])_finger(_tip)?_link", link)
    if finger:
        return (
            f"gripper.{side[finger.group(1)]}_{'pad' if finger.group(2) else 'finger'}"
        )
    if "gripper" in link:
        return "gripper.palm"
    arm = re.fullmatch(r"[lr]_(shoulder|upper_arm|elbow|forearm|wrist)\w*", link)
    if arm:
        return f"arm.{arm.group(1)}"
    if "torso" in link:
        return "torso"
    if any(
        word in link
        for word in ("head", "laser", "sensor", "camera", "kinect", "prosilica")
    ):
        return "head"
    if any(word in link for word in ("base", "caster", "wheel")):
        return "base"
    return "body"


class _PddlScene:
    """What the PR2 and the rover scenes share."""

    touch = TOUCH
    grasp_links: tuple[str, ...] = ()

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.client: int = inner._client  # pylint: disable=protected-access
        self.step_owner = inner
        self.robots: dict[int, Callable[[str], str]] = {}
        self.named: dict[int, tuple[str, str]] = {}

    def held(self) -> list[int]:
        """The bodies that the robot holds now."""
        return []

    def events(self) -> list[tuple[int, int]]:
        """Body pairs of the task events of the step that just finished."""
        return []

    def state(self) -> np.ndarray:
        """The state of the env as one vector (see ``restore``)."""
        return np.asarray(self.inner.get_state(), dtype=np.float64)

    def restore(self, vector: np.ndarray) -> None:
        """Put the env in a recorded state."""
        self.inner.set_state(np.asarray(vector, dtype=np.float32))

    def render(self) -> np.ndarray:
        """The picture of ``env.render()``."""
        return np.asarray(self.inner.render())[..., :3]

    def camera(self) -> dict[str, Any] | None:
        """These envs build their camera themselves: no segmentation mask."""
        return None


class Pr2Scene(_PddlScene):
    """A PR2 tabletop environment (``pr2packed``, ``pr2blocked``)."""

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        robot = inner._robot  # pylint: disable=protected-access
        self.robots = {robot: pr2_part}
        names = _names(
            inner,
            skip=("robot", "client", "tool_link", "obstacles", "movables", "surfaces"),
        )
        movables = list(inner._movables)  # pylint: disable=protected-access
        obstacles = list(inner._obstacles)  # pylint: disable=protected-access
        for i, body in enumerate(movables):
            self.named[body] = (f"env.{names.get(body, f'block{i}')}", "object")
        for i, body in enumerate(obstacles):
            self.named[body] = (f"env.{names.get(body, f'surface{i}')}", "fixture")
        tool = p.getJointInfo(robot, inner._tool_link, physicsClientId=self.client)[
            12
        ].decode()  # pylint: disable=protected-access
        arm = tool[:2]  # "l_" or "r_"
        self.grasp_links = (
            f"{arm}gripper_l_finger_tip_link",
            f"{arm}gripper_r_finger_tip_link",
        )
        self._tool = inner._tool_link  # pylint: disable=protected-access

    def held(self) -> list[int]:
        attachment = self.inner._attachment  # pylint: disable=protected-access
        return [] if attachment is None else [attachment.child]

    def on_blocked(self, report: Callable[[], None]) -> Callable[[], None]:
        """Call ``report`` while the robot is in a configuration that is rejected."""
        check = self.inner._in_collision  # pylint: disable=protected-access

        def wrapped() -> bool:
            hit = check()
            if hit:
                report()
            return bool(hit)

        self.inner._in_collision = wrapped  # pylint: disable=protected-access

        def undo() -> None:
            self.inner.__dict__.pop("_in_collision", None)

        return undo

    def tool_position(self) -> np.ndarray:
        robot = self.inner._robot  # pylint: disable=protected-access
        return np.array(
            p.getLinkState(robot, self._tool, physicsClientId=self.client)[4]
        )


class RoverScene(_PddlScene):
    """The rover environment: two rovers, samples, objectives and a lander."""

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        # pylint: disable=protected-access
        self._rovers = list(inner._rovers)
        for i, body in enumerate(self._rovers):
            self.robots[body] = lambda _link, i=i: f"rover{i}"
        floor = getattr(inner, "_floor", None)
        groups = (
            ("_rocks", "rock", "object"),
            ("_soils", "soil", "object"),
            ("_objectives", "objective", "object"),
            ("_mounds", "mound", "world"),
            ("_walls", "wall", "world"),
            ("_obstacle_bodies", "obstacle", "world"),
        )
        for attr, stem, kind in groups:
            for i, body in enumerate(getattr(inner, attr, []) or []):
                self.named.setdefault(
                    body,
                    (
                        f"env.{stem}{i}" if kind == "object" else f"world.{stem}{i}",
                        kind,
                    ),
                )
        lander = getattr(inner, "_lander", None)
        if lander is not None:
            self.named[lander] = ("env.lander", "fixture")
        for body in inner._fixed:
            if body != floor:
                self.named.setdefault(body, (f"world.body{body}", "world"))
        self.named.pop(floor, None)
        self._lander = lander
        self._seen = self._progress()

    def cls(self, name: str) -> str:
        """The two rovers are one class: who does what is not part of the class."""
        name = re.sub(r"^robot\.rover\d+", "robot.rover", name)
        return re.sub(r"^(env|world)\.([a-z_]+?)_?\d+$", r"\1.\2", name)

    def _progress(self) -> list[tuple[set[int], set[int], int]]:
        inner = self.inner
        # pylint: disable=protected-access
        sent = len(inner._received_image) + len(inner._received_analysis)
        return [
            (set(inner._analyzed[i]), set(inner._have_image[i]), sent)
            for i in range(len(self._rovers))
        ]

    def events(self) -> list[tuple[int, int]]:
        now, pairs = self._progress(), []
        for i, rover in enumerate(self._rovers):
            pairs += [(rover, body) for body in now[i][0] - self._seen[i][0]]  # sample
            pairs += [(rover, body) for body in now[i][1] - self._seen[i][1]]  # picture
            holds = bool(self._seen[i][0] or self._seen[i][1])
            if now[i][2] > self._seen[i][2] and holds and self._lander is not None:
                if self.inner.com_visible(i):
                    pairs.append((rover, self._lander))  # send
        self._seen = now
        return pairs

    def on_blocked(self, report: Callable[[], None]) -> Callable[[], None]:
        """Call ``report`` with the rover put again at the pose that was rejected."""
        inner, cid = self.inner, self.client
        advance = inner._advance  # pylint: disable=protected-access

        def joints(index: int) -> list[float]:
            ids = inner._base_joints[index]  # pylint: disable=protected-access
            return [
                p.getJointState(self._rovers[index], j, physicsClientId=cid)[0]
                for j in ids
            ]

        def put(index: int, conf: list[float]) -> None:
            ids = inner._base_joints[index]  # pylint: disable=protected-access
            for joint, value in zip(ids, conf):
                p.resetJointState(
                    self._rovers[index], joint, value, physicsClientId=cid
                )

        def wrapped(index: int, delta: Any) -> None:
            before = joints(index)
            advance(index, delta)
            after = joints(index)
            moved = float(np.abs(np.asarray(delta, dtype=float)).max()) > 1e-9
            if moved and np.allclose(before, after, atol=1e-9):
                target = np.asarray(before) + np.asarray(delta, dtype=float)
                put(index, target.tolist())
                report()
                put(index, before)

        inner._advance = wrapped  # pylint: disable=protected-access

        def undo() -> None:
            inner.__dict__.pop("_advance", None)

        return undo

    def tool_position(self) -> np.ndarray:
        return np.array(
            p.getBasePositionAndOrientation(
                self._rovers[0], physicsClientId=self.client
            )[0]
        )
