"""The simulator families that the recorder and the replay support.

A *backend* is one way to read contacts from a simulator and to draw a recorded
episode again. The trace files have the same layout for every backend, so all the
analysis (``strategy``, ``histories``, ``diversity``, ``metrics``, ``repeat``) does
not depend on it. ``meta.json`` names the backend of a trace in ``"backend"``
(traces made before this key existed are MuJoCo traces).

To add a simulator: write a recorder (a subclass of ``recorder.BaseRecorder`` and
a ``recording()`` context manager), a way to draw a trace, add both below and in
``replay.main``, and name its scenarios in ``BACKEND_OF_SIMULATOR``.

| backend | scenarios              | recorder       | replay                         |
|---------|------------------------|----------------|--------------------------------|
| mujoco  | dynamic3d (TidyBot)    | recorder.py    | replay.py (needs the env)      |
| pymunk  | dynamic2d              | recorder2d.py  | render2d.py (from the trace)   |
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

# ``catalog.scenario_tags(...)["simulator"]`` -> backend name.
BACKEND_OF_SIMULATOR = {
    "MuJoCo": "mujoco",
    "pymunk": "pymunk",
    "geometric 2D": "kin2d",
    "PyBullet": "bullet",
}


def backend_of_scenario(scenario: str) -> str:
    """The backend that records ``scenario``; raises if there is none yet."""
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.catalog import scenario_tags

    simulator = scenario_tags(scenario)["simulator"]
    if simulator not in BACKEND_OF_SIMULATOR:
        raise SystemExit(
            f"{scenario} uses {simulator}: no contact recorder for this simulator yet"
        )
    return BACKEND_OF_SIMULATOR[simulator]


def backend_of_trace(meta: dict[str, Any]) -> str:
    """The backend that a trace was recorded with."""
    return str(meta.get("backend", "mujoco"))


def recording(backend: str) -> AbstractContextManager[list[Any]]:
    """The context manager that routes the simulator's substeps to a recorder."""
    if backend == "pymunk":
        # pylint: disable-next=import-outside-toplevel
        from experiments.contacts import recorder2d

        return recorder2d.recording()
    if backend == "kin2d":
        # pylint: disable-next=import-outside-toplevel
        from experiments.contacts import recorder_kin2d

        return recorder_kin2d.recording()
    if backend == "bullet":
        # pylint: disable-next=import-outside-toplevel
        from experiments.contacts import recorder_bullet

        return recorder_bullet.recording()
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts import recorder

    return recorder.recording()


def make_recorder(backend: str, env: Any) -> Any:
    """A recorder for ``env`` (after a reset)."""
    if backend == "pymunk":
        # pylint: disable-next=import-outside-toplevel
        from experiments.contacts.recorder2d import PymunkRecorder

        return PymunkRecorder(env)
    if backend == "kin2d":
        # pylint: disable-next=import-outside-toplevel
        from experiments.contacts.recorder_kin2d import Kin2DRecorder

        return Kin2DRecorder(env)
    if backend == "bullet":
        # pylint: disable-next=import-outside-toplevel
        from experiments.contacts.recorder_bullet import BulletRecorder, scene_of

        return BulletRecorder(scene_of(env))
    # pylint: disable-next=import-outside-toplevel
    from experiments.contacts.entities import robot_env
    from experiments.contacts.recorder import ContactRecorder

    return ContactRecorder(robot_env(env))
