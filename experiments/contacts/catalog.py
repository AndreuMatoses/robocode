"""The dataset's scenarios: tags (simulator, robot, ...) and thumbnails.

Tags are derived from each scenario's Hydra environment config. No policy is
involved: a thumbnail is the render of a reset.

    python -m experiments.contacts.catalog            # write missing thumbnails
    python -m experiments.contacts.catalog --force
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import pandas as pd
from omegaconf import OmegaConf

from experiments.render_run import CONF_ROOT, DEFAULT_DATA, REPO, env_config_name

DEFAULT_OUT = REPO / "outputs" / "contact_traces"
SCENARIO_DIR = "_scenarios"

# (simulator, robot, dimension, dynamics), keyed by a substring of the config's
# kinder module path or target class.
_FAMILIES = [
    ("dynamic3d", ("MuJoCo", "TidyBot", "3D", "dynamic")),
    ("kinematic3d", ("PyBullet", "TidyBot (PyBullet)", "3D", "kinematic")),
    ("dynamic2d", ("pymunk", "2D robot", "2D", "dynamic")),
    ("kinematic2d", ("geometric 2D", "2D robot", "2D", "kinematic")),
    ("pr2_tamp", ("PyBullet", "PR2", "3D", "kinematic")),
    ("rovers", ("PyBullet", "Rovers", "3D", "kinematic")),
]
# Fixed (non-generalized) scenarios name a registered env id rather than a module.
_BY_ENV_ID = {
    "BalanceBeam3D": "dynamic3d",
    "Rearrange3D": "dynamic3d",
    "SweepIntoDrawer3D": "dynamic3d",
    "BaseMotion3D": "kinematic3d",
    "PushPullHook2D": "kinematic2d",
    "DynPushT2D": "dynamic2d",
}


SCENES = {
    "plain": "the scene of the experiments (for most scenarios: no background)",
    "lab": "the textured lab room of Kinder (MimicLabs lab2); MuJoCo scenarios only",
}


def make_env(scenario: str, scene: str | None = None) -> Any:
    """Instantiate a scenario's environment from its Hydra config (no policy).

    ``scene`` is for pictures only: "lab" asks Kinder for its textured lab room
    (``scene_bg="mimiclabs-lab2"``), "plain" for no background. It changes the
    visual geoms, not the joints, so a recorded state still fits. Only the MuJoCo
    scenarios have this option; the other scenarios ignore it. ``None`` keeps the
    scene of the experiment config.
    """
    # pylint: disable-next=import-outside-toplevel
    from hydra.utils import instantiate

    # pylint: disable-next=import-outside-toplevel
    from robocode.environments.mujoco_gl import configure_gl_backend

    configure_gl_backend()
    config = OmegaConf.load(
        CONF_ROOT / "environment" / f"{env_config_name(scenario)}.yaml"
    )
    if scene is not None:
        if scene not in SCENES:
            raise ValueError(f"unknown scene {scene!r}")
        value = "mimiclabs-lab2" if scene == "lab" else False
        if "scene_bg" in config.get("constant_object_env_kwargs", {}):
            config.constant_object_env_kwargs.scene_bg = value
        elif "scene_bg" in config:
            config.scene_bg = value
    return instantiate(config)


def scenario_tags(scenario: str) -> dict[str, str]:
    """Simulator, robot, dimension and dynamics of a scenario."""
    config = CONF_ROOT / "environment" / f"{env_config_name(scenario)}.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(config))
    assert isinstance(cfg, dict)
    text = f"{cfg.get('constant_object_env_path', '')} {cfg.get('_target_', '')}"
    env_id = str(cfg.get("env_id", ""))
    for prefix, family in _BY_ENV_ID.items():
        if f"/{prefix}-" in env_id:
            text += f" {family}"
    for needle, (simulator, robot, dim, dynamics) in _FAMILIES:
        if needle in text:
            return {
                "simulator": simulator,
                "robot": robot,
                "dim": dim,
                "dynamics": dynamics,
            }
    return {"simulator": "?", "robot": "?", "dim": "?", "dynamics": "?"}


@functools.lru_cache(maxsize=4)
def runs_table(data: Path) -> pd.DataFrame:
    """All methods' rows of the dataset's ``runs`` table."""
    frames = [
        pd.read_parquet(f) for f in sorted((data / "data/runs").glob("*.parquet"))
    ]
    return pd.concat(frames, ignore_index=True)


def scenarios(data: Path) -> list[dict[str, Any]]:
    """Every scenario in the dataset with its tags."""
    names = sorted(runs_table(data).environment.unique())
    return [{"name": name, **scenario_tags(name)} for name in names]


def thumbnail_path(out: Path, scenario: str) -> Path:
    """Where a scenario's thumbnail is cached."""
    return out / SCENARIO_DIR / scenario / "thumb.png"


def write_thumbnail(out: Path, scenario: str) -> Path:
    """Render the scenario after ``reset(seed=0)``."""
    env = make_env(scenario)
    try:
        env.reset(seed=0)
        frame = env.render()
        path = thumbnail_path(out, scenario)
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, frame[..., :3])
        return path
    finally:
        env.close()


def main() -> None:
    """Write thumbnails, one subprocess per scenario (simulators differ)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("scenarios", nargs="*", help="default: all in the dataset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if len(args.scenarios) == 1:
        print(write_thumbnail(args.out, args.scenarios[0]))
        return
    names = args.scenarios or [s["name"] for s in scenarios(args.data)]
    for name in names:
        if thumbnail_path(args.out, name).exists() and not args.force:
            continue
        cmd = [sys.executable, "-m", "experiments.contacts.catalog", name]
        cmd += ["--out", str(args.out)]
        ok = subprocess.run(cmd, check=False, capture_output=True).returncode == 0
        print(f"{name}: {'ok' if ok else 'FAILED'}", flush=True)


if __name__ == "__main__":
    main()
