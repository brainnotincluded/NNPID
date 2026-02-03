"""Shared utilities for loading SB3 models and VecNormalize."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseAlgorithm = Any  # type: ignore
    VecNormalize = Any  # type: ignore

logger = get_logger(__name__)

DEFAULT_CUSTOM_OBJECTS = {
    "clip_range": lambda _: 0.2,  # Default PPO clip range
    "lr_schedule": lambda _: 0.0003,  # Default learning rate (unused for inference)
}


def _require_sb3() -> None:
    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        )


def _minimal_yaw_env_factory() -> Callable[[], Any]:
    """Create a minimal Gymnasium env for VecNormalize without MuJoCo."""
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    class _MinimalYawTrackingEnv(gym.Env):
        metadata: dict[str, Any] = {}

        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                low=np.array(
                    [
                        -1,
                        -1,  # target_direction (unit vector)
                        -5,  # target_angular_velocity
                        -5,  # current_yaw_rate
                        -np.pi,  # yaw_error
                        -1,
                        -1,  # roll, pitch (normalized)
                        -5,  # altitude_error
                        -1,  # previous_action
                        0,  # time_on_target (normalized)
                        0,  # target_distance (normalized)
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [
                        1,
                        1,  # target_direction
                        5,  # target_angular_velocity
                        5,  # current_yaw_rate
                        np.pi,  # yaw_error
                        1,
                        1,  # roll, pitch
                        5,  # altitude_error
                        1,  # previous_action
                        1,  # time_on_target
                        10,  # target_distance
                    ],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, {}

        def step(self, action):  # type: ignore[override]
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

    return _MinimalYawTrackingEnv


def default_env_factory() -> Callable[[], Any]:
    """Return the best available env factory for VecNormalize."""
    try:
        from ..environments.yaw_tracking_env import YawTrackingEnv

        return lambda: YawTrackingEnv()
    except Exception as exc:
        logger.warning("Falling back to minimal VecNormalize env: %s", exc)
        return _minimal_yaw_env_factory()


def resolve_model_path(model_path: str | Path) -> Path:
    """Resolve a model path to a concrete .zip file."""
    path = Path(model_path)

    if path.is_dir():
        # Direct .zip inside directory
        for name in ("best_model.zip", "final_model.zip"):
            candidate = path / name
            if candidate.exists():
                return candidate

        # Common structure: runs/<run>/best_model/best_model.zip
        for dirname in ("best_model", "final_model"):
            candidate_dir = path / dirname
            if candidate_dir.is_dir():
                for name in ("best_model.zip", "final_model.zip"):
                    candidate = candidate_dir / name
                    if candidate.exists():
                        return candidate

        # Fallback: append .zip if it exists
        candidate = path.with_suffix(".zip")
        if candidate.exists():
            return candidate

        raise FileNotFoundError(f"No model found in {path}")

    if not path.suffix:
        path = path.with_suffix(".zip")

    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    return path


def find_vec_normalize_path(model_path: str | Path) -> Path | None:
    """Locate vec_normalize.pkl relative to a resolved model path."""
    resolved = resolve_model_path(model_path)
    candidates = [
        resolved.parent / "vec_normalize.pkl",
        resolved.parent.parent / "vec_normalize.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_sb3_model(
    model_path: str | Path,
    custom_objects: dict[str, Any] | None = None,
) -> BaseAlgorithm:
    """Load a PPO/SAC model with optional custom objects."""
    _require_sb3()
    resolved = resolve_model_path(model_path)
    objects = DEFAULT_CUSTOM_OBJECTS if custom_objects is None else custom_objects

    try:
        return PPO.load(str(resolved), custom_objects=objects)
    except Exception:
        try:
            return SAC.load(str(resolved), custom_objects=objects)
        except Exception as exc:
            raise ValueError(f"Failed to load model from {resolved}: {exc}") from exc


def load_vec_normalize(
    model_path: str | Path,
    env_factory: Callable[[], Any] | None = None,
) -> VecNormalize | None:
    """Load VecNormalize for a model if present."""
    _require_sb3()
    vec_norm_path = find_vec_normalize_path(model_path)
    if vec_norm_path is None:
        return None

    if env_factory is None:
        env_factory = default_env_factory()

    try:
        dummy_vec_env = make_vec_env(env_factory, n_envs=1)
        vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_vec_env)
        vec_normalize.training = False
        return vec_normalize
    except Exception as exc:
        logger.warning("Could not load VecNormalize from %s: %s", vec_norm_path, exc)
        return None


def load_model_and_vecnormalize(
    model_path: str | Path,
    env_factory: Callable[[], Any] | None = None,
    custom_objects: dict[str, Any] | None = None,
) -> tuple[BaseAlgorithm, VecNormalize | None, Path]:
    """Load model and VecNormalize together."""
    model = load_sb3_model(model_path, custom_objects=custom_objects)
    vec_normalize = load_vec_normalize(model_path, env_factory)
    resolved = resolve_model_path(model_path)
    return model, vec_normalize, resolved
