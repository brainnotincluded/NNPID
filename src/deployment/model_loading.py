"""Shared utilities for loading SB3 models and VecNormalize."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")


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
    env_factory: Callable[[], Any],
) -> VecNormalize | None:
    """Load VecNormalize for a model if present."""
    _require_sb3()
    vec_norm_path = find_vec_normalize_path(model_path)
    if vec_norm_path is None:
        return None

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
    env_factory: Callable[[], Any],
    custom_objects: dict[str, Any] | None = None,
) -> tuple[BaseAlgorithm, VecNormalize | None, Path]:
    """Load model and VecNormalize together."""
    model = load_sb3_model(model_path, custom_objects=custom_objects)
    vec_normalize = load_vec_normalize(model_path, env_factory)
    resolved = resolve_model_path(model_path)
    return model, vec_normalize, resolved
