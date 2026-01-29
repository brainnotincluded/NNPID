"""Trained Yaw Tracking Controller - Wrapper for trained neural network model.

This module provides a simple interface to load and use trained yaw tracking models.
The controller encapsulates model loading, observation normalization, and action prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for imports
if Path(__file__).parent.parent.parent not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseAlgorithm = Any  # type: ignore

from ..environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv


class TrainedYawTracker:
    """Wrapper for trained yaw tracking neural network model.

    This class encapsulates a trained model and provides a simple interface
    for getting yaw rate commands based on current observations. It handles
    model loading, observation normalization (VecNormalize), and action prediction.

    Example:
        ```python
        # Load model
        tracker = TrainedYawTracker.from_path("runs/best_model")

        # In your control loop
        while True:
            # Get observation from environment (11 elements)
            obs = env.get_observation()

            # Get yaw rate command [-1, 1]
            yaw_rate_cmd = tracker.predict(obs, deterministic=True)

            # Scale to actual yaw rate and use in control system
            actual_yaw_rate = yaw_rate_cmd * max_yaw_rate  # e.g., 2.0 rad/s
            motors = stabilizer.compute_motors(state, actual_yaw_rate, dt)
        ```

    Attributes:
        model: Loaded stable-baselines3 model (PPO/SAC)
        vec_normalize: VecNormalize wrapper for observation normalization (optional)
        config: Environment configuration used during training
        observation_space: Size of observation vector (default: 11)

    See Also:
        - [Using Trained Models Guide](../../docs/TRAINED_MODEL_USAGE.md) for detailed documentation
        - [Training Guide](../../docs/TRAINING.md) for training models
    """

    def __init__(
        self,
        model: BaseAlgorithm,
        vec_normalize: VecNormalize | None = None,
        config: YawTrackingConfig | None = None,
    ):
        """Initialize TrainedYawTracker.

        Args:
            model: Loaded stable-baselines3 model (PPO/SAC)
            vec_normalize: Optional VecNormalize wrapper for observation normalization
            config: Optional environment configuration
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )

        self.model = model
        self.vec_normalize = vec_normalize
        self.config = config or YawTrackingConfig()

        # Get observation space from model
        if hasattr(model, "observation_space"):
            self.observation_space = model.observation_space.shape[0]
        else:
            self.observation_space = 11  # Default for YawTrackingEnv

    @classmethod
    def from_path(
        cls,
        model_path: str | Path,
        config: YawTrackingConfig | None = None,
    ) -> TrainedYawTracker:
        """Load model from file path.

        This is the recommended way to create a TrainedYawTracker instance.
        It automatically handles model loading and VecNormalize initialization.

        Args:
            model_path: Path to model directory or .zip file
                - If directory: searches for best_model.zip or final_model.zip
                - If file: loads directly
            config: Optional environment configuration (uses training config if None)

        Returns:
            TrainedYawTracker instance ready to use

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model cannot be loaded
            ImportError: If stable-baselines3 is not installed

        Example:
            ```python
            # Load from directory (searches for best_model.zip)
            tracker = TrainedYawTracker.from_path("runs/best_model")

            # Load from specific file
            tracker = TrainedYawTracker.from_path("runs/model_12345/final_model.zip")
            ```
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required. Install with: pip install stable-baselines3"
            )

        model_path = Path(model_path)

        # Handle directory
        if model_path.is_dir():
            # Look for best_model.zip or final_model.zip
            for name in ["best_model.zip", "final_model.zip", "best_model", "final_model"]:
                candidate = model_path / name
                if candidate.exists():
                    model_path = candidate
                    break
            else:
                # Try direct directory
                if (model_path / "best_model.zip").exists():
                    model_path = model_path / "best_model.zip"
                elif (model_path / "final_model.zip").exists():
                    model_path = model_path / "final_model.zip"
                else:
                    raise FileNotFoundError(f"No model found in {model_path}")

        # Add .zip if needed
        if not model_path.suffix:
            model_path = model_path.with_suffix(".zip")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load model
        try:
            model = PPO.load(str(model_path))
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}") from e

        # Try to load VecNormalize
        vec_normalize = None
        vec_norm_path = model_path.parent.parent / "vec_normalize.pkl"
        if not vec_norm_path.exists():
            vec_norm_path = model_path.parent / "vec_normalize.pkl"

        if vec_norm_path.exists():
            try:
                # Create dummy env for VecNormalize
                from stable_baselines3.common.env_util import make_vec_env

                env_config = config or YawTrackingConfig()
                dummy_vec_env = make_vec_env(
                    lambda: YawTrackingEnv(config=env_config), n_envs=1
                )
                vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_vec_env)
                vec_normalize.training = False
            except Exception as e:
                print(f"Warning: Could not load VecNormalize: {e}")

        return cls(model=model, vec_normalize=vec_normalize, config=config)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> float:
        """Predict yaw rate command from observation.

        This is the main method for getting control commands from the trained model.
        It handles observation normalization automatically if VecNormalize was loaded.

        Args:
            observation: Observation vector from environment
                Shape: (11,) for YawTrackingEnv
                Elements:
                - [0] target_dir_x: X component of target direction
                - [1] target_dir_y: Y component of target direction
                - [2] target_angular_vel: Target angular velocity (rad/s)
                - [3] current_yaw_rate: Current yaw rate (rad/s)
                - [4] yaw_error: Yaw error angle (rad)
                - [5] roll: Current roll angle (rad)
                - [6] pitch: Current pitch angle (rad)
                - [7] altitude_error: Altitude error (m)
                - [8] velocity_x: X velocity (m/s)
                - [9] velocity_y: Y velocity (m/s)
                - [10] previous_action: Previous action value
            deterministic: If True, use deterministic policy (default: True)
                - True: Consistent behavior for deployment
                - False: Stochastic for exploration/testing

        Returns:
            Yaw rate command in range [-1, 1]
                - +1.0: Maximum positive yaw rate
                - -1.0: Maximum negative yaw rate
                - 0.0: No yaw command

        Raises:
            ValueError: If observation shape is incorrect (wrong size or dimension)

        Example:
            ```python
            # Get command
            yaw_cmd = tracker.predict(obs, deterministic=True)

            # Scale to actual yaw rate
            max_yaw_rate = 2.0  # rad/s
            actual_yaw_rate = yaw_cmd * max_yaw_rate
            ```
        """
        obs = np.asarray(observation, dtype=np.float32)

        # Validate shape
        if obs.ndim != 1:
            raise ValueError(f"Observation must be 1D, got shape {obs.shape}")

        if len(obs) != self.observation_space:
            raise ValueError(
                f"Observation size mismatch: expected {self.observation_space}, got {len(obs)}"
            )

        # Normalize observation if VecNormalize is available
        if self.vec_normalize is not None:
            obs_normalized = self.vec_normalize.normalize_obs(obs.reshape(1, -1))
            # Extract first element and convert to numpy array
            obs_normalized_array = np.asarray(obs_normalized)
            obs = np.array(obs_normalized_array[0], dtype=np.float32)

        # Get action from model
        action, _ = self.model.predict(obs, deterministic=deterministic)

        # Extract scalar yaw rate command
        yaw_rate_cmd = float(action[0]) if len(action) > 0 else 0.0

        return yaw_rate_cmd

    def reset(self) -> None:
        """Reset internal state.

        Currently a no-op, but can be extended for models with internal state
        (e.g., recurrent neural networks with hidden states).

        Example:
            ```python
            # Reset before new episode
            tracker.reset()
            ```
        """
        pass

    def get_info(self) -> dict[str, Any]:
        """Get controller information.

        Returns a dictionary with details about the loaded model and configuration.

        Returns:
            Dictionary with model information:
            - model_type: Type of model (e.g., "PPO", "SAC")
            - observation_space: Size of observation vector (default: 11)
            - has_normalization: Whether VecNormalize is loaded
            - config: Environment configuration dictionary with keys:
              - target_patterns: List of target motion patterns
              - target_speed_min/max: Target speed range
              - control_frequency: Control loop frequency
              - yaw_authority: Yaw torque authority

        Example:
            ```python
            info = tracker.get_info()
            print(f"Model: {info['model_type']}")
            print(f"Observation space: {info['observation_space']}")
            print(f"Has normalization: {info['has_normalization']}")
            ```
        """
        info = {
            "model_type": type(self.model).__name__,
            "observation_space": self.observation_space,
            "has_normalization": self.vec_normalize is not None,
        }

        if self.config:
            info["config"] = {
                "target_patterns": self.config.target_patterns,
                "target_speed_min": self.config.target_speed_min,
                "target_speed_max": self.config.target_speed_max,
                "control_frequency": self.config.control_frequency,
                "yaw_authority": self.config.yaw_authority,
            }

        return info

    def __repr__(self) -> str:
        """String representation."""
        norm_str = "with normalization" if self.vec_normalize else "without normalization"
        return f"TrainedYawTracker({self.model.__class__.__name__}, {norm_str})"
