"""Trajectory tracking environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations
from ..utils.trajectory import TrajectoryGenerator, TrajectoryPoint, TrajectoryType
from .base_drone_env import BaseDroneEnv, DroneEnvConfig


@dataclass
class TrajectoryEnvConfig(DroneEnvConfig):
    """Configuration for trajectory tracking environment."""

    # Trajectory parameters
    trajectory_type: str = "circle"  # circle, figure_eight, helix
    trajectory_center: tuple[float, float] = (0.0, 0.0)
    trajectory_radius: float = 2.0
    trajectory_altitude: float = 1.5
    trajectory_speed: float = 0.5  # Angular velocity or linear speed

    # Randomization
    randomize_trajectory: bool = True
    radius_range: tuple[float, float] = (1.0, 3.0)
    altitude_range: tuple[float, float] = (0.5, 2.5)
    speed_range: tuple[float, float] = (0.3, 0.8)

    # Reward weights
    position_tracking_weight: float = 1.0
    velocity_tracking_weight: float = 0.3
    heading_tracking_weight: float = 0.1
    smoothness_weight: float = 0.05
    crash_penalty: float = -100.0


class TrajectoryEnv(BaseDroneEnv):
    """Gymnasium environment for trajectory tracking.

    The goal is to follow a reference trajectory (circle, figure-8, etc.)
    as closely as possible.
    """

    def __init__(
        self,
        config: TrajectoryEnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """Initialize trajectory environment."""
        self.traj_config = config or TrajectoryEnvConfig()
        super().__init__(config=self.traj_config, render_mode=render_mode)

        # Trajectory generator
        self._trajectory_gen = TrajectoryGenerator()

        # Trajectory parameters (set on reset)
        self._traj_type = TrajectoryType.CIRCLE
        self._traj_params: dict[str, Any] = {}
        self._current_reference: TrajectoryPoint | None = None

        # Tracking
        self._total_tracking_error = 0.0

    def _define_spaces(self) -> None:
        """Define observation space with trajectory reference."""
        # Extended observation: base + reference velocity + reference acceleration
        obs_dim = 20 + 3 + 3  # base obs + ref velocity + ref acceleration

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment with new trajectory."""
        self._setup_trajectory()
        self._total_tracking_error = 0.0

        obs, info = super().reset(seed=seed, options=options)

        return self._get_observation(), info

    def _setup_trajectory(self) -> None:
        """Setup trajectory parameters."""
        cfg = self.traj_config

        # Parse trajectory type
        traj_type_map = {
            "circle": TrajectoryType.CIRCLE,
            "figure_eight": TrajectoryType.FIGURE_EIGHT,
            "helix": TrajectoryType.HELIX,
            "hover": TrajectoryType.HOVER,
        }
        self._traj_type = traj_type_map.get(
            cfg.trajectory_type.lower(),
            TrajectoryType.CIRCLE,
        )

        # Set parameters (with optional randomization)
        if cfg.randomize_trajectory:
            radius = self._np_random.uniform(cfg.radius_range[0], cfg.radius_range[1])
            altitude = self._np_random.uniform(cfg.altitude_range[0], cfg.altitude_range[1])
            speed = self._np_random.uniform(cfg.speed_range[0], cfg.speed_range[1])
            center_x = self._np_random.uniform(-1.0, 1.0)
            center_y = self._np_random.uniform(-1.0, 1.0)
        else:
            radius = cfg.trajectory_radius
            altitude = cfg.trajectory_altitude
            speed = cfg.trajectory_speed
            center_x, center_y = cfg.trajectory_center

        self._traj_params = {
            "center": np.array([center_x, center_y]),
            "radius": radius,
            "altitude": altitude,
            "angular_velocity": speed,
            "period": 2 * np.pi / speed,
            "size": radius,
            "start_altitude": altitude,
            "climb_rate": 0.0,
        }

    def _get_target(self) -> np.ndarray:
        """Get current target position from trajectory."""
        ref = self._get_trajectory_reference()
        return ref.position

    def _get_trajectory_reference(self) -> TrajectoryPoint:
        """Get current trajectory reference point."""
        time = self.sim.get_time()

        ref = self._trajectory_gen.sample_trajectory(
            self._traj_type,
            time,
            **self._traj_params,
        )

        self._current_reference = ref
        return ref

    def _get_observation(self) -> np.ndarray:
        """Get observation with trajectory reference."""
        state = self.sim.get_state()
        cfg = self.config
        rng = self._np_random

        # Base observation with noise
        position = state.position + rng.normal(0, cfg.position_noise, 3)
        velocity = state.velocity + rng.normal(0, cfg.velocity_noise, 3)
        quaternion = Rotations.quaternion_normalize(
            state.quaternion + rng.normal(0, cfg.attitude_noise, 4)
        )
        angular_velocity = state.angular_velocity + rng.normal(0, cfg.angular_velocity_noise, 3)

        # Current trajectory reference
        ref = self._get_trajectory_reference()

        # Construct observation
        obs = np.concatenate(
            [
                position,
                velocity,
                quaternion,
                angular_velocity,
                ref.position,
                self._previous_action,
                ref.velocity,
                ref.acceleration,
            ]
        ).astype(np.float32)

        return obs

    def _compute_reward(self, state: QuadrotorState, action: np.ndarray) -> float:
        """Compute trajectory tracking reward."""
        cfg = self.traj_config
        reward = 0.0

        ref = self._current_reference
        if ref is None:
            ref = self._get_trajectory_reference()

        # 1. Position tracking error
        position_error = np.linalg.norm(state.position - ref.position)
        position_reward = np.exp(-2.0 * position_error**2)
        reward += cfg.position_tracking_weight * position_reward

        self._total_tracking_error += position_error

        # 2. Velocity tracking
        velocity_error = np.linalg.norm(state.velocity - ref.velocity)
        velocity_reward = np.exp(-1.0 * velocity_error**2)
        reward += cfg.velocity_tracking_weight * velocity_reward

        # 3. Heading tracking (yaw alignment with trajectory)
        _, _, yaw = Rotations.quaternion_to_euler(state.quaternion)
        yaw_error = abs(self._angle_wrap(yaw - ref.yaw))
        heading_reward = np.exp(-2.0 * yaw_error**2)
        reward += cfg.heading_tracking_weight * heading_reward

        # 4. Control smoothness
        action_rate = np.sum((action - self._previous_action) ** 2)
        reward -= cfg.smoothness_weight * action_rate

        return reward

    def _angle_wrap(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _is_success(self, state: QuadrotorState) -> bool:
        """Check if tracking is successful.

        Success: average tracking error below threshold.
        """
        if self._step_count == 0:
            return False

        avg_error = self._total_tracking_error / self._step_count
        return avg_error < 0.5  # meters

    def _get_info(self) -> dict[str, Any]:
        """Get info with trajectory-specific data."""
        info = super()._get_info()

        ref = self._current_reference
        if ref is not None:
            info["reference_position"] = ref.position.copy()
            info["reference_velocity"] = ref.velocity.copy()

        if self._step_count > 0:
            info["average_tracking_error"] = self._total_tracking_error / self._step_count

        info["trajectory_type"] = self._traj_type.value

        return info


# Need to import gym for registration
import gymnasium as gym
