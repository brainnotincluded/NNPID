"""Fast training environments using setpoint action space.

These environments use a simulated position controller instead of PX4 SITL,
enabling much faster training (1000+ Hz vs ~50 Hz with SITL).

Use these for initial training, then fine-tune with SITL-in-loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .setpoint_base_env import SetpointBaseEnv, SetpointEnvConfig


@dataclass
class SetpointHoverConfig(SetpointEnvConfig):
    """Configuration for setpoint hover environment."""

    # Hover task
    hover_altitude: float = 1.5
    randomize_target: bool = True
    target_range: float = 3.0  # meters

    # Reward weights
    position_weight: float = 1.0
    velocity_weight: float = 0.1
    action_weight: float = 0.05
    stability_weight: float = 0.1

    # Success
    success_threshold: float = 0.15  # meters


class SetpointHoverEnv(SetpointBaseEnv):
    """Hover task using setpoint control.

    The agent outputs velocity setpoints to maintain hover at a target position.
    Uses a simulated position controller for fast training.

    Action space: [vx, vy, vz, yaw_rate] normalized to [-1, 1]
    """

    def __init__(
        self,
        config: SetpointHoverConfig | None = None,
        render_mode: str | None = None,
    ):
        """Initialize hover environment.

        Args:
            config: Environment configuration
            render_mode: Rendering mode
        """
        self.hover_config = config or SetpointHoverConfig()
        super().__init__(config=self.hover_config, render_mode=render_mode)

        self._success_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        self._success_steps = 0
        return obs, info

    def _get_target(self) -> np.ndarray:
        """Get target hover position."""
        if self.hover_config.randomize_target:
            r = self.hover_config.target_range
            return np.array(
                [
                    self._np_random.uniform(-r, r),
                    self._np_random.uniform(-r, r),
                    self._np_random.uniform(1.0, 3.0),
                ]
            )
        else:
            return np.array([0.0, 0.0, self.hover_config.hover_altitude])

    def _compute_reward(self) -> float:
        """Compute hover reward."""
        state = self.sim.get_state()
        cfg = self.hover_config

        # Position error
        pos_error = np.linalg.norm(state.position - self._target_position)
        position_reward = cfg.position_weight * np.exp(-2.0 * pos_error)

        # Velocity penalty (we want to be still at hover)
        vel_magnitude = np.linalg.norm(state.velocity)
        velocity_penalty = cfg.velocity_weight * vel_magnitude

        # Action penalty
        action_magnitude = np.linalg.norm(self._previous_action)
        action_penalty = cfg.action_weight * action_magnitude

        # Stability (penalize angular rates)
        omega_magnitude = np.linalg.norm(state.angular_velocity)
        stability_penalty = cfg.stability_weight * omega_magnitude

        # Success bonus
        success_bonus = 0.0
        if pos_error < cfg.success_threshold and vel_magnitude < 0.3:
            self._success_steps += 1
            success_bonus = 0.5
        else:
            self._success_steps = 0

        # Alive bonus
        alive_bonus = 0.1

        reward = (
            position_reward
            - velocity_penalty
            - action_penalty
            - stability_penalty
            + success_bonus
            + alive_bonus
        )

        return reward

    def _get_info(self) -> dict[str, Any]:
        """Get info with hover-specific data."""
        info = super()._get_info()
        info["success_steps"] = self._success_steps
        info["is_success"] = self._success_steps > 20
        return info


@dataclass
class SetpointWaypointConfig(SetpointEnvConfig):
    """Configuration for setpoint waypoint navigation."""

    # Waypoints
    num_waypoints: int = 5
    waypoint_radius: float = 5.0  # max distance from origin
    waypoint_threshold: float = 0.5  # meters to reach waypoint

    # Reward
    progress_reward: float = 1.0
    waypoint_bonus: float = 5.0

    # Difficulty
    randomize_order: bool = True


class SetpointWaypointEnv(SetpointBaseEnv):
    """Waypoint navigation using setpoint control.

    The agent navigates through a sequence of waypoints by outputting
    velocity setpoints.
    """

    def __init__(
        self,
        config: SetpointWaypointConfig | None = None,
        render_mode: str | None = None,
    ):
        """Initialize waypoint environment.

        Args:
            config: Environment configuration
            render_mode: Rendering mode
        """
        self.wp_config = config or SetpointWaypointConfig()
        super().__init__(config=self.wp_config, render_mode=render_mode)

        self._waypoints = []
        self._current_waypoint_idx = 0
        self._waypoints_reached = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)

        # Generate waypoints
        self._generate_waypoints()
        self._current_waypoint_idx = 0
        self._waypoints_reached = 0

        # Set first waypoint as target
        self._target_position = self._waypoints[0].copy()

        # Update observation with new target
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _generate_waypoints(self) -> None:
        """Generate random waypoints."""
        self._waypoints = []
        r = self.wp_config.waypoint_radius

        for _ in range(self.wp_config.num_waypoints):
            wp = np.array(
                [
                    self._np_random.uniform(-r, r),
                    self._np_random.uniform(-r, r),
                    self._np_random.uniform(1.0, 4.0),
                ]
            )
            self._waypoints.append(wp)

    def _get_target(self) -> np.ndarray:
        """Get current waypoint target."""
        if len(self._waypoints) > 0:
            return self._waypoints[self._current_waypoint_idx].copy()
        return np.array([0.0, 0.0, 1.5])

    def _compute_reward(self) -> float:
        """Compute waypoint navigation reward."""
        state = self.sim.get_state()
        cfg = self.wp_config

        # Distance to current waypoint
        pos_error = np.linalg.norm(state.position - self._target_position)

        # Progress reward (distance reduction)
        progress_reward = cfg.progress_reward * np.exp(-pos_error)

        # Waypoint reached bonus
        waypoint_bonus = 0.0
        if pos_error < cfg.waypoint_threshold:
            waypoint_bonus = cfg.waypoint_bonus
            self._waypoints_reached += 1

            # Move to next waypoint
            self._current_waypoint_idx += 1
            if self._current_waypoint_idx < len(self._waypoints):
                self._target_position = self._waypoints[self._current_waypoint_idx].copy()

        # Penalties
        velocity_penalty = 0.05 * max(0, np.linalg.norm(state.velocity) - 3.0)
        action_penalty = 0.02 * np.linalg.norm(self._previous_action)

        # Alive bonus
        alive_bonus = 0.05

        reward = progress_reward + waypoint_bonus - velocity_penalty - action_penalty + alive_bonus

        return reward

    def _is_terminated(self) -> bool:
        """Check termination."""
        # All waypoints reached
        if self._current_waypoint_idx >= len(self._waypoints):
            return True

        # Other termination conditions from base
        return super()._is_terminated()

    def _get_info(self) -> dict[str, Any]:
        """Get info with waypoint data."""
        info = super()._get_info()
        info["current_waypoint"] = self._current_waypoint_idx
        info["waypoints_reached"] = self._waypoints_reached
        info["total_waypoints"] = len(self._waypoints)
        info["is_success"] = self._current_waypoint_idx >= len(self._waypoints)
        return info


@dataclass
class SetpointTrackingConfig(SetpointEnvConfig):
    """Configuration for trajectory tracking."""

    # Trajectory
    trajectory_type: str = "circle"  # circle, figure8, lissajous
    trajectory_radius: float = 2.0
    trajectory_speed: float = 0.5  # m/s along trajectory
    trajectory_altitude: float = 2.0

    # Tracking
    lookahead_distance: float = 0.5  # meters


class SetpointTrackingEnv(SetpointBaseEnv):
    """Trajectory tracking using setpoint control.

    The agent must follow a moving reference point along a trajectory.
    """

    def __init__(
        self,
        config: SetpointTrackingConfig | None = None,
        render_mode: str | None = None,
    ):
        """Initialize tracking environment.

        Args:
            config: Environment configuration
            render_mode: Rendering mode
        """
        self.track_config = config or SetpointTrackingConfig()
        super().__init__(config=self.track_config, render_mode=render_mode)

        self._trajectory_phase = 0.0
        self._total_tracking_error = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        self._trajectory_phase = 0.0
        self._total_tracking_error = 0.0

        # Set initial target
        self._target_position = self._get_trajectory_point(0.0)
        obs = self._get_observation()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step with trajectory update."""
        # Update trajectory phase
        dt = 1.0 / self.config.control_frequency
        speed = self.track_config.trajectory_speed
        self._trajectory_phase += speed * dt / self.track_config.trajectory_radius

        # Update target
        self._target_position = self._get_trajectory_point(self._trajectory_phase)

        return super().step(action)

    def _get_trajectory_point(self, phase: float) -> np.ndarray:
        """Get point on trajectory at given phase."""
        cfg = self.track_config
        r = cfg.trajectory_radius
        z = cfg.trajectory_altitude

        if cfg.trajectory_type == "circle":
            x = r * np.cos(phase)
            y = r * np.sin(phase)

        elif cfg.trajectory_type == "figure8":
            x = r * np.sin(phase)
            y = r * np.sin(2 * phase) / 2

        elif cfg.trajectory_type == "lissajous":
            x = r * np.sin(phase)
            y = r * np.sin(1.5 * phase + np.pi / 4)

        else:
            x = r * np.cos(phase)
            y = r * np.sin(phase)

        return np.array([x, y, z])

    def _get_target(self) -> np.ndarray:
        """Get current trajectory target."""
        return self._get_trajectory_point(self._trajectory_phase)

    def _compute_reward(self) -> float:
        """Compute tracking reward."""
        state = self.sim.get_state()

        # Tracking error
        tracking_error = np.linalg.norm(state.position - self._target_position)
        self._total_tracking_error += tracking_error

        # Reward for staying close to trajectory
        tracking_reward = np.exp(-3.0 * tracking_error)

        # Velocity should roughly match trajectory velocity
        # (simplified: just penalize very high speeds)
        vel_penalty = 0.05 * max(0, np.linalg.norm(state.velocity) - 2.0)

        # Smoothness
        action_penalty = 0.02 * np.linalg.norm(self._previous_action)

        return tracking_reward - vel_penalty - action_penalty + 0.1

    def _get_info(self) -> dict[str, Any]:
        """Get info with tracking data."""
        info = super()._get_info()
        info["trajectory_phase"] = self._trajectory_phase
        info["mean_tracking_error"] = self._total_tracking_error / max(1, self._step_count)
        return info


# Register environments for gymnasium.make()
def register_setpoint_envs():
    """Register setpoint environments with gymnasium."""
    import gymnasium as gym

    gym.register(
        id="DroneSetpointHover-v0",
        entry_point="src.environments.setpoint_env:SetpointHoverEnv",
    )

    gym.register(
        id="DroneSetpointWaypoint-v0",
        entry_point="src.environments.setpoint_env:SetpointWaypointEnv",
    )

    gym.register(
        id="DroneSetpointTracking-v0",
        entry_point="src.environments.setpoint_env:SetpointTrackingEnv",
    )
