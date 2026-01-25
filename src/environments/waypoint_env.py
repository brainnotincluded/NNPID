"""Waypoint navigation environment."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

from .base_drone_env import BaseDroneEnv, DroneEnvConfig
from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations


@dataclass
class WaypointEnvConfig(DroneEnvConfig):
    """Configuration for waypoint environment."""
    
    # Waypoint task specific
    num_waypoints: int = 5
    waypoint_range: float = 5.0  # meters from origin
    waypoint_height_range: Tuple[float, float] = (0.5, 3.0)
    waypoint_reached_threshold: float = 0.3  # meters
    
    # Reward weights
    progress_reward_weight: float = 1.0
    waypoint_reached_bonus: float = 10.0
    all_waypoints_bonus: float = 50.0
    velocity_penalty_weight: float = 0.05
    action_penalty_weight: float = 0.01
    time_penalty: float = 0.01
    crash_penalty: float = -100.0


class WaypointEnv(BaseDroneEnv):
    """Gymnasium environment for waypoint navigation.
    
    The goal is to visit a sequence of waypoints as quickly as possible.
    """
    
    def __init__(
        self,
        config: Optional[WaypointEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize waypoint environment."""
        self.waypoint_config = config or WaypointEnvConfig()
        super().__init__(config=self.waypoint_config, render_mode=render_mode)
        
        # Waypoint tracking
        self._waypoints: List[np.ndarray] = []
        self._current_waypoint_idx = 0
        self._waypoints_reached = 0
        self._previous_distance = 0.0
    
    def _define_spaces(self) -> None:
        """Define observation space with waypoint info."""
        # Extended observation: base + current waypoint + waypoints remaining
        obs_dim = 20 + 3 + 1  # base obs + next waypoint + num remaining
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Same action space
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment with new waypoints."""
        # Generate waypoints before parent reset
        self._generate_waypoints()
        self._current_waypoint_idx = 0
        self._waypoints_reached = 0
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize distance tracking
        state = self.sim.get_state()
        self._previous_distance = np.linalg.norm(
            state.position - self._waypoints[0]
        )
        
        return self._get_observation(), info
    
    def _generate_waypoints(self) -> None:
        """Generate random waypoints."""
        cfg = self.waypoint_config
        
        self._waypoints = []
        for _ in range(cfg.num_waypoints):
            x = self._np_random.uniform(-cfg.waypoint_range, cfg.waypoint_range)
            y = self._np_random.uniform(-cfg.waypoint_range, cfg.waypoint_range)
            z = self._np_random.uniform(
                cfg.waypoint_height_range[0],
                cfg.waypoint_height_range[1],
            )
            self._waypoints.append(np.array([x, y, z]))
    
    def _get_target(self) -> np.ndarray:
        """Get current waypoint as target."""
        if self._current_waypoint_idx < len(self._waypoints):
            return self._waypoints[self._current_waypoint_idx]
        return self._waypoints[-1]  # Stay at last waypoint
    
    def _get_observation(self) -> np.ndarray:
        """Get observation with waypoint info."""
        state = self.sim.get_state()
        cfg = self.config
        rng = self._np_random
        
        # Base observation with noise
        position = state.position + rng.normal(0, cfg.position_noise, 3)
        velocity = state.velocity + rng.normal(0, cfg.velocity_noise, 3)
        quaternion = Rotations.quaternion_normalize(
            state.quaternion + rng.normal(0, cfg.attitude_noise, 4)
        )
        angular_velocity = state.angular_velocity + rng.normal(
            0, cfg.angular_velocity_noise, 3
        )
        
        # Current waypoint
        current_waypoint = self._get_target()
        
        # Number of remaining waypoints
        remaining = len(self._waypoints) - self._current_waypoint_idx
        
        # Construct observation
        obs = np.concatenate([
            position,
            velocity,
            quaternion,
            angular_velocity,
            current_waypoint,
            self._previous_action,
            current_waypoint,  # Repeat target for compatibility
            [remaining],
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, state: QuadrotorState, action: np.ndarray) -> float:
        """Compute waypoint navigation reward."""
        cfg = self.waypoint_config
        reward = 0.0
        
        current_waypoint = self._get_target()
        current_distance = np.linalg.norm(state.position - current_waypoint)
        
        # 1. Progress reward (getting closer to waypoint)
        progress = self._previous_distance - current_distance
        reward += cfg.progress_reward_weight * progress
        
        # 2. Check if waypoint reached
        if current_distance < cfg.waypoint_reached_threshold:
            reward += cfg.waypoint_reached_bonus
            self._waypoints_reached += 1
            self._current_waypoint_idx += 1
            
            # Check if all waypoints reached
            if self._current_waypoint_idx >= len(self._waypoints):
                reward += cfg.all_waypoints_bonus
            else:
                # Update distance for next waypoint
                current_distance = np.linalg.norm(
                    state.position - self._get_target()
                )
        
        # 3. Velocity penalty (excessive speed)
        max_velocity = 5.0
        velocity_magnitude = np.linalg.norm(state.velocity)
        if velocity_magnitude > max_velocity:
            reward -= cfg.velocity_penalty_weight * (velocity_magnitude - max_velocity)**2
        
        # 4. Action penalty
        action_magnitude = np.sum(action**2)
        reward -= cfg.action_penalty_weight * action_magnitude
        
        # 5. Time penalty (encourage speed)
        reward -= cfg.time_penalty
        
        self._previous_distance = current_distance
        
        return reward
    
    def _is_success(self, state: QuadrotorState) -> bool:
        """Check if all waypoints visited."""
        return self._current_waypoint_idx >= len(self._waypoints)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info with waypoint-specific data."""
        info = super()._get_info()
        
        info["waypoints_reached"] = self._waypoints_reached
        info["total_waypoints"] = len(self._waypoints)
        info["current_waypoint_idx"] = self._current_waypoint_idx
        info["all_waypoints_completed"] = self._is_success(self.sim.get_state())
        
        return info


# Need to import gym for registration
import gymnasium as gym
