"""Hover task environment - maintain position."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .base_drone_env import BaseDroneEnv, DroneEnvConfig
from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations


@dataclass
class HoverEnvConfig(DroneEnvConfig):
    """Configuration for hover environment."""
    
    # Hover task specific
    hover_position: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    randomize_hover_position: bool = True
    hover_position_range: float = 2.0  # meters
    hover_height_range: Tuple[float, float] = (0.5, 2.0)
    
    # Reward weights
    position_reward_weight: float = 1.0
    position_reward_scale: float = 2.0  # exp(-scale * error^2)
    velocity_penalty_weight: float = 0.1
    angular_velocity_penalty_weight: float = 0.05
    action_penalty_weight: float = 0.01
    action_rate_penalty_weight: float = 0.01
    orientation_penalty_weight: float = 0.1
    alive_bonus: float = 0.1
    success_bonus: float = 10.0
    crash_penalty: float = -100.0
    
    # Success threshold
    success_position_threshold: float = 0.1  # meters
    success_velocity_threshold: float = 0.1  # m/s


class HoverEnv(BaseDroneEnv):
    """Gymnasium environment for hover task.
    
    The goal is to maintain a stable hover at a target position.
    
    Reward shaping:
    - Exponential reward for proximity to target position
    - Penalties for high velocity, angular rates, and action magnitude
    - Bonus for being alive
    - Large penalty for crashing
    """
    
    def __init__(
        self,
        config: Optional[HoverEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize hover environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode
        """
        self.hover_config = config or HoverEnvConfig()
        super().__init__(config=self.hover_config, render_mode=render_mode)
        
        # Track previous action for rate penalty
        self._last_action = np.zeros(4)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        obs, info = super().reset(seed=seed, options=options)
        self._last_action = np.zeros(4)
        return obs, info
    
    def _get_target(self) -> np.ndarray:
        """Get hover target position.
        
        Returns:
            Target hover position
        """
        cfg = self.hover_config
        
        if cfg.randomize_hover_position:
            # Random position within range
            x = self._np_random.uniform(-cfg.hover_position_range, cfg.hover_position_range)
            y = self._np_random.uniform(-cfg.hover_position_range, cfg.hover_position_range)
            z = self._np_random.uniform(cfg.hover_height_range[0], cfg.hover_height_range[1])
            return np.array([x, y, z])
        else:
            return np.array(cfg.hover_position)
    
    def _compute_reward(self, state: QuadrotorState, action: np.ndarray) -> float:
        """Compute hover reward.
        
        Reward components:
        1. Position tracking (exponential)
        2. Velocity penalty (quadratic)
        3. Angular velocity penalty (quadratic)
        4. Action magnitude penalty
        5. Action rate penalty
        6. Orientation penalty (tilt)
        7. Alive bonus
        8. Crash/success bonuses
        
        Args:
            state: Current quadrotor state
            action: Motor commands
            
        Returns:
            Total reward
        """
        cfg = self.hover_config
        reward = 0.0
        
        # 1. Position tracking reward (exponential)
        position_error = np.linalg.norm(state.position - self._target_position)
        position_reward = np.exp(-cfg.position_reward_scale * position_error**2)
        reward += cfg.position_reward_weight * position_reward
        
        # 2. Velocity penalty
        velocity_magnitude = np.linalg.norm(state.velocity)
        velocity_penalty = velocity_magnitude**2
        reward -= cfg.velocity_penalty_weight * velocity_penalty
        
        # 3. Angular velocity penalty
        angular_velocity_magnitude = np.linalg.norm(state.angular_velocity)
        angular_penalty = angular_velocity_magnitude**2
        reward -= cfg.angular_velocity_penalty_weight * angular_penalty
        
        # 4. Action magnitude penalty (prefer lower throttle)
        action_magnitude = np.sum(action**2)
        reward -= cfg.action_penalty_weight * action_magnitude
        
        # 5. Action rate penalty (smooth control)
        action_rate = np.sum((action - self._last_action)**2)
        reward -= cfg.action_rate_penalty_weight * action_rate
        
        # 6. Orientation penalty (prefer level flight)
        roll, pitch, _ = Rotations.quaternion_to_euler(state.quaternion)
        tilt_penalty = roll**2 + pitch**2
        reward -= cfg.orientation_penalty_weight * tilt_penalty
        
        # 7. Alive bonus
        reward += cfg.alive_bonus
        
        # 8. Success bonus
        if self._is_success(state):
            reward += cfg.success_bonus
        
        # Store action for next rate penalty calculation
        self._last_action = action.copy()
        
        return reward
    
    def _check_termination(self, state: QuadrotorState) -> bool:
        """Check termination with crash penalty.
        
        Args:
            state: Current state
            
        Returns:
            True if terminated
        """
        terminated = super()._check_termination(state)
        
        if terminated:
            # Apply crash penalty (added in reward computation next step)
            pass
        
        return terminated
    
    def _is_success(self, state: QuadrotorState) -> bool:
        """Check if hover is successful.
        
        Success: within position and velocity thresholds.
        
        Args:
            state: Current state
            
        Returns:
            True if hovering successfully
        """
        cfg = self.hover_config
        
        # Check position
        position_error = np.linalg.norm(state.position - self._target_position)
        if position_error > cfg.success_position_threshold:
            return False
        
        # Check velocity
        velocity_magnitude = np.linalg.norm(state.velocity)
        if velocity_magnitude > cfg.success_velocity_threshold:
            return False
        
        return True
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info with hover-specific data."""
        info = super()._get_info()
        
        state = self.sim.get_state()
        
        # Add hover-specific info
        info["hover_success"] = self._is_success(state)
        info["distance_to_target"] = np.linalg.norm(
            state.position - self._target_position
        )
        info["velocity_magnitude"] = np.linalg.norm(state.velocity)
        
        return info
