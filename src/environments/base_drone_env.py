"""Base Gymnasium environment for drone simulation."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, SupportsFloat
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from ..core.mujoco_sim import MuJoCoSimulator, QuadrotorState, create_simulator
from ..core.quadrotor import QuadrotorDynamics, QuadrotorConfig
from ..utils.transforms import CoordinateTransforms
from ..utils.rotations import Rotations


@dataclass
class DroneEnvConfig:
    """Configuration for drone environment."""
    
    # Simulation
    model: str = "generic"
    physics_timestep: float = 0.002  # 500 Hz
    control_frequency: float = 50.0  # 50 Hz control
    max_episode_steps: int = 1000
    
    # Initial state randomization
    init_position_range: Tuple[float, float, float] = (0.5, 0.5, 0.2)  # meters
    init_velocity_range: float = 0.5  # m/s
    init_angle_range: float = 0.1  # radians
    init_angular_velocity_range: float = 0.1  # rad/s
    
    # Observation noise
    position_noise: float = 0.01
    velocity_noise: float = 0.01
    attitude_noise: float = 0.005
    angular_velocity_noise: float = 0.01
    
    # Termination conditions
    max_position_error: float = 10.0  # meters
    max_velocity: float = 20.0  # m/s
    max_tilt_angle: float = 1.2  # radians (~70 degrees)
    min_altitude: float = 0.0  # meters (ground)
    max_altitude: float = 50.0  # meters
    
    # Domain randomization
    randomize_mass: bool = False
    mass_range: Tuple[float, float] = (0.9, 1.1)
    randomize_inertia: bool = False
    inertia_range: Tuple[float, float] = (0.9, 1.1)


class BaseDroneEnv(gym.Env):
    """Base Gymnasium environment for quadrotor control.
    
    Observation space:
        - position (3): World frame position [x, y, z]
        - velocity (3): World frame velocity [vx, vy, vz]
        - quaternion (4): Orientation [w, x, y, z]
        - angular_velocity (3): Body frame angular velocity [p, q, r]
        - target_position (3): Goal position [x, y, z]
        - previous_action (4): Last motor commands
        Total: 20 dimensions
    
    Action space:
        - motor_commands (4): Normalized motor commands [0, 1]
    
    Subclasses should implement:
        - _compute_reward(): Task-specific reward function
        - _get_target(): Target/goal for the task
        - _is_success(): Success condition
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[DroneEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize drone environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.config = config or DroneEnvConfig()
        self.render_mode = render_mode
        
        # Create simulator
        self.sim = create_simulator(model=self.config.model)
        
        # Dynamics helper
        self.dynamics = QuadrotorDynamics(QuadrotorConfig())
        
        # Steps between control updates
        self._physics_steps_per_control = int(
            1.0 / (self.config.control_frequency * self.config.physics_timestep)
        )
        
        # State tracking
        self._step_count = 0
        self._previous_action = np.zeros(4)
        self._target_position = np.array([0.0, 0.0, 1.0])  # Default hover at 1m
        self._episode_reward = 0.0
        
        # Random number generator
        self._np_random: Optional[np.random.Generator] = None
        
        # Define observation and action spaces
        self._define_spaces()
        
        # Rendering
        self._renderer = None
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation space bounds
        obs_low = np.array([
            -100, -100, -100,  # position
            -50, -50, -50,     # velocity
            -1, -1, -1, -1,    # quaternion
            -20, -20, -20,     # angular velocity
            -100, -100, -100,  # target position
            0, 0, 0, 0,        # previous action
        ], dtype=np.float32)
        
        obs_high = np.array([
            100, 100, 100,     # position
            50, 50, 50,        # velocity
            1, 1, 1, 1,        # quaternion
            20, 20, 20,        # angular velocity
            100, 100, 100,     # target position
            1, 1, 1, 1,        # previous action
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )
        
        # Action space: normalized motor commands [0, 1]
        self.action_space = spaces.Box(
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
        """Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (e.g., "target_position")
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Initialize RNG
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        
        # Generate initial state
        init_pos, init_vel, init_quat, init_omega = self._sample_initial_state()
        
        # Reset simulator
        self.sim.reset(
            position=init_pos,
            velocity=init_vel,
            quaternion=init_quat,
            angular_velocity=init_omega,
        )
        
        # Reset tracking variables
        self._step_count = 0
        self._previous_action = np.zeros(4)
        self._episode_reward = 0.0
        
        # Set target
        if options is not None and "target_position" in options:
            self._target_position = np.array(options["target_position"])
        else:
            self._target_position = self._get_target()
        
        # Apply domain randomization if enabled
        if self.config.randomize_mass:
            self._randomize_dynamics()
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Motor commands [0, 1], shape (4,)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        
        # Step physics multiple times at higher rate
        for _ in range(self._physics_steps_per_control):
            self.sim.step(action)
        
        self._step_count += 1
        self._previous_action = action.copy()
        
        # Get current state
        state = self.sim.get_state()
        
        # Check termination conditions
        terminated = self._check_termination(state)
        truncated = self._step_count >= self.config.max_episode_steps
        
        # Compute reward
        reward = self._compute_reward(state, action)
        self._episode_reward += reward
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["episode_length"] = self._step_count
        
        return obs, reward, terminated, truncated, info
    
    def _sample_initial_state(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample random initial state.
        
        Returns:
            Tuple of (position, velocity, quaternion, angular_velocity)
        """
        rng = self._np_random
        cfg = self.config
        
        # Random position around origin
        position = rng.uniform(
            -np.array(cfg.init_position_range),
            np.array(cfg.init_position_range),
        )
        position[2] = abs(position[2]) + 0.5  # Ensure above ground (Z-up in MuJoCo)
        
        # Random velocity
        velocity = rng.uniform(
            -cfg.init_velocity_range,
            cfg.init_velocity_range,
            size=3,
        )
        
        # Random small tilt
        roll = rng.uniform(-cfg.init_angle_range, cfg.init_angle_range)
        pitch = rng.uniform(-cfg.init_angle_range, cfg.init_angle_range)
        yaw = rng.uniform(-np.pi, np.pi)  # Full yaw range
        
        quaternion = Rotations.euler_to_quaternion(roll, pitch, yaw)
        
        # Random angular velocity
        angular_velocity = rng.uniform(
            -cfg.init_angular_velocity_range,
            cfg.init_angular_velocity_range,
            size=3,
        )
        
        return position, velocity, quaternion, angular_velocity
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation array
        """
        state = self.sim.get_state()
        cfg = self.config
        rng = self._np_random
        
        # Add observation noise
        position = state.position + rng.normal(0, cfg.position_noise, 3)
        velocity = state.velocity + rng.normal(0, cfg.velocity_noise, 3)
        quaternion = Rotations.quaternion_normalize(
            state.quaternion + rng.normal(0, cfg.attitude_noise, 4)
        )
        angular_velocity = state.angular_velocity + rng.normal(
            0, cfg.angular_velocity_noise, 3
        )
        
        # Construct observation
        obs = np.concatenate([
            position,
            velocity,
            quaternion,
            angular_velocity,
            self._target_position,
            self._previous_action,
        ]).astype(np.float32)
        
        return obs
    
    def _check_termination(self, state: QuadrotorState) -> bool:
        """Check if episode should terminate.
        
        Args:
            state: Current quadrotor state
            
        Returns:
            True if episode should terminate
        """
        cfg = self.config
        
        # Position error too large
        position_error = np.linalg.norm(state.position - self._target_position)
        if position_error > cfg.max_position_error:
            return True
        
        # Velocity too high
        velocity_magnitude = np.linalg.norm(state.velocity)
        if velocity_magnitude > cfg.max_velocity:
            return True
        
        # Tilt angle too large
        roll, pitch, _ = Rotations.quaternion_to_euler(state.quaternion)
        tilt_angle = np.sqrt(roll**2 + pitch**2)
        if tilt_angle > cfg.max_tilt_angle:
            return True
        
        # Below ground (MuJoCo Z-up)
        if state.position[2] < cfg.min_altitude:
            return True
        
        # Too high
        if state.position[2] > cfg.max_altitude:
            return True
        
        return False
    
    def _compute_reward(self, state: QuadrotorState, action: np.ndarray) -> float:
        """Compute reward for current state.
        
        Override in subclasses for task-specific rewards.
        
        Args:
            state: Current quadrotor state
            action: Action taken
            
        Returns:
            Reward value
        """
        # Base reward: negative distance to target
        position_error = np.linalg.norm(state.position - self._target_position)
        reward = -position_error
        
        return reward
    
    def _get_target(self) -> np.ndarray:
        """Get target position for the task.
        
        Override in subclasses for different tasks.
        
        Returns:
            Target position [x, y, z]
        """
        return np.array([0.0, 0.0, 1.0])  # Default: hover at 1m
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary.
        
        Returns:
            Info dictionary with additional state information
        """
        state = self.sim.get_state()
        
        position_error = np.linalg.norm(state.position - self._target_position)
        roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)
        
        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "position_error": position_error,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "angular_velocity": state.angular_velocity.copy(),
            "target_position": self._target_position.copy(),
            "step_count": self._step_count,
            "is_success": self._is_success(state),
        }
    
    def _is_success(self, state: QuadrotorState) -> bool:
        """Check if task is successfully completed.
        
        Override in subclasses.
        
        Args:
            state: Current state
            
        Returns:
            True if task is successful
        """
        # Default: within 0.1m of target
        position_error = np.linalg.norm(state.position - self._target_position)
        return position_error < 0.1
    
    def _randomize_dynamics(self) -> None:
        """Apply domain randomization to dynamics."""
        cfg = self.config
        rng = self._np_random
        
        if cfg.randomize_mass:
            mass_scale = rng.uniform(cfg.mass_range[0], cfg.mass_range[1])
            # Note: MuJoCo model mass is read-only after loading
            # This would require model modification
            pass
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        if self.render_mode is None:
            return None
        
        if self._renderer is None:
            self._renderer = self.sim.create_renderer(width=640, height=480)
        
        pixels = self.sim.render(self._renderer)
        
        if self.render_mode == "rgb_array":
            return pixels
        elif self.render_mode == "human":
            # Would need to display with cv2/matplotlib/etc
            pass
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None


# Register environment with Gymnasium
gym.register(
    id="DroneHover-v0",
    entry_point="src.environments.hover_env:HoverEnv",
)

gym.register(
    id="DroneWaypoint-v0",
    entry_point="src.environments.waypoint_env:WaypointEnv",
)

gym.register(
    id="DroneTrajectory-v0",
    entry_point="src.environments.trajectory_env:TrajectoryEnv",
)
