"""Base Gymnasium environment for setpoint-based drone control.

This environment uses position/velocity setpoints as actions instead of
direct motor commands. Designed for training neural networks that will
control the drone through PX4 offboard mode.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
from gymnasium import spaces

from ..core.mujoco_sim import MuJoCoSimulator, QuadrotorState, create_simulator
from ..utils.transforms import CoordinateTransforms
from ..utils.rotations import Rotations


class SetpointMode(Enum):
    """Action modes for setpoint environment."""
    VELOCITY = "velocity"  # [vx, vy, vz, yaw_rate]
    POSITION_DELTA = "position_delta"  # [dx, dy, dz, dyaw]
    POSITION_ABSOLUTE = "position_absolute"  # [x, y, z, yaw]


@dataclass
class SetpointEnvConfig:
    """Configuration for setpoint-based environment."""
    
    # Simulation
    model: str = "generic"
    physics_timestep: float = 0.002  # 500 Hz
    control_frequency: float = 50.0  # 50 Hz control (20ms between actions)
    max_episode_steps: int = 500  # 10 seconds at 50Hz
    
    # Action mode
    action_mode: SetpointMode = SetpointMode.VELOCITY
    
    # Action limits
    max_velocity: float = 5.0  # m/s
    max_position_delta: float = 2.0  # m per step
    max_yaw_rate: float = 1.0  # rad/s
    max_altitude: float = 50.0  # m
    min_altitude: float = 0.1  # m
    
    # Simulated position controller gains
    pos_p_gain: float = 3.0  # Position P gain
    vel_p_gain: float = 5.0  # Velocity P gain
    vel_i_gain: float = 0.5  # Velocity I gain
    
    # Initial state randomization
    init_position_range: Tuple[float, float, float] = (0.5, 0.5, 0.2)
    init_velocity_range: float = 0.3
    init_angle_range: float = 0.1
    
    # Observation noise
    position_noise: float = 0.01
    velocity_noise: float = 0.01
    
    # Termination conditions
    max_tilt_angle: float = 1.2  # radians (~70 degrees)
    max_position_error: float = 15.0  # meters from target
    
    # Use body frame for velocity commands
    use_body_frame: bool = False


class SetpointBaseEnv(gym.Env):
    """Base Gymnasium environment for setpoint-based control.
    
    Instead of outputting motor commands, the agent outputs velocity or
    position setpoints. A simulated position controller converts these
    to motor commands for the physics simulation.
    
    This architecture matches how a neural network would control a real
    drone through PX4's offboard mode.
    
    Observation space (18 dims):
        - position (3): World frame position [x, y, z]
        - velocity (3): World frame velocity [vx, vy, vz]
        - orientation (3): Roll, pitch, yaw [rad]
        - angular_velocity (3): Body frame angular rates [p, q, r]
        - target_position (3): Goal position [x, y, z]
        - previous_action (3-4): Last setpoint command
    
    Action space (4 dims for velocity mode):
        - [vx, vy, vz, yaw_rate] normalized to [-1, 1]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[SetpointEnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize setpoint environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config or SetpointEnvConfig()
        self.render_mode = render_mode
        
        # Create simulator
        self.sim = create_simulator(model=self.config.model)
        
        # Steps between control updates
        self._physics_steps_per_control = int(
            1.0 / (self.config.control_frequency * self.config.physics_timestep)
        )
        
        # State tracking
        self._step_count = 0
        self._previous_action = np.zeros(4)
        self._target_position = np.array([0.0, 0.0, 1.0])
        self._episode_reward = 0.0
        
        # Position controller state
        self._velocity_integral = np.zeros(3)
        self._current_setpoint_pos = np.zeros(3)
        self._current_setpoint_vel = np.zeros(3)
        self._current_yaw_setpoint = 0.0
        
        # RNG
        self._np_random: Optional[np.random.Generator] = None
        
        # Define spaces
        self._define_spaces()
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: position, velocity, euler angles, angular velocity, target, prev action
        # Total: 3 + 3 + 3 + 3 + 3 + 4 = 19 dims
        obs_dim = 19
        
        obs_low = np.array([
            -100, -100, 0,      # position (z >= 0)
            -20, -20, -20,      # velocity
            -np.pi, -np.pi/2, -np.pi,  # euler angles (roll, pitch, yaw)
            -10, -10, -10,      # angular velocity
            -100, -100, 0,      # target position
            -1, -1, -1, -1,     # previous action
        ], dtype=np.float32)
        
        obs_high = np.array([
            100, 100, 100,      # position
            20, 20, 20,         # velocity
            np.pi, np.pi/2, np.pi,  # euler angles
            10, 10, 10,         # angular velocity
            100, 100, 100,      # target position
            1, 1, 1, 1,         # previous action
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )
        
        # Action space: normalized setpoint commands [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
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
        """Reset environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        
        # Sample initial state
        init_pos, init_vel, init_quat, init_omega = self._sample_initial_state()
        
        # Reset simulator
        self.sim.reset(
            position=init_pos,
            velocity=init_vel,
            quaternion=init_quat,
            angular_velocity=init_omega,
        )
        
        # Reset tracking
        self._step_count = 0
        self._previous_action = np.zeros(4)
        self._episode_reward = 0.0
        
        # Reset controller state
        self._velocity_integral = np.zeros(3)
        self._current_setpoint_pos = init_pos.copy()
        self._current_setpoint_vel = np.zeros(3)
        self._current_yaw_setpoint = 0.0
        
        # Set target
        if options is not None and "target_position" in options:
            self._target_position = np.array(options["target_position"])
        else:
            self._target_position = self._get_target()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.
        
        Args:
            action: Normalized setpoint command [-1, 1]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Convert action to setpoint
        setpoint_vel, yaw_rate = self._action_to_setpoint(action)
        
        # Run physics simulation with position controller
        for _ in range(self._physics_steps_per_control):
            motor_commands = self._position_controller(setpoint_vel, yaw_rate)
            self.sim.step(motor_commands)
        
        # Update state
        self._step_count += 1
        self._previous_action = action.copy()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        self._episode_reward += reward
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self._step_count >= self.config.max_episode_steps
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _action_to_setpoint(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Convert normalized action to velocity setpoint.
        
        Args:
            action: Normalized action [-1, 1]
            
        Returns:
            Tuple of (velocity_setpoint, yaw_rate)
        """
        if self.config.action_mode == SetpointMode.VELOCITY:
            # Direct velocity command
            velocity = action[:3] * self.config.max_velocity
            yaw_rate = action[3] * self.config.max_yaw_rate
            
            self._current_setpoint_vel = velocity
            return velocity, yaw_rate
        
        elif self.config.action_mode == SetpointMode.POSITION_DELTA:
            # Relative position change
            delta = action[:3] * self.config.max_position_delta
            delta_yaw = action[3] * self.config.max_yaw_rate
            
            # Update position setpoint
            self._current_setpoint_pos += delta
            
            # Clip altitude
            self._current_setpoint_pos[2] = np.clip(
                self._current_setpoint_pos[2],
                self.config.min_altitude,
                self.config.max_altitude,
            )
            
            # Convert to velocity command (position controller)
            state = self.sim.get_state()
            pos_error = self._current_setpoint_pos - state.position
            velocity = pos_error * self.config.pos_p_gain
            velocity = np.clip(velocity, -self.config.max_velocity, self.config.max_velocity)
            
            self._current_setpoint_vel = velocity
            return velocity, delta_yaw
        
        else:  # POSITION_ABSOLUTE
            # Absolute position target
            position = action[:3] * np.array([10.0, 10.0, 5.0])  # Scale to reasonable range
            position[2] = np.clip(position[2], self.config.min_altitude, self.config.max_altitude)
            yaw = action[3] * np.pi
            
            self._current_setpoint_pos = position
            self._current_yaw_setpoint = yaw
            
            # Convert to velocity
            state = self.sim.get_state()
            pos_error = position - state.position
            velocity = pos_error * self.config.pos_p_gain
            velocity = np.clip(velocity, -self.config.max_velocity, self.config.max_velocity)
            
            return velocity, 0.0
    
    def _position_controller(
        self,
        target_velocity: np.ndarray,
        yaw_rate: float,
    ) -> np.ndarray:
        """Simple position/velocity controller.
        
        Converts velocity setpoint to motor commands.
        This simulates PX4's position controller behavior.
        
        Args:
            target_velocity: Desired velocity [m/s]
            yaw_rate: Desired yaw rate [rad/s]
            
        Returns:
            Motor commands [0, 1]
        """
        state = self.sim.get_state()
        dt = self.config.physics_timestep
        
        # Velocity error
        vel_error = target_velocity - state.velocity
        
        # Velocity controller (PI)
        self._velocity_integral += vel_error * dt
        self._velocity_integral = np.clip(self._velocity_integral, -2.0, 2.0)
        
        # Desired acceleration
        accel_cmd = (
            vel_error * self.config.vel_p_gain +
            self._velocity_integral * self.config.vel_i_gain
        )
        
        # Convert to thrust and attitude
        # Simplified: assume we want to tilt to achieve horizontal acceleration
        # and use collective thrust for vertical
        
        # Get current attitude
        euler = Rotations.quaternion_to_euler(state.quaternion)
        roll, pitch, yaw = euler
        
        # Desired roll/pitch from horizontal acceleration
        # Simplified dynamics: accel_x ~ pitch, accel_y ~ -roll
        pitch_cmd = np.clip(accel_cmd[0] / 10.0, -0.5, 0.5)
        roll_cmd = np.clip(-accel_cmd[1] / 10.0, -0.5, 0.5)
        
        # Thrust for vertical (gravity compensation + z control)
        gravity_comp = 0.5  # Approximate hover thrust
        z_thrust = gravity_comp + accel_cmd[2] / 20.0
        z_thrust = np.clip(z_thrust, 0.1, 0.9)
        
        # Simple mixer: convert to motor commands
        # Attitude errors
        roll_error = roll_cmd - roll
        pitch_error = pitch_cmd - pitch
        yaw_error = yaw_rate - state.angular_velocity[2]
        
        # PD attitude control
        roll_torque = roll_error * 2.0 - state.angular_velocity[0] * 0.3
        pitch_torque = pitch_error * 2.0 - state.angular_velocity[1] * 0.3
        yaw_torque = yaw_error * 1.0
        
        # Motor mixing (X configuration)
        m1 = z_thrust + roll_torque - pitch_torque + yaw_torque  # Front-right
        m2 = z_thrust - roll_torque - pitch_torque - yaw_torque  # Front-left
        m3 = z_thrust - roll_torque + pitch_torque + yaw_torque  # Back-left
        m4 = z_thrust + roll_torque + pitch_torque - yaw_torque  # Back-right
        
        motors = np.array([m1, m2, m3, m4])
        motors = np.clip(motors, 0.0, 1.0)
        
        return motors
    
    def _sample_initial_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample random initial state.
        
        Returns:
            Tuple of (position, velocity, quaternion, angular_velocity)
        """
        # Position
        pos_range = self.config.init_position_range
        position = np.array([
            self._np_random.uniform(-pos_range[0], pos_range[0]),
            self._np_random.uniform(-pos_range[1], pos_range[1]),
            self._np_random.uniform(0.5, 0.5 + pos_range[2]),
        ])
        
        # Velocity
        velocity = self._np_random.uniform(
            -self.config.init_velocity_range,
            self.config.init_velocity_range,
            size=3,
        )
        
        # Attitude (small random angles)
        roll = self._np_random.uniform(-self.config.init_angle_range, self.config.init_angle_range)
        pitch = self._np_random.uniform(-self.config.init_angle_range, self.config.init_angle_range)
        yaw = self._np_random.uniform(-np.pi, np.pi)
        quaternion = Rotations.euler_to_quaternion(roll, pitch, yaw)
        
        # Angular velocity
        angular_velocity = self._np_random.uniform(-0.1, 0.1, size=3)
        
        return position, velocity, quaternion, angular_velocity
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation.
        
        Returns:
            Observation array
        """
        state = self.sim.get_state()
        
        # Convert quaternion to euler
        euler = np.array(Rotations.quaternion_to_euler(state.quaternion))
        
        # Add observation noise
        position = state.position + self._np_random.normal(0, self.config.position_noise, 3)
        velocity = state.velocity + self._np_random.normal(0, self.config.velocity_noise, 3)
        
        obs = np.concatenate([
            position,
            velocity,
            euler,
            state.angular_velocity,
            self._target_position,
            self._previous_action,
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary.
        
        Returns:
            Info dict
        """
        state = self.sim.get_state()
        euler = Rotations.quaternion_to_euler(state.quaternion)
        
        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "euler_angles": np.array(euler),
            "target_position": self._target_position.copy(),
            "position_error": np.linalg.norm(state.position - self._target_position),
            "step_count": self._step_count,
            "episode_reward": self._episode_reward,
            "setpoint_velocity": self._current_setpoint_vel.copy(),
        }
    
    def _get_target(self) -> np.ndarray:
        """Get target position for current episode.
        
        Override in subclasses for task-specific targets.
        
        Returns:
            Target position [x, y, z]
        """
        # Default: random hover target
        return np.array([
            self._np_random.uniform(-3, 3),
            self._np_random.uniform(-3, 3),
            self._np_random.uniform(1, 3),
        ])
    
    def _compute_reward(self) -> float:
        """Compute reward for current step.
        
        Override in subclasses for task-specific rewards.
        
        Returns:
            Reward value
        """
        state = self.sim.get_state()
        
        # Position error reward
        pos_error = np.linalg.norm(state.position - self._target_position)
        position_reward = np.exp(-pos_error)
        
        # Velocity penalty
        vel_penalty = 0.01 * np.linalg.norm(state.velocity)
        
        # Action smoothness penalty
        action_penalty = 0.01 * np.linalg.norm(self._previous_action)
        
        # Alive bonus
        alive_bonus = 0.1
        
        reward = position_reward - vel_penalty - action_penalty + alive_bonus
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate.
        
        Returns:
            True if terminated
        """
        state = self.sim.get_state()
        
        # Ground crash
        if state.position[2] < 0.05:
            return True
        
        # Too high
        if state.position[2] > self.config.max_altitude:
            return True
        
        # Too tilted
        euler = Rotations.quaternion_to_euler(state.quaternion)
        if abs(euler[0]) > self.config.max_tilt_angle or abs(euler[1]) > self.config.max_tilt_angle:
            return True
        
        # Too far from target
        pos_error = np.linalg.norm(state.position - self._target_position)
        if pos_error > self.config.max_position_error:
            return True
        
        return False
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Return image array
            return None  # TODO: Implement offscreen rendering
        elif self.render_mode == "human":
            # Interactive viewing handled externally
            pass
    
    def close(self):
        """Clean up environment."""
        pass
