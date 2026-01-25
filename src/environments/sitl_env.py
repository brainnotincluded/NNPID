"""SITL-in-the-loop Gymnasium environment.

This environment runs the full loop:
NN action -> MAVLink setpoint -> PX4 SITL -> Motor commands -> MuJoCo physics

Requires PX4 SITL to be running externally.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from ..core.mujoco_sim import MuJoCoSimulator, QuadrotorState, create_simulator
from ..core.sensors import SensorSimulator, SensorConfig
from ..communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig
from ..communication.messages import (
    HILSensorMessage,
    HILGPSMessage,
    SetpointCommand,
)
from ..controllers.offboard_controller import OffboardController, OffboardConfig, OffboardControlMode
from ..utils.transforms import CoordinateTransforms, GPSTransforms
from ..utils.rotations import Rotations


@dataclass
class SITLEnvConfig:
    """Configuration for SITL environment."""
    
    # Simulation
    model: str = "generic"
    physics_timestep: float = 0.002  # 500 Hz
    control_frequency: float = 50.0  # 50 Hz NN control
    max_episode_steps: int = 500  # 10 seconds
    
    # MAVLink
    mavlink_host: str = "127.0.0.1"
    mavlink_port: int = 4560
    connection_timeout: float = 30.0
    
    # Control mode
    control_mode: OffboardControlMode = OffboardControlMode.VELOCITY
    
    # Action limits
    max_velocity: float = 5.0  # m/s
    max_yaw_rate: float = 1.0  # rad/s
    
    # GPS reference
    ref_lat: float = 47.397742  # Zurich
    ref_lon: float = 8.545594
    ref_alt: float = 488.0  # meters
    
    # Initial state
    init_altitude: float = 0.5  # Starting altitude
    
    # Termination
    max_altitude: float = 50.0
    min_altitude: float = 0.05
    max_tilt: float = 1.2  # radians
    max_position_error: float = 20.0
    
    # Reward shaping
    position_reward_scale: float = 2.0
    velocity_penalty_scale: float = 0.1
    action_penalty_scale: float = 0.05


class SITLEnv(gym.Env):
    """Gymnasium environment with PX4 SITL in the loop.
    
    This environment:
    1. Takes velocity/position setpoint actions from the NN
    2. Sends setpoints to PX4 via MAVLink offboard mode
    3. Receives motor commands from PX4
    4. Runs MuJoCo physics with those motor commands
    5. Sends sensor data back to PX4
    6. Returns observation and reward to the NN
    
    This provides the most realistic training environment as it includes
    PX4's actual control loops, state estimation, and timing.
    
    Requirements:
        - PX4 SITL must be running and configured for external simulation
        - Start PX4 with: make px4_sitl none_iris
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[SITLEnvConfig] = None,
        render_mode: Optional[str] = None,
        auto_connect: bool = False,
    ):
        """Initialize SITL environment.
        
        Args:
            config: Environment configuration
            render_mode: Rendering mode
            auto_connect: Whether to auto-connect to PX4 on init
        """
        super().__init__()
        
        self.config = config or SITLEnvConfig()
        self.render_mode = render_mode
        
        # Physics simulation
        self.sim = create_simulator(model=self.config.model)
        
        # Sensor simulation
        self.sensors = SensorSimulator(SensorConfig())
        
        # MAVLink connection
        mavlink_config = MAVLinkConfig(
            host=self.config.mavlink_host,
            port=self.config.mavlink_port,
            connection_timeout=self.config.connection_timeout,
        )
        
        # Offboard controller
        offboard_config = OffboardConfig(
            mode=self.config.control_mode,
            max_velocity=self.config.max_velocity,
            max_yaw_rate=self.config.max_yaw_rate,
        )
        self._offboard = OffboardController(
            config=offboard_config,
            mavlink_config=mavlink_config,
        )
        
        # State tracking
        self._step_count = 0
        self._previous_action = np.zeros(4)
        self._target_position = np.array([0.0, 0.0, 2.0])
        self._episode_reward = 0.0
        self._connected = False
        self._armed = False
        
        # Timing
        self._physics_steps_per_control = int(
            1.0 / (self.config.control_frequency * self.config.physics_timestep)
        )
        self._last_sensor_time = 0.0
        
        # RNG
        self._np_random: Optional[np.random.Generator] = None
        
        # Define spaces
        self._define_spaces()
        
        # Auto-connect if requested
        if auto_connect:
            self.connect()
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: position, velocity, euler, angular velocity, target, prev action
        obs_dim = 19
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action: velocity setpoint + yaw rate
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )
    
    def connect(self) -> bool:
        """Connect to PX4 SITL.
        
        Returns:
            True if connection successful
        """
        print("Connecting to PX4 SITL...")
        
        if self._offboard.connect(wait_for_px4=True):
            self._connected = True
            print("Connected to PX4 SITL")
            return True
        else:
            print("Failed to connect to PX4 SITL")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from PX4."""
        self._offboard.disconnect()
        self._connected = False
        self._armed = False
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment.
        
        This will:
        1. Reset MuJoCo simulation
        2. Disarm if armed
        3. Re-initialize offboard mode
        4. Arm the vehicle
        
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
        
        # Ensure connected
        if not self._connected:
            if not self.connect():
                raise RuntimeError("Cannot reset: not connected to PX4 SITL")
        
        # Disarm if armed
        if self._armed:
            self._offboard.disarm()
            self._armed = False
            time.sleep(0.5)
        
        # Reset MuJoCo simulation
        init_position = np.array([0.0, 0.0, self.config.init_altitude])
        self.sim.reset(position=init_position)
        self.sensors.reset()
        
        # Reset tracking
        self._step_count = 0
        self._previous_action = np.zeros(4)
        self._episode_reward = 0.0
        self._last_sensor_time = 0.0
        
        # Set target
        if options is not None and "target_position" in options:
            self._target_position = np.array(options["target_position"])
        else:
            self._target_position = self._get_target()
        
        # Send initial sensor data to PX4
        self._send_sensors()
        
        # Initialize offboard mode
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(init_position)
        if not self._offboard.initialize_offboard(pos_ned):
            print("Warning: Failed to initialize offboard mode")
        
        # Arm
        if not self._offboard.arm():
            print("Warning: Failed to arm")
        else:
            self._armed = True
        
        # Let things settle
        for _ in range(10):
            self._send_sensors()
            self._step_physics(np.zeros(4))
            time.sleep(0.02)
        
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
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        
        # Send action to PX4 as setpoint
        state = self.sim.get_state()
        self._offboard.send_nn_action(action, state)
        
        # Step physics with motor commands from PX4
        self._step_physics(action)
        
        # Update tracking
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
    
    def _step_physics(self, action: np.ndarray) -> None:
        """Step physics simulation and handle SITL communication.
        
        Args:
            action: Current action for context
        """
        bridge = self._offboard.bridge
        if bridge is None:
            return
        
        for i in range(self._physics_steps_per_control):
            # Get motor commands from PX4
            motors = bridge.get_motor_commands(timeout=0.001)
            if motors is None:
                motors = np.zeros(4)
            
            # Step MuJoCo
            self.sim.step(motors)
            
            # Send sensors at physics rate
            self._send_sensors()
            
            # Send heartbeat
            bridge.send_heartbeat()
    
    def _send_sensors(self) -> None:
        """Send sensor data to PX4."""
        bridge = self._offboard.bridge
        if bridge is None:
            return
        
        state = self.sim.get_state()
        timestamp = self.sim.get_time()
        
        # Get IMU data
        gyro, accel = self.sim.get_imu_data()
        
        # Convert to FRD/NED frames
        gyro_frd = CoordinateTransforms.angular_velocity_mujoco_to_frd(gyro)
        accel_frd = CoordinateTransforms.acceleration_mujoco_to_frd(accel)
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(state.velocity)
        
        # Add sensor noise
        imu_data = self.sensors.get_imu(gyro_frd, accel_frd, timestamp)
        
        # Send HIL_SENSOR
        sensor_msg = HILSensorMessage.from_sensor_data(
            time_sec=timestamp,
            gyro=imu_data.gyro,
            accel=imu_data.accel,
            mag=np.array([0.21, 0.0, 0.42]),
            pressure=101325.0,
            temperature=20.0,
            altitude=-pos_ned[2],
        )
        bridge.send_hil_sensor(sensor_msg)
        
        # Send GPS at lower rate
        if timestamp - self._last_sensor_time > 0.1:  # 10 Hz
            gps_data = self.sensors.get_gps(pos_ned, vel_ned, timestamp)
            gps_msg = HILGPSMessage.from_gps_data(
                time_sec=timestamp,
                lat=gps_data.latitude,
                lon=gps_data.longitude,
                alt=gps_data.altitude,
                vel_ned=np.array([
                    gps_data.velocity_north,
                    gps_data.velocity_east,
                    gps_data.velocity_down,
                ]),
            )
            bridge.send_hil_gps(gps_msg)
            self._last_sensor_time = timestamp
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        state = self.sim.get_state()
        euler = np.array(Rotations.quaternion_to_euler(state.quaternion))
        
        obs = np.concatenate([
            state.position,
            state.velocity,
            euler,
            state.angular_velocity,
            self._target_position,
            self._previous_action,
        ]).astype(np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        state = self.sim.get_state()
        
        return {
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "target_position": self._target_position.copy(),
            "position_error": np.linalg.norm(state.position - self._target_position),
            "step_count": self._step_count,
            "episode_reward": self._episode_reward,
            "connected": self._connected,
            "armed": self._armed,
        }
    
    def _get_target(self) -> np.ndarray:
        """Get random target position."""
        return np.array([
            self._np_random.uniform(-3, 3),
            self._np_random.uniform(-3, 3),
            self._np_random.uniform(1, 3),
        ])
    
    def _compute_reward(self) -> float:
        """Compute step reward."""
        state = self.sim.get_state()
        
        # Position tracking
        pos_error = np.linalg.norm(state.position - self._target_position)
        position_reward = np.exp(-self.config.position_reward_scale * pos_error)
        
        # Velocity penalty
        vel_penalty = self.config.velocity_penalty_scale * np.linalg.norm(state.velocity)
        
        # Action smoothness
        action_penalty = self.config.action_penalty_scale * np.linalg.norm(self._previous_action)
        
        # Alive bonus
        alive_bonus = 0.1
        
        return position_reward - vel_penalty - action_penalty + alive_bonus
    
    def _is_terminated(self) -> bool:
        """Check termination conditions."""
        state = self.sim.get_state()
        
        # Crash
        if state.position[2] < self.config.min_altitude:
            return True
        
        # Too high
        if state.position[2] > self.config.max_altitude:
            return True
        
        # Too tilted
        euler = Rotations.quaternion_to_euler(state.quaternion)
        if abs(euler[0]) > self.config.max_tilt or abs(euler[1]) > self.config.max_tilt:
            return True
        
        # Too far
        if np.linalg.norm(state.position - self._target_position) > self.config.max_position_error:
            return True
        
        return False
    
    def close(self) -> None:
        """Clean up environment."""
        self.disconnect()
