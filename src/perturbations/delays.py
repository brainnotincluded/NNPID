"""Delay perturbations for simulating latency in sensors, actuators, and communication."""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

from .base import BasePerturbation, DelayConfig

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


@dataclass
class DelayedSample:
    """A sample with timestamp for delay buffer."""
    timestamp: float
    data: np.ndarray


class DelayBuffer:
    """Ring buffer for implementing delays.
    
    Stores samples with timestamps and retrieves delayed values.
    """
    
    def __init__(self, max_delay_ms: float, sample_rate_hz: float = 500.0):
        """Initialize delay buffer.
        
        Args:
            max_delay_ms: Maximum delay in milliseconds
            sample_rate_hz: Expected sample rate
        """
        self.max_delay_ms = max_delay_ms
        self.max_delay_s = max_delay_ms / 1000.0
        
        # Calculate buffer size
        buffer_size = int(np.ceil(max_delay_ms / 1000.0 * sample_rate_hz * 2)) + 10
        buffer_size = max(buffer_size, 100)
        
        self._buffer: deque = deque(maxlen=buffer_size)
        self._last_value: Optional[np.ndarray] = None
    
    def push(self, timestamp: float, data: np.ndarray) -> None:
        """Add a sample to the buffer.
        
        Args:
            timestamp: Sample timestamp in seconds
            data: Sample data
        """
        self._buffer.append(DelayedSample(timestamp, data.copy()))
        self._last_value = data.copy()
    
    def get_delayed(self, current_time: float, delay_ms: float) -> Optional[np.ndarray]:
        """Get delayed sample.
        
        Args:
            current_time: Current timestamp in seconds
            delay_ms: Delay in milliseconds
            
        Returns:
            Delayed sample data or None if buffer is empty
        """
        if len(self._buffer) == 0:
            return self._last_value
        
        delay_s = delay_ms / 1000.0
        target_time = current_time - delay_s
        
        # Find sample closest to target time
        best_sample = None
        best_diff = float('inf')
        
        for sample in self._buffer:
            diff = abs(sample.timestamp - target_time)
            if diff < best_diff:
                best_diff = diff
                best_sample = sample
            # Early exit if we've passed the target (buffer is ordered)
            if sample.timestamp > target_time and best_sample is not None:
                break
        
        if best_sample is not None:
            return best_sample.data.copy()
        
        # Fallback to oldest or newest sample
        if target_time < self._buffer[0].timestamp:
            return self._buffer[0].data.copy()
        return self._buffer[-1].data.copy()
    
    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._last_value = None
    
    def __len__(self) -> int:
        return len(self._buffer)


class FirstOrderFilter:
    """First-order low-pass filter for motor dynamics.
    
    Simulates motor response time constant.
    """
    
    def __init__(self, time_constant_ms: float):
        """Initialize filter.
        
        Args:
            time_constant_ms: Time constant in milliseconds
        """
        self.time_constant_ms = time_constant_ms
        self.time_constant_s = time_constant_ms / 1000.0
        self._state: Optional[np.ndarray] = None
    
    def update(self, input_value: np.ndarray, dt: float) -> np.ndarray:
        """Update filter with new input.
        
        Args:
            input_value: New input value
            dt: Time step in seconds
            
        Returns:
            Filtered output
        """
        if self._state is None:
            self._state = input_value.copy()
            return self._state.copy()
        
        if self.time_constant_s <= 0:
            self._state = input_value.copy()
            return self._state.copy()
        
        # First-order filter: y' = (u - y) / tau
        alpha = dt / (self.time_constant_s + dt)
        self._state = (1 - alpha) * self._state + alpha * input_value
        
        return self._state.copy()
    
    def reset(self, initial_value: Optional[np.ndarray] = None) -> None:
        """Reset filter state.
        
        Args:
            initial_value: Optional initial state
        """
        self._state = initial_value.copy() if initial_value is not None else None
    
    @property
    def state(self) -> Optional[np.ndarray]:
        """Get current filter state."""
        return self._state.copy() if self._state is not None else None


class DelayPerturbation(BasePerturbation):
    """Simulates various delays in the control system.
    
    Components:
    - Sensor delays: IMU, GPS, barometer, magnetometer
    - Actuator delays: Motor command delay and response time
    - Communication delays: Command and telemetry latency
    - Jitter: Random variation in delays
    - Sample dropout: Occasional lost samples
    """
    
    def __init__(self, config: Optional[DelayConfig] = None):
        """Initialize delay perturbation.
        
        Args:
            config: Delay configuration. Uses defaults if None.
        """
        super().__init__(config or DelayConfig())
        self.delay_config: DelayConfig = self.config
        
        # Maximum delay for buffer sizing
        max_delay = max(
            self.delay_config.imu_delay,
            self.delay_config.gps_delay,
            self.delay_config.barometer_delay,
            self.delay_config.magnetometer_delay,
            self.delay_config.motor_delay,
            self.delay_config.command_delay,
            self.delay_config.telemetry_delay,
        ) + self.delay_config.jitter_max
        
        # Observation delay buffers (for different sensor types)
        # Buffer indices: 0-2: position, 3-5: velocity, 6-9: quaternion, 10-12: angular_velocity
        self._observation_buffer = DelayBuffer(max_delay + 100, sample_rate_hz=500.0)
        self._position_buffer = DelayBuffer(self.delay_config.gps_delay + 50)
        self._velocity_buffer = DelayBuffer(self.delay_config.gps_delay + 50)
        self._quaternion_buffer = DelayBuffer(self.delay_config.imu_delay + 50)
        self._angular_velocity_buffer = DelayBuffer(self.delay_config.imu_delay + 50)
        
        # Action delay buffer
        self._action_buffer = DelayBuffer(max_delay + 100)
        
        # Motor dynamics filter
        self._motor_filter = FirstOrderFilter(self.delay_config.motor_time_constant)
        
        # Jitter state
        self._current_jitter = np.zeros(4)  # Per-sensor jitter
        
        # Dropout state
        self._consecutive_drops = 0
        self._is_dropped = False
        self._last_valid_observation: Optional[np.ndarray] = None
        self._last_valid_action: Optional[np.ndarray] = None
        
        # Effective delays (with jitter)
        self._effective_imu_delay = 0.0
        self._effective_gps_delay = 0.0
        self._effective_motor_delay = 0.0
    
    def reset(self, rng: np.random.Generator) -> None:
        """Reset delay state."""
        super().reset(rng)
        
        # Clear all buffers
        self._observation_buffer.clear()
        self._position_buffer.clear()
        self._velocity_buffer.clear()
        self._quaternion_buffer.clear()
        self._angular_velocity_buffer.clear()
        self._action_buffer.clear()
        
        # Reset filter
        self._motor_filter.reset()
        
        # Reset state
        self._current_jitter = np.zeros(4)
        self._consecutive_drops = 0
        self._is_dropped = False
        self._last_valid_observation = None
        self._last_valid_action = None
        
        # Initial jitter
        self._update_jitter()
    
    def _update_jitter(self) -> None:
        """Update jitter values."""
        cfg = self.delay_config
        
        if not cfg.jitter_enabled:
            self._current_jitter = np.zeros(4)
            return
        
        if cfg.jitter_distribution == "uniform":
            self._current_jitter = self._rng.uniform(
                cfg.jitter_base, cfg.jitter_max, 4
            )
        elif cfg.jitter_distribution == "gaussian":
            mean = (cfg.jitter_base + cfg.jitter_max) / 2
            std = (cfg.jitter_max - cfg.jitter_base) / 4
            self._current_jitter = np.clip(
                self._rng.normal(mean, std, 4),
                cfg.jitter_base,
                cfg.jitter_max
            )
        elif cfg.jitter_distribution == "exponential":
            scale = (cfg.jitter_max - cfg.jitter_base) / 2
            self._current_jitter = np.clip(
                cfg.jitter_base + self._rng.exponential(scale, 4),
                cfg.jitter_base,
                cfg.jitter_max
            )
        else:
            self._current_jitter = np.zeros(4)
    
    def _check_dropout(self) -> bool:
        """Check if current sample should be dropped.
        
        Returns:
            True if sample should be dropped
        """
        cfg = self.delay_config
        
        if not cfg.dropout_enabled:
            return False
        
        # Check probability
        if self._rng.random() < cfg.dropout_probability:
            if self._consecutive_drops < cfg.dropout_max_consecutive:
                self._consecutive_drops += 1
                return True
        
        # Reset consecutive drops
        self._consecutive_drops = 0
        return False
    
    def update(self, dt: float, state: "QuadrotorState") -> None:
        """Update delay perturbation.
        
        Args:
            dt: Time step
            state: Current quadrotor state
        """
        if not self.enabled:
            return
        
        self._time += dt
        
        # Periodically update jitter (every ~0.1 seconds)
        if self._rng.random() < dt * 10:
            self._update_jitter()
        
        # Update effective delays with jitter
        cfg = self.delay_config
        self._effective_imu_delay = cfg.imu_delay + self._current_jitter[0]
        self._effective_gps_delay = cfg.gps_delay + self._current_jitter[1]
        self._effective_motor_delay = cfg.motor_delay + self._current_jitter[2]
        
        # Store state in buffers
        self._position_buffer.push(self._time, state.position)
        self._velocity_buffer.push(self._time, state.velocity)
        self._quaternion_buffer.push(self._time, state.quaternion)
        self._angular_velocity_buffer.push(self._time, state.angular_velocity)
        
        # Check dropout
        self._is_dropped = self._check_dropout()
    
    def apply_to_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply delay to observation.
        
        The observation is assumed to have the format from YawTrackingEnv:
        [target_dir(2), target_angular_vel(1), current_yaw_rate(1), 
         yaw_error(1), roll(1), pitch(1), altitude_error(1), 
         previous_action(1), time_on_target(1), target_distance(1)]
        
        Args:
            obs: Original observation
            
        Returns:
            Delayed observation
        """
        if not self.enabled:
            return obs
        
        cfg = self.delay_config
        
        # Handle dropout
        if self._is_dropped:
            if self._last_valid_observation is not None:
                return self._last_valid_observation.copy()
            return obs
        
        # Store observation in buffer
        self._observation_buffer.push(self._time, obs)
        
        # Get delayed observation
        # For yaw tracking env, we apply different delays to different components
        result = obs.copy()
        
        # Angular velocity (yaw_rate at index 3) - IMU delay
        delayed_obs = self._observation_buffer.get_delayed(
            self._time, self._effective_imu_delay
        )
        if delayed_obs is not None and len(delayed_obs) == len(obs):
            result[3] = delayed_obs[3]  # current_yaw_rate
            result[5] = delayed_obs[5]  # roll
            result[6] = delayed_obs[6]  # pitch
        
        # Store as last valid
        self._last_valid_observation = result.copy()
        
        return result
    
    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply delay and motor dynamics to action.
        
        Args:
            action: Original action
            
        Returns:
            Delayed and filtered action
        """
        if not self.enabled:
            return action
        
        cfg = self.delay_config
        
        # Handle dropout
        if self._is_dropped:
            if self._last_valid_action is not None:
                return self._last_valid_action.copy()
            return action
        
        # Store action in buffer
        self._action_buffer.push(self._time, action)
        
        # Get delayed action
        delayed_action = self._action_buffer.get_delayed(
            self._time, self._effective_motor_delay + cfg.command_delay
        )
        
        if delayed_action is None:
            delayed_action = action.copy()
        
        # Apply motor dynamics filter
        # Use approximate dt based on typical control rate
        dt = 0.02  # 50 Hz control rate
        filtered_action = self._motor_filter.update(delayed_action, dt)
        
        # Store as last valid
        self._last_valid_action = filtered_action.copy()
        
        return filtered_action
    
    def get_delayed_state_component(
        self,
        component: str,
        current_time: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Get a delayed state component.
        
        Args:
            component: One of "position", "velocity", "quaternion", "angular_velocity"
            current_time: Current time (uses internal time if None)
            
        Returns:
            Delayed state component or None
        """
        if current_time is None:
            current_time = self._time
        
        cfg = self.delay_config
        
        if component == "position":
            delay = self._effective_gps_delay
            buffer = self._position_buffer
        elif component == "velocity":
            delay = self._effective_gps_delay
            buffer = self._velocity_buffer
        elif component == "quaternion":
            delay = self._effective_imu_delay
            buffer = self._quaternion_buffer
        elif component == "angular_velocity":
            delay = self._effective_imu_delay
            buffer = self._angular_velocity_buffer
        else:
            return None
        
        return buffer.get_delayed(current_time, delay)
    
    def get_info(self) -> Dict[str, Any]:
        """Get delay perturbation information."""
        info = super().get_info()
        info.update({
            "effective_imu_delay_ms": float(self._effective_imu_delay),
            "effective_gps_delay_ms": float(self._effective_gps_delay),
            "effective_motor_delay_ms": float(self._effective_motor_delay),
            "current_jitter": self._current_jitter.tolist(),
            "is_dropped": self._is_dropped,
            "consecutive_drops": self._consecutive_drops,
            "buffer_sizes": {
                "observation": len(self._observation_buffer),
                "action": len(self._action_buffer),
            },
        })
        return info


# Convenience factory functions
def create_low_latency() -> DelayPerturbation:
    """Create low latency delay configuration."""
    config = DelayConfig(
        enabled=True,
        intensity=1.0,
        imu_delay=1.0,
        gps_delay=50.0,
        motor_delay=2.0,
        motor_time_constant=10.0,
        jitter_enabled=True,
        jitter_base=0.5,
        jitter_max=2.0,
        dropout_enabled=False,
    )
    return DelayPerturbation(config)


def create_typical_latency() -> DelayPerturbation:
    """Create typical latency delay configuration."""
    config = DelayConfig(
        enabled=True,
        intensity=1.0,
        imu_delay=5.0,
        gps_delay=100.0,
        motor_delay=10.0,
        motor_time_constant=20.0,
        jitter_enabled=True,
        jitter_base=1.0,
        jitter_max=5.0,
        dropout_enabled=False,
    )
    return DelayPerturbation(config)


def create_high_latency() -> DelayPerturbation:
    """Create high latency delay configuration."""
    config = DelayConfig(
        enabled=True,
        intensity=1.0,
        imu_delay=10.0,
        gps_delay=200.0,
        motor_delay=20.0,
        motor_time_constant=50.0,
        jitter_enabled=True,
        jitter_base=5.0,
        jitter_max=20.0,
        dropout_enabled=True,
        dropout_probability=0.02,
        dropout_max_consecutive=2,
    )
    return DelayPerturbation(config)


def create_unreliable_connection() -> DelayPerturbation:
    """Create unreliable connection with high dropout."""
    config = DelayConfig(
        enabled=True,
        intensity=1.0,
        imu_delay=5.0,
        gps_delay=150.0,
        motor_delay=15.0,
        motor_time_constant=30.0,
        jitter_enabled=True,
        jitter_base=5.0,
        jitter_max=50.0,
        jitter_distribution="exponential",
        dropout_enabled=True,
        dropout_probability=0.05,
        dropout_max_consecutive=5,
    )
    return DelayPerturbation(config)
