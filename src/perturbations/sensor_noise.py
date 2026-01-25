"""Sensor noise perturbations for realistic sensor simulation."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BasePerturbation, SensorNoiseConfig

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


class SensorType(Enum):
    """Types of sensors that can be affected."""

    GYRO = "gyro"
    ACCEL = "accel"
    POSITION = "position"
    VELOCITY = "velocity"
    QUATERNION = "quaternion"
    MAGNETOMETER = "magnetometer"
    BAROMETER = "barometer"


class BiasRandomWalk:
    """Simulates sensor bias drift using random walk process.

    Models slowly drifting sensor biases that occur due to
    temperature changes, sensor aging, etc.
    """

    def __init__(self, drift_rate: float, initial_bias: np.ndarray | None = None):
        """Initialize bias random walk.

        Args:
            drift_rate: Drift rate (units per sqrt(second))
            initial_bias: Initial bias values (optional)
        """
        self.drift_rate = drift_rate
        self._bias: np.ndarray | None = initial_bias
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator, size: int = 3) -> None:
        """Reset bias state.

        Args:
            rng: Random number generator
            size: Dimension of bias vector
        """
        self._rng = rng
        # Start with small random bias
        self._bias = rng.normal(0, self.drift_rate * 10, size)

    def update(self, dt: float) -> np.ndarray:
        """Update bias using random walk.

        Args:
            dt: Time step in seconds

        Returns:
            Current bias vector
        """
        if self._bias is None or self._rng is None:
            return np.zeros(3)

        # Random walk: bias += N(0, drift_rate * sqrt(dt))
        noise = self._rng.normal(0, self.drift_rate * np.sqrt(dt), len(self._bias))
        self._bias = self._bias + noise

        return self._bias.copy()

    @property
    def bias(self) -> np.ndarray:
        """Get current bias."""
        return self._bias.copy() if self._bias is not None else np.zeros(3)


class GPSOutage:
    """Simulates GPS signal loss or degradation."""

    def __init__(
        self,
        loss_probability: float = 0.001,
        min_duration: float = 1.0,
        max_duration: float = 5.0,
    ):
        """Initialize GPS outage simulation.

        Args:
            loss_probability: Probability of GPS loss per timestep
            min_duration: Minimum outage duration in seconds
            max_duration: Maximum outage duration in seconds
        """
        self.loss_probability = loss_probability
        self.min_duration = min_duration
        self.max_duration = max_duration

        self._is_lost = False
        self._outage_start_time = 0.0
        self._outage_duration = 0.0
        self._last_valid_position: np.ndarray | None = None
        self._last_valid_velocity: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset GPS outage state."""
        self._rng = rng
        self._is_lost = False
        self._outage_start_time = 0.0
        self._outage_duration = 0.0
        self._last_valid_position = None
        self._last_valid_velocity = None

    def update(self, time: float, position: np.ndarray, velocity: np.ndarray) -> None:
        """Update GPS outage state.

        Args:
            time: Current simulation time
            position: Current position
            velocity: Current velocity
        """
        if self._rng is None:
            return

        if self._is_lost:
            # Check if outage should end
            if time - self._outage_start_time >= self._outage_duration:
                self._is_lost = False
        else:
            # Store valid values
            self._last_valid_position = position.copy()
            self._last_valid_velocity = velocity.copy()

            # Check for new outage
            if self._rng.random() < self.loss_probability:
                self._is_lost = True
                self._outage_start_time = time
                self._outage_duration = self._rng.uniform(self.min_duration, self.max_duration)

    @property
    def is_lost(self) -> bool:
        """Check if GPS is currently lost."""
        return self._is_lost

    def get_last_valid(self) -> tuple:
        """Get last valid GPS values.

        Returns:
            Tuple of (position, velocity) or (None, None)
        """
        return self._last_valid_position, self._last_valid_velocity


class MagneticInterference:
    """Simulates magnetic interference effects on magnetometer."""

    def __init__(
        self,
        random_probability: float = 0.01,
        max_magnitude: float = 0.5,
    ):
        """Initialize magnetic interference.

        Args:
            random_probability: Probability of random interference
            max_magnitude: Maximum interference magnitude (Gauss)
        """
        self.random_probability = random_probability
        self.max_magnitude = max_magnitude

        self._current_interference = np.zeros(3)
        self._interference_active = False
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset interference state."""
        self._rng = rng
        self._current_interference = np.zeros(3)
        self._interference_active = False

    def update(self, dt: float, position: np.ndarray) -> np.ndarray:
        """Update interference and return current value.

        Args:
            dt: Time step
            position: Current position (for zone-based interference)

        Returns:
            Interference vector to add to magnetometer reading
        """
        if self._rng is None:
            return np.zeros(3)

        # Random interference bursts
        if self._rng.random() < self.random_probability * dt * 100:
            self._interference_active = True
            # Random direction and magnitude
            direction = self._rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction) + 1e-6
            magnitude = self._rng.uniform(0, self.max_magnitude)
            self._current_interference = direction * magnitude
        elif self._rng.random() < 0.1:  # Decay probability
            self._interference_active = False
            self._current_interference *= 0.9

        return self._current_interference.copy()


class SensorNoisePerturbation(BasePerturbation):
    """Comprehensive sensor noise simulation.

    Components:
    - Gaussian noise: Random noise on all sensor readings
    - Sensor drift: Slowly varying bias (random walk)
    - Outliers: Occasional large erroneous readings
    - GPS loss: Complete GPS signal loss
    - Magnetic interference: Magnetometer disturbances
    - Temperature effects: Temperature-dependent sensor behavior
    """

    def __init__(self, config: SensorNoiseConfig | None = None):
        """Initialize sensor noise perturbation.

        Args:
            config: Sensor noise configuration. Uses defaults if None.
        """
        super().__init__(config or SensorNoiseConfig())
        self.noise_config: SensorNoiseConfig = self.config

        # Bias drift for different sensors
        self._gyro_bias = BiasRandomWalk(self.noise_config.gyro_bias_drift)
        self._accel_bias = BiasRandomWalk(self.noise_config.accel_bias_drift)

        # GPS outage
        self._gps_outage = GPSOutage(
            loss_probability=self.noise_config.gps_loss_probability,
            min_duration=self.noise_config.gps_loss_min_duration,
            max_duration=self.noise_config.gps_loss_max_duration,
        )

        # Magnetic interference
        self._mag_interference = MagneticInterference(
            random_probability=self.noise_config.magnetic_random_probability,
            max_magnitude=self.noise_config.magnetic_max_magnitude,
        )

        # Outlier state
        self._outlier_active = False
        self._outlier_sensor: SensorType | None = None
        self._outlier_value = np.zeros(3)

        # Temperature state
        self._current_temperature = 20.0
        self._temperature_offset = 0.0

        # Current noise values (for info)
        self._current_gyro_noise = np.zeros(3)
        self._current_accel_noise = np.zeros(3)
        self._current_position_noise = np.zeros(3)

    def reset(self, rng: np.random.Generator) -> None:
        """Reset sensor noise state."""
        super().reset(rng)

        # Reset bias drifts
        self._gyro_bias.reset(rng, 3)
        self._accel_bias.reset(rng, 3)

        # Reset GPS
        self._gps_outage.reset(rng)

        # Reset magnetic interference
        self._mag_interference.reset(rng)

        # Reset outlier
        self._outlier_active = False
        self._outlier_sensor = None
        self._outlier_value = np.zeros(3)

        # Reset temperature
        cfg = self.noise_config
        if cfg.temperature_enabled:
            self._current_temperature = cfg.ambient_temperature + rng.normal(0, 5)
            self._temperature_offset = (
                self._current_temperature - 20.0
            ) * cfg.temperature_coefficient
        else:
            self._current_temperature = 20.0
            self._temperature_offset = 0.0

    def update(self, dt: float, state: QuadrotorState) -> None:
        """Update sensor noise state.

        Args:
            dt: Time step
            state: Current quadrotor state
        """
        if not self.enabled:
            return

        self._time += dt
        cfg = self.noise_config

        # Update bias drifts
        if cfg.drift_enabled:
            self._gyro_bias.update(dt)
            self._accel_bias.update(dt)

        # Update GPS outage
        if cfg.gps_loss_enabled:
            self._gps_outage.update(self._time, state.position, state.velocity)

        # Update magnetic interference
        if cfg.magnetic_interference_enabled:
            self._mag_interference.update(dt, state.position)

        # Update outliers
        if cfg.outliers_enabled:
            self._update_outliers(dt)

        # Update temperature
        if cfg.temperature_enabled:
            self._update_temperature(dt)

    def _update_outliers(self, dt: float) -> None:
        """Update outlier state."""
        cfg = self.noise_config

        if self._outlier_active:
            # Outliers are instantaneous, reset on next step
            self._outlier_active = False
            self._outlier_sensor = None
        elif self._rng.random() < cfg.outlier_probability:
            # Generate new outlier
            self._outlier_active = True
            # Randomly select sensor type
            sensor_types = [
                SensorType.GYRO,
                SensorType.ACCEL,
                SensorType.POSITION,
                SensorType.VELOCITY,
            ]
            self._outlier_sensor = self._rng.choice(sensor_types)

            # Generate outlier value
            if self._outlier_sensor == SensorType.GYRO:
                base = cfg.gyro_noise_std
            elif self._outlier_sensor == SensorType.ACCEL:
                base = cfg.accel_noise_std
            elif self._outlier_sensor == SensorType.POSITION:
                base = cfg.position_noise_std
            else:
                base = cfg.velocity_noise_std

            # Outlier is large deviation
            direction = self._rng.choice([-1, 1], 3)
            self._outlier_value = (
                direction * base * cfg.outlier_magnitude * self._rng.uniform(0.5, 1.5, 3)
            )

    def _update_temperature(self, dt: float) -> None:
        """Update temperature-related effects."""
        cfg = self.noise_config

        # Slow temperature drift
        temp_change = self._rng.normal(0, 0.1 * dt)
        self._current_temperature += temp_change
        self._current_temperature = np.clip(self._current_temperature, -20, 60)

        # Update temperature offset
        self._temperature_offset = (self._current_temperature - 20.0) * cfg.temperature_coefficient

    def apply_to_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply sensor noise to observation.

        For YawTrackingEnv observation format:
        [target_dir(2), target_angular_vel(1), current_yaw_rate(1),
         yaw_error(1), roll(1), pitch(1), altitude_error(1),
         previous_action(1), time_on_target(1), target_distance(1)]

        Args:
            obs: Original observation

        Returns:
            Noisy observation
        """
        if not self.enabled:
            return obs

        cfg = self.noise_config
        result = obs.copy()
        intensity = cfg.intensity

        # Apply noise to relevant components
        # Note: We add noise to sensor-derived quantities

        # current_yaw_rate (index 3) - from gyroscope
        gyro_noise = self._rng.normal(0, cfg.gyro_noise_std) * intensity
        if cfg.drift_enabled:
            gyro_noise += self._gyro_bias.bias[2]  # Z-axis bias for yaw rate
        gyro_noise *= 1.0 + self._temperature_offset
        result[3] += gyro_noise
        self._current_gyro_noise = np.array([0, 0, gyro_noise])

        # roll, pitch (indices 5, 6) - from IMU
        attitude_noise = self._rng.normal(0, cfg.quaternion_noise_std, 2) * intensity
        attitude_noise *= 1.0 + self._temperature_offset
        result[5] += attitude_noise[0]
        result[6] += attitude_noise[1]

        # altitude_error (index 7) - from barometer/GPS
        if cfg.gps_loss_enabled and self._gps_outage.is_lost:
            # During GPS loss, altitude reading might freeze or drift
            pass  # Keep previous value (already in observation)
        else:
            altitude_noise = self._rng.normal(0, cfg.position_noise_std) * intensity
            result[7] += altitude_noise

        # yaw_error (index 4) - derived from position sensors
        if not (cfg.gps_loss_enabled and self._gps_outage.is_lost):
            yaw_error_noise = self._rng.normal(0, cfg.quaternion_noise_std * 2) * intensity
            result[4] += yaw_error_noise

        # target_direction (indices 0, 1) - from position sensors
        if not (cfg.gps_loss_enabled and self._gps_outage.is_lost):
            dir_noise = self._rng.normal(0, cfg.position_noise_std * 0.1, 2) * intensity
            result[0:2] += dir_noise
            # Re-normalize direction
            norm = np.sqrt(result[0] ** 2 + result[1] ** 2)
            if norm > 0.01:
                result[0:2] /= norm

        # Apply outliers if active
        if cfg.outliers_enabled and self._outlier_active:
            if self._outlier_sensor == SensorType.GYRO:
                result[3] += self._outlier_value[2]
            elif self._outlier_sensor == SensorType.POSITION:
                result[7] += self._outlier_value[2]

        return result

    def apply_noise_to_state(self, state: QuadrotorState) -> QuadrotorState:
        """Apply sensor noise directly to a QuadrotorState.

        Useful for low-level sensor simulation.

        Args:
            state: Original state

        Returns:
            Noisy state copy
        """
        from ..core.mujoco_sim import QuadrotorState

        if not self.enabled:
            return state

        cfg = self.noise_config
        intensity = cfg.intensity

        # Create noisy copies
        noisy_position = state.position.copy()
        noisy_velocity = state.velocity.copy()
        noisy_quaternion = state.quaternion.copy()
        noisy_angular_velocity = state.angular_velocity.copy()

        # Position noise (GPS)
        if cfg.gps_loss_enabled and self._gps_outage.is_lost:
            last_pos, last_vel = self._gps_outage.get_last_valid()
            if last_pos is not None:
                noisy_position = last_pos
            if last_vel is not None:
                noisy_velocity = last_vel
        else:
            noisy_position += self._rng.normal(0, cfg.position_noise_std, 3) * intensity
            noisy_velocity += self._rng.normal(0, cfg.velocity_noise_std, 3) * intensity

        # Quaternion noise (IMU)
        quat_noise = self._rng.normal(0, cfg.quaternion_noise_std, 4) * intensity
        noisy_quaternion += quat_noise
        # Renormalize
        noisy_quaternion /= np.linalg.norm(noisy_quaternion)

        # Angular velocity noise (gyro)
        gyro_noise = self._rng.normal(0, cfg.gyro_noise_std, 3) * intensity
        if cfg.drift_enabled:
            gyro_noise += self._gyro_bias.bias
        gyro_noise *= 1.0 + self._temperature_offset
        noisy_angular_velocity += gyro_noise

        return QuadrotorState(
            position=noisy_position,
            velocity=noisy_velocity,
            quaternion=noisy_quaternion,
            angular_velocity=noisy_angular_velocity,
            motor_speeds=state.motor_speeds.copy(),
        )

    def get_info(self) -> dict[str, Any]:
        """Get sensor noise perturbation information."""
        info = super().get_info()
        cfg = self.noise_config

        info.update(
            {
                "gyro_bias": self._gyro_bias.bias.tolist(),
                "accel_bias": self._accel_bias.bias.tolist(),
                "gps_lost": self._gps_outage.is_lost if cfg.gps_loss_enabled else False,
                "outlier_active": self._outlier_active,
                "outlier_sensor": self._outlier_sensor.value if self._outlier_sensor else None,
                "current_temperature": float(self._current_temperature),
                "temperature_offset": float(self._temperature_offset),
            }
        )
        return info


# Convenience factory functions
def create_low_noise() -> SensorNoisePerturbation:
    """Create low noise configuration (high quality sensors)."""
    config = SensorNoiseConfig(
        enabled=True,
        intensity=0.5,
        gyro_noise_std=0.001,
        accel_noise_std=0.01,
        position_noise_std=0.01,
        velocity_noise_std=0.02,
        drift_enabled=False,
        outliers_enabled=False,
        gps_loss_enabled=False,
    )
    return SensorNoisePerturbation(config)


def create_typical_noise() -> SensorNoisePerturbation:
    """Create typical sensor noise configuration."""
    config = SensorNoiseConfig(
        enabled=True,
        intensity=1.0,
        gyro_noise_std=0.005,
        accel_noise_std=0.05,
        position_noise_std=0.02,
        velocity_noise_std=0.05,
        drift_enabled=True,
        gyro_bias_drift=0.0001,
        accel_bias_drift=0.001,
        outliers_enabled=False,
        gps_loss_enabled=False,
    )
    return SensorNoisePerturbation(config)


def create_noisy_sensors() -> SensorNoisePerturbation:
    """Create noisy sensor configuration."""
    config = SensorNoiseConfig(
        enabled=True,
        intensity=1.0,
        gyro_noise_std=0.01,
        accel_noise_std=0.1,
        position_noise_std=0.1,
        velocity_noise_std=0.1,
        drift_enabled=True,
        gyro_bias_drift=0.0005,
        accel_bias_drift=0.005,
        outliers_enabled=True,
        outlier_probability=0.005,
        outlier_magnitude=5.0,
        gps_loss_enabled=False,
    )
    return SensorNoisePerturbation(config)


def create_harsh_conditions() -> SensorNoisePerturbation:
    """Create harsh conditions with GPS loss and interference."""
    config = SensorNoiseConfig(
        enabled=True,
        intensity=1.0,
        gyro_noise_std=0.02,
        accel_noise_std=0.2,
        position_noise_std=0.5,
        velocity_noise_std=0.2,
        drift_enabled=True,
        gyro_bias_drift=0.001,
        accel_bias_drift=0.01,
        outliers_enabled=True,
        outlier_probability=0.01,
        outlier_magnitude=10.0,
        gps_loss_enabled=True,
        gps_loss_probability=0.005,
        gps_loss_min_duration=2.0,
        gps_loss_max_duration=10.0,
        magnetic_interference_enabled=True,
        magnetic_random_probability=0.05,
        magnetic_max_magnitude=1.0,
        temperature_enabled=True,
        ambient_temperature=35.0,
    )
    return SensorNoisePerturbation(config)
