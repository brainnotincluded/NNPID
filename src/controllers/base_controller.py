"""Base controller interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..core.mujoco_sim import QuadrotorState


class BaseController(ABC):
    """Abstract base class for drone controllers.

    All controllers should inherit from this class and implement
    the compute_action method.
    """

    def __init__(self, name: str = "BaseController"):
        """Initialize controller.

        Args:
            name: Controller name for logging
        """
        self.name = name
        self._is_initialized = False

    @abstractmethod
    def compute_action(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute motor commands given current state and target.

        Args:
            state: Current quadrotor state
            target_position: Desired position [x, y, z]
            dt: Time step in seconds

        Returns:
            Motor commands [0, 1] for 4 motors
        """
        pass

    def reset(self) -> None:  # noqa: B027
        """Reset controller state.

        Override in subclasses to reset any internal state.
        This is intentionally not abstract - default is no-op.
        """
        pass

    def initialize(self, **kwargs) -> None:
        """Initialize controller with parameters.

        Override in subclasses for custom initialization.
        """
        self._is_initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._is_initialized

    def get_info(self) -> dict[str, Any]:
        """Get controller information.

        Override in subclasses to provide additional info.

        Returns:
            Dictionary of controller information
        """
        return {
            "name": self.name,
            "initialized": self._is_initialized,
        }


class PIDController(BaseController):
    """Simple PID position controller for reference.

    Uses separate PID loops for position and attitude.
    """

    def __init__(
        self,
        kp_pos: np.ndarray = np.array([2.0, 2.0, 4.0]),
        kd_pos: np.ndarray = np.array([1.5, 1.5, 2.0]),
        ki_pos: np.ndarray = np.array([0.1, 0.1, 0.2]),
        kp_att: np.ndarray = np.array([8.0, 8.0, 4.0]),
        kd_att: np.ndarray = np.array([2.0, 2.0, 1.0]),
        max_tilt: float = 0.5,  # radians
        mass: float = 2.0,
        gravity: float = 9.81,
    ):
        """Initialize PID controller.

        Args:
            kp_pos: Position proportional gains [x, y, z]
            kd_pos: Position derivative gains
            ki_pos: Position integral gains
            kp_att: Attitude proportional gains [roll, pitch, yaw]
            kd_att: Attitude derivative gains
            max_tilt: Maximum tilt angle
            mass: Quadrotor mass
            gravity: Gravity magnitude
        """
        super().__init__(name="PIDController")

        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.ki_pos = ki_pos
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.max_tilt = max_tilt
        self.mass = mass
        self.gravity = gravity

        # Integral error accumulator
        self._integral_error = np.zeros(3)
        self._max_integral = np.array([2.0, 2.0, 5.0])

        # Previous error for derivative
        self._prev_error = np.zeros(3)

        self._is_initialized = True

    def reset(self) -> None:
        """Reset controller state."""
        self._integral_error = np.zeros(3)
        self._prev_error = np.zeros(3)

    def compute_action(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute motor commands using cascaded PID.

        Outer loop: Position -> Desired acceleration
        Middle: Desired acceleration -> Desired attitude
        Inner loop: Attitude -> Motor commands
        """
        # Position error
        pos_error = target_position - state.position

        # Update integral
        self._integral_error += pos_error * dt
        self._integral_error = np.clip(
            self._integral_error,
            -self._max_integral,
            self._max_integral,
        )

        # Derivative
        pos_error_dot = (pos_error - self._prev_error) / dt if dt > 0 else np.zeros(3)
        self._prev_error = pos_error.copy()

        # Desired acceleration (PID)
        desired_accel = (
            self.kp_pos * pos_error
            + self.kd_pos * pos_error_dot
            + self.ki_pos * self._integral_error
        )

        # Add gravity compensation
        desired_accel[2] += self.gravity

        # Compute desired thrust (magnitude of acceleration)
        thrust = self.mass * np.linalg.norm(desired_accel)

        # Compute desired roll and pitch from desired acceleration
        # Assuming yaw = 0 for simplicity
        accel_norm = np.linalg.norm(desired_accel)
        if accel_norm > 0.01:
            # Desired tilt angles
            desired_roll = np.arcsin(np.clip(desired_accel[1] / accel_norm, -1, 1))
            desired_pitch = np.arcsin(np.clip(-desired_accel[0] / accel_norm, -1, 1))
        else:
            desired_roll = 0.0
            desired_pitch = 0.0

        # Limit tilt
        desired_roll = np.clip(desired_roll, -self.max_tilt, self.max_tilt)
        desired_pitch = np.clip(desired_pitch, -self.max_tilt, self.max_tilt)

        # Current Euler angles (approximate from quaternion)
        w, x, y, z = state.quaternion
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        # Attitude errors
        roll_error = desired_roll - roll
        pitch_error = desired_pitch - pitch
        yaw_error = -yaw  # Target yaw = 0

        # Angular velocity (body frame)
        p, q, r = state.angular_velocity

        # Attitude PD control -> torques
        roll_torque = self.kp_att[0] * roll_error - self.kd_att[0] * p
        pitch_torque = self.kp_att[1] * pitch_error - self.kd_att[1] * q
        yaw_torque = self.kp_att[2] * yaw_error - self.kd_att[2] * r

        # Motor mixing (X configuration)
        # Motor 1 (FR): + thrust + roll + pitch + yaw
        # Motor 2 (FL): + thrust - roll + pitch - yaw
        # Motor 3 (BL): + thrust - roll - pitch + yaw
        # Motor 4 (BR): + thrust + roll - pitch - yaw

        base_thrust = thrust / 4.0

        m1 = base_thrust + roll_torque + pitch_torque + yaw_torque
        m2 = base_thrust - roll_torque + pitch_torque - yaw_torque
        m3 = base_thrust - roll_torque - pitch_torque + yaw_torque
        m4 = base_thrust + roll_torque - pitch_torque - yaw_torque

        motors = np.array([m1, m2, m3, m4])

        # Normalize to [0, 1]
        max_thrust_per_motor = 8.0  # N (from config)
        motors = motors / max_thrust_per_motor
        motors = np.clip(motors, 0.0, 1.0)

        return motors

    def get_info(self) -> dict[str, Any]:
        """Get controller info."""
        info = super().get_info()
        info.update(
            {
                "kp_pos": self.kp_pos.tolist(),
                "kd_pos": self.kd_pos.tolist(),
                "ki_pos": self.ki_pos.tolist(),
                "integral_error": self._integral_error.tolist(),
            }
        )
        return info
