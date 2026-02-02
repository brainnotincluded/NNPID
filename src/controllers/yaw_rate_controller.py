"""Yaw rate controller with internal stabilization.

This controller takes a yaw rate command and outputs motor commands,
while internally stabilizing roll, pitch, and altitude using PD control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.motor_mixer import mix_x_configuration
from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations
from .base_controller import BaseController


@dataclass
class YawRateControllerConfig:
    """Configuration for yaw rate controller."""

    # Altitude control
    altitude_kp: float = 2.0
    altitude_kd: float = 1.0
    altitude_ki: float = 0.1
    max_altitude_integral: float = 2.0

    # Roll/pitch stabilization
    attitude_kp: float = 5.0
    attitude_kd: float = 1.0

    # Yaw rate control
    yaw_rate_kp: float = 2.0
    yaw_rate_ki: float = 0.1
    max_yaw_rate_integral: float = 1.0

    # Output limits
    base_thrust: float = 0.5  # Normalized [0, 1]
    max_thrust_adjustment: float = 0.3
    max_attitude_cmd: float = 0.3
    max_yaw_cmd: float = 0.2

    # Physical parameters
    hover_thrust: float = 0.5  # Thrust to hover


class YawRateController(BaseController):
    """Controller that tracks a yaw rate while stabilizing hover.

    Takes a desired yaw rate and outputs motor commands.
    Internally uses PD control for:
    - Altitude hold
    - Roll stabilization (to zero)
    - Pitch stabilization (to zero)
    - Yaw rate tracking

    This allows a neural network to only learn yaw tracking,
    while this controller handles the stabilization.
    """

    def __init__(
        self,
        config: YawRateControllerConfig | None = None,
        hover_height: float = 1.0,
    ):
        """Initialize controller.

        Args:
            config: Controller configuration
            hover_height: Target hover altitude
        """
        super().__init__(name="YawRateController")

        self.config = config or YawRateControllerConfig()
        self.hover_height = hover_height

        # Integral accumulators
        self._altitude_integral = 0.0
        self._yaw_rate_integral = 0.0

        # Previous values for derivative computation
        self._prev_altitude_error = 0.0
        self._prev_yaw_rate_error = 0.0

        # Current command
        self._current_yaw_rate_cmd = 0.0

        self._is_initialized = True

    def reset(self) -> None:
        """Reset controller state."""
        self._altitude_integral = 0.0
        self._yaw_rate_integral = 0.0
        self._prev_altitude_error = 0.0
        self._prev_yaw_rate_error = 0.0
        self._current_yaw_rate_cmd = 0.0

    def set_hover_height(self, height: float) -> None:
        """Set target hover height.

        Args:
            height: Target altitude in meters
        """
        self.hover_height = height

    def compute_action(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute motor commands.

        Note: target_position is not used for position control,
        only its z-component is used if hover_height is not set.
        Instead, use compute_yaw_rate_action for yaw rate control.

        Args:
            state: Current quadrotor state
            target_position: Target position (z used for altitude)
            dt: Time step

        Returns:
            Motor commands [0, 1]
        """
        # Use target z for hover height if provided
        if target_position is not None:
            self.hover_height = target_position[2]

        return self.compute_yaw_rate_action(
            state=state,
            yaw_rate_cmd=self._current_yaw_rate_cmd,
            dt=dt,
        )

    def compute_yaw_rate_action(
        self,
        state: QuadrotorState,
        yaw_rate_cmd: float,
        dt: float,
    ) -> np.ndarray:
        """Compute motor commands for given yaw rate command.

        Args:
            state: Current quadrotor state
            yaw_rate_cmd: Desired yaw rate in rad/s
            dt: Time step in seconds

        Returns:
            Motor commands [0, 1] for 4 motors
        """
        cfg = self.config
        self._current_yaw_rate_cmd = yaw_rate_cmd

        # Get current state
        roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)
        omega = state.angular_velocity

        # === Altitude Control (PID) ===
        altitude_error = self.hover_height - state.position[2]
        altitude_rate = -state.velocity[2]  # Negative because z-up

        # Update integral
        self._altitude_integral += altitude_error * dt
        self._altitude_integral = np.clip(
            self._altitude_integral,
            -cfg.max_altitude_integral,
            cfg.max_altitude_integral,
        )

        # Compute thrust adjustment
        thrust_adj = (
            cfg.altitude_kp * altitude_error
            + cfg.altitude_kd * altitude_rate
            + cfg.altitude_ki * self._altitude_integral
        )
        thrust_adj = np.clip(thrust_adj, -cfg.max_thrust_adjustment, cfg.max_thrust_adjustment)

        total_thrust = cfg.base_thrust + thrust_adj

        # === Roll Stabilization (PD to zero) ===
        roll_error = 0.0 - roll
        roll_rate = -omega[0]

        roll_cmd = cfg.attitude_kp * roll_error + cfg.attitude_kd * roll_rate
        roll_cmd = np.clip(roll_cmd, -cfg.max_attitude_cmd, cfg.max_attitude_cmd)

        # === Pitch Stabilization (PD to zero) ===
        pitch_error = 0.0 - pitch
        pitch_rate = -omega[1]

        pitch_cmd = cfg.attitude_kp * pitch_error + cfg.attitude_kd * pitch_rate
        pitch_cmd = np.clip(pitch_cmd, -cfg.max_attitude_cmd, cfg.max_attitude_cmd)

        # === Yaw Rate Control (PI) ===
        yaw_rate_error = yaw_rate_cmd - omega[2]

        # Update integral
        self._yaw_rate_integral += yaw_rate_error * dt
        self._yaw_rate_integral = np.clip(
            self._yaw_rate_integral,
            -cfg.max_yaw_rate_integral,
            cfg.max_yaw_rate_integral,
        )

        yaw_cmd = cfg.yaw_rate_kp * yaw_rate_error + cfg.yaw_rate_ki * self._yaw_rate_integral
        yaw_cmd = np.clip(yaw_cmd, -cfg.max_yaw_cmd, cfg.max_yaw_cmd)

        # === Motor Mixing (X configuration) ===
        motors = mix_x_configuration(total_thrust, roll_cmd, pitch_cmd, yaw_cmd)
        motors = np.clip(motors, 0.0, 1.0)

        return motors

    def get_info(self) -> dict[str, Any]:
        """Get controller information."""
        info = super().get_info()
        info.update(
            {
                "hover_height": self.hover_height,
                "altitude_integral": self._altitude_integral,
                "yaw_rate_integral": self._yaw_rate_integral,
                "current_yaw_rate_cmd": self._current_yaw_rate_cmd,
            }
        )
        return info


class YawRateStabilizer:
    """Lightweight stabilizer for use in environments.

    Provides the same functionality as YawRateController but
    with a simpler interface for environment integration.
    """

    def __init__(
        self,
        hover_height: float = 1.0,
        altitude_kp: float = 2.0,
        altitude_kd: float = 1.0,
        attitude_kp: float = 5.0,
        attitude_kd: float = 1.0,
        yaw_rate_kp: float = 2.0,
    ):
        """Initialize stabilizer.

        Args:
            hover_height: Target altitude
            altitude_kp: Altitude proportional gain
            altitude_kd: Altitude derivative gain
            attitude_kp: Attitude proportional gain
            attitude_kd: Attitude derivative gain
            yaw_rate_kp: Yaw rate proportional gain
        """
        self.hover_height = hover_height
        self.altitude_kp = altitude_kp
        self.altitude_kd = altitude_kd
        self.attitude_kp = attitude_kp
        self.attitude_kd = attitude_kd
        self.yaw_rate_kp = yaw_rate_kp

    def compute_motors(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        yaw_rate_cmd: float,
    ) -> np.ndarray:
        """Compute motor commands.

        Args:
            position: Current position [x, y, z]
            velocity: Current velocity [vx, vy, vz]
            quaternion: Current orientation [w, x, y, z]
            angular_velocity: Current angular velocity [p, q, r]
            yaw_rate_cmd: Desired yaw rate in rad/s

        Returns:
            Motor commands [0, 1] for 4 motors
        """
        # Get Euler angles
        roll, pitch, yaw = Rotations.quaternion_to_euler(quaternion)

        # Altitude control
        alt_error = self.hover_height - position[2]
        alt_rate = -velocity[2]
        thrust = 0.5 + self.altitude_kp * alt_error + self.altitude_kd * alt_rate

        # Attitude stabilization (to zero)
        roll_cmd = self.attitude_kp * (-roll) + self.attitude_kd * (-angular_velocity[0])
        pitch_cmd = self.attitude_kp * (-pitch) + self.attitude_kd * (-angular_velocity[1])

        # Yaw rate tracking
        yaw_rate_error = yaw_rate_cmd - angular_velocity[2]
        yaw_cmd = self.yaw_rate_kp * yaw_rate_error

        # Motor mixing
        motors = mix_x_configuration(thrust, roll_cmd, pitch_cmd, yaw_cmd)
        return np.clip(motors, 0.0, 1.0)
