"""Hover stabilizer for guaranteed drone stability.

This stabilizer uses direct PID control (angle → torque) with aggressive
gains to maintain hover. It is designed to be robust to disturbances
and prevent the drone from crashing regardless of yaw commands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations


@dataclass
class HoverStabilizerConfig:
    """Configuration for hover stabilizer.

    Tuned for X500 quadrotor model in MuJoCo:
    - Mass: 2.0 kg
    - Motor max thrust: 8N each
    """

    # Target hover state
    hover_height: float = 1.0

    # Altitude PID gains
    altitude_kp: float = 15.0
    altitude_ki: float = 3.0
    altitude_kd: float = 8.0

    # Attitude PID gains (direct: angle → torque)
    # Very aggressive for strong stabilization
    attitude_kp: float = 40.0
    attitude_ki: float = 2.0
    attitude_kd: float = 15.0

    # Yaw rate control
    yaw_rate_kp: float = 2.0

    # Physical parameters
    base_thrust: float = 0.62

    # Safety limits
    safety_tilt_threshold: float = 0.5  # ~28 degrees - ignore yaw above this
    yaw_authority: float = 0.03  # Max yaw torque
    max_integral: float = 0.5  # Anti-windup limit


class HoverStabilizer:
    """Direct PID hover stabilizer.

    Uses direct PID control (angle error → motor torque) with aggressive
    gains to maintain stable hover. The neural network controls yaw rate
    only, and even aggressive yaw commands cannot crash the drone.
    """

    def __init__(self, config: HoverStabilizerConfig | None = None):
        self.config = config or HoverStabilizerConfig()

        # PID integral states
        self._alt_integral = 0.0
        self._roll_integral = 0.0
        self._pitch_integral = 0.0

        # Diagnostics
        self._safety_mode = False

    def reset(self) -> None:
        """Reset all internal states."""
        self._alt_integral = 0.0
        self._roll_integral = 0.0
        self._pitch_integral = 0.0
        self._safety_mode = False

    def compute_motors(
        self,
        state: QuadrotorState,
        yaw_rate_cmd: float,
        dt: float,
    ) -> np.ndarray:
        """Compute motor commands for stable hover with yaw control.

        Args:
            state: Current quadrotor state
            yaw_rate_cmd: Desired yaw rate from NN (rad/s)
            dt: Time step (seconds)

        Returns:
            Motor commands [0, 1] for 4 motors
        """
        cfg = self.config

        # Get current state
        roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)
        omega = state.angular_velocity

        # === SAFETY CHECK ===
        # If tilting too much, prioritize stabilization over yaw control
        tilt = np.sqrt(roll**2 + pitch**2)
        if tilt > cfg.safety_tilt_threshold:
            yaw_rate_cmd = 0.0  # Ignore NN yaw command, focus on recovery
            self._safety_mode = True
        else:
            self._safety_mode = False

        # === ALTITUDE PID ===
        alt_error = cfg.hover_height - state.position[2]
        alt_rate = -state.velocity[2]

        # Update integral with anti-windup
        self._alt_integral += alt_error * dt
        self._alt_integral = np.clip(self._alt_integral, -cfg.max_integral, cfg.max_integral)

        thrust = (
            cfg.base_thrust
            + cfg.altitude_kp * alt_error
            + cfg.altitude_ki * self._alt_integral
            + cfg.altitude_kd * alt_rate
        )
        thrust = np.clip(thrust, 0.3, 0.9)  # Safety limits

        # === ATTITUDE PID (Roll) ===
        roll_error = 0.0 - roll
        self._roll_integral += roll_error * dt
        self._roll_integral = np.clip(self._roll_integral, -cfg.max_integral, cfg.max_integral)

        roll_torque = (
            cfg.attitude_kp * roll_error
            + cfg.attitude_ki * self._roll_integral
            - cfg.attitude_kd * omega[0]
        )

        # === ATTITUDE PID (Pitch) ===
        pitch_error = 0.0 - pitch
        self._pitch_integral += pitch_error * dt
        self._pitch_integral = np.clip(self._pitch_integral, -cfg.max_integral, cfg.max_integral)

        pitch_torque = (
            cfg.attitude_kp * pitch_error
            + cfg.attitude_ki * self._pitch_integral
            - cfg.attitude_kd * omega[1]
        )

        # Clamp attitude torques (high limits for strong stabilization)
        roll_torque = np.clip(roll_torque, -0.5, 0.5)
        pitch_torque = np.clip(pitch_torque, -0.5, 0.5)

        # === YAW RATE CONTROL ===
        # Only P control with reduced authority (NN cannot destabilize drone)
        yaw_rate_error = yaw_rate_cmd - omega[2]
        yaw_torque = np.clip(
            cfg.yaw_rate_kp * yaw_rate_error, -cfg.yaw_authority, cfg.yaw_authority
        )

        # === MOTOR MIXING (X configuration) ===
        # MuJoCo model uses X=forward, Y=left, Z=up coordinate system
        # Motor positions:
        #   Motor 1: (+X, +Y) = Front-Left,  CCW (+yaw torque)
        #   Motor 2: (-X, +Y) = Back-Left,   CW  (-yaw torque)
        #   Motor 3: (-X, -Y) = Back-Right,  CCW (+yaw torque)
        #   Motor 4: (+X, -Y) = Front-Right, CW  (-yaw torque)
        #
        # Roll: positive roll = right side down
        #   → increase right motors (3, 4), decrease left motors (1, 2)
        #   → roll_torque < 0 when roll > 0, so: m1,m2 use +roll, m3,m4 use -roll
        #
        # Pitch: positive pitch = nose down
        #   → increase front motors (1, 4), decrease back motors (2, 3)
        #   → pitch_torque < 0 when pitch > 0, so: m1,m4 use -pitch, m2,m3 use +pitch
        #
        # Yaw: positive yaw_torque = want CCW rotation
        #   → increase CCW motors (1, 3), decrease CW motors (2, 4)

        m1 = thrust + roll_torque - pitch_torque + yaw_torque  # Front-Left CCW
        m2 = thrust + roll_torque + pitch_torque - yaw_torque  # Back-Left CW
        m3 = thrust - roll_torque + pitch_torque + yaw_torque  # Back-Right CCW
        m4 = thrust - roll_torque - pitch_torque - yaw_torque  # Front-Right CW

        motors = np.array([m1, m2, m3, m4])
        return np.clip(motors, 0.0, 1.0)

    @property
    def is_safety_mode_active(self) -> bool:
        """Check if safety mode is currently active."""
        return self._safety_mode

    def get_debug_info(self) -> dict:
        """Get internal state for debugging."""
        return {
            "alt_integral": self._alt_integral,
            "roll_integral": self._roll_integral,
            "pitch_integral": self._pitch_integral,
            "safety_mode": self._safety_mode,
        }
