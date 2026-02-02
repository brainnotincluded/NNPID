"""Position/velocity controller used by setpoint environments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.motor_mixer import mix_x_configuration
from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations


@dataclass
class PositionControllerConfig:
    """Configuration for the simple position controller."""

    vel_p_gain: float = 5.0
    vel_i_gain: float = 0.5
    physics_timestep: float = 0.002
    max_integral: float = 2.0
    max_tilt: float = 0.5
    gravity_comp: float = 0.5


class PositionController:
    """Simple PI velocity controller that outputs motor commands."""

    def __init__(self, config: PositionControllerConfig):
        self.config = config
        self._velocity_integral = np.zeros(3)

    def reset(self) -> None:
        """Reset controller state."""
        self._velocity_integral = np.zeros(3)

    def compute_motors(
        self,
        state: QuadrotorState,
        target_velocity: np.ndarray,
        yaw_rate: float,
    ) -> np.ndarray:
        """Compute motor commands for a velocity setpoint."""
        cfg = self.config
        dt = cfg.physics_timestep

        # Velocity error
        vel_error = target_velocity - state.velocity

        # Velocity controller (PI)
        self._velocity_integral += vel_error * dt
        self._velocity_integral = np.clip(
            self._velocity_integral, -cfg.max_integral, cfg.max_integral
        )

        # Desired acceleration
        accel_cmd = vel_error * cfg.vel_p_gain + self._velocity_integral * cfg.vel_i_gain

        # Get current attitude
        roll, pitch, _yaw = Rotations.quaternion_to_euler(state.quaternion)

        # Desired roll/pitch from horizontal acceleration
        pitch_cmd = np.clip(accel_cmd[0] / 10.0, -cfg.max_tilt, cfg.max_tilt)
        roll_cmd = np.clip(-accel_cmd[1] / 10.0, -cfg.max_tilt, cfg.max_tilt)

        # Thrust for vertical (gravity compensation + z control)
        z_thrust = cfg.gravity_comp + accel_cmd[2] / 20.0
        z_thrust = np.clip(z_thrust, 0.1, 0.9)

        # Attitude errors
        roll_error = roll_cmd - roll
        pitch_error = pitch_cmd - pitch
        yaw_error = yaw_rate - state.angular_velocity[2]

        # PD attitude control
        roll_torque = roll_error * 2.0 - state.angular_velocity[0] * 0.3
        pitch_torque = pitch_error * 2.0 - state.angular_velocity[1] * 0.3
        yaw_torque = yaw_error * 1.0

        motors = mix_x_configuration(z_thrust, roll_torque, pitch_torque, yaw_torque)
        return np.clip(motors, 0.0, 1.0)
