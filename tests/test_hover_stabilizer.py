"""Tests for HoverStabilizer controller."""

import numpy as np

from src.controllers.hover_stabilizer import HoverStabilizer, HoverStabilizerConfig
from src.core.mujoco_sim import QuadrotorState
from src.utils.rotations import Rotations


def _make_state(roll: float, pitch: float, yaw: float = 0.0, z: float = 1.0) -> QuadrotorState:
    return QuadrotorState(
        position=np.array([0.0, 0.0, z]),
        velocity=np.zeros(3),
        quaternion=Rotations.euler_to_quaternion(roll, pitch, yaw),
        angular_velocity=np.zeros(3),
        motor_speeds=np.zeros(4),
    )


def test_safety_mode_ignores_yaw():
    """Yaw command should be ignored when tilt exceeds threshold."""
    cfg = HoverStabilizerConfig(hover_height=1.0, safety_tilt_threshold=0.2)
    state = _make_state(roll=0.3, pitch=0.0, z=cfg.hover_height)

    stabilizer_with_yaw = HoverStabilizer(cfg)
    motors_with_yaw = stabilizer_with_yaw.compute_motors(state, yaw_rate_cmd=1.0, dt=0.01)
    assert stabilizer_with_yaw.is_safety_mode_active

    stabilizer_no_yaw = HoverStabilizer(cfg)
    motors_no_yaw = stabilizer_no_yaw.compute_motors(state, yaw_rate_cmd=0.0, dt=0.01)

    np.testing.assert_allclose(motors_with_yaw, motors_no_yaw, atol=1e-6)


def test_motor_mixing_yaw_sign():
    """Positive yaw command should increase CCW motors (1, 3)."""
    cfg = HoverStabilizerConfig(hover_height=1.0)
    state = _make_state(roll=0.0, pitch=0.0, z=cfg.hover_height)

    stabilizer = HoverStabilizer(cfg)
    motors = stabilizer.compute_motors(state, yaw_rate_cmd=1.0, dt=0.01)

    assert motors[0] > motors[1]
    assert motors[2] > motors[3]
