"""Shared utilities for environment implementations."""

from __future__ import annotations

import numpy as np

from ..core.mujoco_sim import QuadrotorState
from ..utils.rotations import Rotations


def sample_initial_state(
    rng: np.random.Generator,
    position_range: tuple[float, float, float],
    velocity_range: float,
    angle_range: float,
    angular_velocity_range: float,
    min_z: float = 0.5,
    z_span: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a randomized initial state for an episode."""
    if z_span is None:
        position = rng.uniform(-np.array(position_range), np.array(position_range))
        position[2] = abs(position[2]) + min_z
    else:
        position = np.array(
            [
                rng.uniform(-position_range[0], position_range[0]),
                rng.uniform(-position_range[1], position_range[1]),
                rng.uniform(min_z, min_z + z_span),
            ]
        )

    velocity = rng.uniform(-velocity_range, velocity_range, size=3)

    roll = rng.uniform(-angle_range, angle_range)
    pitch = rng.uniform(-angle_range, angle_range)
    yaw = rng.uniform(-np.pi, np.pi)
    quaternion = Rotations.euler_to_quaternion(roll, pitch, yaw)

    angular_velocity = rng.uniform(-angular_velocity_range, angular_velocity_range, size=3)

    return position, velocity, quaternion, angular_velocity


def build_setpoint_observation(
    state: QuadrotorState,
    target_position: np.ndarray,
    previous_action: np.ndarray,
    rng: np.random.Generator,
    position_noise: float,
    velocity_noise: float,
) -> np.ndarray:
    """Construct a setpoint observation vector with noise."""
    euler = np.array(Rotations.quaternion_to_euler(state.quaternion))
    position = state.position + rng.normal(0, position_noise, 3)
    velocity = state.velocity + rng.normal(0, velocity_noise, 3)

    obs = np.concatenate(
        [
            position,
            velocity,
            euler,
            state.angular_velocity,
            target_position,
            previous_action,
        ]
    ).astype(np.float32)

    return obs
