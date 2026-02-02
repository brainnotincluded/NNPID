"""Motor mixing utilities for X-configuration quadrotors."""

from __future__ import annotations

import numpy as np


def mix_x_configuration(
    base_thrust: float,
    roll_torque: float,
    pitch_torque: float,
    yaw_torque: float,
) -> np.ndarray:
    """Mix roll/pitch/yaw torques into per-motor thrusts.

    Motor order (MuJoCo X-forward, Y-left):
    - m1: Front-Left  (CCW)
    - m2: Back-Left   (CW)
    - m3: Back-Right  (CCW)
    - m4: Front-Right (CW)
    """
    m1 = base_thrust + roll_torque - pitch_torque + yaw_torque
    m2 = base_thrust + roll_torque + pitch_torque - yaw_torque
    m3 = base_thrust - roll_torque + pitch_torque + yaw_torque
    m4 = base_thrust - roll_torque - pitch_torque - yaw_torque
    return np.array([m1, m2, m3, m4])
