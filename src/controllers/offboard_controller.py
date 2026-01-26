"""Offboard controller for neural network setpoint control.

This controller converts neural network outputs (position/velocity setpoints)
to MAVLink messages for PX4 offboard control mode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ..communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig
from ..communication.messages import SetpointCommand
from ..core.mujoco_sim import QuadrotorState
from ..utils.transforms import CoordinateTransforms
from .base_controller import BaseController


class OffboardControlMode(Enum):
    """Control modes for offboard controller."""

    POSITION = "position"  # NN outputs position setpoints
    VELOCITY = "velocity"  # NN outputs velocity setpoints
    POSITION_VELOCITY = "position_velocity"  # NN outputs both


@dataclass
class OffboardConfig:
    """Configuration for offboard controller."""

    # Control mode
    mode: OffboardControlMode = OffboardControlMode.VELOCITY

    # Setpoint limits
    max_velocity: float = 5.0  # m/s
    max_position_delta: float = 10.0  # m
    max_yaw_rate: float = 1.0  # rad/s

    # Action scaling (NN outputs are [-1, 1])
    velocity_scale: float = 3.0  # m/s per unit action
    position_scale: float = 2.0  # m per unit action
    yaw_rate_scale: float = 0.5  # rad/s per unit action

    # Offboard mode settings
    setpoint_rate: float = 50.0  # Hz - rate at which to send setpoints
    arm_check_timeout: float = 5.0  # seconds to wait for arming

    # Coordinate frame
    use_body_frame: bool = False  # If True, velocity is in body frame


class OffboardController(BaseController):
    """Controller for offboard mode with neural network setpoints.

    The neural network outputs normalized actions [-1, 1] which are
    converted to position/velocity setpoints and sent to PX4 via MAVLink.

    Action space depends on mode:
        - VELOCITY: [vx, vy, vz, yaw_rate] normalized
        - POSITION: [dx, dy, dz, dyaw] relative position change
        - POSITION_VELOCITY: [x, y, z, yaw, vx, vy, vz] (7 dims)
    """

    def __init__(
        self,
        config: OffboardConfig | None = None,
        mavlink_config: MAVLinkConfig | None = None,
    ):
        """Initialize offboard controller.

        Args:
            config: Offboard controller configuration
            mavlink_config: MAVLink connection configuration
        """
        super().__init__(name="OffboardController")

        self.config = config or OffboardConfig()
        self._mavlink_config = mavlink_config or MAVLinkConfig()

        # State
        self._bridge: MAVLinkBridge | None = None
        self._connected = False
        self._armed = False
        self._in_offboard = False

        # Current setpoint tracking
        self._current_position = np.zeros(3)  # NED
        self._current_yaw = 0.0
        self._last_setpoint_time = 0.0

        # Statistics
        self._setpoints_sent = 0
        self._last_action = np.zeros(4)

    @property
    def action_dim(self) -> int:
        """Get action space dimension based on mode."""
        if self.config.mode == OffboardControlMode.POSITION_VELOCITY:
            return 7  # [x, y, z, yaw, vx, vy, vz]
        else:
            return 4  # [vx, vy, vz, yaw_rate] or [dx, dy, dz, dyaw]

    def connect(self, wait_for_px4: bool = True) -> bool:
        """Connect to PX4 SITL.

        Args:
            wait_for_px4: Whether to wait for PX4 connection

        Returns:
            True if connection successful
        """
        if self._bridge is not None:
            self._bridge.stop()

        self._bridge = MAVLinkBridge(self._mavlink_config)

        if wait_for_px4 and not self._bridge.start_server():
            return False

        self._connected = True
        self._is_initialized = True
        return True

    def disconnect(self) -> None:
        """Disconnect from PX4."""
        if self._armed:
            self.disarm()

        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None

        self._connected = False
        self._in_offboard = False

    def initialize_offboard(self, initial_position: np.ndarray) -> bool:
        """Initialize offboard mode.

        Must send setpoints before entering offboard mode.

        Args:
            initial_position: Initial position in NED [m]

        Returns:
            True if offboard mode activated
        """
        if not self._connected or self._bridge is None:
            return False

        # Store initial position as reference
        self._current_position = initial_position.copy()

        # Send setpoints for 1 second before switching modes
        # PX4 requires streaming setpoints before offboard mode
        print("Streaming setpoints before offboard mode...")
        for _ in range(50):  # 50 setpoints at 50Hz = 1 second
            self._bridge.send_position_setpoint(self._current_position)
            self._bridge.send_heartbeat()
            time.sleep(0.02)

        # Switch to offboard mode
        print("Switching to offboard mode...")
        if not self._bridge.set_offboard_mode():
            print("Failed to set offboard mode")
            return False

        self._in_offboard = True
        return True

    def arm(self) -> bool:
        """Arm the vehicle.

        Returns:
            True if arming successful
        """
        if not self._connected or self._bridge is None:
            return False

        print("Arming vehicle...")
        if self._bridge.arm():
            self._armed = True
            return True
        return False

    def disarm(self, force: bool = False) -> bool:
        """Disarm the vehicle.

        Args:
            force: Force disarm even in flight

        Returns:
            True if disarming successful
        """
        if not self._connected or self._bridge is None:
            return False

        if self._bridge.disarm(force=force):
            self._armed = False
            return True
        return False

    def compute_action(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute motor commands (not used in offboard mode).

        In offboard mode, this method returns zeros since PX4
        computes the motor commands. Use send_setpoint() instead.

        Args:
            state: Current quadrotor state
            target_position: Target position (used for internal reference)
            dt: Time step

        Returns:
            Zero motor commands (PX4 handles motors)
        """
        # In offboard mode, we don't compute motor commands
        # This is just for API compatibility
        return np.zeros(4)

    def process_nn_action(
        self,
        action: np.ndarray,
        current_state: QuadrotorState,
    ) -> SetpointCommand:
        """Convert neural network action to setpoint command.

        Args:
            action: Normalized action from NN [-1, 1]
            current_state: Current drone state

        Returns:
            SetpointCommand for MAVLink
        """
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.copy()

        # Convert current state to NED
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(current_state.position)

        if self.config.mode == OffboardControlMode.VELOCITY:
            # Action: [vx, vy, vz, yaw_rate]
            velocity = action[:3] * self.config.velocity_scale
            yaw_rate = action[3] * self.config.yaw_rate_scale if len(action) > 3 else 0.0

            # Clip velocity
            velocity = np.clip(velocity, -self.config.max_velocity, self.config.max_velocity)
            yaw_rate = np.clip(yaw_rate, -self.config.max_yaw_rate, self.config.max_yaw_rate)

            # Convert to NED if in body frame
            if self.config.use_body_frame:
                # Rotate velocity from body to NED using current yaw
                yaw = CoordinateTransforms.euler_from_quaternion_ned(current_state.quaternion)[2]
                c, s = np.cos(yaw), np.sin(yaw)
                velocity_ned = np.array(
                    [
                        c * velocity[0] - s * velocity[1],
                        s * velocity[0] + c * velocity[1],
                        velocity[2],
                    ]
                )
            else:
                velocity_ned = velocity

            return SetpointCommand(
                velocity=velocity_ned,
                yaw_rate=yaw_rate,
            )

        elif self.config.mode == OffboardControlMode.POSITION:
            # Action: [dx, dy, dz, dyaw] relative position change
            delta_pos = action[:3] * self.config.position_scale
            delta_yaw = action[3] * self.config.yaw_rate_scale if len(action) > 3 else 0.0

            # Clip position change
            delta_pos = np.clip(
                delta_pos,
                -self.config.max_position_delta,
                self.config.max_position_delta,
            )

            # Update target position
            new_position = pos_ned + delta_pos
            new_yaw = self._current_yaw + delta_yaw

            self._current_position = new_position
            self._current_yaw = new_yaw

            return SetpointCommand(
                position=new_position,
                yaw=new_yaw,
            )

        else:  # POSITION_VELOCITY
            # Action: [x, y, z, yaw, vx, vy, vz]
            position = action[:3] * self.config.position_scale
            yaw = action[3] * np.pi if len(action) > 3 else 0.0
            velocity = action[4:7] * self.config.velocity_scale if len(action) > 4 else np.zeros(3)

            return SetpointCommand(
                position=position,
                velocity=velocity,
                yaw=yaw,
            )

    def send_nn_action(
        self,
        action: np.ndarray,
        current_state: QuadrotorState,
    ) -> bool:
        """Send neural network action as setpoint to PX4.

        Args:
            action: Normalized action from NN [-1, 1]
            current_state: Current drone state

        Returns:
            True if setpoint sent successfully
        """
        if not self._connected or self._bridge is None:
            return False

        # Convert NN action to setpoint
        setpoint = self.process_nn_action(action, current_state)

        # Send to PX4
        success = self._bridge.send_setpoint(setpoint)

        if success:
            self._setpoints_sent += 1
            self._last_setpoint_time = time.time()

        # Send heartbeat periodically
        self._bridge.send_heartbeat()

        return success

    def reset(self) -> None:
        """Reset controller state."""
        self._current_position = np.zeros(3)
        self._current_yaw = 0.0
        self._last_action = np.zeros(4)
        self._setpoints_sent = 0

    def get_info(self) -> dict[str, Any]:
        """Get controller info."""
        info = super().get_info()
        info.update(
            {
                "connected": self._connected,
                "armed": self._armed,
                "in_offboard": self._in_offboard,
                "mode": self.config.mode.value,
                "setpoints_sent": self._setpoints_sent,
                "last_action": self._last_action.tolist(),
                "current_position_ned": self._current_position.tolist(),
            }
        )
        return info

    @property
    def is_connected(self) -> bool:
        """Check if connected to PX4."""
        return self._connected

    @property
    def is_armed(self) -> bool:
        """Check if vehicle is armed."""
        return self._armed

    @property
    def is_in_offboard(self) -> bool:
        """Check if in offboard mode."""
        return self._in_offboard

    @property
    def bridge(self) -> MAVLinkBridge | None:
        """Get MAVLink bridge instance."""
        return self._bridge
