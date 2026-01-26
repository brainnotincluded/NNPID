"""MAVLink client for connecting to real drones.

This module provides a client interface for deploying trained neural
network controllers to real drones running PX4 or ArduPilot.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    from pymavlink import mavutil

    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    mavutil = None

from ..communication.messages import PX4Mode


@dataclass
class DroneState:
    """Current drone state from MAVLink telemetry."""

    # Position (NED frame, meters)
    position: np.ndarray  # [x, y, z]

    # Velocity (NED frame, m/s)
    velocity: np.ndarray  # [vx, vy, vz]

    # Attitude (radians)
    roll: float
    pitch: float
    yaw: float

    # Angular velocity (rad/s)
    angular_velocity: np.ndarray  # [p, q, r]

    # Status
    armed: bool = False
    mode: str = "UNKNOWN"
    battery_voltage: float = 0.0

    # Timestamp
    timestamp: float = 0.0

    def to_observation(self, target_position: np.ndarray) -> np.ndarray:
        """Convert state to observation vector for NN.

        Args:
            target_position: Current target position

        Returns:
            Observation array matching SetpointBaseEnv format
        """
        euler = np.array([self.roll, self.pitch, self.yaw])

        # Matches SetpointBaseEnv observation format (19 dims)
        obs = np.concatenate(
            [
                self.position,
                self.velocity,
                euler,
                self.angular_velocity,
                target_position,
                np.zeros(4),  # Previous action (will be set by controller)
            ]
        )

        return obs.astype(np.float32)


@dataclass
class ClientConfig:
    """Configuration for MAVLink client."""

    # Connection
    connection_string: str = "udp:127.0.0.1:14540"  # Default PX4 SITL
    baudrate: int = 57600  # For serial connections

    # Timeouts
    connection_timeout: float = 30.0
    heartbeat_timeout: float = 5.0

    # Control settings
    setpoint_rate: float = 50.0  # Hz

    # Safety
    max_velocity: float = 5.0  # m/s
    max_altitude: float = 50.0  # m
    geofence_radius: float = 100.0  # m from home


class MAVLinkClient:
    """Client for connecting to real drones via MAVLink.

    Provides:
    - Connection management
    - State telemetry reception
    - Offboard setpoint sending
    - Safety monitoring

    Example usage:
        client = MAVLinkClient()
        client.connect()

        # Wait for drone to be ready
        while not client.state.armed:
            time.sleep(0.1)

        # Send setpoints
        while running:
            obs = client.state.to_observation(target)
            action = model.predict(obs)
            client.send_velocity_setpoint(action[:3], action[3])
            time.sleep(0.02)
    """

    def __init__(self, config: ClientConfig | None = None):
        """Initialize MAVLink client.

        Args:
            config: Client configuration
        """
        if not PYMAVLINK_AVAILABLE:
            raise ImportError("pymavlink is required")

        self.config = config or ClientConfig()

        # Connection
        self._connection = None
        self._connected = False

        # State
        self._state = DroneState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            angular_velocity=np.zeros(3),
        )
        self._home_position = np.zeros(3)

        # Threading
        self._running = False
        self._receive_thread = None
        self._state_lock = threading.Lock()

        # Callbacks
        self._on_state_update: Callable[[DroneState], None] | None = None

        # Timing
        self._last_heartbeat = 0.0
        self._last_setpoint = 0.0

    @property
    def state(self) -> DroneState:
        """Get current drone state (thread-safe)."""
        with self._state_lock:
            return DroneState(
                position=self._state.position.copy(),
                velocity=self._state.velocity.copy(),
                roll=self._state.roll,
                pitch=self._state.pitch,
                yaw=self._state.yaw,
                angular_velocity=self._state.angular_velocity.copy(),
                armed=self._state.armed,
                mode=self._state.mode,
                battery_voltage=self._state.battery_voltage,
                timestamp=self._state.timestamp,
            )

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    def connect(self) -> bool:
        """Connect to drone.

        Returns:
            True if connection successful
        """
        print(f"Connecting to: {self.config.connection_string}")

        try:
            self._connection = mavutil.mavlink_connection(
                self.config.connection_string,
                baud=self.config.baudrate,
            )

            # Wait for heartbeat
            print("Waiting for heartbeat...")
            msg = self._connection.wait_heartbeat(timeout=self.config.connection_timeout)

            if msg is None:
                print("No heartbeat received")
                return False

            print(f"Connected to system {self._connection.target_system}")

            # Start receive thread
            self._connected = True
            self._running = True
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()

            # Request data streams
            self._request_data_streams()

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from drone."""
        self._running = False
        self._connected = False

        if self._receive_thread is not None:
            self._receive_thread.join(timeout=2.0)

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _request_data_streams(self) -> None:
        """Request high-rate data streams from autopilot."""
        if self._connection is None:
            return

        # Request position and attitude at high rate
        self._connection.mav.request_data_stream_send(
            self._connection.target_system,
            self._connection.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            50,  # 50 Hz
            1,  # Start
        )

    def _receive_loop(self) -> None:
        """Background thread for receiving messages."""
        while self._running:
            try:
                msg = self._connection.recv_match(blocking=True, timeout=0.1)
                if msg is not None:
                    self._handle_message(msg)
            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}")

    def _handle_message(self, msg) -> None:
        """Handle received MAVLink message."""
        msg_type = msg.get_type()

        if msg_type == "LOCAL_POSITION_NED":
            with self._state_lock:
                self._state.position = np.array([msg.x, msg.y, msg.z])
                self._state.velocity = np.array([msg.vx, msg.vy, msg.vz])
                self._state.timestamp = time.time()

        elif msg_type == "ATTITUDE":
            with self._state_lock:
                self._state.roll = msg.roll
                self._state.pitch = msg.pitch
                self._state.yaw = msg.yaw
                self._state.angular_velocity = np.array(
                    [msg.rollspeed, msg.pitchspeed, msg.yawspeed]
                )

        elif msg_type == "HEARTBEAT":
            with self._state_lock:
                self._state.armed = (
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                ) != 0
                # Decode mode
                if msg.custom_mode == PX4Mode.offboard_mode():
                    self._state.mode = "OFFBOARD"
                else:
                    self._state.mode = "OTHER"
            self._last_heartbeat = time.time()

        elif msg_type == "SYS_STATUS":
            with self._state_lock:
                self._state.battery_voltage = msg.voltage_battery / 1000.0

        elif msg_type == "HOME_POSITION":
            self._home_position = np.array([0.0, 0.0, 0.0])  # Local frame

        # Call callback
        if self._on_state_update is not None:
            self._on_state_update(self.state)

    def send_velocity_setpoint(
        self,
        velocity: np.ndarray,
        yaw_rate: float = 0.0,
    ) -> bool:
        """Send velocity setpoint.

        Args:
            velocity: Velocity [vx, vy, vz] in NED [m/s]
            yaw_rate: Yaw rate [rad/s]

        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False

        # Safety check
        velocity = np.clip(velocity, -self.config.max_velocity, self.config.max_velocity)

        # Type mask for velocity control
        type_mask = 0b0000_0001_1111_1000  # Ignore position, acceleration, yaw

        try:
            self._connection.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0,
                0,
                0,  # position (ignored)
                velocity[0],
                velocity[1],
                velocity[2],
                0,
                0,
                0,  # acceleration (ignored)
                0,  # yaw (ignored)
                yaw_rate,
            )
            self._last_setpoint = time.time()
            return True
        except Exception as e:
            print(f"Failed to send setpoint: {e}")
            return False

    def send_position_setpoint(
        self,
        position: np.ndarray,
        yaw: float = 0.0,
    ) -> bool:
        """Send position setpoint.

        Args:
            position: Position [x, y, z] in NED [m]
            yaw: Yaw angle [rad]

        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False

        # Safety checks
        if abs(position[2]) > self.config.max_altitude:
            print(f"Altitude {position[2]} exceeds limit")
            return False

        dist_from_home = np.linalg.norm(position[:2] - self._home_position[:2])
        if dist_from_home > self.config.geofence_radius:
            print("Position outside geofence")
            return False

        # Type mask for position control
        type_mask = 0b0000_1111_1111_1000  # Ignore velocity, acceleration, yaw rate

        try:
            self._connection.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                position[0],
                position[1],
                position[2],
                0,
                0,
                0,  # velocity
                0,
                0,
                0,  # acceleration
                yaw,
                0,  # yaw_rate
            )
            self._last_setpoint = time.time()
            return True
        except Exception as e:
            print(f"Failed to send setpoint: {e}")
            return False

    def arm(self) -> bool:
        """Arm the drone.

        Returns:
            True if command sent
        """
        if not self._connected or self._connection is None:
            return False

        try:
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,  # confirmation
                1,  # arm
                0,
                0,
                0,
                0,
                0,
                0,
            )
            return True
        except Exception as e:
            print(f"Arm failed: {e}")
            return False

    def disarm(self, force: bool = False) -> bool:
        """Disarm the drone.

        Args:
            force: Force disarm

        Returns:
            True if command sent
        """
        if not self._connected or self._connection is None:
            return False

        try:
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,  # disarm
                21196.0 if force else 0,  # force disarm magic
                0,
                0,
                0,
                0,
                0,
            )
            return True
        except Exception as e:
            print(f"Disarm failed: {e}")
            return False

    def set_offboard_mode(self) -> bool:
        """Switch to offboard mode.

        Note: Must be streaming setpoints before calling this.

        Returns:
            True if command sent
        """
        if not self._connected or self._connection is None:
            return False

        try:
            # Send DO_SET_MODE command
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                1,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
                PX4Mode.offboard_mode(),
                0,
                0,
                0,
                0,
                0,
            )
            return True
        except Exception as e:
            print(f"Mode change failed: {e}")
            return False

    def land(self) -> bool:
        """Command landing.

        Returns:
            True if command sent
        """
        if not self._connected or self._connection is None:
            return False

        try:
            self._connection.mav.command_long_send(
                self._connection.target_system,
                self._connection.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            return True
        except Exception as e:
            print(f"Land failed: {e}")
            return False

    def set_state_callback(self, callback: Callable[[DroneState], None]) -> None:
        """Set callback for state updates.

        Args:
            callback: Function called with new state
        """
        self._on_state_update = callback


class NNDeploymentController:
    """Deploy neural network controller to real drone.

    Combines:
    - ONNX model inference
    - MAVLink communication
    - Safety monitoring
    """

    def __init__(
        self,
        model_path: str,
        client_config: ClientConfig | None = None,
    ):
        """Initialize deployment controller.

        Args:
            model_path: Path to ONNX model
            client_config: MAVLink client configuration
        """
        from .model_export import ONNXInference

        self.model = ONNXInference(model_path)
        self.client = MAVLinkClient(client_config)

        self._target_position = np.array([0.0, 0.0, -2.0])  # Default hover
        self._previous_action = np.zeros(4)
        self._running = False

    @property
    def target_position(self) -> np.ndarray:
        """Get current target position."""
        return self._target_position.copy()

    @target_position.setter
    def target_position(self, value: np.ndarray) -> None:
        """Set target position."""
        self._target_position = np.array(value)

    def start(self) -> bool:
        """Start the controller.

        Returns:
            True if started successfully
        """
        if not self.client.connect():
            return False

        self._running = True
        return True

    def stop(self) -> None:
        """Stop the controller."""
        self._running = False
        self.client.land()
        time.sleep(0.5)
        self.client.disconnect()

    def run_step(self) -> np.ndarray:
        """Run one control step.

        Returns:
            Action taken
        """
        # Get current state
        state = self.client.state

        # Build observation
        obs = state.to_observation(self._target_position)
        obs[-4:] = self._previous_action  # Add previous action

        # Get action from NN
        action = self.model.predict(obs)
        action = np.clip(action, -1.0, 1.0)

        # Scale to velocity
        max_vel = self.client.config.max_velocity
        velocity = action[:3] * max_vel
        yaw_rate = action[3] * 1.0  # rad/s

        # Send to drone
        self.client.send_velocity_setpoint(velocity, yaw_rate)

        self._previous_action = action
        return action

    def run_loop(self, rate: float = 50.0) -> None:
        """Run control loop.

        Args:
            rate: Control rate in Hz
        """
        dt = 1.0 / rate

        print("Starting control loop...")
        print("Press Ctrl+C to stop")

        try:
            while self._running:
                start = time.time()

                self.run_step()

                # Maintain rate
                elapsed = time.time() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
