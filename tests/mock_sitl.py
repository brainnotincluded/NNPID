"""Mock SITL server for automated testing without PX4.

This module provides a mock PX4 SITL that:
- Accepts MAVLink connections
- Responds to sensor messages with motor commands
- Simulates basic flight controller behavior

Usage:
    # In tests
    from tests.mock_sitl import MockSITL
    
    mock = MockSITL(port=4560)
    mock.start()
    
    # ... run tests ...
    
    mock.stop()
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any
from queue import Queue, Empty
from enum import Enum

import numpy as np

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink2
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    mavutil = None
    mavlink2 = None


class FlightMode(Enum):
    """Simulated flight modes."""
    MANUAL = 0
    STABILIZE = 1
    ALTITUDE_HOLD = 2
    POSITION_HOLD = 3
    OFFBOARD = 6
    LAND = 7


@dataclass
class MockState:
    """Simulated vehicle state."""
    
    # Position (NED, meters)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Velocity (NED, m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Attitude (quaternion w, x, y, z)
    quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    
    # Angular velocity (rad/s)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Control state
    armed: bool = False
    mode: FlightMode = FlightMode.MANUAL
    
    # Setpoints
    position_setpoint: Optional[np.ndarray] = None
    velocity_setpoint: Optional[np.ndarray] = None
    
    # Timestamp
    timestamp: float = 0.0


@dataclass
class MockConfig:
    """Configuration for mock SITL."""
    
    # Network
    host: str = "127.0.0.1"
    port: int = 4560
    
    # Simulation
    update_rate: float = 250.0  # Hz
    
    # Simple dynamics
    max_thrust: float = 20.0  # N
    mass: float = 2.0  # kg
    gravity: float = 9.81  # m/s²
    
    # Position controller gains
    pos_p: float = 2.0
    vel_p: float = 3.0
    
    # Response characteristics
    motor_response_time: float = 0.02  # seconds
    
    # System IDs
    system_id: int = 1
    component_id: int = 1


class MockSITL:
    """Mock PX4 SITL for testing.
    
    Simulates basic PX4 behavior:
    - Accepts MAVLink connections
    - Processes HIL_SENSOR messages
    - Sends HIL_ACTUATOR_CONTROLS
    - Responds to arm/disarm commands
    - Handles offboard mode
    """
    
    def __init__(self, config: Optional[MockConfig] = None):
        """Initialize mock SITL.
        
        Args:
            config: Mock configuration
        """
        if not PYMAVLINK_AVAILABLE:
            raise ImportError("pymavlink is required")
        
        self.config = config or MockConfig()
        
        # State
        self.state = MockState()
        
        # Network
        self._socket: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._mav = None
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Message queues
        self._sensor_queue: Queue = Queue(maxsize=100)
        
        # Motor outputs
        self._motor_outputs = np.zeros(4)
        
        # Timing
        self._last_update = 0.0
        self._last_heartbeat = 0.0
        
        # Statistics
        self.stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "sensor_updates": 0,
        }
    
    def start(self, timeout: float = 5.0) -> bool:
        """Start mock SITL server.
        
        Args:
            timeout: Connection timeout
            
        Returns:
            True if started successfully
        """
        print(f"[MockSITL] Starting on {self.config.host}:{self.config.port}")
        
        # Create server socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.config.host, self.config.port))
        self._socket.listen(1)
        self._socket.settimeout(timeout)
        
        try:
            print("[MockSITL] Waiting for connection...")
            self._client, addr = self._socket.accept()
            self._client.setblocking(False)
            print(f"[MockSITL] Connected from {addr}")
            
            # Create MAVLink encoder/decoder
            self._mav = mavutil.mavlink.MAVLink(
                file=None,
                srcSystem=self.config.system_id,
                srcComponent=self.config.component_id,
            )
            
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            
            return True
            
        except socket.timeout:
            print("[MockSITL] Connection timeout")
            self.stop()
            return False
        except Exception as e:
            print(f"[MockSITL] Error: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop mock SITL."""
        print("[MockSITL] Stopping...")
        
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        if self._client is not None:
            try:
                self._client.close()
            except:
                pass
        
        if self._socket is not None:
            try:
                self._socket.close()
            except:
                pass
        
        print("[MockSITL] Stopped")
    
    def _run_loop(self) -> None:
        """Main update loop."""
        dt = 1.0 / self.config.update_rate
        
        while self._running:
            start = time.time()
            
            # Receive messages
            self._receive_messages()
            
            # Update simulation
            self._update_simulation(dt)
            
            # Send actuator outputs
            self._send_actuators()
            
            # Send heartbeat
            if time.time() - self._last_heartbeat > 1.0:
                self._send_heartbeat()
                self._last_heartbeat = time.time()
            
            # Rate limit
            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    def _receive_messages(self) -> None:
        """Receive and process MAVLink messages."""
        if self._client is None:
            return
        
        try:
            data = self._client.recv(1024)
            if data:
                # Parse MAVLink messages
                for byte in data:
                    msg = self._mav.parse_char(bytes([byte]))
                    if msg is not None:
                        self._handle_message(msg)
                        self.stats["messages_received"] += 1
        except BlockingIOError:
            pass
        except Exception as e:
            if self._running:
                print(f"[MockSITL] Receive error: {e}")
    
    def _handle_message(self, msg) -> None:
        """Handle received MAVLink message."""
        msg_type = msg.get_type()
        
        if msg_type == "HIL_SENSOR":
            self._handle_sensor(msg)
        elif msg_type == "HIL_GPS":
            self._handle_gps(msg)
        elif msg_type == "COMMAND_LONG":
            self._handle_command(msg)
        elif msg_type == "SET_POSITION_TARGET_LOCAL_NED":
            self._handle_position_target(msg)
        elif msg_type == "HEARTBEAT":
            pass  # Ignore incoming heartbeats
    
    def _handle_sensor(self, msg) -> None:
        """Handle HIL_SENSOR message."""
        with self._lock:
            # Update timestamp
            self.state.timestamp = msg.time_usec / 1e6
            
            # Use accelerometer to estimate attitude (simplified)
            # In reality, PX4 uses a proper state estimator
            accel = np.array([msg.xacc, msg.yacc, msg.zacc])
            
            self.stats["sensor_updates"] += 1
    
    def _handle_gps(self, msg) -> None:
        """Handle HIL_GPS message."""
        with self._lock:
            # Update position from GPS
            # Convert from lat/lon to NED (simplified: use as offset from origin)
            self.state.position = np.array([
                (msg.lat / 1e7 - 47.397742) * 111000,  # ~111km per degree
                (msg.lon / 1e7 - 8.545594) * 111000 * np.cos(np.radians(47.4)),
                -(msg.alt / 1000 - 488.0),  # NED, down is positive
            ])
            
            self.state.velocity = np.array([
                msg.vn / 100,
                msg.ve / 100,
                msg.vd / 100,
            ])
    
    def _handle_command(self, msg) -> None:
        """Handle COMMAND_LONG message."""
        cmd = msg.command
        
        if cmd == 400:  # MAV_CMD_COMPONENT_ARM_DISARM
            arm = msg.param1 > 0.5
            with self._lock:
                self.state.armed = arm
            print(f"[MockSITL] {'Armed' if arm else 'Disarmed'}")
            
            # Send ACK
            self._send_command_ack(cmd, 0)  # MAV_RESULT_ACCEPTED
        
        elif cmd == 176:  # MAV_CMD_DO_SET_MODE
            mode = int(msg.param2)
            with self._lock:
                # Check for offboard mode
                if mode == (6 << 16):  # PX4 offboard custom mode
                    self.state.mode = FlightMode.OFFBOARD
                    print("[MockSITL] Offboard mode")
            
            self._send_command_ack(cmd, 0)
        
        elif cmd == 21:  # MAV_CMD_NAV_LAND
            with self._lock:
                self.state.mode = FlightMode.LAND
            print("[MockSITL] Land command")
            self._send_command_ack(cmd, 0)
    
    def _handle_position_target(self, msg) -> None:
        """Handle SET_POSITION_TARGET_LOCAL_NED message."""
        with self._lock:
            # Extract position setpoint
            type_mask = msg.type_mask
            
            # Check if position fields are valid
            if not (type_mask & 0x07):  # Position bits not ignored
                self.state.position_setpoint = np.array([msg.x, msg.y, msg.z])
            
            # Check if velocity fields are valid
            if not (type_mask & 0x38):  # Velocity bits not ignored
                self.state.velocity_setpoint = np.array([msg.vx, msg.vy, msg.vz])
    
    def _update_simulation(self, dt: float) -> None:
        """Update simulated state.
        
        Args:
            dt: Time step
        """
        with self._lock:
            if not self.state.armed:
                self._motor_outputs = np.zeros(4)
                return
            
            if self.state.mode == FlightMode.LAND:
                # Simple landing: descend at 0.5 m/s
                target_vel = np.array([0.0, 0.0, 0.5])  # Down is positive in NED
                self._compute_motors_for_velocity(target_vel)
                
                # Check if landed
                if self.state.position[2] > -0.1:  # Near ground (NED: down positive)
                    self.state.armed = False
                    self._motor_outputs = np.zeros(4)
            
            elif self.state.mode == FlightMode.OFFBOARD:
                if self.state.velocity_setpoint is not None:
                    self._compute_motors_for_velocity(self.state.velocity_setpoint)
                elif self.state.position_setpoint is not None:
                    # Position control
                    pos_error = self.state.position_setpoint - self.state.position
                    target_vel = pos_error * self.config.pos_p
                    target_vel = np.clip(target_vel, -5.0, 5.0)
                    self._compute_motors_for_velocity(target_vel)
                else:
                    # Hover
                    self._compute_motors_for_velocity(np.zeros(3))
            else:
                # Default: hover
                self._compute_motors_for_velocity(np.zeros(3))
    
    def _compute_motors_for_velocity(self, target_vel: np.ndarray) -> None:
        """Compute motor outputs for target velocity.
        
        Args:
            target_vel: Target velocity in NED (m/s)
        """
        # Velocity error
        vel_error = target_vel - self.state.velocity
        
        # Thrust for vertical (gravity + z control)
        # NED: positive z is down
        hover_thrust = 0.5  # Normalized hover thrust
        z_cmd = -vel_error[2] * self.config.vel_p / 20.0  # Negative because down is positive
        
        thrust = hover_thrust + z_cmd
        thrust = np.clip(thrust, 0.1, 0.9)
        
        # Simple attitude control for horizontal
        roll_cmd = -vel_error[1] * 0.1  # Roll for east velocity
        pitch_cmd = vel_error[0] * 0.1  # Pitch for north velocity
        
        # Motor mixing (simplified X quad)
        m1 = thrust + roll_cmd - pitch_cmd
        m2 = thrust - roll_cmd - pitch_cmd
        m3 = thrust - roll_cmd + pitch_cmd
        m4 = thrust + roll_cmd + pitch_cmd
        
        self._motor_outputs = np.clip([m1, m2, m3, m4], 0.0, 1.0)
    
    def _send_actuators(self) -> None:
        """Send HIL_ACTUATOR_CONTROLS message."""
        if self._client is None or self._mav is None:
            return
        
        # Create message
        controls = np.zeros(16)
        controls[:4] = self._motor_outputs * 2 - 1  # Convert [0,1] to [-1,1]
        
        try:
            msg = self._mav.hil_actuator_controls_encode(
                int(self.state.timestamp * 1e6),
                controls.tolist(),
                0,  # mode
                0,  # flags
            )
            self._client.send(msg.get_msgbuf())
            self.stats["messages_sent"] += 1
        except Exception as e:
            if self._running:
                print(f"[MockSITL] Send error: {e}")
    
    def _send_heartbeat(self) -> None:
        """Send HEARTBEAT message."""
        if self._client is None or self._mav is None:
            return
        
        try:
            # Determine mode flags
            base_mode = 0
            if self.state.armed:
                base_mode |= 128  # MAV_MODE_FLAG_SAFETY_ARMED
            
            custom_mode = self.state.mode.value << 16
            
            msg = self._mav.heartbeat_encode(
                2,  # MAV_TYPE_QUADROTOR
                12,  # MAV_AUTOPILOT_PX4
                base_mode,
                custom_mode,
                4,  # MAV_STATE_ACTIVE
            )
            self._client.send(msg.get_msgbuf())
        except Exception as e:
            if self._running:
                print(f"[MockSITL] Heartbeat error: {e}")
    
    def _send_command_ack(self, command: int, result: int) -> None:
        """Send COMMAND_ACK message."""
        if self._client is None or self._mav is None:
            return
        
        try:
            msg = self._mav.command_ack_encode(
                command,
                result,
                0,  # progress
                0,  # result_param2
                self.config.system_id,
                self.config.component_id,
            )
            self._client.send(msg.get_msgbuf())
        except Exception as e:
            if self._running:
                print(f"[MockSITL] ACK error: {e}")
    
    # =========================================================================
    # Test helpers
    # =========================================================================
    
    def get_state(self) -> MockState:
        """Get current state (thread-safe copy)."""
        with self._lock:
            return MockState(
                position=self.state.position.copy(),
                velocity=self.state.velocity.copy(),
                quaternion=self.state.quaternion.copy(),
                angular_velocity=self.state.angular_velocity.copy(),
                armed=self.state.armed,
                mode=self.state.mode,
                position_setpoint=self.state.position_setpoint.copy() if self.state.position_setpoint is not None else None,
                velocity_setpoint=self.state.velocity_setpoint.copy() if self.state.velocity_setpoint is not None else None,
                timestamp=self.state.timestamp,
            )
    
    def get_motor_outputs(self) -> np.ndarray:
        """Get current motor outputs."""
        with self._lock:
            return self._motor_outputs.copy()
    
    def is_armed(self) -> bool:
        """Check if armed."""
        with self._lock:
            return self.state.armed
    
    def get_mode(self) -> FlightMode:
        """Get current mode."""
        with self._lock:
            return self.state.mode


# Pytest fixture
import pytest

@pytest.fixture
def mock_sitl():
    """Pytest fixture for mock SITL."""
    mock = MockSITL()
    yield mock
    mock.stop()
