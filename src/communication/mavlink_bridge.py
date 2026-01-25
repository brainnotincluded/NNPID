"""MAVLink bridge for PX4 SITL communication."""

from __future__ import annotations

import asyncio
import socket
import struct
import time
import threading
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
from queue import Queue, Empty

import numpy as np

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink2
    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    mavutil = None
    mavlink2 = None

from .messages import (
    HILSensorMessage,
    HILGPSMessage,
    HILActuatorControls,
    HILStateQuaternion,
    SetPositionTargetLocalNED,
    CommandLong,
    SetpointCommand,
    PX4Mode,
)
from .lockstep import LockstepController


@dataclass
class MAVLinkConfig:
    """Configuration for MAVLink bridge."""
    
    host: str = "127.0.0.1"
    port: int = 4560
    lockstep: bool = True
    
    # Message rates (Hz)
    hil_sensor_rate: int = 250
    hil_gps_rate: int = 10
    hil_state_rate: int = 50
    
    # Timeouts
    connection_timeout: float = 30.0
    receive_timeout: float = 0.1
    heartbeat_interval: float = 1.0
    
    # System IDs
    system_id: int = 1
    component_id: int = 1


class MAVLinkBridge:
    """Bridge between MuJoCo simulator and PX4 SITL via MAVLink.
    
    Implements TCP server for PX4 connection and handles HIL messages:
    - Receives HIL_ACTUATOR_CONTROLS from PX4
    - Sends HIL_SENSOR, HIL_GPS, HIL_STATE_QUATERNION to PX4
    
    Supports lockstep mode for deterministic simulation.
    """
    
    def __init__(self, config: Optional[MAVLinkConfig] = None):
        """Initialize MAVLink bridge.
        
        Args:
            config: MAVLink configuration. Uses defaults if None.
        """
        if not PYMAVLINK_AVAILABLE:
            raise ImportError(
                "pymavlink is required for MAVLink communication. "
                "Install with: pip install pymavlink"
            )
        
        self.config = config or MAVLinkConfig()
        
        # Connection state
        self._socket: Optional[socket.socket] = None
        self._connection: Optional[mavutil.mavlink_connection] = None
        self._connected = False
        self._running = False
        
        # Lockstep controller
        self.lockstep = LockstepController(
            enabled=self.config.lockstep,
            timeout=self.config.receive_timeout,
        )
        
        # Message queues
        self._actuator_queue: Queue[HILActuatorControls] = Queue(maxsize=10)
        self._last_actuators: Optional[HILActuatorControls] = None
        
        # Callbacks
        self._on_actuators: Optional[Callable[[HILActuatorControls], None]] = None
        
        # Thread for receiving messages
        self._receive_thread: Optional[threading.Thread] = None
        
        # Heartbeat timing
        self._last_heartbeat_time = 0.0
    
    def start_server(self) -> bool:
        """Start TCP server and wait for PX4 connection.
        
        Returns:
            True if connection established, False on timeout
        """
        print(f"Starting MAVLink server on {self.config.host}:{self.config.port}")
        
        # Create TCP server socket
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.config.host, self.config.port))
        self._socket.listen(1)
        self._socket.settimeout(self.config.connection_timeout)
        
        print(f"Waiting for PX4 SITL connection (timeout: {self.config.connection_timeout}s)...")
        
        try:
            conn, addr = self._socket.accept()
            print(f"PX4 connected from {addr}")
            
            # Create MAVLink connection
            self._connection = mavutil.mavlink_connection(
                f"tcp:{self.config.host}:{self.config.port}",
                source_system=255,  # GCS system ID
                source_component=0,
                input=False,  # We're not reading from this
            )
            
            # Store the connection socket
            self._connection.port = conn
            
            self._connected = True
            self._running = True
            
            # Start receive thread
            self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._receive_thread.start()
            
            return True
            
        except socket.timeout:
            print("Connection timeout - no PX4 connection received")
            self.stop()
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the MAVLink bridge."""
        self._running = False
        self._connected = False
        
        if self._connection is not None:
            try:
                self._connection.close()
            except:
                pass
            self._connection = None
        
        if self._socket is not None:
            try:
                self._socket.close()
            except:
                pass
            self._socket = None
        
        # Wait for receive thread
        if self._receive_thread is not None and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=1.0)
    
    def _receive_loop(self) -> None:
        """Background thread for receiving MAVLink messages."""
        while self._running and self._connected:
            try:
                msg = self._receive_message(timeout=0.01)
                if msg is not None:
                    self._handle_message(msg)
            except Exception as e:
                if self._running:
                    print(f"Receive error: {e}")
                break
    
    def _receive_message(self, timeout: float = 0.1):
        """Receive a MAVLink message.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            MAVLink message or None
        """
        if not self._connected or self._connection is None:
            return None
        
        try:
            self._connection.port.settimeout(timeout)
            msg = self._connection.recv_match(blocking=True, timeout=timeout)
            return msg
        except socket.timeout:
            return None
        except Exception as e:
            return None
    
    def _handle_message(self, msg) -> None:
        """Handle received MAVLink message.
        
        Args:
            msg: MAVLink message
        """
        msg_type = msg.get_type()
        
        if msg_type == "HIL_ACTUATOR_CONTROLS":
            actuators = HILActuatorControls(
                time_usec=msg.time_usec,
                controls=np.array(msg.controls),
                mode=msg.mode,
                flags=msg.flags,
            )
            
            self._last_actuators = actuators
            
            try:
                self._actuator_queue.put_nowait(actuators)
            except:
                pass  # Queue full, drop oldest
            
            if self._on_actuators is not None:
                self._on_actuators(actuators)
            
            if self.config.lockstep:
                self.lockstep.actuators_received(msg.time_usec)
        
        elif msg_type == "HEARTBEAT":
            # PX4 heartbeat received
            pass
    
    def send_hil_sensor(self, sensor: HILSensorMessage) -> bool:
        """Send HIL_SENSOR message to PX4.
        
        Args:
            sensor: Sensor data message
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        try:
            self._connection.mav.hil_sensor_send(
                sensor.time_usec,
                sensor.xacc,
                sensor.yacc,
                sensor.zacc,
                sensor.xgyro,
                sensor.ygyro,
                sensor.zgyro,
                sensor.xmag,
                sensor.ymag,
                sensor.zmag,
                sensor.abs_pressure,
                sensor.diff_pressure,
                sensor.pressure_alt,
                sensor.temperature,
                sensor.fields_updated,
                sensor.id,
            )
            return True
        except Exception as e:
            print(f"Failed to send HIL_SENSOR: {e}")
            return False
    
    def send_hil_gps(self, gps: HILGPSMessage) -> bool:
        """Send HIL_GPS message to PX4.
        
        Args:
            gps: GPS data message
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        try:
            self._connection.mav.hil_gps_send(
                gps.time_usec,
                gps.fix_type,
                gps.lat,
                gps.lon,
                gps.alt,
                gps.eph,
                gps.epv,
                gps.vel,
                gps.vn,
                gps.ve,
                gps.vd,
                gps.cog,
                gps.satellites_visible,
                gps.id,
                gps.yaw,
            )
            return True
        except Exception as e:
            print(f"Failed to send HIL_GPS: {e}")
            return False
    
    def send_hil_state_quaternion(
        self,
        time_usec: int,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        lat: int,
        lon: int,
        alt: int,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        airspeed: float = 0.0,
    ) -> bool:
        """Send HIL_STATE_QUATERNION message to PX4.
        
        Args:
            time_usec: Timestamp in microseconds
            quaternion: Attitude quaternion [w, x, y, z]
            angular_velocity: Angular velocity [rad/s]
            lat: Latitude [degE7]
            lon: Longitude [degE7]
            alt: Altitude [mm]
            velocity: Velocity [m/s] NED
            acceleration: Acceleration [m/s²]
            airspeed: Airspeed [m/s]
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        try:
            # Convert to expected units
            self._connection.mav.hil_state_quaternion_send(
                time_usec,
                list(quaternion),  # attitude_quaternion
                angular_velocity[0],  # rollspeed
                angular_velocity[1],  # pitchspeed
                angular_velocity[2],  # yawspeed
                lat,
                lon,
                alt,
                int(velocity[0] * 100),  # vx [cm/s]
                int(velocity[1] * 100),  # vy [cm/s]
                int(velocity[2] * 100),  # vz [cm/s]
                int(airspeed * 100),  # ind_airspeed [cm/s]
                int(airspeed * 100),  # true_airspeed [cm/s]
                int(acceleration[0] * 1000 / 9.81),  # xacc [mG]
                int(acceleration[1] * 1000 / 9.81),  # yacc [mG]
                int(acceleration[2] * 1000 / 9.81),  # zacc [mG]
            )
            return True
        except Exception as e:
            print(f"Failed to send HIL_STATE_QUATERNION: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to PX4.
        
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        now = time.time()
        if now - self._last_heartbeat_time < self.config.heartbeat_interval:
            return True
        
        try:
            self._connection.mav.heartbeat_send(
                6,  # MAV_TYPE_GCS
                8,  # MAV_AUTOPILOT_INVALID
                0,  # base_mode
                0,  # custom_mode
                4,  # MAV_STATE_ACTIVE
            )
            self._last_heartbeat_time = now
            return True
        except Exception as e:
            print(f"Failed to send heartbeat: {e}")
            return False
    
    def get_actuators(self, timeout: float = 0.1) -> Optional[HILActuatorControls]:
        """Get latest actuator commands.
        
        Args:
            timeout: How long to wait for new commands
            
        Returns:
            Actuator controls or None if not available
        """
        try:
            return self._actuator_queue.get(timeout=timeout)
        except Empty:
            return self._last_actuators
    
    def get_motor_commands(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get motor commands as normalized array [0, 1].
        
        Args:
            timeout: How long to wait for new commands
            
        Returns:
            Motor commands array or None
        """
        actuators = self.get_actuators(timeout)
        if actuators is None:
            return None
        return actuators.motor_commands
    
    def set_actuator_callback(
        self,
        callback: Callable[[HILActuatorControls], None],
    ) -> None:
        """Set callback for when actuator commands are received.
        
        Args:
            callback: Function to call with actuator data
        """
        self._on_actuators = callback
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to PX4."""
        return self._connected
    
    @property
    def last_actuators(self) -> Optional[HILActuatorControls]:
        """Get last received actuator commands."""
        return self._last_actuators
    
    # =========================================================================
    # Offboard Control Methods
    # =========================================================================
    
    def send_position_setpoint(
        self,
        position: np.ndarray,
        yaw: float = 0.0,
        time_boot_ms: Optional[int] = None,
    ) -> bool:
        """Send position setpoint for offboard control.
        
        Args:
            position: Target position [x, y, z] in NED frame [m]
            yaw: Target yaw angle [rad]
            time_boot_ms: Timestamp in milliseconds (auto-generated if None)
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        if time_boot_ms is None:
            time_boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
        
        msg = SetPositionTargetLocalNED.from_position(
            time_ms=time_boot_ms,
            position=position,
            yaw=yaw,
            target_system=self.config.system_id,
        )
        
        return self._send_position_target(msg)
    
    def send_velocity_setpoint(
        self,
        velocity: np.ndarray,
        yaw_rate: float = 0.0,
        time_boot_ms: Optional[int] = None,
    ) -> bool:
        """Send velocity setpoint for offboard control.
        
        Args:
            velocity: Target velocity [vx, vy, vz] in NED frame [m/s]
            yaw_rate: Target yaw rate [rad/s]
            time_boot_ms: Timestamp in milliseconds (auto-generated if None)
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        if time_boot_ms is None:
            time_boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
        
        msg = SetPositionTargetLocalNED.from_velocity(
            time_ms=time_boot_ms,
            velocity=velocity,
            yaw_rate=yaw_rate,
            target_system=self.config.system_id,
        )
        
        return self._send_position_target(msg)
    
    def send_setpoint(
        self,
        setpoint: SetpointCommand,
        time_boot_ms: Optional[int] = None,
    ) -> bool:
        """Send setpoint command for offboard control.
        
        Args:
            setpoint: SetpointCommand with position/velocity targets
            time_boot_ms: Timestamp in milliseconds (auto-generated if None)
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        if time_boot_ms is None:
            time_boot_ms = int(time.time() * 1000) & 0xFFFFFFFF
        
        msg = setpoint.to_mavlink_message(time_boot_ms)
        return self._send_position_target(msg)
    
    def _send_position_target(self, msg: SetPositionTargetLocalNED) -> bool:
        """Send SET_POSITION_TARGET_LOCAL_NED message.
        
        Args:
            msg: Position target message
            
        Returns:
            True if sent successfully
        """
        try:
            self._connection.mav.set_position_target_local_ned_send(
                msg.time_boot_ms,
                msg.target_system,
                msg.target_component,
                msg.coordinate_frame,
                msg.type_mask,
                msg.x, msg.y, msg.z,
                msg.vx, msg.vy, msg.vz,
                msg.afx, msg.afy, msg.afz,
                msg.yaw, msg.yaw_rate,
            )
            return True
        except Exception as e:
            print(f"Failed to send position target: {e}")
            return False
    
    def send_command(self, cmd: CommandLong) -> bool:
        """Send COMMAND_LONG message.
        
        Args:
            cmd: Command to send
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._connection is None:
            return False
        
        try:
            self._connection.mav.command_long_send(
                cmd.target_system,
                cmd.target_component,
                cmd.command,
                cmd.confirmation,
                cmd.param1,
                cmd.param2,
                cmd.param3,
                cmd.param4,
                cmd.param5,
                cmd.param6,
                cmd.param7,
            )
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False
    
    def arm(self, arm: bool = True) -> bool:
        """Arm or disarm the vehicle.
        
        Args:
            arm: True to arm, False to disarm
            
        Returns:
            True if command sent successfully
        """
        cmd = CommandLong.arm(arm=arm, target_system=self.config.system_id)
        return self.send_command(cmd)
    
    def disarm(self, force: bool = False) -> bool:
        """Disarm the vehicle.
        
        Args:
            force: Force disarm even in flight
            
        Returns:
            True if command sent successfully
        """
        cmd = CommandLong.arm(arm=False, target_system=self.config.system_id)
        if force:
            cmd.param2 = 21196.0  # Force disarm magic number
        return self.send_command(cmd)
    
    def set_offboard_mode(self) -> bool:
        """Switch to offboard control mode.
        
        Note: Setpoints must be streaming before calling this,
        otherwise PX4 will reject the mode change.
        
        Returns:
            True if command sent successfully
        """
        cmd = CommandLong.set_offboard_mode(target_system=self.config.system_id)
        return self.send_command(cmd)
    
    def takeoff(self, altitude: float) -> bool:
        """Command takeoff to specified altitude.
        
        Args:
            altitude: Target altitude [m]
            
        Returns:
            True if command sent successfully
        """
        cmd = CommandLong.takeoff(altitude=altitude, target_system=self.config.system_id)
        return self.send_command(cmd)
    
    def land(self) -> bool:
        """Command landing.
        
        Returns:
            True if command sent successfully
        """
        cmd = CommandLong.land(target_system=self.config.system_id)
        return self.send_command(cmd)


class SimpleTCPBridge:
    """Simplified TCP bridge for testing without full MAVLink.
    
    Uses a simple binary protocol for motor commands and sensor data.
    Useful for debugging when pymavlink is not available.
    """
    
    # Simple protocol:
    # Motor commands: 4 floats (16 bytes)
    # Sensor data: struct with timestamp, gyro, accel, position, velocity
    
    MOTOR_FORMAT = "<4f"  # 4 floats, little-endian
    SENSOR_FORMAT = "<d 3f 3f 3f 3f 4f"  # timestamp, gyro, accel, pos, vel, quat
    
    def __init__(self, host: str = "127.0.0.1", port: int = 9002):
        """Initialize simple bridge.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self._socket: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
    
    def start(self) -> bool:
        """Start TCP server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self.host, self.port))
        self._socket.settimeout(0.1)
        return True
    
    def stop(self) -> None:
        """Stop server."""
        if self._socket:
            self._socket.close()
            self._socket = None
    
    def receive_motors(self) -> Optional[np.ndarray]:
        """Receive motor commands."""
        if self._socket is None:
            return None
        
        try:
            data, addr = self._socket.recvfrom(1024)
            if len(data) >= 16:
                motors = struct.unpack(self.MOTOR_FORMAT, data[:16])
                return np.array(motors)
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Receive error: {e}")
        
        return None
    
    def send_sensors(
        self,
        timestamp: float,
        gyro: np.ndarray,
        accel: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        addr: Tuple[str, int],
    ) -> bool:
        """Send sensor data."""
        if self._socket is None:
            return False
        
        try:
            data = struct.pack(
                self.SENSOR_FORMAT,
                timestamp,
                *gyro, *accel, *position, *velocity, *quaternion,
            )
            self._socket.sendto(data, addr)
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
