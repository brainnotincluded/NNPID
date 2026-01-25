"""SITL passthrough controller."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any

from .base_controller import BaseController
from ..core.mujoco_sim import QuadrotorState
from ..communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig


class SITLController(BaseController):
    """Controller that gets commands from PX4 SITL.
    
    This controller doesn't compute actions itself - it receives
    motor commands from PX4 SITL via MAVLink and passes them through.
    
    The simulator sends sensor data to PX4, and PX4 sends back
    motor commands which this controller forwards to the simulation.
    """
    
    def __init__(
        self,
        mavlink_config: Optional[MAVLinkConfig] = None,
        auto_connect: bool = False,
    ):
        """Initialize SITL controller.
        
        Args:
            mavlink_config: MAVLink connection configuration
            auto_connect: Whether to auto-connect on initialization
        """
        super().__init__(name="SITLController")
        
        self._mavlink_config = mavlink_config or MAVLinkConfig()
        self._bridge: Optional[MAVLinkBridge] = None
        self._last_motor_commands = np.array([0.0, 0.0, 0.0, 0.0])
        self._connected = False
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """Connect to PX4 SITL.
        
        Returns:
            True if connection successful
        """
        if self._bridge is not None:
            self._bridge.stop()
        
        self._bridge = MAVLinkBridge(self._mavlink_config)
        
        if self._bridge.start_server():
            self._connected = True
            self._is_initialized = True
            return True
        
        return False
    
    def disconnect(self) -> None:
        """Disconnect from PX4 SITL."""
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge = None
        self._connected = False
    
    def compute_action(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Get motor commands from PX4 SITL.
        
        Note: This doesn't use target_position - PX4 handles its own
        navigation and control. The target should be sent to PX4
        via MAVLink mission commands.
        
        Args:
            state: Current quadrotor state (used for sending sensors)
            target_position: Ignored (PX4 has its own target)
            dt: Time step
            
        Returns:
            Motor commands from PX4
        """
        if not self._connected or self._bridge is None:
            # Return hover command if not connected
            return np.array([0.5, 0.5, 0.5, 0.5])
        
        # Try to get motor commands from PX4
        motors = self._bridge.get_motor_commands(timeout=0.001)
        
        if motors is not None:
            self._last_motor_commands = motors
        
        return self._last_motor_commands.copy()
    
    def send_sensors(
        self,
        state: QuadrotorState,
        gyro: np.ndarray,
        accel: np.ndarray,
        timestamp: float,
    ) -> bool:
        """Send sensor data to PX4.
        
        Args:
            state: Current quadrotor state
            gyro: Gyroscope reading [rad/s]
            accel: Accelerometer reading [m/s²]
            timestamp: Simulation time [s]
            
        Returns:
            True if sent successfully
        """
        if not self._connected or self._bridge is None:
            return False
        
        # Import here to avoid circular imports
        from ..communication.messages import HILSensorMessage, HILGPSMessage
        from ..core.sensors import SensorSimulator
        
        # Create sensor message
        sensor_msg = HILSensorMessage.from_sensor_data(
            time_sec=timestamp,
            gyro=gyro,
            accel=accel,
            mag=np.array([0.21, 0.0, 0.42]),  # Default mag field
            pressure=101325.0,
            temperature=20.0,
            altitude=state.position[2],
        )
        
        success = self._bridge.send_hil_sensor(sensor_msg)
        
        # Send GPS at lower rate (handled by lockstep controller)
        if self._bridge.lockstep.should_send_gps():
            # Create GPS message (simplified)
            gps_msg = HILGPSMessage.from_gps_data(
                time_sec=timestamp,
                lat=47.397742,  # Reference location
                lon=8.545594,
                alt=488.0 + state.position[2],
                vel_ned=state.velocity,
            )
            self._bridge.send_hil_gps(gps_msg)
        
        return success
    
    def reset(self) -> None:
        """Reset controller state."""
        self._last_motor_commands = np.array([0.0, 0.0, 0.0, 0.0])
        
        if self._bridge is not None:
            self._bridge.lockstep.reset()
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to PX4."""
        return self._connected
    
    @property
    def bridge(self) -> Optional[MAVLinkBridge]:
        """Get MAVLink bridge instance."""
        return self._bridge
    
    def get_info(self) -> Dict[str, Any]:
        """Get controller info."""
        info = super().get_info()
        info.update({
            "connected": self._connected,
            "last_motors": self._last_motor_commands.tolist(),
        })
        
        if self._bridge is not None:
            info["lockstep_stats"] = {
                "total_steps": self._bridge.lockstep.statistics.total_steps,
                "avg_wait_time": self._bridge.lockstep.statistics.avg_wait_time,
            }
        
        return info
