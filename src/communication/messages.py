"""MAVLink message data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HILSensorMessage:
    """HIL_SENSOR message data for PX4.

    Contains IMU, magnetometer, barometer, and other sensor data.
    All values in SI units.
    """

    time_usec: int  # Timestamp in microseconds

    # Accelerometer [m/s²], body frame FRD
    xacc: float
    yacc: float
    zacc: float

    # Gyroscope [rad/s], body frame FRD
    xgyro: float
    ygyro: float
    zgyro: float

    # Magnetometer [Gauss], body frame FRD
    xmag: float
    ymag: float
    zmag: float

    # Barometer
    abs_pressure: float  # Absolute pressure [mbar]
    diff_pressure: float  # Differential pressure [mbar]
    pressure_alt: float  # Pressure altitude [m]
    temperature: float  # Temperature [°C]

    # Bitmap of updated fields (0x1FFF = all fields)
    fields_updated: int = 0x1FFF

    # Optional: sensor IDs for multi-IMU
    id: int = 0

    @classmethod
    def from_sensor_data(
        cls,
        time_sec: float,
        gyro: np.ndarray,
        accel: np.ndarray,
        mag: np.ndarray,
        pressure: float,
        temperature: float,
        altitude: float,
    ) -> HILSensorMessage:
        """Create HIL_SENSOR message from sensor readings.

        Args:
            time_sec: Simulation time in seconds
            gyro: Gyroscope [rad/s], FRD body frame
            accel: Accelerometer [m/s²], FRD body frame
            mag: Magnetometer [Gauss], FRD body frame
            pressure: Absolute pressure [Pa]
            temperature: Temperature [°C]
            altitude: Altitude [m]

        Returns:
            HILSensorMessage instance
        """
        return cls(
            time_usec=int(time_sec * 1e6),
            xacc=accel[0],
            yacc=accel[1],
            zacc=accel[2],
            xgyro=gyro[0],
            ygyro=gyro[1],
            zgyro=gyro[2],
            xmag=mag[0],
            ymag=mag[1],
            zmag=mag[2],
            abs_pressure=pressure / 100.0,  # Pa to mbar
            diff_pressure=0.0,
            pressure_alt=altitude,
            temperature=temperature,
        )


@dataclass
class HILGPSMessage:
    """HIL_GPS message data for PX4.

    Contains GPS position, velocity, and status.
    """

    time_usec: int  # Timestamp in microseconds

    # Position
    lat: int  # Latitude [degE7]
    lon: int  # Longitude [degE7]
    alt: int  # Altitude [mm] AMSL

    # Accuracy
    eph: int  # GPS HDOP [cm]
    epv: int  # GPS VDOP [cm]

    # Velocity
    vel: int  # GPS ground speed [cm/s]
    vn: int  # GPS North velocity [cm/s]
    ve: int  # GPS East velocity [cm/s]
    vd: int  # GPS Down velocity [cm/s]
    cog: int  # Course over ground [cdeg]

    # Status
    fix_type: int  # 0-1: no fix, 2: 2D, 3: 3D
    satellites_visible: int

    # Optional
    id: int = 0
    yaw: int = 0  # Yaw [cdeg], 0 if unknown

    @classmethod
    def from_gps_data(
        cls,
        time_sec: float,
        lat: float,
        lon: float,
        alt: float,
        vel_ned: np.ndarray,
        hdop: float = 1.0,
        vdop: float = 1.5,
        fix_type: int = 3,
        satellites: int = 10,
    ) -> HILGPSMessage:
        """Create HIL_GPS message from GPS readings.

        Args:
            time_sec: Simulation time in seconds
            lat: Latitude [degrees]
            lon: Longitude [degrees]
            alt: Altitude [m] AMSL
            vel_ned: Velocity [m/s] in NED frame
            hdop: Horizontal dilution of precision
            vdop: Vertical dilution of precision
            fix_type: GPS fix type
            satellites: Number of visible satellites

        Returns:
            HILGPSMessage instance
        """
        # Ground speed
        ground_speed = np.sqrt(vel_ned[0] ** 2 + vel_ned[1] ** 2)

        # Course over ground
        if ground_speed > 0.1:
            cog = np.degrees(np.arctan2(vel_ned[1], vel_ned[0]))
            if cog < 0:
                cog += 360
        else:
            cog = 0.0

        return cls(
            time_usec=int(time_sec * 1e6),
            lat=int(lat * 1e7),
            lon=int(lon * 1e7),
            alt=int(alt * 1000),  # m to mm
            eph=int(hdop * 100),  # m to cm
            epv=int(vdop * 100),
            vel=int(ground_speed * 100),  # m/s to cm/s
            vn=int(vel_ned[0] * 100),
            ve=int(vel_ned[1] * 100),
            vd=int(vel_ned[2] * 100),
            cog=int(cog * 100),  # deg to cdeg
            fix_type=fix_type,
            satellites_visible=satellites,
        )


@dataclass
class HILStateQuaternion:
    """HIL_STATE_QUATERNION message for ground truth state.

    Provides full vehicle state for simulator feedback.
    """

    time_usec: int  # Timestamp in microseconds

    # Attitude quaternion [w, x, y, z]
    attitude_quaternion: np.ndarray

    # Angular velocity [rad/s], body frame
    rollspeed: float
    pitchspeed: float
    yawspeed: float

    # Position [m], NED frame (but as lat/lon/alt)
    lat: int  # Latitude [degE7]
    lon: int  # Longitude [degE7]
    alt: int  # Altitude [mm] AMSL

    # Velocity [m/s], NED frame
    vx: int  # [cm/s]
    vy: int  # [cm/s]
    vz: int  # [cm/s]

    # Airspeed
    ind_airspeed: int  # [cm/s]
    true_airspeed: int  # [cm/s]

    # Accelerations [mG]
    xacc: int
    yacc: int
    zacc: int


@dataclass
class HILActuatorControls:
    """HIL_ACTUATOR_CONTROLS message from PX4.

    Contains motor/actuator commands from flight controller.
    """

    time_usec: int  # Timestamp in microseconds
    controls: np.ndarray  # Control outputs [-1, 1], shape (16,)
    mode: int  # System mode (MAV_MODE)
    flags: int  # Flags

    @property
    def motor_commands(self) -> np.ndarray:
        """Get motor commands (first 4 controls).

        Returns:
            Motor commands [0, 1] for 4 motors
        """
        # PX4 outputs [-1, 1], convert to [0, 1] for motors
        return (self.controls[:4] + 1.0) / 2.0

    @classmethod
    def from_controls(
        cls,
        time_usec: int,
        controls: np.ndarray,
        mode: int = 0,
        flags: int = 0,
    ) -> HILActuatorControls:
        """Create from control array.

        Args:
            time_usec: Timestamp
            controls: Control values
            mode: System mode
            flags: Flags

        Returns:
            HILActuatorControls instance
        """
        full_controls = np.zeros(16)
        full_controls[: len(controls)] = controls

        return cls(
            time_usec=time_usec,
            controls=full_controls,
            mode=mode,
            flags=flags,
        )


@dataclass
class Heartbeat:
    """MAVLink HEARTBEAT message."""

    type: int  # MAV_TYPE
    autopilot: int  # MAV_AUTOPILOT
    base_mode: int  # MAV_MODE_FLAG
    custom_mode: int
    system_status: int  # MAV_STATE
    mavlink_version: int = 3

    @classmethod
    def simulator_heartbeat(cls) -> Heartbeat:
        """Create heartbeat for simulator."""
        return cls(
            type=6,  # MAV_TYPE_GCS (simulator acts as GCS)
            autopilot=8,  # MAV_AUTOPILOT_INVALID
            base_mode=0,
            custom_mode=0,
            system_status=4,  # MAV_STATE_ACTIVE
        )


# ============================================================================
# Offboard Control Messages
# ============================================================================


# PX4 Custom Mode Constants
class PX4Mode:
    """PX4 flight mode constants."""

    MAIN_MODE_MANUAL = 1
    MAIN_MODE_ALTCTL = 2
    MAIN_MODE_POSCTL = 3
    MAIN_MODE_AUTO = 4
    MAIN_MODE_OFFBOARD = 6

    SUB_MODE_AUTO_LOITER = 3
    SUB_MODE_AUTO_MISSION = 4
    SUB_MODE_AUTO_LAND = 6

    @staticmethod
    def custom_mode(main_mode: int, sub_mode: int = 0) -> int:
        """Create PX4 custom mode value."""
        return (main_mode << 16) | (sub_mode << 24)

    @staticmethod
    def offboard_mode() -> int:
        """Get offboard mode custom value."""
        return PX4Mode.custom_mode(PX4Mode.MAIN_MODE_OFFBOARD)


# MAVLink type masks for SET_POSITION_TARGET_LOCAL_NED
class PositionTargetTypeMask:
    """Type mask bits for position target messages."""

    IGNORE_PX = 1
    IGNORE_PY = 2
    IGNORE_PZ = 4
    IGNORE_VX = 8
    IGNORE_VY = 16
    IGNORE_VZ = 32
    IGNORE_AFX = 64
    IGNORE_AFY = 128
    IGNORE_AFZ = 256
    FORCE = 512  # Use force instead of acceleration
    IGNORE_YAW = 1024
    IGNORE_YAW_RATE = 2048

    # Common masks
    POSITION_ONLY = (
        IGNORE_VX | IGNORE_VY | IGNORE_VZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | IGNORE_YAW_RATE
    )
    VELOCITY_ONLY = (
        IGNORE_PX | IGNORE_PY | IGNORE_PZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | IGNORE_YAW
    )
    POSITION_AND_VELOCITY = IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | IGNORE_YAW_RATE


@dataclass
class SetPositionTargetLocalNED:
    """SET_POSITION_TARGET_LOCAL_NED message for offboard control.

    Sends position, velocity, and/or acceleration setpoints to PX4.
    All values in NED frame (North-East-Down).
    """

    time_boot_ms: int  # Timestamp in milliseconds

    # Target system/component
    target_system: int = 1
    target_component: int = 1

    # Coordinate frame (MAV_FRAME)
    coordinate_frame: int = 1  # MAV_FRAME_LOCAL_NED

    # Type mask - specifies which fields to use
    type_mask: int = 0

    # Position [m] NED
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Velocity [m/s] NED
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Acceleration [m/s²] NED (or force if FORCE bit set)
    afx: float = 0.0
    afy: float = 0.0
    afz: float = 0.0

    # Yaw and yaw rate
    yaw: float = 0.0  # [rad]
    yaw_rate: float = 0.0  # [rad/s]

    @classmethod
    def from_position(
        cls,
        time_ms: int,
        position: np.ndarray,
        yaw: float = 0.0,
        target_system: int = 1,
    ) -> SetPositionTargetLocalNED:
        """Create position setpoint message.

        Args:
            time_ms: Timestamp in milliseconds
            position: Position [x, y, z] in NED [m]
            yaw: Yaw angle [rad]
            target_system: Target system ID

        Returns:
            SetPositionTargetLocalNED instance
        """
        return cls(
            time_boot_ms=time_ms,
            target_system=target_system,
            type_mask=PositionTargetTypeMask.POSITION_ONLY,
            x=position[0],
            y=position[1],
            z=position[2],
            yaw=yaw,
        )

    @classmethod
    def from_velocity(
        cls,
        time_ms: int,
        velocity: np.ndarray,
        yaw_rate: float = 0.0,
        target_system: int = 1,
    ) -> SetPositionTargetLocalNED:
        """Create velocity setpoint message.

        Args:
            time_ms: Timestamp in milliseconds
            velocity: Velocity [vx, vy, vz] in NED [m/s]
            yaw_rate: Yaw rate [rad/s]
            target_system: Target system ID

        Returns:
            SetPositionTargetLocalNED instance
        """
        return cls(
            time_boot_ms=time_ms,
            target_system=target_system,
            type_mask=PositionTargetTypeMask.VELOCITY_ONLY,
            vx=velocity[0],
            vy=velocity[1],
            vz=velocity[2],
            yaw_rate=yaw_rate,
        )

    @classmethod
    def from_position_velocity(
        cls,
        time_ms: int,
        position: np.ndarray,
        velocity: np.ndarray,
        yaw: float = 0.0,
        target_system: int = 1,
    ) -> SetPositionTargetLocalNED:
        """Create combined position and velocity setpoint.

        Args:
            time_ms: Timestamp in milliseconds
            position: Position [x, y, z] in NED [m]
            velocity: Velocity [vx, vy, vz] in NED [m/s]
            yaw: Yaw angle [rad]
            target_system: Target system ID

        Returns:
            SetPositionTargetLocalNED instance
        """
        return cls(
            time_boot_ms=time_ms,
            target_system=target_system,
            type_mask=PositionTargetTypeMask.POSITION_AND_VELOCITY,
            x=position[0],
            y=position[1],
            z=position[2],
            vx=velocity[0],
            vy=velocity[1],
            vz=velocity[2],
            yaw=yaw,
        )


@dataclass
class CommandLong:
    """COMMAND_LONG message for sending commands to PX4.

    Used for arming, mode changes, and other commands.
    """

    target_system: int = 1
    target_component: int = 1
    command: int = 0  # MAV_CMD
    confirmation: int = 0
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    param5: float = 0.0
    param6: float = 0.0
    param7: float = 0.0

    # Common MAV_CMD values
    MAV_CMD_COMPONENT_ARM_DISARM = 400
    MAV_CMD_DO_SET_MODE = 176
    MAV_CMD_NAV_TAKEOFF = 22
    MAV_CMD_NAV_LAND = 21

    @classmethod
    def arm(cls, arm: bool = True, target_system: int = 1) -> CommandLong:
        """Create arm/disarm command.

        Args:
            arm: True to arm, False to disarm
            target_system: Target system ID

        Returns:
            CommandLong instance
        """
        return cls(
            target_system=target_system,
            command=cls.MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1.0 if arm else 0.0,
            param2=21196.0 if not arm else 0.0,  # Force disarm magic number
        )

    @classmethod
    def set_mode(
        cls,
        mode: int,
        custom_mode: int = 0,
        target_system: int = 1,
    ) -> CommandLong:
        """Create set mode command.

        Args:
            mode: Base mode (MAV_MODE)
            custom_mode: Custom mode (PX4 specific)
            target_system: Target system ID

        Returns:
            CommandLong instance
        """
        return cls(
            target_system=target_system,
            command=cls.MAV_CMD_DO_SET_MODE,
            param1=float(mode),
            param2=float(custom_mode),
        )

    @classmethod
    def set_offboard_mode(cls, target_system: int = 1) -> CommandLong:
        """Create command to set offboard mode.

        Args:
            target_system: Target system ID

        Returns:
            CommandLong instance
        """
        # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
        return cls.set_mode(
            mode=1,  # Custom mode enabled
            custom_mode=PX4Mode.offboard_mode(),
            target_system=target_system,
        )

    @classmethod
    def takeoff(
        cls,
        altitude: float,
        target_system: int = 1,
    ) -> CommandLong:
        """Create takeoff command.

        Args:
            altitude: Target altitude [m]
            target_system: Target system ID

        Returns:
            CommandLong instance
        """
        return cls(
            target_system=target_system,
            command=cls.MAV_CMD_NAV_TAKEOFF,
            param7=altitude,
        )

    @classmethod
    def land(cls, target_system: int = 1) -> CommandLong:
        """Create land command.

        Args:
            target_system: Target system ID

        Returns:
            CommandLong instance
        """
        return cls(
            target_system=target_system,
            command=cls.MAV_CMD_NAV_LAND,
        )


@dataclass
class SetpointCommand:
    """Neural network output converted to PX4 setpoint.

    High-level setpoint that can be converted to MAVLink messages.
    """

    # Position setpoint (NED frame, meters)
    position: np.ndarray | None = None  # [x, y, z]

    # Velocity setpoint (NED frame, m/s)
    velocity: np.ndarray | None = None  # [vx, vy, vz]

    # Yaw control
    yaw: float | None = None  # radians
    yaw_rate: float | None = None  # rad/s

    def to_mavlink_message(self, time_ms: int) -> SetPositionTargetLocalNED:
        """Convert to MAVLink SET_POSITION_TARGET_LOCAL_NED message.

        Args:
            time_ms: Timestamp in milliseconds

        Returns:
            SetPositionTargetLocalNED message
        """
        if self.position is not None and self.velocity is not None:
            return SetPositionTargetLocalNED.from_position_velocity(
                time_ms=time_ms,
                position=self.position,
                velocity=self.velocity,
                yaw=self.yaw or 0.0,
            )
        elif self.position is not None:
            return SetPositionTargetLocalNED.from_position(
                time_ms=time_ms,
                position=self.position,
                yaw=self.yaw or 0.0,
            )
        elif self.velocity is not None:
            return SetPositionTargetLocalNED.from_velocity(
                time_ms=time_ms,
                velocity=self.velocity,
                yaw_rate=self.yaw_rate or 0.0,
            )
        else:
            # Default: hold position at origin
            return SetPositionTargetLocalNED.from_position(
                time_ms=time_ms,
                position=np.zeros(3),
            )
