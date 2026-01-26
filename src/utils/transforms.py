"""Coordinate frame transformations.

Frame Definitions:
- MuJoCo: X-forward, Y-left, Z-up (ENU-like)
- NED: X-North, Y-East, Z-Down (aviation standard)
- FRD: X-Forward, Y-Right, Z-Down (body frame)
- ENU: X-East, Y-North, Z-Up (robotics common)
"""

from __future__ import annotations

import numpy as np


class CoordinateTransforms:
    """Static methods for coordinate frame transformations.

    MuJoCo uses a right-handed coordinate system with Z-up.
    PX4/MAVLink uses NED (North-East-Down) for world frame and
    FRD (Forward-Right-Down) for body frame.
    """

    # Rotation matrix from MuJoCo to NED frame
    # MuJoCo: X-forward, Y-left, Z-up
    # NED: X-north, Y-east, Z-down
    # Assuming MuJoCo X aligns with NED X (North):
    # NED_x = MuJoCo_x
    # NED_y = -MuJoCo_y (left -> right = east)
    # NED_z = -MuJoCo_z (up -> down)
    R_MUJOCO_TO_NED = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    R_NED_TO_MUJOCO = R_MUJOCO_TO_NED.T  # Transpose for inverse

    # MuJoCo body frame to FRD body frame
    # MuJoCo body: X-forward, Y-left, Z-up
    # FRD: X-forward, Y-right, Z-down
    R_MUJOCO_BODY_TO_FRD = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    R_FRD_TO_MUJOCO_BODY = R_MUJOCO_BODY_TO_FRD.T

    @staticmethod
    def position_mujoco_to_ned(pos_mujoco: np.ndarray) -> np.ndarray:
        """Convert position from MuJoCo frame to NED frame.

        Args:
            pos_mujoco: Position in MuJoCo frame [x, y, z]

        Returns:
            Position in NED frame [north, east, down]
        """
        return CoordinateTransforms.R_MUJOCO_TO_NED @ pos_mujoco

    @staticmethod
    def position_ned_to_mujoco(pos_ned: np.ndarray) -> np.ndarray:
        """Convert position from NED frame to MuJoCo frame.

        Args:
            pos_ned: Position in NED frame [north, east, down]

        Returns:
            Position in MuJoCo frame [x, y, z]
        """
        return CoordinateTransforms.R_NED_TO_MUJOCO @ pos_ned

    @staticmethod
    def velocity_mujoco_to_ned(vel_mujoco: np.ndarray) -> np.ndarray:
        """Convert velocity from MuJoCo frame to NED frame."""
        return CoordinateTransforms.R_MUJOCO_TO_NED @ vel_mujoco

    @staticmethod
    def velocity_ned_to_mujoco(vel_ned: np.ndarray) -> np.ndarray:
        """Convert velocity from NED frame to MuJoCo frame."""
        return CoordinateTransforms.R_NED_TO_MUJOCO @ vel_ned

    @staticmethod
    def angular_velocity_mujoco_to_frd(omega_mujoco: np.ndarray) -> np.ndarray:
        """Convert angular velocity from MuJoCo body to FRD body frame.

        Args:
            omega_mujoco: Angular velocity in MuJoCo body frame [p, q, r]

        Returns:
            Angular velocity in FRD frame [p, q, r]
        """
        return CoordinateTransforms.R_MUJOCO_BODY_TO_FRD @ omega_mujoco

    @staticmethod
    def angular_velocity_frd_to_mujoco(omega_frd: np.ndarray) -> np.ndarray:
        """Convert angular velocity from FRD to MuJoCo body frame."""
        return CoordinateTransforms.R_FRD_TO_MUJOCO_BODY @ omega_frd

    @staticmethod
    def acceleration_mujoco_to_frd(accel_mujoco: np.ndarray) -> np.ndarray:
        """Convert acceleration from MuJoCo body to FRD body frame."""
        return CoordinateTransforms.R_MUJOCO_BODY_TO_FRD @ accel_mujoco

    @staticmethod
    def quaternion_mujoco_to_ned(quat_mujoco: np.ndarray) -> np.ndarray:
        """Convert quaternion from MuJoCo convention to NED convention.

        MuJoCo quaternion represents rotation from world to body.
        We need to account for the frame difference.

        Args:
            quat_mujoco: Quaternion [w, x, y, z] in MuJoCo convention

        Returns:
            Quaternion [w, x, y, z] in NED convention
        """
        # The quaternion transformation involves composing rotations
        # q_ned = R_frame * q_mujoco * R_frame.T

        w, x, y, z = quat_mujoco

        # For our frame transformation (flip Y and Z):
        # q_ned = [w, x, -y, -z]
        return np.array([w, x, -y, -z])

    @staticmethod
    def quaternion_ned_to_mujoco(quat_ned: np.ndarray) -> np.ndarray:
        """Convert quaternion from NED convention to MuJoCo convention."""
        w, x, y, z = quat_ned
        return np.array([w, x, -y, -z])

    @staticmethod
    def euler_from_quaternion_ned(quat: np.ndarray) -> np.ndarray:
        """Convert NED quaternion to Euler angles.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        w, x, y, z = quat

        # Roll (rotation around x-axis)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (rotation around y-axis)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)

        # Yaw (rotation around z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    @staticmethod
    def quaternion_from_euler_ned(euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to NED quaternion.

        Args:
            euler: Euler angles [roll, pitch, yaw] in radians

        Returns:
            Quaternion [w, x, y, z]
        """
        roll, pitch, yaw = euler

        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return np.array([w, x, y, z])

    @staticmethod
    def rotation_matrix_from_quaternion(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Rotation matrix (3, 3)
        """
        w, x, y, z = quat

        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

        return R

    @staticmethod
    def state_mujoco_to_ned(
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert full state from MuJoCo to NED convention.

        Args:
            position: Position in MuJoCo frame
            velocity: Velocity in MuJoCo frame
            quaternion: Orientation quaternion (MuJoCo convention)
            angular_velocity: Angular velocity in MuJoCo body frame

        Returns:
            Tuple of (position_ned, velocity_ned, quaternion_ned, angular_velocity_frd)
        """
        ct = CoordinateTransforms

        pos_ned = ct.position_mujoco_to_ned(position)
        vel_ned = ct.velocity_mujoco_to_ned(velocity)
        quat_ned = ct.quaternion_mujoco_to_ned(quaternion)
        omega_frd = ct.angular_velocity_mujoco_to_frd(angular_velocity)

        return pos_ned, vel_ned, quat_ned, omega_frd

    @staticmethod
    def state_ned_to_mujoco(
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert full state from NED to MuJoCo convention.

        Args:
            position: Position in NED frame
            velocity: Velocity in NED frame
            quaternion: Orientation quaternion (NED convention)
            angular_velocity: Angular velocity in FRD body frame

        Returns:
            Tuple of (position_mujoco, velocity_mujoco, quaternion_mujoco, angular_velocity_mujoco)
        """
        ct = CoordinateTransforms

        pos_mujoco = ct.position_ned_to_mujoco(position)
        vel_mujoco = ct.velocity_ned_to_mujoco(velocity)
        quat_mujoco = ct.quaternion_ned_to_mujoco(quaternion)
        omega_mujoco = ct.angular_velocity_frd_to_mujoco(angular_velocity)

        return pos_mujoco, vel_mujoco, quat_mujoco, omega_mujoco


class GPSTransforms:
    """GPS coordinate transformations."""

    EARTH_RADIUS = 6371000.0  # meters

    @staticmethod
    def ned_to_geodetic(
        position_ned: np.ndarray,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
    ) -> tuple[float, float, float]:
        """Convert NED position to geodetic coordinates.

        Args:
            position_ned: Position in NED frame [north, east, down] in meters
            ref_lat: Reference latitude in degrees
            ref_lon: Reference longitude in degrees
            ref_alt: Reference altitude in meters (AMSL)

        Returns:
            Tuple of (latitude, longitude, altitude) in (degrees, degrees, meters)
        """
        north, east, down = position_ned

        # Latitude change
        lat = ref_lat + np.degrees(north / GPSTransforms.EARTH_RADIUS)

        # Longitude change (accounts for latitude)
        lon = ref_lon + np.degrees(
            east / (GPSTransforms.EARTH_RADIUS * np.cos(np.radians(ref_lat)))
        )

        # Altitude (NED down is negative altitude)
        alt = ref_alt - down

        return lat, lon, alt

    @staticmethod
    def geodetic_to_ned(
        lat: float,
        lon: float,
        alt: float,
        ref_lat: float,
        ref_lon: float,
        ref_alt: float,
    ) -> np.ndarray:
        """Convert geodetic coordinates to NED position.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters (AMSL)
            ref_lat: Reference latitude in degrees
            ref_lon: Reference longitude in degrees
            ref_alt: Reference altitude in meters (AMSL)

        Returns:
            Position in NED frame [north, east, down] in meters
        """
        north = np.radians(lat - ref_lat) * GPSTransforms.EARTH_RADIUS
        east = np.radians(lon - ref_lon) * GPSTransforms.EARTH_RADIUS * np.cos(np.radians(ref_lat))
        down = ref_alt - alt

        return np.array([north, east, down])
