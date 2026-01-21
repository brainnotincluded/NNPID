"""
Coordinate transformation utilities for drone control.
All transformations between Body Frame, NED (North-East-Down), and Global (GPS) coordinates.

Body Frame: x-forward, y-right, z-down (relative to drone orientation)
NED Frame: North-East-Down (local tangent plane)
Global Frame: Latitude, Longitude, Altitude

Critical for neural network: All observations MUST be in Body Frame for
location-invariant learning.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R


@dataclass
class Pose:
    """Represents a 6DOF pose (position + orientation)"""
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # Quaternion [w, x, y, z] or Euler [roll, pitch, yaw]
    frame: str = "NED"  # "NED", "BODY", or "GLOBAL"
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=np.float64)
        self.orientation = np.array(self.orientation, dtype=np.float64)


class CoordinateTransform:
    """
    Handles all coordinate transformations for drone navigation.
    Primary use: Convert target position to Body Frame for neural network input.
    """
    
    @staticmethod
    def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles (ZYX convention) to rotation matrix.
        
        Args:
            roll: Rotation around x-axis (radians)
            pitch: Rotation around y-axis (radians)
            yaw: Rotation around z-axis (radians)
            
        Returns:
            3x3 rotation matrix from NED to Body frame
        """
        # Using scipy for numerical stability
        rotation = R.from_euler('xyz', [roll, pitch, yaw])
        return rotation.as_matrix()
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        
        Args:
            quat: Quaternion [w, x, y, z] or [x, y, z, w] (auto-detected)
            
        Returns:
            3x3 rotation matrix
        """
        quat = np.array(quat, dtype=np.float64)
        
        # Auto-detect quaternion format
        if np.abs(quat[0]) > 0.9:  # Likely [w, x, y, z]
            quat = np.roll(quat, -1)  # Convert to [x, y, z, w]
        
        rotation = R.from_quat(quat)
        return rotation.as_matrix()
    
    @staticmethod
    def ned_to_body(
        position_ned: np.ndarray,
        drone_position_ned: np.ndarray,
        drone_orientation: np.ndarray,
        orientation_type: str = "euler"
    ) -> np.ndarray:
        """
        Transform position from NED frame to drone Body frame.
        
        This is THE CRITICAL transformation for neural network input.
        Network must see target position relative to drone's current orientation.
        
        Args:
            position_ned: Target position in NED [x, y, z] (meters)
            drone_position_ned: Drone position in NED [x, y, z] (meters)
            drone_orientation: Drone orientation (euler [r,p,y] or quat [w,x,y,z])
            orientation_type: "euler" or "quaternion"
            
        Returns:
            Position in Body frame [x_forward, y_right, z_down] (meters)
            
        Example:
            >>> target_ned = np.array([10.0, 5.0, -2.0])  # 10m North, 5m East, 2m up
            >>> drone_ned = np.array([0.0, 0.0, -1.5])    # Origin, 1.5m up
            >>> drone_euler = np.array([0.0, 0.0, np.pi/4])  # Yawed 45° to NE
            >>> target_body = ned_to_body(target_ned, drone_ned, drone_euler, "euler")
            >>> # target_body ≈ [10.6, -3.5, -0.5] (forward-right, slightly down)
        """
        # Relative position in NED
        relative_ned = position_ned - drone_position_ned
        
        # Get rotation matrix NED -> Body
        if orientation_type == "euler":
            R_matrix = CoordinateTransform.euler_to_rotation_matrix(*drone_orientation)
        elif orientation_type == "quaternion":
            R_matrix = CoordinateTransform.quaternion_to_rotation_matrix(drone_orientation)
        else:
            raise ValueError(f"Unknown orientation type: {orientation_type}")
        
        # Transform: Body = R @ NED
        position_body = R_matrix @ relative_ned
        
        return position_body
    
    @staticmethod
    def body_to_ned(
        position_body: np.ndarray,
        drone_position_ned: np.ndarray,
        drone_orientation: np.ndarray,
        orientation_type: str = "euler"
    ) -> np.ndarray:
        """
        Transform position from drone Body frame to NED frame.
        Used for converting neural network outputs back to world coordinates.
        
        Args:
            position_body: Position in Body frame [x, y, z] (meters)
            drone_position_ned: Drone position in NED [x, y, z] (meters)
            drone_orientation: Drone orientation
            orientation_type: "euler" or "quaternion"
            
        Returns:
            Position in NED frame [north, east, down] (meters)
        """
        # Get rotation matrix
        if orientation_type == "euler":
            R_matrix = CoordinateTransform.euler_to_rotation_matrix(*drone_orientation)
        else:
            R_matrix = CoordinateTransform.quaternion_to_rotation_matrix(drone_orientation)
        
        # Transform: NED = R^T @ Body (R is orthogonal, so R^T = R^-1)
        relative_ned = R_matrix.T @ position_body
        
        # Add drone position
        position_ned = drone_position_ned + relative_ned
        
        return position_ned
    
    @staticmethod
    def velocity_body_to_ned(
        velocity_body: np.ndarray,
        drone_orientation: np.ndarray,
        orientation_type: str = "euler"
    ) -> np.ndarray:
        """
        Transform velocity from Body frame to NED frame.
        
        Args:
            velocity_body: Velocity in Body frame [vx, vy, vz] (m/s)
            drone_orientation: Drone orientation
            orientation_type: "euler" or "quaternion"
            
        Returns:
            Velocity in NED frame [vn, ve, vd] (m/s)
        """
        if orientation_type == "euler":
            R_matrix = CoordinateTransform.euler_to_rotation_matrix(*drone_orientation)
        else:
            R_matrix = CoordinateTransform.quaternion_to_rotation_matrix(drone_orientation)
        
        return R_matrix.T @ velocity_body
    
    @staticmethod
    def velocity_ned_to_body(
        velocity_ned: np.ndarray,
        drone_orientation: np.ndarray,
        orientation_type: str = "euler"
    ) -> np.ndarray:
        """
        Transform velocity from NED frame to Body frame.
        
        Args:
            velocity_ned: Velocity in NED frame [vn, ve, vd] (m/s)
            drone_orientation: Drone orientation
            orientation_type: "euler" or "quaternion"
            
        Returns:
            Velocity in Body frame [vx, vy, vz] (m/s)
        """
        if orientation_type == "euler":
            R_matrix = CoordinateTransform.euler_to_rotation_matrix(*drone_orientation)
        else:
            R_matrix = CoordinateTransform.quaternion_to_rotation_matrix(drone_orientation)
        
        return R_matrix @ velocity_ned
    
    @staticmethod
    def global_to_ned(
        lat: float, 
        lon: float, 
        alt: float,
        home_lat: float,
        home_lon: float,
        home_alt: float
    ) -> np.ndarray:
        """
        Convert GPS coordinates to local NED frame.
        
        Args:
            lat, lon, alt: Target GPS coordinates (degrees, degrees, meters)
            home_lat, home_lon, home_alt: Home/origin GPS coordinates
            
        Returns:
            NED position [north, east, down] in meters
        """
        # Earth radius (meters)
        R_EARTH = 6371000.0
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        home_lat_rad = np.radians(home_lat)
        home_lon_rad = np.radians(home_lon)
        
        # Compute NED
        d_lat = lat_rad - home_lat_rad
        d_lon = lon_rad - home_lon_rad
        
        north = d_lat * R_EARTH
        east = d_lon * R_EARTH * np.cos(home_lat_rad)
        down = -(alt - home_alt)  # NED: down is positive
        
        return np.array([north, east, down], dtype=np.float64)
    
    @staticmethod
    def ned_to_global(
        north: float,
        east: float,
        down: float,
        home_lat: float,
        home_lon: float,
        home_alt: float
    ) -> Tuple[float, float, float]:
        """
        Convert local NED position to GPS coordinates.
        
        Args:
            north, east, down: NED position (meters)
            home_lat, home_lon, home_alt: Home GPS coordinates
            
        Returns:
            (latitude, longitude, altitude) in degrees, degrees, meters
        """
        R_EARTH = 6371000.0
        
        home_lat_rad = np.radians(home_lat)
        home_lon_rad = np.radians(home_lon)
        
        # Compute lat/lon
        d_lat = north / R_EARTH
        d_lon = east / (R_EARTH * np.cos(home_lat_rad))
        
        lat = np.degrees(home_lat_rad + d_lat)
        lon = np.degrees(home_lon_rad + d_lon)
        alt = home_alt - down  # NED down is negative altitude
        
        return lat, lon, alt
    
    @staticmethod
    def pixel_to_body(
        pixel_x: float,
        pixel_y: float,
        image_width: int,
        image_height: int,
        camera_fov_horizontal: float,
        camera_fov_vertical: float,
        target_distance: float,
        camera_pitch: float = 0.0
    ) -> np.ndarray:
        """
        Convert pixel coordinates from camera to Body frame position.
        Assumes camera is mounted on drone facing forward.
        
        Args:
            pixel_x, pixel_y: Target pixel coordinates
            image_width, image_height: Image dimensions (pixels)
            camera_fov_horizontal: Horizontal FOV (radians)
            camera_fov_vertical: Vertical FOV (radians)
            target_distance: Estimated distance to target (meters)
            camera_pitch: Camera pitch angle relative to body (radians, positive = down)
            
        Returns:
            Position in Body frame [x, y, z] (meters)
        """
        # Normalize pixel coordinates to [-1, 1]
        u = (2 * pixel_x / image_width) - 1
        v = (2 * pixel_y / image_height) - 1
        
        # Compute angles
        angle_horizontal = u * (camera_fov_horizontal / 2)
        angle_vertical = v * (camera_fov_vertical / 2)
        
        # Project to 3D (assuming target at distance D)
        # Camera frame: x-right, y-down, z-forward
        x_cam = target_distance * np.tan(angle_horizontal)
        y_cam = target_distance * np.tan(angle_vertical)
        z_cam = target_distance
        
        # Rotate by camera pitch to get Body frame
        # Body frame: x-forward, y-right, z-down
        cos_p = np.cos(camera_pitch)
        sin_p = np.sin(camera_pitch)
        
        x_body = z_cam * cos_p - y_cam * sin_p
        y_body = x_cam
        z_body = z_cam * sin_p + y_cam * cos_p
        
        return np.array([x_body, y_body, z_body], dtype=np.float64)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))


# ============================================================================
# USAGE EXAMPLES (can be run as tests)
# ============================================================================

if __name__ == "__main__":
    print("=== Coordinate Transform Tests ===\n")
    
    # Test 1: NED to Body (target ahead and to the right)
    print("Test 1: NED to Body")
    target_ned = np.array([10.0, 5.0, -2.0])  # 10m North, 5m East, 2m up
    drone_ned = np.array([0.0, 0.0, -1.5])     # At origin, 1.5m up
    drone_euler = np.array([0.0, 0.0, 0.0])    # No rotation
    
    target_body = CoordinateTransform.ned_to_body(
        target_ned, drone_ned, drone_euler, "euler"
    )
    print(f"Target NED: {target_ned}")
    print(f"Drone NED: {drone_ned}, Euler: {drone_euler}")
    print(f"Target Body: {target_body}")
    print(f"Expected: [10.0, 5.0, -0.5] (forward, right, up)")
    print()
    
    # Test 2: NED to Body with yaw rotation
    print("Test 2: NED to Body with 45° yaw")
    drone_euler_yawed = np.array([0.0, 0.0, np.pi/4])  # 45° yaw
    target_body_yawed = CoordinateTransform.ned_to_body(
        target_ned, drone_ned, drone_euler_yawed, "euler"
    )
    print(f"Drone Euler: {drone_euler_yawed} (45° yaw)")
    print(f"Target Body: {target_body_yawed}")
    print(f"Expected: [~10.6, -3.5, -0.5]")
    print()
    
    # Test 3: Velocity transformation
    print("Test 3: Velocity Body to NED")
    vel_body = np.array([2.0, 0.0, 0.0])  # 2 m/s forward
    drone_euler_90 = np.array([0.0, 0.0, np.pi/2])  # 90° yaw (facing East)
    vel_ned = CoordinateTransform.velocity_body_to_ned(vel_body, drone_euler_90, "euler")
    print(f"Velocity Body: {vel_body} (2 m/s forward)")
    print(f"Drone yaw: 90° (facing East)")
    print(f"Velocity NED: {vel_ned}")
    print(f"Expected: [0, 2, 0] (moving East)")
    print()
    
    print("=== All tests complete ===")
