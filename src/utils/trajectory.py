"""Trajectory generation utilities."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Callable
from enum import Enum


class TrajectoryType(Enum):
    """Types of trajectories."""
    HOVER = "hover"
    WAYPOINT = "waypoint"
    CIRCLE = "circle"
    FIGURE_EIGHT = "figure_eight"
    HELIX = "helix"
    LEMNISCATE = "lemniscate"
    CUSTOM = "custom"


@dataclass
class TrajectoryPoint:
    """A single point in a trajectory."""
    
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    yaw: float = 0.0  # yaw angle in radians
    yaw_rate: float = 0.0  # yaw rate in rad/s
    time: float = 0.0  # time in seconds


class TrajectoryGenerator:
    """Generate various reference trajectories for drone control."""
    
    def __init__(self):
        """Initialize trajectory generator."""
        self._custom_trajectory: Optional[Callable[[float], TrajectoryPoint]] = None
    
    def hover(
        self,
        position: np.ndarray,
        yaw: float = 0.0,
        time: float = 0.0,
    ) -> TrajectoryPoint:
        """Generate hover trajectory at a fixed position.
        
        Args:
            position: Hover position [x, y, z]
            yaw: Desired yaw angle
            time: Current time (unused for hover)
            
        Returns:
            TrajectoryPoint at hover position
        """
        return TrajectoryPoint(
            position=np.array(position),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=yaw,
            yaw_rate=0.0,
            time=time,
        )
    
    def circle(
        self,
        center: np.ndarray,
        radius: float,
        altitude: float,
        angular_velocity: float,
        time: float,
        phase: float = 0.0,
    ) -> TrajectoryPoint:
        """Generate circular trajectory.
        
        Args:
            center: Circle center [x, y]
            radius: Circle radius in meters
            altitude: Flight altitude (z in NED is negative for up)
            angular_velocity: Angular velocity in rad/s
            time: Current time
            phase: Initial phase offset in radians
            
        Returns:
            TrajectoryPoint on circular path
        """
        theta = angular_velocity * time + phase
        
        # Position
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = altitude
        
        # Velocity
        vx = -radius * angular_velocity * np.sin(theta)
        vy = radius * angular_velocity * np.cos(theta)
        vz = 0.0
        
        # Acceleration
        ax = -radius * angular_velocity**2 * np.cos(theta)
        ay = -radius * angular_velocity**2 * np.sin(theta)
        az = 0.0
        
        # Yaw points in direction of motion
        yaw = theta + np.pi / 2
        
        return TrajectoryPoint(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            acceleration=np.array([ax, ay, az]),
            yaw=yaw,
            yaw_rate=angular_velocity,
            time=time,
        )
    
    def figure_eight(
        self,
        center: np.ndarray,
        size: float,
        altitude: float,
        period: float,
        time: float,
    ) -> TrajectoryPoint:
        """Generate figure-eight trajectory (lemniscate of Bernoulli).
        
        Args:
            center: Center of figure-eight [x, y]
            size: Size parameter (half-width)
            altitude: Flight altitude
            period: Time for one complete figure-eight
            time: Current time
            
        Returns:
            TrajectoryPoint on figure-eight path
        """
        omega = 2 * np.pi / period
        t = time
        
        # Parametric equations for figure-eight
        # x = a * cos(t) / (1 + sin^2(t))
        # y = a * sin(t) * cos(t) / (1 + sin^2(t))
        
        sin_t = np.sin(omega * t)
        cos_t = np.cos(omega * t)
        denom = 1 + sin_t**2
        
        x = center[0] + size * cos_t / denom
        y = center[1] + size * sin_t * cos_t / denom
        z = altitude
        
        # Velocity (numerical approximation for simplicity)
        dt = 0.001
        sin_t2 = np.sin(omega * (t + dt))
        cos_t2 = np.cos(omega * (t + dt))
        denom2 = 1 + sin_t2**2
        
        x2 = center[0] + size * cos_t2 / denom2
        y2 = center[1] + size * sin_t2 * cos_t2 / denom2
        
        vx = (x2 - x) / dt
        vy = (y2 - y) / dt
        vz = 0.0
        
        # Yaw from velocity direction
        yaw = np.arctan2(vy, vx) if np.sqrt(vx**2 + vy**2) > 0.01 else 0.0
        
        return TrajectoryPoint(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            acceleration=np.zeros(3),  # Simplified
            yaw=yaw,
            yaw_rate=0.0,
            time=time,
        )
    
    def helix(
        self,
        center: np.ndarray,
        radius: float,
        start_altitude: float,
        climb_rate: float,
        angular_velocity: float,
        time: float,
    ) -> TrajectoryPoint:
        """Generate helical (spiral) trajectory.
        
        Args:
            center: Helix center [x, y]
            radius: Helix radius
            start_altitude: Starting altitude
            climb_rate: Vertical climb rate (negative for up in NED)
            angular_velocity: Angular velocity in rad/s
            time: Current time
            
        Returns:
            TrajectoryPoint on helical path
        """
        theta = angular_velocity * time
        
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        z = start_altitude + climb_rate * time
        
        vx = -radius * angular_velocity * np.sin(theta)
        vy = radius * angular_velocity * np.cos(theta)
        vz = climb_rate
        
        yaw = theta + np.pi / 2
        
        return TrajectoryPoint(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            acceleration=np.zeros(3),
            yaw=yaw,
            yaw_rate=angular_velocity,
            time=time,
        )
    
    def waypoint_interpolation(
        self,
        waypoints: List[np.ndarray],
        velocities: Optional[List[np.ndarray]] = None,
        segment_durations: Optional[List[float]] = None,
        time: float = 0.0,
    ) -> TrajectoryPoint:
        """Generate trajectory by interpolating between waypoints.
        
        Args:
            waypoints: List of waypoint positions [x, y, z]
            velocities: Optional list of velocities at each waypoint
            segment_durations: Time to travel each segment. If None, 1s each.
            time: Current time
            
        Returns:
            Interpolated TrajectoryPoint
        """
        n_waypoints = len(waypoints)
        if n_waypoints < 2:
            return self.hover(waypoints[0] if waypoints else np.zeros(3), time=time)
        
        if segment_durations is None:
            segment_durations = [1.0] * (n_waypoints - 1)
        
        # Find current segment
        total_time = 0.0
        segment_idx = 0
        segment_time = time
        
        for i, duration in enumerate(segment_durations):
            if time <= total_time + duration:
                segment_idx = i
                segment_time = time - total_time
                break
            total_time += duration
        else:
            # Past last waypoint, stay at end
            return self.hover(waypoints[-1], time=time)
        
        # Linear interpolation within segment
        t_normalized = segment_time / segment_durations[segment_idx]
        t_normalized = np.clip(t_normalized, 0, 1)
        
        p0 = np.array(waypoints[segment_idx])
        p1 = np.array(waypoints[segment_idx + 1])
        
        position = p0 + t_normalized * (p1 - p0)
        velocity = (p1 - p0) / segment_durations[segment_idx]
        
        return TrajectoryPoint(
            position=position,
            velocity=velocity,
            acceleration=np.zeros(3),
            yaw=0.0,
            yaw_rate=0.0,
            time=time,
        )
    
    def polynomial_trajectory(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
        time: float,
        start_vel: Optional[np.ndarray] = None,
        end_vel: Optional[np.ndarray] = None,
    ) -> TrajectoryPoint:
        """Generate minimum-jerk (5th order polynomial) trajectory.
        
        Args:
            start_pos: Starting position
            end_pos: Ending position
            duration: Total trajectory duration
            time: Current time (0 to duration)
            start_vel: Starting velocity (default: zero)
            end_vel: Ending velocity (default: zero)
            
        Returns:
            TrajectoryPoint on polynomial path
        """
        if start_vel is None:
            start_vel = np.zeros(3)
        if end_vel is None:
            end_vel = np.zeros(3)
        
        t = np.clip(time / duration, 0, 1)
        
        # 5th order polynomial coefficients for minimum jerk
        # s(t) = 10*t^3 - 15*t^4 + 6*t^5
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        s_dot = (30 * t**2 - 60 * t**3 + 30 * t**4) / duration
        s_ddot = (60 * t - 180 * t**2 + 120 * t**3) / duration**2
        
        position = start_pos + s * (end_pos - start_pos)
        velocity = s_dot * (end_pos - start_pos)
        acceleration = s_ddot * (end_pos - start_pos)
        
        return TrajectoryPoint(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            yaw=0.0,
            yaw_rate=0.0,
            time=time,
        )
    
    def set_custom_trajectory(
        self,
        trajectory_func: Callable[[float], TrajectoryPoint],
    ) -> None:
        """Set a custom trajectory function.
        
        Args:
            trajectory_func: Function taking time and returning TrajectoryPoint
        """
        self._custom_trajectory = trajectory_func
    
    def get_custom_trajectory(self, time: float) -> TrajectoryPoint:
        """Get point from custom trajectory.
        
        Args:
            time: Current time
            
        Returns:
            TrajectoryPoint from custom function
        """
        if self._custom_trajectory is None:
            return self.hover(np.zeros(3), time=time)
        return self._custom_trajectory(time)
    
    def sample_trajectory(
        self,
        trajectory_type: TrajectoryType,
        time: float,
        **kwargs,
    ) -> TrajectoryPoint:
        """Sample a trajectory of the specified type.
        
        Args:
            trajectory_type: Type of trajectory to generate
            time: Current time
            **kwargs: Additional parameters for the trajectory type
            
        Returns:
            TrajectoryPoint at the specified time
        """
        if trajectory_type == TrajectoryType.HOVER:
            return self.hover(
                position=kwargs.get("position", np.zeros(3)),
                yaw=kwargs.get("yaw", 0.0),
                time=time,
            )
        elif trajectory_type == TrajectoryType.CIRCLE:
            return self.circle(
                center=kwargs.get("center", np.zeros(2)),
                radius=kwargs.get("radius", 1.0),
                altitude=kwargs.get("altitude", -1.0),
                angular_velocity=kwargs.get("angular_velocity", 0.5),
                time=time,
            )
        elif trajectory_type == TrajectoryType.FIGURE_EIGHT:
            return self.figure_eight(
                center=kwargs.get("center", np.zeros(2)),
                size=kwargs.get("size", 2.0),
                altitude=kwargs.get("altitude", -1.0),
                period=kwargs.get("period", 10.0),
                time=time,
            )
        elif trajectory_type == TrajectoryType.HELIX:
            return self.helix(
                center=kwargs.get("center", np.zeros(2)),
                radius=kwargs.get("radius", 1.0),
                start_altitude=kwargs.get("start_altitude", -1.0),
                climb_rate=kwargs.get("climb_rate", -0.1),
                angular_velocity=kwargs.get("angular_velocity", 0.5),
                time=time,
            )
        elif trajectory_type == TrajectoryType.CUSTOM:
            return self.get_custom_trajectory(time)
        else:
            return self.hover(np.zeros(3), time=time)
