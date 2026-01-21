"""
Safety layer for drone control.

CRITICAL: Never trust neural network outputs blindly on real hardware.
This module provides:
- Geofence (spatial boundaries)
- Action clipping (velocity/acceleration limits)
- Fallback PID controller
- Emergency stop conditions

"Safety first, performance second" - always.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class SafetyStatus(Enum):
    """Safety system status"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class SafetyLimits:
    """Safety limits for drone operation"""
    # Spatial limits (geofence) in NED frame
    max_distance_from_home: float = 50.0  # meters
    min_altitude: float = 0.5  # meters above ground
    max_altitude: float = 50.0  # meters above ground
    
    # Velocity limits
    max_horizontal_velocity: float = 5.0  # m/s
    max_vertical_velocity: float = 3.0  # m/s
    max_velocity_magnitude: float = 6.0  # m/s
    
    # Acceleration limits
    max_acceleration: float = 5.0  # m/s²
    
    # Angular limits
    max_roll: float = np.radians(30)  # radians
    max_pitch: float = np.radians(30)  # radians
    max_yaw_rate: float = np.radians(90)  # rad/s
    
    # Neural network monitoring
    max_inference_time: float = 0.05  # seconds (20Hz = 50ms max)
    max_nan_tolerance: int = 0  # Number of NaN outputs tolerated
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'max_distance_from_home': self.max_distance_from_home,
            'min_altitude': self.min_altitude,
            'max_altitude': self.max_altitude,
            'max_horizontal_velocity': self.max_horizontal_velocity,
            'max_vertical_velocity': self.max_vertical_velocity,
            'max_velocity_magnitude': self.max_velocity_magnitude,
            'max_acceleration': self.max_acceleration,
            'max_roll': self.max_roll,
            'max_pitch': self.max_pitch,
            'max_yaw_rate': self.max_yaw_rate,
            'max_inference_time': self.max_inference_time
        }


class GeofenceChecker:
    """
    Checks if drone is within safe spatial boundaries.
    """
    
    def __init__(self, limits: SafetyLimits, home_position: np.ndarray):
        """
        Initialize geofence checker.
        
        Args:
            limits: Safety limits
            home_position: Home position in NED frame [north, east, down]
        """
        self.limits = limits
        self.home_position = home_position
    
    def check(self, position: np.ndarray) -> Tuple[bool, str]:
        """
        Check if position is within geofence.
        
        Args:
            position: Current position in NED frame [north, east, down]
            
        Returns:
            is_safe: True if within geofence
            message: Description of violation if any
        """
        # Distance from home (horizontal)
        horizontal_distance = np.linalg.norm(position[:2] - self.home_position[:2])
        if horizontal_distance > self.limits.max_distance_from_home:
            return False, f"Too far from home: {horizontal_distance:.1f}m > {self.limits.max_distance_from_home}m"
        
        # Altitude (NED: down is positive, so negative values are altitude)
        altitude = -position[2]  # Convert NED down to altitude
        home_altitude = -self.home_position[2]
        height_above_ground = altitude - home_altitude
        
        if height_above_ground < self.limits.min_altitude:
            return False, f"Too low: {height_above_ground:.1f}m < {self.limits.min_altitude}m"
        
        if height_above_ground > self.limits.max_altitude:
            return False, f"Too high: {height_above_ground:.1f}m > {self.limits.max_altitude}m"
        
        return True, "Within geofence"
    
    def get_closest_safe_position(self, position: np.ndarray) -> np.ndarray:
        """
        Project position to closest point within geofence.
        
        Args:
            position: Desired position in NED frame
            
        Returns:
            Clamped position within geofence
        """
        safe_position = position.copy()
        
        # Clamp horizontal distance
        horizontal_offset = safe_position[:2] - self.home_position[:2]
        horizontal_distance = np.linalg.norm(horizontal_offset)
        
        if horizontal_distance > self.limits.max_distance_from_home:
            # Scale down to max distance
            horizontal_offset = horizontal_offset / horizontal_distance * self.limits.max_distance_from_home
            safe_position[:2] = self.home_position[:2] + horizontal_offset
        
        # Clamp altitude
        home_altitude = -self.home_position[2]
        altitude = -safe_position[2]
        height_above_ground = altitude - home_altitude
        
        if height_above_ground < self.limits.min_altitude:
            safe_position[2] = -(home_altitude + self.limits.min_altitude)
        elif height_above_ground > self.limits.max_altitude:
            safe_position[2] = -(home_altitude + self.limits.max_altitude)
        
        return safe_position


class ActionSafetyFilter:
    """
    Filters/clips neural network actions to safe values.
    """
    
    def __init__(self, limits: SafetyLimits):
        """
        Initialize action safety filter.
        
        Args:
            limits: Safety limits
        """
        self.limits = limits
        self.prev_action = None
    
    def filter_velocity_command(
        self,
        velocity_command: np.ndarray,
        current_velocity: Optional[np.ndarray] = None,
        dt: float = 0.05
    ) -> Tuple[np.ndarray, bool]:
        """
        Filter velocity command to ensure safety.
        
        Args:
            velocity_command: Desired velocity [vx, vy, vz] in body frame (m/s)
            current_velocity: Current velocity (for acceleration limiting)
            dt: Time step (seconds)
            
        Returns:
            filtered_command: Safe velocity command
            was_modified: True if command was modified
        """
        original = velocity_command.copy()
        filtered = velocity_command.copy()
        
        # Clip horizontal velocity
        horizontal_vel = np.linalg.norm(filtered[:2])
        if horizontal_vel > self.limits.max_horizontal_velocity:
            scale = self.limits.max_horizontal_velocity / horizontal_vel
            filtered[:2] *= scale
        
        # Clip vertical velocity
        filtered[2] = np.clip(
            filtered[2],
            -self.limits.max_vertical_velocity,
            self.limits.max_vertical_velocity
        )
        
        # Clip total velocity magnitude
        vel_magnitude = np.linalg.norm(filtered)
        if vel_magnitude > self.limits.max_velocity_magnitude:
            filtered = filtered / vel_magnitude * self.limits.max_velocity_magnitude
        
        # Acceleration limiting
        if current_velocity is not None and dt > 0:
            desired_accel = (filtered - current_velocity) / dt
            accel_magnitude = np.linalg.norm(desired_accel)
            
            if accel_magnitude > self.limits.max_acceleration:
                # Limit acceleration
                safe_accel = desired_accel / accel_magnitude * self.limits.max_acceleration
                filtered = current_velocity + safe_accel * dt
        
        # Check if was modified
        was_modified = not np.allclose(original, filtered, atol=1e-6)
        
        # Check for NaN
        if np.any(np.isnan(filtered)):
            filtered = np.zeros(3)
            was_modified = True
        
        self.prev_action = filtered.copy()
        return filtered, was_modified
    
    def reset(self):
        """Reset filter state"""
        self.prev_action = None


class FallbackPIDController:
    """
    Simple PID controller as fallback when neural network fails.
    
    This is a SAFETY CRITICAL component.
    """
    
    def __init__(
        self,
        kp: np.ndarray = np.array([1.0, 1.0, 1.5]),
        ki: np.ndarray = np.array([0.1, 0.1, 0.2]),
        kd: np.ndarray = np.array([0.5, 0.5, 0.7]),
        max_integral: float = 2.0
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gains [x, y, z]
            ki: Integral gains [x, y, z]
            kd: Derivative gains [x, y, z]
            max_integral: Maximum integral term (anti-windup)
        """
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        self.max_integral = max_integral
        
        # State
        self.integral = np.zeros(3)
        self.prev_error = None
    
    def compute(
        self,
        target_position: np.ndarray,
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute PID control output.
        
        Args:
            target_position: Desired position [x, y, z] in body frame
            current_position: Current position [x, y, z] in body frame
            current_velocity: Current velocity [vx, vy, vz] in body frame
            dt: Time step (seconds)
            
        Returns:
            velocity_command: [vx, vy, vz] in body frame
        """
        # Error
        error = target_position - current_position
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.prev_error is not None:
            d_error = (error - self.prev_error) / dt
        else:
            d_error = np.zeros(3)
        d_term = self.kd * d_error
        
        # PID output
        velocity_command = p_term + i_term + d_term
        
        # Update state
        self.prev_error = error.copy()
        
        return velocity_command
    
    def reset(self):
        """Reset PID state"""
        self.integral = np.zeros(3)
        self.prev_error = None


class SafetyMonitor:
    """
    Main safety monitoring system.
    Integrates all safety checks and decides when to use fallback.
    """
    
    def __init__(
        self,
        limits: SafetyLimits,
        home_position: np.ndarray,
        enable_fallback: bool = True
    ):
        """
        Initialize safety monitor.
        
        Args:
            limits: Safety limits
            home_position: Home position in NED frame
            enable_fallback: Enable automatic fallback to PID
        """
        self.limits = limits
        self.geofence = GeofenceChecker(limits, home_position)
        self.action_filter = ActionSafetyFilter(limits)
        self.fallback_controller = FallbackPIDController() if enable_fallback else None
        
        # Status
        self.status = SafetyStatus.SAFE
        self.violation_count = 0
        self.nan_count = 0
        self.using_fallback = False
        self.last_inference_time = 0.0
        
        # Logging
        self.violation_history = []
    
    def check_and_filter(
        self,
        position: np.ndarray,
        velocity_command: np.ndarray,
        current_velocity: np.ndarray,
        target_position: np.ndarray,
        inference_time: float,
        dt: float = 0.05
    ) -> Tuple[np.ndarray, SafetyStatus, Dict]:
        """
        Perform all safety checks and filter action.
        
        Args:
            position: Current position in NED frame
            velocity_command: Neural network velocity command (body frame)
            current_velocity: Current velocity (body frame)
            target_position: Target position (body frame)
            inference_time: Time taken for neural network inference (seconds)
            dt: Time step (seconds)
            
        Returns:
            safe_command: Safe velocity command (may be from PID if fallback triggered)
            status: Safety status
            info: Dict with safety information
        """
        info = {
            'geofence_ok': True,
            'action_filtered': False,
            'used_fallback': False,
            'violations': []
        }
        
        # Check geofence
        geofence_ok, geofence_msg = self.geofence.check(position)
        info['geofence_ok'] = geofence_ok
        if not geofence_ok:
            info['violations'].append(geofence_msg)
            self.violation_count += 1
            self.status = SafetyStatus.WARNING
        
        # Check for NaN in neural network output
        if np.any(np.isnan(velocity_command)):
            self.nan_count += 1
            info['violations'].append(f"NaN in network output (count: {self.nan_count})")
            self.status = SafetyStatus.CRITICAL
            
            if self.nan_count > self.limits.max_nan_tolerance:
                self.using_fallback = True
        
        # Check inference time
        self.last_inference_time = inference_time
        if inference_time > self.limits.max_inference_time:
            info['violations'].append(f"Inference too slow: {inference_time*1000:.1f}ms")
            self.status = SafetyStatus.WARNING
        
        # Decide whether to use fallback
        if self.using_fallback and self.fallback_controller is not None:
            # Use PID fallback
            velocity_command = self.fallback_controller.compute(
                target_position,
                np.zeros(3),  # In body frame, target is relative
                current_velocity,
                dt
            )
            info['used_fallback'] = True
            self.status = SafetyStatus.CRITICAL
        
        # Filter action for safety
        safe_command, was_filtered = self.action_filter.filter_velocity_command(
            velocity_command,
            current_velocity,
            dt
        )
        info['action_filtered'] = was_filtered
        
        if was_filtered:
            info['violations'].append("Action was clipped to safety limits")
        
        # Update status
        if len(info['violations']) == 0:
            self.status = SafetyStatus.SAFE
            self.violation_count = max(0, self.violation_count - 1)  # Decay
        
        # Log violations
        if len(info['violations']) > 0:
            self.violation_history.append({
                'status': self.status.value,
                'violations': info['violations'].copy()
            })
        
        return safe_command, self.status, info
    
    def reset(self):
        """Reset safety monitor state"""
        self.status = SafetyStatus.SAFE
        self.violation_count = 0
        self.nan_count = 0
        self.using_fallback = False
        self.action_filter.reset()
        if self.fallback_controller:
            self.fallback_controller.reset()
    
    def get_statistics(self) -> Dict:
        """Get safety statistics"""
        return {
            'current_status': self.status.value,
            'violation_count': self.violation_count,
            'nan_count': self.nan_count,
            'using_fallback': self.using_fallback,
            'last_inference_time_ms': self.last_inference_time * 1000,
            'total_violations': len(self.violation_history)
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== Safety System Tests ===\n")
    
    # Setup
    limits = SafetyLimits(
        max_distance_from_home=20.0,
        min_altitude=1.0,
        max_altitude=10.0,
        max_horizontal_velocity=3.0,
        max_vertical_velocity=2.0
    )
    
    home_position = np.array([0.0, 0.0, -1.5])  # 1.5m altitude in NED
    
    # Test 1: Geofence
    print("Test 1: Geofence Checker")
    geofence = GeofenceChecker(limits, home_position)
    
    test_positions = [
        np.array([0.0, 0.0, -2.5]),  # Safe (1m above home)
        np.array([25.0, 0.0, -2.5]),  # Too far
        np.array([0.0, 0.0, -0.5]),  # Too low
        np.array([0.0, 0.0, -15.0]),  # Too high
    ]
    
    for pos in test_positions:
        is_safe, msg = geofence.check(pos)
        status = "✓ SAFE" if is_safe else "✗ UNSAFE"
        print(f"  Position {pos}: {status} - {msg}")
    print()
    
    # Test 2: Action Filter
    print("Test 2: Action Safety Filter")
    action_filter = ActionSafetyFilter(limits)
    
    test_actions = [
        np.array([2.0, 1.0, -0.5]),  # Safe
        np.array([10.0, 5.0, -3.0]),  # Too fast
        np.array([np.nan, 1.0, 0.0]),  # NaN
    ]
    
    for action in test_actions:
        filtered, modified = action_filter.filter_velocity_command(action)
        status = "MODIFIED" if modified else "OK"
        print(f"  Input: {action} -> Output: {filtered} [{status}]")
    print()
    
    # Test 3: PID Controller
    print("Test 3: Fallback PID Controller")
    pid = FallbackPIDController()
    
    target = np.array([5.0, 0.0, 0.0])  # 5m forward
    current_pos = np.array([0.0, 0.0, 0.0])
    current_vel = np.array([0.0, 0.0, 0.0])
    
    print("  Simulating PID tracking over 2 seconds:")
    for t in np.arange(0, 2.0, 0.1):
        vel_cmd = pid.compute(target, current_pos, current_vel, 0.1)
        current_pos += vel_cmd * 0.1
        current_vel = vel_cmd
        
        error = np.linalg.norm(target - current_pos)
        if t % 0.5 < 0.1:  # Print every 0.5s
            print(f"    t={t:.1f}s: pos={current_pos}, error={error:.2f}m")
    print()
    
    # Test 4: Safety Monitor
    print("Test 4: Integrated Safety Monitor")
    monitor = SafetyMonitor(limits, home_position, enable_fallback=True)
    
    # Simulate neural network failure
    position = np.array([5.0, 0.0, -2.5])
    velocity_command = np.array([np.nan, 0.0, 0.0])  # Network outputs NaN
    current_velocity = np.array([1.0, 0.0, 0.0])
    target_position = np.array([5.0, 0.0, 0.0])
    
    safe_cmd, status, info = monitor.check_and_filter(
        position, velocity_command, current_velocity,
        target_position, inference_time=0.03, dt=0.05
    )
    
    print(f"  Status: {status.value}")
    print(f"  Used fallback: {info['used_fallback']}")
    print(f"  Safe command: {safe_cmd}")
    print(f"  Violations: {info['violations']}")
    
    stats = monitor.get_statistics()
    print(f"  Statistics: {stats}")
    
    print("\n=== All tests complete ===")
