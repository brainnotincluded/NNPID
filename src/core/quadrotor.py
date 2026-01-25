"""Quadrotor dynamics and physical properties."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QuadrotorConfig:
    """Configuration for quadrotor physical properties.
    
    Default values match the X500 frame used in PX4.
    """
    
    # Mass and inertia
    mass: float = 2.0  # kg
    inertia: np.ndarray = field(
        default_factory=lambda: np.array([0.029125, 0.029125, 0.055225])
    )  # Diagonal inertia [Ixx, Iyy, Izz]
    
    # Geometry
    arm_length: float = 0.25  # meters, center to motor
    
    # Motor properties
    max_thrust_per_motor: float = 8.0  # Newtons
    min_thrust_per_motor: float = 0.0
    motor_time_constant: float = 0.02  # seconds
    prop_torque_coefficient: float = 0.01  # Nm per N of thrust
    
    # Motor positions relative to CoM (X-config)
    # Order: front-right, front-left, back-left, back-right
    motor_positions: np.ndarray = field(
        default_factory=lambda: np.array([
            [0.1768, 0.1768, 0.0],   # Motor 1: Front-right (CCW)
            [-0.1768, 0.1768, 0.0],  # Motor 2: Front-left (CW)
            [-0.1768, -0.1768, 0.0], # Motor 3: Back-left (CCW)
            [0.1768, -0.1768, 0.0],  # Motor 4: Back-right (CW)
        ])
    )
    
    # Motor spin directions (1 = CCW, -1 = CW)
    motor_directions: np.ndarray = field(
        default_factory=lambda: np.array([1, -1, 1, -1])
    )
    
    def __post_init__(self):
        """Convert lists to numpy arrays if needed."""
        if isinstance(self.inertia, list):
            self.inertia = np.array(self.inertia)
        if isinstance(self.motor_positions, list):
            self.motor_positions = np.array(self.motor_positions)
        if isinstance(self.motor_directions, list):
            self.motor_directions = np.array(self.motor_directions)
    
    @property
    def hover_thrust(self) -> float:
        """Thrust per motor needed to hover (assuming level flight)."""
        return (self.mass * 9.81) / 4.0
    
    @property
    def hover_throttle(self) -> float:
        """Normalized throttle [0, 1] needed to hover."""
        return self.hover_thrust / self.max_thrust_per_motor
    
    @property
    def max_total_thrust(self) -> float:
        """Maximum combined thrust from all motors."""
        return 4.0 * self.max_thrust_per_motor
    
    @property
    def thrust_to_weight_ratio(self) -> float:
        """Maximum thrust to weight ratio."""
        return self.max_total_thrust / (self.mass * 9.81)


class QuadrotorDynamics:
    """Analytical quadrotor dynamics model.
    
    Used for computing control allocation and dynamics analysis.
    Not used during MuJoCo simulation (MuJoCo handles physics).
    
    Coordinate frame:
    - Body frame: X-forward, Y-right, Z-down (FRD)
    - World frame: North-East-Down (NED)
    """
    
    def __init__(self, config: Optional[QuadrotorConfig] = None):
        """Initialize dynamics model.
        
        Args:
            config: Quadrotor configuration. Uses defaults if None.
        """
        self.config = config or QuadrotorConfig()
        
        # Precompute allocation matrix
        self._compute_allocation_matrix()
    
    def _compute_allocation_matrix(self) -> None:
        """Compute motor mixing matrix.
        
        Maps motor thrusts to body wrench [Fx, Fy, Fz, Mx, My, Mz].
        """
        cfg = self.config
        
        # Allocation matrix: wrench = A @ thrusts
        # For each motor i:
        # - Fz contribution: thrust_i (all motors push in +Z body frame)
        # - Mx contribution: y_i * thrust_i (pitch moment from y-offset)
        # - My contribution: -x_i * thrust_i (roll moment from x-offset)
        # - Mz contribution: direction_i * kappa * thrust_i (yaw moment from reaction torque)
        
        A = np.zeros((6, 4))
        
        for i in range(4):
            pos = cfg.motor_positions[i]
            direction = cfg.motor_directions[i]
            
            # Force (all in body Z direction for copters)
            A[2, i] = 1.0  # Fz
            
            # Moments
            A[3, i] = pos[1]  # Mx (roll) from y-position
            A[4, i] = -pos[0]  # My (pitch) from x-position
            A[5, i] = direction * cfg.prop_torque_coefficient  # Mz (yaw) from prop torque
        
        self._allocation_matrix = A
        
        # Compute pseudo-inverse for control allocation
        self._allocation_pinv = np.linalg.pinv(A)
    
    def motor_thrusts_to_wrench(self, thrusts: np.ndarray) -> np.ndarray:
        """Convert motor thrusts to body wrench.
        
        Args:
            thrusts: Motor thrust values [N], shape (4,)
            
        Returns:
            Body wrench [Fx, Fy, Fz, Mx, My, Mz], shape (6,)
        """
        return self._allocation_matrix @ thrusts
    
    def wrench_to_motor_thrusts(self, wrench: np.ndarray) -> np.ndarray:
        """Convert desired body wrench to motor thrusts.
        
        Uses pseudo-inverse for allocation (may not be exactly achievable).
        
        Args:
            wrench: Desired body wrench [Fx, Fy, Fz, Mx, My, Mz], shape (6,)
            
        Returns:
            Motor thrusts [N], shape (4,)
        """
        thrusts = self._allocation_pinv @ wrench
        # Clip to valid range
        thrusts = np.clip(
            thrusts,
            self.config.min_thrust_per_motor,
            self.config.max_thrust_per_motor
        )
        return thrusts
    
    def collective_to_thrusts(
        self,
        thrust: float,
        roll_moment: float,
        pitch_moment: float,
        yaw_moment: float,
    ) -> np.ndarray:
        """Convert collective thrust and moments to individual motor thrusts.
        
        Args:
            thrust: Total desired thrust [N]
            roll_moment: Desired roll moment [Nm]
            pitch_moment: Desired pitch moment [Nm]
            yaw_moment: Desired yaw moment [Nm]
            
        Returns:
            Motor thrusts [N], shape (4,)
        """
        wrench = np.array([0.0, 0.0, thrust, roll_moment, pitch_moment, yaw_moment])
        return self.wrench_to_motor_thrusts(wrench)
    
    def normalize_thrusts(self, thrusts: np.ndarray) -> np.ndarray:
        """Convert thrusts [N] to normalized commands [0, 1].
        
        Args:
            thrusts: Motor thrusts [N], shape (4,)
            
        Returns:
            Normalized motor commands [0, 1], shape (4,)
        """
        normalized = (thrusts - self.config.min_thrust_per_motor) / (
            self.config.max_thrust_per_motor - self.config.min_thrust_per_motor
        )
        return np.clip(normalized, 0.0, 1.0)
    
    def denormalize_thrusts(self, commands: np.ndarray) -> np.ndarray:
        """Convert normalized commands [0, 1] to thrusts [N].
        
        Args:
            commands: Normalized motor commands [0, 1], shape (4,)
            
        Returns:
            Motor thrusts [N], shape (4,)
        """
        commands = np.clip(commands, 0.0, 1.0)
        return (
            commands * (self.config.max_thrust_per_motor - self.config.min_thrust_per_motor)
            + self.config.min_thrust_per_motor
        )
    
    def compute_hover_command(self) -> np.ndarray:
        """Compute motor commands for hovering.
        
        Returns:
            Normalized motor commands for hover [0, 1], shape (4,)
        """
        hover_thrust = self.config.hover_thrust
        return np.full(4, self.config.hover_throttle)
    
    def rotation_matrix_body_to_world(self, quaternion: np.ndarray) -> np.ndarray:
        """Compute rotation matrix from body to world frame.
        
        Args:
            quaternion: Orientation quaternion [w, x, y, z]
            
        Returns:
            Rotation matrix (3, 3)
        """
        w, x, y, z = quaternion
        
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        
        return R
    
    def euler_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            quaternion: Orientation quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        w, x, y, z = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
