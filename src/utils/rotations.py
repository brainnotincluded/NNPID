"""Rotation and quaternion utilities."""

from __future__ import annotations

import numpy as np


class Rotations:
    """Static methods for rotation operations."""

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    @staticmethod
    def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate (inverse for unit quaternions).

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Conjugate quaternion [w, -x, -y, -z]
        """
        return np.array([q[0], -q[1], -q[2], -q[3]])

    @staticmethod
    def quaternion_inverse(q: np.ndarray) -> np.ndarray:
        """Compute quaternion inverse.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Inverse quaternion
        """
        norm_sq = np.sum(q**2)
        return Rotations.quaternion_conjugate(q) / norm_sq

    @staticmethod
    def quaternion_normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit length.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Normalized quaternion
        """
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return q / norm

    @staticmethod
    def quaternion_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis-angle representation.

        Args:
            axis: Unit rotation axis [x, y, z]
            angle: Rotation angle in radians

        Returns:
            Quaternion [w, x, y, z]
        """
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])

    @staticmethod
    def quaternion_to_axis_angle(q: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert quaternion to axis-angle representation.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Tuple of (axis, angle) where axis is unit vector
        """
        q = Rotations.quaternion_normalize(q)

        angle = 2 * np.arccos(np.clip(q[0], -1.0, 1.0))

        s = np.sqrt(1 - q[0] ** 2)
        axis = np.array([1.0, 0.0, 0.0]) if s < 1e-10 else q[1:4] / s

        return axis, angle

    @staticmethod
    def quaternion_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion.

        Args:
            R: Rotation matrix (3, 3)

        Returns:
            Quaternion [w, x, y, z]
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return Rotations.quaternion_normalize(np.array([w, x, y, z]))

    @staticmethod
    def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Rotation matrix (3, 3)
        """
        q = Rotations.quaternion_normalize(q)
        w, x, y, z = q

        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def quaternion_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion.

        Args:
            q: Quaternion [w, x, y, z]
            v: Vector [x, y, z]

        Returns:
            Rotated vector [x, y, z]
        """
        # Convert vector to quaternion form
        v_quat = np.array([0.0, v[0], v[1], v[2]])

        # q * v * q^-1
        q_conj = Rotations.quaternion_conjugate(q)
        result = Rotations.quaternion_multiply(Rotations.quaternion_multiply(q, v_quat), q_conj)

        return result[1:4]

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles (ZYX convention) to quaternion.

        Args:
            roll: Roll angle (x-axis) in radians
            pitch: Pitch angle (y-axis) in radians
            yaw: Yaw angle (z-axis) in radians

        Returns:
            Quaternion [w, x, y, z]
        """
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
    def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
        """Convert quaternion to Euler angles (ZYX convention).

        Args:
            q: Quaternion [w, x, y, z]

        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        w, x, y, z = q

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

        return roll, pitch, yaw

    @staticmethod
    def quaternion_error(q_desired: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Compute quaternion error (rotation from current to desired).

        Args:
            q_desired: Desired quaternion [w, x, y, z]
            q_current: Current quaternion [w, x, y, z]

        Returns:
            Error quaternion [w, x, y, z]
        """
        q_error = Rotations.quaternion_multiply(
            q_desired, Rotations.quaternion_conjugate(q_current)
        )

        # Ensure shortest path
        if q_error[0] < 0:
            q_error = -q_error

        return q_error

    @staticmethod
    def angular_velocity_from_quaternion_derivative(
        q: np.ndarray,
        q_dot: np.ndarray,
    ) -> np.ndarray:
        """Compute angular velocity from quaternion and its derivative.

        Args:
            q: Quaternion [w, x, y, z]
            q_dot: Quaternion time derivative

        Returns:
            Angular velocity [wx, wy, wz] in body frame
        """
        # omega = 2 * q^-1 * q_dot
        q_inv = Rotations.quaternion_conjugate(q)
        omega_quat = 2 * Rotations.quaternion_multiply(q_inv, q_dot)
        return omega_quat[1:4]

    @staticmethod
    def quaternion_derivative_from_angular_velocity(
        q: np.ndarray,
        omega: np.ndarray,
    ) -> np.ndarray:
        """Compute quaternion derivative from angular velocity.

        Args:
            q: Quaternion [w, x, y, z]
            omega: Angular velocity [wx, wy, wz] in body frame

        Returns:
            Quaternion time derivative
        """
        # q_dot = 0.5 * q * omega_quat
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        return 0.5 * Rotations.quaternion_multiply(q, omega_quat)

    @staticmethod
    def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between quaternions.

        Args:
            q0: Start quaternion [w, x, y, z]
            q1: End quaternion [w, x, y, z]
            t: Interpolation parameter [0, 1]

        Returns:
            Interpolated quaternion
        """
        q0 = Rotations.quaternion_normalize(q0)
        q1 = Rotations.quaternion_normalize(q1)

        dot = np.sum(q0 * q1)

        # If dot is negative, negate one quaternion for shortest path
        if dot < 0:
            q1 = -q1
            dot = -dot

        # If quaternions are very close, use linear interpolation
        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return Rotations.quaternion_normalize(result)

        # Calculate interpolation
        theta_0 = np.arccos(dot)
        theta = theta_0 * t

        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0 * q0 + s1 * q1
