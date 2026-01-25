"""Tests for coordinate transforms."""

import pytest
import numpy as np

from src.utils.transforms import CoordinateTransforms, GPSTransforms
from src.utils.rotations import Rotations


class TestCoordinateTransforms:
    """Test coordinate frame transformations."""
    
    def test_position_mujoco_to_ned_identity(self):
        """Test position transform for axis-aligned vectors."""
        # X-axis (forward)
        pos = np.array([1.0, 0.0, 0.0])
        ned = CoordinateTransforms.position_mujoco_to_ned(pos)
        # MuJoCo X-forward -> NED X-north
        np.testing.assert_array_almost_equal(ned, [1.0, 0.0, 0.0])
    
    def test_position_ned_to_mujoco_inverse(self):
        """Test that transforms are inverses."""
        original = np.array([1.5, -2.3, 0.8])
        
        ned = CoordinateTransforms.position_mujoco_to_ned(original)
        recovered = CoordinateTransforms.position_ned_to_mujoco(ned)
        
        np.testing.assert_array_almost_equal(original, recovered)
    
    def test_velocity_transform(self):
        """Test velocity transform."""
        vel_mujoco = np.array([1.0, 2.0, 3.0])
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(vel_mujoco)
        vel_recovered = CoordinateTransforms.velocity_ned_to_mujoco(vel_ned)
        
        np.testing.assert_array_almost_equal(vel_mujoco, vel_recovered)
    
    def test_angular_velocity_transform(self):
        """Test angular velocity transform."""
        omega_mujoco = np.array([0.1, 0.2, 0.3])
        omega_frd = CoordinateTransforms.angular_velocity_mujoco_to_frd(omega_mujoco)
        omega_recovered = CoordinateTransforms.angular_velocity_frd_to_mujoco(omega_frd)
        
        np.testing.assert_array_almost_equal(omega_mujoco, omega_recovered)
    
    def test_quaternion_transform(self):
        """Test quaternion transform."""
        # Identity quaternion
        quat_mujoco = np.array([1.0, 0.0, 0.0, 0.0])
        quat_ned = CoordinateTransforms.quaternion_mujoco_to_ned(quat_mujoco)
        quat_recovered = CoordinateTransforms.quaternion_ned_to_mujoco(quat_ned)
        
        np.testing.assert_array_almost_equal(quat_mujoco, quat_recovered)
    
    def test_euler_from_quaternion(self):
        """Test Euler angle extraction."""
        # Identity quaternion -> zero Euler angles
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        euler = CoordinateTransforms.euler_from_quaternion_ned(quat)
        
        np.testing.assert_array_almost_equal(euler, [0.0, 0.0, 0.0])
    
    def test_euler_quaternion_roundtrip(self):
        """Test Euler to quaternion and back."""
        euler_original = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw
        
        quat = CoordinateTransforms.quaternion_from_euler_ned(euler_original)
        euler_recovered = CoordinateTransforms.euler_from_quaternion_ned(quat)
        
        np.testing.assert_array_almost_equal(euler_original, euler_recovered)
    
    def test_rotation_matrix_from_quaternion(self):
        """Test rotation matrix from quaternion."""
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        R = CoordinateTransforms.rotation_matrix_from_quaternion(quat)
        
        # Identity quaternion -> identity matrix
        np.testing.assert_array_almost_equal(R, np.eye(3))
    
    def test_full_state_transform(self):
        """Test full state transformation."""
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.1, 0.2, 0.3])
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        omega = np.array([0.01, 0.02, 0.03])
        
        pos_ned, vel_ned, quat_ned, omega_frd = CoordinateTransforms.state_mujoco_to_ned(
            pos, vel, quat, omega
        )
        
        pos_r, vel_r, quat_r, omega_r = CoordinateTransforms.state_ned_to_mujoco(
            pos_ned, vel_ned, quat_ned, omega_frd
        )
        
        np.testing.assert_array_almost_equal(pos, pos_r)
        np.testing.assert_array_almost_equal(vel, vel_r)
        np.testing.assert_array_almost_equal(omega, omega_r)


class TestGPSTransforms:
    """Test GPS coordinate transformations."""
    
    def test_ned_to_geodetic_at_origin(self):
        """Test NED to geodetic at reference point."""
        ref_lat, ref_lon, ref_alt = 47.397742, 8.545594, 488.0
        
        # Zero NED offset
        ned = np.array([0.0, 0.0, 0.0])
        lat, lon, alt = GPSTransforms.ned_to_geodetic(ned, ref_lat, ref_lon, ref_alt)
        
        assert abs(lat - ref_lat) < 1e-6
        assert abs(lon - ref_lon) < 1e-6
        assert abs(alt - ref_alt) < 0.1
    
    def test_geodetic_to_ned_roundtrip(self):
        """Test geodetic to NED and back."""
        ref_lat, ref_lon, ref_alt = 47.397742, 8.545594, 488.0
        
        original_ned = np.array([100.0, 50.0, -10.0])  # North, East, Down
        
        lat, lon, alt = GPSTransforms.ned_to_geodetic(
            original_ned, ref_lat, ref_lon, ref_alt
        )
        
        recovered_ned = GPSTransforms.geodetic_to_ned(
            lat, lon, alt, ref_lat, ref_lon, ref_alt
        )
        
        np.testing.assert_array_almost_equal(original_ned, recovered_ned, decimal=1)
    
    def test_altitude_conversion(self):
        """Test altitude conversion."""
        ref_lat, ref_lon, ref_alt = 47.0, 8.0, 500.0
        
        # 10m above reference (NED down = -10)
        ned = np.array([0.0, 0.0, -10.0])
        lat, lon, alt = GPSTransforms.ned_to_geodetic(ned, ref_lat, ref_lon, ref_alt)
        
        assert alt == pytest.approx(510.0, rel=0.01)


class TestRotations:
    """Test rotation utilities."""
    
    def test_quaternion_multiply_identity(self):
        """Test quaternion multiplication with identity."""
        q = np.array([0.707, 0.707, 0.0, 0.0])
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        
        result = Rotations.quaternion_multiply(q, identity)
        np.testing.assert_array_almost_equal(result, q)
    
    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_conj = Rotations.quaternion_conjugate(q)
        
        expected = np.array([0.5, -0.5, -0.5, -0.5])
        np.testing.assert_array_almost_equal(q_conj, expected)
    
    def test_quaternion_normalize(self):
        """Test quaternion normalization."""
        q = np.array([2.0, 0.0, 0.0, 0.0])
        q_norm = Rotations.quaternion_normalize(q)
        
        assert np.linalg.norm(q_norm) == pytest.approx(1.0)
    
    def test_quaternion_inverse(self):
        """Test quaternion inverse."""
        q = np.array([0.707, 0.707, 0.0, 0.0])
        q_inv = Rotations.quaternion_inverse(q)
        
        # q * q^-1 = identity
        result = Rotations.quaternion_multiply(q, q_inv)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        
        np.testing.assert_array_almost_equal(result, identity, decimal=5)
    
    def test_axis_angle_conversion(self):
        """Test axis-angle conversion."""
        axis = np.array([0.0, 0.0, 1.0])  # Z-axis
        angle = np.pi / 2  # 90 degrees
        
        q = Rotations.quaternion_from_axis_angle(axis, angle)
        axis_r, angle_r = Rotations.quaternion_to_axis_angle(q)
        
        np.testing.assert_array_almost_equal(axis, axis_r)
        assert angle == pytest.approx(angle_r)
    
    def test_euler_to_quaternion(self):
        """Test Euler to quaternion conversion."""
        # Zero rotation
        q = Rotations.euler_to_quaternion(0, 0, 0)
        np.testing.assert_array_almost_equal(q, [1, 0, 0, 0])
    
    def test_quaternion_rotate_vector(self):
        """Test vector rotation by quaternion."""
        # 90 degree rotation around Z
        q = Rotations.euler_to_quaternion(0, 0, np.pi / 2)
        v = np.array([1.0, 0.0, 0.0])  # X-axis
        
        v_rotated = Rotations.quaternion_rotate_vector(q, v)
        
        # Should now point in Y direction
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(v_rotated, expected, decimal=5)
    
    def test_slerp_endpoints(self):
        """Test SLERP at endpoints."""
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.707, 0.707, 0.0, 0.0])
        
        # At t=0, should be q0
        result = Rotations.slerp(q0, q1, 0.0)
        np.testing.assert_array_almost_equal(result, q0, decimal=3)
        
        # At t=1, should be q1
        result = Rotations.slerp(q0, q1, 1.0)
        np.testing.assert_array_almost_equal(result, q1, decimal=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
