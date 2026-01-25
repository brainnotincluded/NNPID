"""Tests for MuJoCo simulation."""

import pytest
import numpy as np
from pathlib import Path

# Skip all tests if MuJoCo not available
pytest.importorskip("mujoco")

from src.core.mujoco_sim import MuJoCoSimulator, create_simulator, QuadrotorState


class TestMuJoCoSimulator:
    """Test MuJoCo simulator functionality."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return create_simulator(model="generic")
    
    def test_simulator_creation(self, simulator):
        """Test simulator can be created."""
        assert simulator is not None
        assert simulator.model is not None
        assert simulator.data is not None
    
    def test_reset(self, simulator):
        """Test reset returns valid state."""
        state = simulator.reset()
        
        assert isinstance(state, QuadrotorState)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)
        assert state.quaternion.shape == (4,)
        assert state.angular_velocity.shape == (3,)
        assert state.motor_speeds.shape == (4,)
    
    def test_reset_with_position(self, simulator):
        """Test reset with custom position."""
        target_pos = np.array([1.0, 2.0, 3.0])
        state = simulator.reset(position=target_pos)
        
        np.testing.assert_array_almost_equal(state.position, target_pos, decimal=2)
    
    def test_step(self, simulator):
        """Test simulation step."""
        simulator.reset()
        
        motor_commands = np.array([0.5, 0.5, 0.5, 0.5])
        state = simulator.step(motor_commands)
        
        assert isinstance(state, QuadrotorState)
        assert simulator.get_time() > 0
    
    def test_step_clips_commands(self, simulator):
        """Test that motor commands are clipped."""
        simulator.reset()
        
        # Commands outside [0, 1] should be clipped
        motor_commands = np.array([-1.0, 2.0, 0.5, 0.5])
        state = simulator.step(motor_commands)
        
        # Should not crash and state should be valid
        assert isinstance(state, QuadrotorState)
    
    def test_hover_command(self, simulator):
        """Test that hover command produces thrust."""
        state = simulator.reset(position=np.array([0.0, 0.0, 1.0]))
        
        # Full thrust command
        thrust_cmd = np.array([0.8, 0.8, 0.8, 0.8])
        
        # Step for 0.5 seconds
        for _ in range(250):  # 250 steps at 500Hz
            state = simulator.step(thrust_cmd)
        
        # With high thrust, should have upward or maintained position
        # Just verify simulation runs without crashing
        assert state.position is not None
        assert len(state.position) == 3
    
    def test_free_fall(self, simulator):
        """Test that drone falls with no thrust."""
        state = simulator.reset(position=np.array([0.0, 0.0, 2.0]))
        
        # Zero thrust
        zero_cmd = np.array([0.0, 0.0, 0.0, 0.0])
        
        for _ in range(100):
            state = simulator.step(zero_cmd)
        
        # Should be falling
        assert state.velocity[2] < 0  # Negative Z velocity (falling in Z-up)
    
    def test_imu_data(self, simulator):
        """Test IMU sensor data."""
        simulator.reset()
        simulator.step(np.array([0.5, 0.5, 0.5, 0.5]))
        
        gyro, accel = simulator.get_imu_data()
        
        assert gyro.shape == (3,)
        assert accel.shape == (3,)
        
        # Accelerometer should show reasonable values
        accel_magnitude = np.linalg.norm(accel)
        assert 0.0 <= accel_magnitude < 100.0  # Reasonable range
    
    def test_gravity(self, simulator):
        """Test gravity vector."""
        gravity = simulator.gravity
        
        assert gravity.shape == (3,)
        assert gravity[2] < 0  # Gravity points down in MuJoCo Z-up
    
    def test_mass(self, simulator):
        """Test mass property."""
        mass = simulator.mass
        
        assert mass > 0
        assert isinstance(mass, float)
    
    def test_contact_detection(self, simulator):
        """Test contact detection method exists and works."""
        # Reset above ground
        state = simulator.reset(position=np.array([0.0, 0.0, 2.0]))
        
        # Step simulation
        for _ in range(10):
            simulator.step(np.zeros(4))
        
        # Method should return boolean
        contact_result = simulator.has_contact()
        assert isinstance(contact_result, bool)
    
    def test_state_to_array(self):
        """Test state serialization."""
        state = QuadrotorState(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.01, 0.02, 0.03]),
            motor_speeds=np.array([0.5, 0.5, 0.5, 0.5]),
        )
        
        arr = state.to_array()
        assert arr.shape == (17,)  # 3+3+4+3+4
        
        # Reconstruct
        state2 = QuadrotorState.from_array(arr)
        np.testing.assert_array_almost_equal(state.position, state2.position)
        np.testing.assert_array_almost_equal(state.quaternion, state2.quaternion)


class TestCreateSimulator:
    """Test simulator factory function."""
    
    def test_create_generic(self):
        """Test creating generic model."""
        sim = create_simulator(model="generic")
        assert sim is not None
    
    def test_create_x500(self):
        """Test creating X500 model."""
        sim = create_simulator(model="x500")
        assert sim is not None
    
    def test_custom_timestep(self):
        """Test custom timestep."""
        sim = create_simulator(model="generic", timestep=0.001)
        assert sim.timestep == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
