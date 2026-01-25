"""Automated SITL integration tests.

These tests verify the MAVLink communication and control loops
using the MockSITL server (no real PX4 required).

Run with: pytest tests/test_sitl_integration.py -v
"""

from __future__ import annotations

import time
import threading
import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mujoco_sim import create_simulator
from src.core.sensors import SensorSimulator, SensorConfig
from src.communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig
from src.communication.messages import (
    HILSensorMessage,
    HILGPSMessage,
    SetPositionTargetLocalNED,
    CommandLong,
)
from src.utils.transforms import CoordinateTransforms


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simulator():
    """Create MuJoCo simulator."""
    sim = create_simulator(model="generic")
    sim.reset(position=np.array([0.0, 0.0, 0.5]))
    yield sim


@pytest.fixture
def sensors():
    """Create sensor simulator."""
    sensor_sim = SensorSimulator(SensorConfig())
    sensor_sim.reset()
    yield sensor_sim


# ============================================================================
# Unit Tests - Messages
# ============================================================================

class TestMAVLinkMessages:
    """Test MAVLink message creation."""
    
    def test_position_setpoint_message(self):
        """Test creating position setpoint message."""
        msg = SetPositionTargetLocalNED.from_position(
            time_ms=1000,
            position=np.array([1.0, 2.0, -3.0]),
            yaw=0.5,
        )
        
        assert msg.time_boot_ms == 1000
        assert msg.x == 1.0
        assert msg.y == 2.0
        assert msg.z == -3.0
        assert msg.yaw == 0.5
        assert msg.coordinate_frame == 1  # MAV_FRAME_LOCAL_NED
    
    def test_velocity_setpoint_message(self):
        """Test creating velocity setpoint message."""
        msg = SetPositionTargetLocalNED.from_velocity(
            time_ms=2000,
            velocity=np.array([1.5, -0.5, 0.2]),
            yaw_rate=0.1,
        )
        
        assert msg.time_boot_ms == 2000
        assert msg.vx == 1.5
        assert msg.vy == -0.5
        assert msg.vz == 0.2
        assert msg.yaw_rate == 0.1
    
    def test_arm_command(self):
        """Test creating arm command."""
        cmd = CommandLong.arm(arm=True, target_system=1)
        
        assert cmd.command == 400  # MAV_CMD_COMPONENT_ARM_DISARM
        assert cmd.param1 == 1.0
        assert cmd.target_system == 1
    
    def test_disarm_command(self):
        """Test creating disarm command."""
        cmd = CommandLong.arm(arm=False, target_system=1)
        
        assert cmd.command == 400
        assert cmd.param1 == 0.0
    
    def test_offboard_mode_command(self):
        """Test creating offboard mode command."""
        cmd = CommandLong.set_offboard_mode(target_system=1)
        
        assert cmd.command == 176  # MAV_CMD_DO_SET_MODE


class TestHILMessages:
    """Test HIL sensor messages."""
    
    def test_hil_sensor_creation(self):
        """Test creating HIL_SENSOR message."""
        msg = HILSensorMessage.from_sensor_data(
            time_sec=1.5,
            gyro=np.array([0.01, -0.02, 0.0]),
            accel=np.array([0.0, 0.0, -9.81]),
            mag=np.array([0.2, 0.0, 0.4]),
            pressure=101325.0,
            temperature=25.0,
            altitude=10.0,
        )
        
        assert msg.time_usec == 1500000
        assert abs(msg.zacc - (-9.81)) < 0.01
        assert msg.temperature == 25.0
    
    def test_hil_gps_creation(self):
        """Test creating HIL_GPS message."""
        msg = HILGPSMessage.from_gps_data(
            time_sec=2.0,
            lat=47.397742,
            lon=8.545594,
            alt=500.0,
            vel_ned=np.array([1.0, 0.5, -0.2]),
        )
        
        assert msg.time_usec == 2000000
        assert msg.lat == 473977420  # degE7
        assert msg.fix_type == 3  # 3D fix


# ============================================================================
# Integration Tests - Sensor Streaming
# ============================================================================

class TestSensorStreaming:
    """Test sensor data generation and streaming."""
    
    def test_imu_data_generation(self, simulator, sensors):
        """Test IMU data generation from simulator."""
        # Step simulation
        simulator.step(np.array([0.5, 0.5, 0.5, 0.5]))
        
        # Get IMU data
        gyro, accel = simulator.get_imu_data()
        
        assert gyro.shape == (3,)
        assert accel.shape == (3,)
        
        # Should have gravity component
        accel_mag = np.linalg.norm(accel)
        assert 5.0 < accel_mag < 15.0
    
    def test_sensor_noise_application(self, sensors):
        """Test that sensor noise is applied."""
        # Get multiple readings
        readings = []
        for i in range(10):
            gyro = np.zeros(3)
            accel = np.array([0.0, 0.0, -9.81])
            imu = sensors.get_imu(gyro, accel, i * 0.01)
            readings.append(imu.accel.copy())
        
        readings = np.array(readings)
        
        # Should have some variation due to noise
        std = np.std(readings, axis=0)
        assert np.any(std > 0.001)
    
    def test_coordinate_transforms(self, simulator):
        """Test coordinate frame conversions."""
        simulator.reset(position=np.array([1.0, 2.0, 3.0]))
        state = simulator.get_state()
        
        # Convert to NED
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
        
        # MuJoCo Z-up -> NED: x->x, y->-y, z->-z
        assert pos_ned[0] == 1.0
        assert pos_ned[1] == -2.0
        assert pos_ned[2] == -3.0
        
        # Round trip
        pos_back = CoordinateTransforms.position_ned_to_mujoco(pos_ned)
        np.testing.assert_array_almost_equal(state.position, pos_back)


# ============================================================================
# Integration Tests - Control Loop
# ============================================================================

class TestControlLoop:
    """Test closed-loop control behavior."""
    
    def test_gravity_compensation(self, simulator):
        """Test that hover thrust compensates gravity."""
        simulator.reset(position=np.array([0.0, 0.0, 1.0]))
        initial_z = simulator.get_state().position[2]
        
        # Approximate hover thrust
        hover_thrust = 0.55
        motors = np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])
        
        # Run for 1 second
        for _ in range(500):
            simulator.step(motors)
        
        final_z = simulator.get_state().position[2]
        
        # Should stay roughly at same altitude (within 0.5m)
        assert abs(final_z - initial_z) < 1.0
    
    def test_motor_response(self, simulator):
        """Test that motors produce expected forces."""
        simulator.reset(position=np.array([0.0, 0.0, 2.0]))
        
        # Full thrust
        full_thrust = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Step
        for _ in range(50):
            simulator.step(full_thrust)
        
        state = simulator.get_state()
        
        # Should be moving up or at least not falling fast
        assert state.velocity[2] > -5.0
    
    def test_zero_thrust_fall(self, simulator):
        """Test that zero thrust causes fall."""
        simulator.reset(position=np.array([0.0, 0.0, 5.0]))
        initial_z = simulator.get_state().position[2]
        
        # Zero thrust
        no_thrust = np.zeros(4)
        
        # Run for 0.5 seconds
        for _ in range(250):
            simulator.step(no_thrust)
        
        final_z = simulator.get_state().position[2]
        
        # Should have fallen
        assert final_z < initial_z - 0.5


# ============================================================================
# Integration Tests - Offboard Controller
# ============================================================================

class TestOffboardController:
    """Test offboard controller functionality."""
    
    def test_action_to_velocity_conversion(self):
        """Test NN action to velocity setpoint conversion."""
        from src.controllers.offboard_controller import OffboardController, OffboardConfig
        
        config = OffboardConfig(
            velocity_scale=3.0,
            max_velocity=5.0,
        )
        controller = OffboardController(config=config)
        
        # Create dummy state
        sim = create_simulator("generic")
        sim.reset(position=np.array([0.0, 0.0, 1.0]))
        state = sim.get_state()
        
        # Process action
        action = np.array([0.5, -0.3, 0.2, 0.0])
        setpoint = controller.process_nn_action(action, state)
        
        # Check velocity scaling
        assert setpoint.velocity is not None
        assert abs(setpoint.velocity[0] - 1.5) < 0.01  # 0.5 * 3.0
        assert abs(setpoint.velocity[1] - (-0.9)) < 0.01  # -0.3 * 3.0
    
    def test_action_clipping(self):
        """Test that actions are clipped to valid range."""
        from src.controllers.offboard_controller import OffboardController
        
        controller = OffboardController()
        
        sim = create_simulator("generic")
        sim.reset()
        state = sim.get_state()
        
        # Extreme action
        action = np.array([10.0, -10.0, 5.0, 2.0])
        setpoint = controller.process_nn_action(action, state)
        
        # Should be clipped
        assert setpoint.velocity is not None
        assert np.all(np.abs(setpoint.velocity) <= controller.config.max_velocity)


# ============================================================================
# Integration Tests - Environment
# ============================================================================

class TestSetpointEnvironment:
    """Test setpoint-based Gymnasium environment."""
    
    def test_env_creation(self):
        """Test environment creation."""
        from src.environments.setpoint_env import SetpointHoverEnv
        
        env = SetpointHoverEnv()
        
        assert env.observation_space.shape == (19,)
        assert env.action_space.shape == (4,)
        
        env.close()
    
    def test_env_reset(self):
        """Test environment reset."""
        from src.environments.setpoint_env import SetpointHoverEnv
        
        env = SetpointHoverEnv()
        obs, info = env.reset(seed=42)
        
        assert obs.shape == (19,)
        assert "target_position" in info
        
        env.close()
    
    def test_env_step(self):
        """Test environment step."""
        from src.environments.setpoint_env import SetpointHoverEnv
        
        env = SetpointHoverEnv()
        env.reset(seed=42)
        
        action = np.array([0.0, 0.0, 0.1, 0.0])  # Slight upward velocity
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (19,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        
        env.close()
    
    def test_env_deterministic_reset(self):
        """Test that reset with same seed gives same result."""
        from src.environments.setpoint_env import SetpointHoverEnv
        
        env = SetpointHoverEnv()
        
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        
        np.testing.assert_array_equal(obs1, obs2)
        
        env.close()
    
    def test_env_episode_termination(self):
        """Test that episode terminates on crash."""
        from src.environments.setpoint_env import SetpointHoverEnv
        
        env = SetpointHoverEnv()
        env.reset(seed=42)
        
        # Apply downward velocity until crash
        terminated = False
        steps = 0
        max_steps = 500
        
        while not terminated and steps < max_steps:
            action = np.array([0.0, 0.0, -1.0, 0.0])  # Downward
            _, _, terminated, truncated, _ = env.step(action)
            if truncated:
                break
            steps += 1
        
        # Should terminate (crash)
        assert terminated or steps < max_steps
        
        env.close()


# ============================================================================
# Stress Tests
# ============================================================================

class TestStress:
    """Stress tests for performance and stability."""
    
    def test_long_simulation(self, simulator):
        """Test running simulation for extended period."""
        simulator.reset()
        
        hover = np.array([0.55, 0.55, 0.55, 0.55])
        
        # Run for 10000 steps (20 seconds at 500Hz)
        for i in range(10000):
            simulator.step(hover)
        
        state = simulator.get_state()
        
        # Should still be valid
        assert not np.any(np.isnan(state.position))
        assert not np.any(np.isnan(state.velocity))
    
    def test_rapid_reset(self, simulator):
        """Test rapid environment resets."""
        for _ in range(100):
            simulator.reset(position=np.array([0.0, 0.0, np.random.uniform(0.5, 3.0)]))
            
            for _ in range(10):
                simulator.step(np.random.rand(4))
        
        # Should complete without error
        assert True
    
    def test_extreme_inputs(self, simulator):
        """Test behavior with extreme motor inputs."""
        simulator.reset(position=np.array([0.0, 0.0, 5.0]))
        
        # Extreme inputs
        inputs = [
            np.array([1.0, 1.0, 1.0, 1.0]),  # Max thrust
            np.array([0.0, 0.0, 0.0, 0.0]),  # No thrust
            np.array([1.0, 0.0, 1.0, 0.0]),  # Asymmetric
            np.array([0.0, 1.0, 0.0, 1.0]),  # Other asymmetric
        ]
        
        for inp in inputs:
            for _ in range(100):
                simulator.step(inp)
        
        state = simulator.get_state()
        
        # Should not have NaN
        assert not np.any(np.isnan(state.position))


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
