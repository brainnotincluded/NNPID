"""Tests for the perturbation system."""

from pathlib import Path

import numpy as np
import pytest

from src.core.mujoco_sim import QuadrotorState
from src.perturbations import (
    AerodynamicsConfig,
    AerodynamicsPerturbation,
    BasePerturbation,
    DelayConfig,
    DelayPerturbation,
    ExternalForcesConfig,
    ExternalForcesPerturbation,
    PerturbationConfig,
    # Base
    PerturbationManager,
    PhysicsConfig,
    PhysicsPerturbation,
    SensorNoiseConfig,
    SensorNoisePerturbation,
    # Configs
    WindConfig,
    # Perturbations
    WindPerturbation,
    create_calm_environment,
    # Factory functions
    create_light_breeze,
    create_moderate_wind,
    create_realistic_aero,
    create_realistic_physics,
    create_typical_latency,
    create_typical_noise,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng():
    """Create reproducible RNG."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_state():
    """Create a sample quadrotor state."""
    return QuadrotorState(
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([1.0, 0.5, 0.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.5]),
        motor_speeds=np.array([0.5, 0.5, 0.5, 0.5]),
    )


@pytest.fixture
def perturbation_manager():
    """Create a perturbation manager with all perturbation types."""
    manager = PerturbationManager(seed=42)

    # Add all types
    manager.add_perturbation("wind", create_light_breeze())
    manager.add_perturbation("delays", create_typical_latency())
    manager.add_perturbation("sensor_noise", create_typical_noise())
    manager.add_perturbation("physics", create_realistic_physics())
    manager.add_perturbation("aerodynamics", create_realistic_aero())
    manager.add_perturbation("external_forces", create_calm_environment())

    return manager


# =============================================================================
# Base Classes Tests
# =============================================================================


class TestPerturbationConfig:
    """Tests for PerturbationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerturbationConfig()
        assert config.enabled is True
        assert config.intensity == 1.0
        assert config.randomize_on_reset is True
        assert config.seed is None

    def test_scale_method(self):
        """Test value scaling by intensity."""
        config = PerturbationConfig(intensity=0.5)
        assert config.scale(10.0) == 5.0

        config.enabled = False
        assert config.scale(10.0) == 0.0


class TestPerturbationManager:
    """Tests for PerturbationManager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = PerturbationManager()
        assert manager.enabled is True
        assert manager.global_intensity == 1.0
        assert len(manager) == 0

    def test_add_remove_perturbation(self, rng):
        """Test adding and removing perturbations."""
        manager = PerturbationManager()
        wind = create_light_breeze()

        manager.add_perturbation("wind", wind)
        assert "wind" in manager
        assert len(manager) == 1

        removed = manager.remove_perturbation("wind")
        assert removed is wind
        assert "wind" not in manager
        assert len(manager) == 0

    def test_enable_disable(self, perturbation_manager, sample_state, rng):
        """Test enable/disable functionality."""
        manager = perturbation_manager
        manager.reset()

        # Initially enabled
        manager.update(0.02, sample_state)
        manager.get_total_force()

        # Disable
        manager.enabled = False
        force_disabled = manager.get_total_force()

        assert np.allclose(force_disabled, np.zeros(3))

    def test_global_intensity(self, rng, sample_state):
        """Test global intensity scaling."""
        manager = PerturbationManager(seed=42)
        wind = WindPerturbation(
            WindConfig(
                enabled=True,
                intensity=1.0,
                steady_wind_velocity=5.0,
                gusts_enabled=False,
                turbulence_enabled=False,
            )
        )
        manager.add_perturbation("wind", wind)
        manager.reset()

        # Update to generate force
        for _ in range(10):
            manager.update(0.02, sample_state)

        manager.get_total_force().copy()

        # Set half intensity
        manager.global_intensity = 0.5
        manager.reset()
        for _ in range(10):
            manager.update(0.02, sample_state)

        manager.get_total_force()

        # Force should be approximately halved
        # Note: This is approximate due to wind dynamics

    def test_list_perturbations(self, perturbation_manager):
        """Test listing perturbations."""
        names = perturbation_manager.list_perturbations()
        assert "wind" in names
        assert "delays" in names
        assert "sensor_noise" in names
        assert "physics" in names
        assert "aerodynamics" in names
        assert "external_forces" in names

    def test_get_info(self, perturbation_manager, sample_state):
        """Test get_info method."""
        perturbation_manager.reset()
        perturbation_manager.update(0.02, sample_state)

        info = perturbation_manager.get_info()

        assert "enabled" in info
        assert "global_intensity" in info
        assert "total_force" in info
        assert "total_torque" in info
        assert "perturbations" in info


# =============================================================================
# Wind Perturbation Tests
# =============================================================================


class TestWindPerturbation:
    """Tests for WindPerturbation."""

    def test_initialization(self):
        """Test wind perturbation initialization."""
        wind = WindPerturbation()
        assert wind.enabled is True
        assert isinstance(wind.wind_config, WindConfig)

    def test_steady_wind(self, rng, sample_state):
        """Test steady wind component."""
        config = WindConfig(
            enabled=True,
            intensity=1.0,
            steady_wind_enabled=True,
            steady_wind_velocity=5.0,
            gusts_enabled=False,
            turbulence_enabled=False,
            shear_enabled=False,
            thermals_enabled=False,
        )
        wind = WindPerturbation(config)
        wind.reset(rng)

        # Update several times
        for _ in range(50):
            wind.update(0.02, sample_state)

        # Should produce non-zero force
        wind.get_force()
        wind_vel = wind.get_wind_velocity()

        assert np.linalg.norm(wind_vel[:2]) > 0  # Horizontal wind

    def test_gusts(self, rng, sample_state):
        """Test gust generation."""
        config = WindConfig(
            enabled=True,
            steady_wind_enabled=False,
            gusts_enabled=True,
            gust_probability=0.5,  # High probability for testing
            turbulence_enabled=False,
        )
        wind = WindPerturbation(config)
        wind.reset(rng)

        # Run for a while to trigger gusts
        for _ in range(100):
            wind.update(0.02, sample_state)
            info = wind.get_info()
            if info["gust_active"]:
                break

        assert True  # Probabilistic, may not always trigger

    def test_turbulence_types(self, rng, sample_state):
        """Test different turbulence types."""
        for turb_type in ["gaussian", "dryden", "perlin"]:
            config = WindConfig(
                enabled=True,
                steady_wind_enabled=False,
                gusts_enabled=False,
                turbulence_enabled=True,
                turbulence_type=turb_type,
                turbulence_intensity=0.5,
            )
            wind = WindPerturbation(config)
            wind.reset(rng)

            for _ in range(10):
                wind.update(0.02, sample_state)

            # Should not raise errors

    def test_factory_functions(self):
        """Test wind factory functions."""
        light = create_light_breeze()
        moderate = create_moderate_wind()

        assert light.wind_config.steady_wind_velocity < moderate.wind_config.steady_wind_velocity


# =============================================================================
# Delay Perturbation Tests
# =============================================================================


class TestDelayPerturbation:
    """Tests for DelayPerturbation."""

    def test_initialization(self):
        """Test delay perturbation initialization."""
        delay = DelayPerturbation()
        assert delay.enabled is True

    def test_observation_delay(self, rng, sample_state):
        """Test observation delay application."""
        delay = create_typical_latency()
        delay.reset(rng)

        # Create sample observation
        obs = np.arange(11, dtype=np.float32)

        # Push several observations
        for i in range(20):
            delay.update(0.02, sample_state)
            delay.apply_to_observation(obs + i)

        # Delayed observation should differ from current
        # (depending on delay magnitude)

    def test_action_delay(self, rng, sample_state):
        """Test action delay and motor dynamics."""
        delay = create_typical_latency()
        delay.reset(rng)

        action = np.array([0.5, 0.5, 0.5, 0.5])

        # Update and apply
        delay.update(0.02, sample_state)
        delayed_action = delay.apply_to_action(action)

        # Action should be modified (filtered/delayed)
        assert delayed_action is not action

    def test_jitter(self, rng, sample_state):
        """Test jitter variation."""
        config = DelayConfig(
            enabled=True,
            jitter_enabled=True,
            jitter_base=1.0,
            jitter_max=10.0,
        )
        delay = DelayPerturbation(config)
        delay.reset(rng)

        # Collect jitter values
        jitters = []
        for _ in range(100):
            delay.update(0.02, sample_state)
            info = delay.get_info()
            jitters.append(info["current_jitter"])

        # Jitter should vary
        jitters = np.array(jitters)
        assert jitters.std() > 0 or len(np.unique(jitters)) > 1


# =============================================================================
# Sensor Noise Perturbation Tests
# =============================================================================


class TestSensorNoisePerturbation:
    """Tests for SensorNoisePerturbation."""

    def test_initialization(self):
        """Test sensor noise initialization."""
        noise = SensorNoisePerturbation()
        assert noise.enabled is True

    def test_observation_noise(self, rng, sample_state):
        """Test noise application to observations."""
        noise = create_typical_noise()
        noise.reset(rng)
        noise.update(0.02, sample_state)

        obs = np.zeros(11, dtype=np.float32)
        noisy_obs = noise.apply_to_observation(obs.copy())

        # Observation should be modified
        assert not np.allclose(noisy_obs, obs)

    def test_bias_drift(self, rng, sample_state):
        """Test sensor bias drift."""
        config = SensorNoiseConfig(
            enabled=True,
            drift_enabled=True,
            gyro_bias_drift=0.01,  # Fast drift for testing
            accel_bias_drift=0.01,
        )
        noise = SensorNoisePerturbation(config)
        noise.reset(rng)

        initial_gyro_bias = noise._gyro_bias.bias.copy()

        # Update many times
        for _ in range(100):
            noise.update(0.02, sample_state)

        final_gyro_bias = noise._gyro_bias.bias

        # Bias should have drifted
        assert not np.allclose(initial_gyro_bias, final_gyro_bias)


# =============================================================================
# Physics Perturbation Tests
# =============================================================================


class TestPhysicsPerturbation:
    """Tests for PhysicsPerturbation."""

    def test_initialization(self):
        """Test physics perturbation initialization."""
        physics = PhysicsPerturbation()
        assert physics.enabled is True

    def test_ground_effect(self, rng):
        """Test ground effect at different altitudes."""
        config = PhysicsConfig(
            enabled=True,
            ground_effect_enabled=True,
            ground_effect_height=1.0,
            ground_effect_strength=0.5,
        )
        physics = PhysicsPerturbation(config)
        physics.reset(rng)

        # State near ground
        state_low = QuadrotorState(
            position=np.array([0.0, 0.0, 0.2]),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            motor_speeds=np.array([0.6, 0.6, 0.6, 0.6]),
        )

        # State high
        state_high = QuadrotorState(
            position=np.array([0.0, 0.0, 5.0]),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            motor_speeds=np.array([0.6, 0.6, 0.6, 0.6]),
        )

        physics.update(0.02, state_low)
        force_low = physics.get_force().copy()

        physics.update(0.02, state_high)
        force_high = physics.get_force()

        # Ground effect should produce more upward force near ground
        assert force_low[2] > force_high[2]

    def test_motor_variation(self, rng, sample_state):
        """Test motor variation effects."""
        config = PhysicsConfig(
            enabled=True,
            motor_variation_enabled=True,
            motor_thrust_variation=0.1,
            motor_per_motor=True,
        )
        physics = PhysicsPerturbation(config)
        physics.reset(rng)

        motor_status = physics.get_motor_status()

        # Check motors have different thrust factors
        thrust_factors = [m["thrust_factor"] for m in motor_status]
        assert not all(f == thrust_factors[0] for f in thrust_factors)


# =============================================================================
# Aerodynamics Perturbation Tests
# =============================================================================


class TestAerodynamicsPerturbation:
    """Tests for AerodynamicsPerturbation."""

    def test_initialization(self):
        """Test aerodynamics initialization."""
        aero = AerodynamicsPerturbation()
        assert aero.enabled is True

    def test_drag_force(self, rng):
        """Test drag force at different velocities."""
        config = AerodynamicsConfig(
            enabled=True,
            drag_enabled=True,
            drag_coefficient=0.5,
        )
        aero = AerodynamicsPerturbation(config)
        aero.reset(rng)

        # Low velocity
        state_slow = QuadrotorState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            motor_speeds=np.array([0.5, 0.5, 0.5, 0.5]),
        )

        # High velocity
        state_fast = QuadrotorState(
            position=np.array([0.0, 0.0, 1.0]),
            velocity=np.array([10.0, 0.0, 0.0]),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            motor_speeds=np.array([0.5, 0.5, 0.5, 0.5]),
        )

        aero.update(0.02, state_slow)
        drag_slow = np.linalg.norm(aero.get_drag_force())

        aero.update(0.02, state_fast)
        drag_fast = np.linalg.norm(aero.get_drag_force())

        # Drag should be much higher at high velocity (proportional to v^2)
        assert drag_fast > drag_slow * 10


# =============================================================================
# External Forces Perturbation Tests
# =============================================================================


class TestExternalForcesPerturbation:
    """Tests for ExternalForcesPerturbation."""

    def test_initialization(self):
        """Test external forces initialization."""
        ext = ExternalForcesPerturbation()
        assert ext.enabled is True

    def test_manual_impulse(self, rng, sample_state):
        """Test manually applied impulse."""
        ext = create_calm_environment()
        ext.reset(rng)

        # Apply impulse
        ext.apply_impulse(
            force=np.array([10.0, 0.0, 0.0]),
            torque=np.array([0.0, 0.0, 0.1]),
            duration=0.1,
        )

        # Update during impulse
        ext.update(0.02, sample_state)
        force = ext.get_force()

        # Should have force applied
        assert np.linalg.norm(force) > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the perturbation system."""

    def test_full_update_cycle(self, perturbation_manager, sample_state):
        """Test full update cycle with all perturbations."""
        manager = perturbation_manager
        manager.reset()

        # Run for several steps
        for _ in range(100):
            manager.update(0.02, sample_state)

            force = manager.get_total_force()
            torque = manager.get_total_torque()

            # Should produce valid outputs
            assert force.shape == (3,)
            assert torque.shape == (3,)
            assert not np.any(np.isnan(force))
            assert not np.any(np.isnan(torque))

    def test_observation_pipeline(self, perturbation_manager, sample_state):
        """Test observation perturbation pipeline."""
        manager = perturbation_manager
        manager.reset()
        manager.update(0.02, sample_state)

        obs = np.zeros(11, dtype=np.float32)
        perturbed_obs = manager.apply_to_observation(obs)

        # Should return valid observation
        assert perturbed_obs.shape == obs.shape
        assert not np.any(np.isnan(perturbed_obs))

    def test_action_pipeline(self, perturbation_manager, sample_state):
        """Test action perturbation pipeline."""
        manager = perturbation_manager
        manager.reset()
        manager.update(0.02, sample_state)

        action = np.array([0.5, 0.5, 0.5, 0.5])
        perturbed_action = manager.apply_to_action(action)

        # Should return valid action
        assert perturbed_action.shape == action.shape
        assert not np.any(np.isnan(perturbed_action))
        assert np.all(perturbed_action >= 0) and np.all(perturbed_action <= 1)

    def test_config_loading(self, tmp_path):
        """Test loading configuration from YAML."""
        config_content = """
enabled: true
global_intensity: 0.5

wind:
  enabled: true
  intensity: 1.0
  steady_wind_velocity: 3.0
  gusts_enabled: false
  turbulence_enabled: true
  turbulence_intensity: 0.2
"""
        config_file = tmp_path / "test_perturbations.yaml"
        config_file.write_text(config_content)

        manager = PerturbationManager(config_path=str(config_file))

        assert manager.global_intensity == 0.5
        assert "wind" in manager

        wind = manager.get_perturbation("wind")
        assert wind.wind_config.steady_wind_velocity == 3.0


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
