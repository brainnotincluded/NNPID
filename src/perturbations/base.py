"""Base classes for perturbation system."""

from __future__ import annotations

import numpy as np
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from collections import deque
from copy import deepcopy

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


@dataclass
class PerturbationConfig:
    """Base configuration for all perturbations."""
    
    enabled: bool = True
    intensity: float = 1.0  # Global intensity multiplier (0-1)
    randomize_on_reset: bool = True
    seed: Optional[int] = None
    
    def scale(self, value: float) -> float:
        """Scale a value by intensity."""
        return value * self.intensity if self.enabled else 0.0


@dataclass
class WindConfig(PerturbationConfig):
    """Configuration for wind perturbations."""
    
    # Steady wind
    steady_wind_enabled: bool = True
    steady_wind_velocity: float = 2.0  # m/s
    steady_wind_direction: float = 0.0  # radians
    steady_wind_vertical: float = 0.0  # m/s
    direction_change_rate: float = 0.1  # rad/s
    
    # Gusts
    gusts_enabled: bool = True
    gust_probability: float = 0.02  # per timestep
    gust_min_duration: float = 0.5  # seconds
    gust_max_duration: float = 2.0  # seconds
    gust_min_intensity: float = 1.0  # m/s
    gust_max_intensity: float = 5.0  # m/s
    gust_rise_time: float = 0.3  # seconds
    gust_fall_time: float = 0.5  # seconds
    
    # Turbulence
    turbulence_enabled: bool = True
    turbulence_intensity: float = 0.3  # 0-1
    turbulence_frequency: float = 2.0  # Hz
    turbulence_type: str = "gaussian"  # gaussian, perlin, dryden
    
    # Wind shear (gradient with altitude)
    shear_enabled: bool = False
    shear_ground_velocity: float = 0.0  # m/s at ground
    shear_gradient: float = 0.5  # m/s per meter height
    shear_reference_height: float = 10.0  # meters
    
    # Thermals
    thermals_enabled: bool = False
    thermal_probability: float = 0.01
    thermal_radius: float = 5.0  # meters
    thermal_strength: float = 2.0  # m/s upward
    thermal_height_min: float = 0.5  # meters
    thermal_height_max: float = 50.0  # meters


@dataclass
class DelayConfig(PerturbationConfig):
    """Configuration for delay perturbations."""
    
    # Sensor delays (milliseconds)
    imu_delay: float = 2.0  # ms
    gps_delay: float = 100.0  # ms
    barometer_delay: float = 20.0  # ms
    magnetometer_delay: float = 10.0  # ms
    
    # Actuator delays
    motor_delay: float = 5.0  # ms
    motor_time_constant: float = 20.0  # ms (first-order filter)
    
    # Communication delays
    command_delay: float = 10.0  # ms
    telemetry_delay: float = 50.0  # ms
    
    # Jitter
    jitter_enabled: bool = True
    jitter_base: float = 1.0  # ms
    jitter_max: float = 5.0  # ms
    jitter_distribution: str = "gaussian"  # uniform, gaussian, exponential
    
    # Sample dropout
    dropout_enabled: bool = False
    dropout_probability: float = 0.01
    dropout_max_consecutive: int = 3


@dataclass
class SensorNoiseConfig(PerturbationConfig):
    """Configuration for sensor noise perturbations."""
    
    # Gaussian noise
    gyro_noise_std: float = 0.005  # rad/s
    accel_noise_std: float = 0.05  # m/s^2
    position_noise_std: float = 0.02  # meters
    velocity_noise_std: float = 0.05  # m/s
    quaternion_noise_std: float = 0.001  # per component
    
    # Sensor drift (bias random walk)
    drift_enabled: bool = True
    gyro_bias_drift: float = 0.0001  # rad/s per sqrt(s)
    accel_bias_drift: float = 0.001  # m/s^2 per sqrt(s)
    
    # Outliers
    outliers_enabled: bool = False
    outlier_probability: float = 0.001
    outlier_magnitude: float = 10.0  # multiplier
    
    # GPS loss
    gps_loss_enabled: bool = False
    gps_loss_probability: float = 0.001
    gps_loss_min_duration: float = 1.0  # seconds
    gps_loss_max_duration: float = 5.0  # seconds
    
    # Magnetic interference
    magnetic_interference_enabled: bool = False
    magnetic_random_probability: float = 0.01
    magnetic_max_magnitude: float = 0.5  # Gauss
    
    # Temperature effects
    temperature_enabled: bool = False
    ambient_temperature: float = 20.0  # Celsius
    temperature_coefficient: float = 0.01  # per degree


@dataclass
class PhysicsConfig(PerturbationConfig):
    """Configuration for physics perturbations."""
    
    # Center of mass offset
    com_offset_enabled: bool = False
    com_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    com_randomize: bool = True
    com_max_offset: float = 0.02  # meters
    
    # Motor variation
    motor_variation_enabled: bool = True
    motor_thrust_variation: float = 0.05  # 5% variation
    motor_response_variation: float = 0.1  # 10% variation
    motor_per_motor: bool = True  # Individual variation per motor
    
    # Motor degradation
    degradation_enabled: bool = False
    degradation_rate: float = 0.0001  # % per second
    degradation_initial: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    degradation_failure_probability: float = 0.0  # per episode
    
    # Mass change
    mass_change_enabled: bool = False
    mass_variation_min: float = 0.9  # multiplier
    mass_variation_max: float = 1.1
    mass_dynamic_change: bool = False
    mass_change_rate: float = 0.0  # kg/s
    
    # Ground effect
    ground_effect_enabled: bool = True
    ground_effect_height: float = 0.5  # meters
    ground_effect_strength: float = 0.3  # 0-1
    ground_effect_thrust_mult: float = 1.2  # at ground level
    
    # Wall proximity effect
    proximity_enabled: bool = False
    proximity_distance: float = 1.0  # meters
    proximity_coefficient: float = 0.1


@dataclass
class AerodynamicsConfig(PerturbationConfig):
    """Configuration for aerodynamic perturbations."""
    
    # Air drag
    drag_enabled: bool = True
    drag_coefficient: float = 0.5
    air_density: float = 1.225  # kg/m^3 at sea level
    reference_area: float = 0.1  # m^2
    
    # Blade flapping
    blade_flapping_enabled: bool = False
    flapping_coefficient: float = 0.01
    
    # Vortex ring state
    vrs_enabled: bool = False
    vrs_descent_threshold: float = 2.0  # m/s descent
    vrs_intensity: float = 0.3
    
    # Air density variation
    density_variation_enabled: bool = False
    density_altitude_coeff: float = 0.00012  # per meter
    density_temperature_coeff: float = 0.004  # per degree C
    density_random_variation: float = 0.02  # 2%


@dataclass
class ExternalForcesConfig(PerturbationConfig):
    """Configuration for external force perturbations."""
    
    # Random impulses
    impulses_enabled: bool = False
    impulse_probability: float = 0.001
    impulse_min_interval: float = 2.0  # seconds
    impulse_force_min: float = 0.5  # N
    impulse_force_max: float = 5.0  # N
    impulse_torque_min: float = 0.01  # Nm
    impulse_torque_max: float = 0.1  # Nm
    impulse_duration: float = 0.05  # seconds
    
    # Periodic disturbance
    periodic_enabled: bool = False
    periodic_frequency: float = 1.0  # Hz
    periodic_amplitude: float = 0.5  # N
    periodic_direction: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    
    # Vibrations
    vibrations_enabled: bool = False
    vibration_frequencies: List[float] = field(default_factory=lambda: [50.0, 100.0])
    vibration_amplitudes: List[float] = field(default_factory=lambda: [0.1, 0.05])
    vibration_motor_coupled: bool = True


class BasePerturbation(ABC):
    """Abstract base class for all perturbations.
    
    Perturbations can affect:
    - Forces and torques applied to the drone
    - Sensor readings (observations)
    - Action/actuator commands
    """
    
    def __init__(self, config: PerturbationConfig):
        """Initialize perturbation.
        
        Args:
            config: Perturbation configuration
        """
        self.config = config
        self._rng: Optional[np.random.Generator] = None
        self._time = 0.0
        self._is_active = config.enabled
        
        # Current perturbation values
        self._current_force = np.zeros(3)
        self._current_torque = np.zeros(3)
    
    @property
    def enabled(self) -> bool:
        """Check if perturbation is enabled."""
        return self._is_active and self.config.enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable perturbation."""
        self._is_active = value
    
    @property
    def intensity(self) -> float:
        """Get current intensity."""
        return self.config.intensity
    
    @intensity.setter
    def intensity(self, value: float) -> None:
        """Set intensity (0-1)."""
        self.config.intensity = np.clip(value, 0.0, 1.0)
    
    def reset(self, rng: np.random.Generator) -> None:
        """Reset perturbation state.
        
        Args:
            rng: Random number generator
        """
        self._rng = rng
        self._time = 0.0
        self._current_force = np.zeros(3)
        self._current_torque = np.zeros(3)
        
        if self.config.randomize_on_reset:
            self._randomize_parameters()
    
    def _randomize_parameters(self) -> None:
        """Randomize perturbation parameters. Override in subclasses."""
        pass
    
    @abstractmethod
    def update(self, dt: float, state: "QuadrotorState") -> None:
        """Update perturbation state.
        
        Args:
            dt: Time step in seconds
            state: Current quadrotor state
        """
        pass
    
    def get_force(self) -> np.ndarray:
        """Get current perturbation force in world frame.
        
        Returns:
            Force vector [fx, fy, fz] in Newtons
        """
        if not self.enabled:
            return np.zeros(3)
        return self._current_force * self.config.intensity
    
    def get_torque(self) -> np.ndarray:
        """Get current perturbation torque in body frame.
        
        Returns:
            Torque vector [tx, ty, tz] in Newton-meters
        """
        if not self.enabled:
            return np.zeros(3)
        return self._current_torque * self.config.intensity
    
    def apply_to_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply perturbation to observation/sensor data.
        
        Override in subclasses for sensor perturbations.
        
        Args:
            obs: Original observation
            
        Returns:
            Perturbed observation
        """
        return obs
    
    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply perturbation to action/actuator commands.
        
        Override in subclasses for actuator perturbations.
        
        Args:
            action: Original action
            
        Returns:
            Perturbed action
        """
        return action
    
    def get_info(self) -> Dict[str, Any]:
        """Get perturbation information.
        
        Returns:
            Dictionary with perturbation state
        """
        return {
            "type": self.__class__.__name__,
            "enabled": self.enabled,
            "intensity": self.intensity,
            "force": self._current_force.tolist(),
            "torque": self._current_torque.tolist(),
        }


class PerturbationManager:
    """Manages multiple perturbations and applies them to simulation.
    
    The manager coordinates all perturbations, handles their lifecycle,
    and provides a unified interface for the environment.
    """
    
    def __init__(self, config_path: Optional[str] = None, seed: Optional[int] = None):
        """Initialize perturbation manager.
        
        Args:
            config_path: Path to YAML configuration file
            seed: Random seed
        """
        self._perturbations: Dict[str, BasePerturbation] = {}
        self._rng = np.random.default_rng(seed)
        self._global_intensity = 1.0
        self._enabled = True
        
        # State history for delays
        self._state_history: deque = deque(maxlen=1000)
        self._observation_history: deque = deque(maxlen=1000)
        self._action_history: deque = deque(maxlen=100)
        
        # Load config if provided
        if config_path is not None:
            self.load_config(config_path)
    
    @property
    def enabled(self) -> bool:
        """Check if manager is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable all perturbations."""
        self._enabled = value
    
    @property
    def global_intensity(self) -> float:
        """Get global intensity multiplier."""
        return self._global_intensity
    
    @global_intensity.setter
    def global_intensity(self, value: float) -> None:
        """Set global intensity multiplier (0-1)."""
        self._global_intensity = np.clip(value, 0.0, 1.0)
    
    def add_perturbation(self, name: str, perturbation: BasePerturbation) -> None:
        """Add a perturbation to the manager.
        
        Args:
            name: Unique name for the perturbation
            perturbation: Perturbation instance
        """
        self._perturbations[name] = perturbation
    
    def remove_perturbation(self, name: str) -> Optional[BasePerturbation]:
        """Remove a perturbation from the manager.
        
        Args:
            name: Name of perturbation to remove
            
        Returns:
            Removed perturbation or None if not found
        """
        return self._perturbations.pop(name, None)
    
    def get_perturbation(self, name: str) -> Optional[BasePerturbation]:
        """Get a perturbation by name.
        
        Args:
            name: Perturbation name
            
        Returns:
            Perturbation instance or None
        """
        return self._perturbations.get(name)
    
    def enable_perturbation(self, name: str) -> None:
        """Enable a specific perturbation.
        
        Args:
            name: Perturbation name
        """
        if name in self._perturbations:
            self._perturbations[name].enabled = True
    
    def disable_perturbation(self, name: str) -> None:
        """Disable a specific perturbation.
        
        Args:
            name: Perturbation name
        """
        if name in self._perturbations:
            self._perturbations[name].enabled = False
    
    def set_intensity(self, name: str, intensity: float) -> None:
        """Set intensity for a specific perturbation.
        
        Args:
            name: Perturbation name
            intensity: Intensity value (0-1)
        """
        if name in self._perturbations:
            self._perturbations[name].intensity = intensity
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all perturbations.
        
        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Clear histories
        self._state_history.clear()
        self._observation_history.clear()
        self._action_history.clear()
        
        # Reset all perturbations
        for perturbation in self._perturbations.values():
            # Create child RNG for each perturbation
            child_rng = np.random.default_rng(self._rng.integers(0, 2**32))
            perturbation.reset(child_rng)
    
    def update(self, dt: float, state: "QuadrotorState") -> None:
        """Update all perturbations.
        
        Args:
            dt: Time step in seconds
            state: Current quadrotor state
        """
        if not self._enabled:
            return
        
        # Store state in history
        self._state_history.append((state, dt))
        
        # Update each perturbation
        for perturbation in self._perturbations.values():
            if perturbation.enabled:
                perturbation.update(dt, state)
    
    def get_total_force(self) -> np.ndarray:
        """Get total force from all perturbations.
        
        Returns:
            Sum of all perturbation forces [fx, fy, fz]
        """
        if not self._enabled:
            return np.zeros(3)
        
        total_force = np.zeros(3)
        for perturbation in self._perturbations.values():
            total_force += perturbation.get_force()
        
        return total_force * self._global_intensity
    
    def get_total_torque(self) -> np.ndarray:
        """Get total torque from all perturbations.
        
        Returns:
            Sum of all perturbation torques [tx, ty, tz]
        """
        if not self._enabled:
            return np.zeros(3)
        
        total_torque = np.zeros(3)
        for perturbation in self._perturbations.values():
            total_torque += perturbation.get_torque()
        
        return total_torque * self._global_intensity
    
    def apply_to_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply all perturbations to observation.
        
        Args:
            obs: Original observation
            
        Returns:
            Perturbed observation
        """
        if not self._enabled:
            return obs
        
        # Store in history
        self._observation_history.append(obs.copy())
        
        result = obs.copy()
        for perturbation in self._perturbations.values():
            if perturbation.enabled:
                result = perturbation.apply_to_observation(result)
        
        return result
    
    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply all perturbations to action.
        
        Args:
            action: Original action
            
        Returns:
            Perturbed action
        """
        if not self._enabled:
            return action
        
        # Store in history
        self._action_history.append(action.copy())
        
        result = action.copy()
        for perturbation in self._perturbations.values():
            if perturbation.enabled:
                result = perturbation.apply_to_action(result)
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about all perturbations.
        
        Returns:
            Dictionary with manager and perturbation info
        """
        return {
            "enabled": self._enabled,
            "global_intensity": self._global_intensity,
            "total_force": self.get_total_force().tolist(),
            "total_torque": self.get_total_torque().tolist(),
            "perturbations": {
                name: p.get_info() for name, p in self._perturbations.items()
            },
        }
    
    def load_config(self, config_path: str) -> None:
        """Load perturbations from YAML configuration.
        
        Args:
            config_path: Path to YAML file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path) as f:
            config = yaml.safe_load(f)
        
        self._load_from_dict(config)
    
    def _load_from_dict(self, config: Dict[str, Any]) -> None:
        """Load perturbations from configuration dictionary.
        
        Args:
            config: Configuration dictionary
        """
        # Import perturbation classes here to avoid circular imports
        from .wind import WindPerturbation
        from .delays import DelayPerturbation
        from .sensor_noise import SensorNoisePerturbation
        from .physics import PhysicsPerturbation
        from .aerodynamics import AerodynamicsPerturbation
        from .external_forces import ExternalForcesPerturbation
        
        # Global settings
        self._enabled = config.get("enabled", True)
        self._global_intensity = config.get("global_intensity", 1.0)
        
        # Wind
        if "wind" in config:
            wind_config = self._dict_to_config(config["wind"], WindConfig)
            self.add_perturbation("wind", WindPerturbation(wind_config))
        
        # Delays
        if "delays" in config:
            delay_config = self._dict_to_config(config["delays"], DelayConfig)
            self.add_perturbation("delays", DelayPerturbation(delay_config))
        
        # Sensor noise
        if "sensor_noise" in config:
            noise_config = self._dict_to_config(config["sensor_noise"], SensorNoiseConfig)
            self.add_perturbation("sensor_noise", SensorNoisePerturbation(noise_config))
        
        # Physics
        if "physics" in config:
            physics_config = self._dict_to_config(config["physics"], PhysicsConfig)
            self.add_perturbation("physics", PhysicsPerturbation(physics_config))
        
        # Aerodynamics
        if "aerodynamics" in config:
            aero_config = self._dict_to_config(config["aerodynamics"], AerodynamicsConfig)
            self.add_perturbation("aerodynamics", AerodynamicsPerturbation(aero_config))
        
        # External forces
        if "external_forces" in config:
            forces_config = self._dict_to_config(config["external_forces"], ExternalForcesConfig)
            self.add_perturbation("external_forces", ExternalForcesPerturbation(forces_config))
    
    @staticmethod
    def _dict_to_config(d: Dict[str, Any], config_class: type) -> Any:
        """Convert dictionary to config dataclass.
        
        Args:
            d: Dictionary with config values
            config_class: Dataclass type
            
        Returns:
            Config instance
        """
        # Filter to only include valid fields
        valid_fields = {f.name for f in config_class.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return config_class(**filtered)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], seed: Optional[int] = None) -> "PerturbationManager":
        """Create manager from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            seed: Random seed
            
        Returns:
            Configured PerturbationManager
        """
        manager = cls(seed=seed)
        manager._load_from_dict(config)
        return manager
    
    def list_perturbations(self) -> List[str]:
        """List all registered perturbation names.
        
        Returns:
            List of perturbation names
        """
        return list(self._perturbations.keys())
    
    def __contains__(self, name: str) -> bool:
        """Check if perturbation exists."""
        return name in self._perturbations
    
    def __len__(self) -> int:
        """Get number of perturbations."""
        return len(self._perturbations)
