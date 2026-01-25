"""Perturbation system for realistic drone simulation.

This module provides a comprehensive set of perturbations to simulate
real-world conditions including wind, sensor delays, noise, and more.

Available perturbation types:
- WindPerturbation: Atmospheric wind effects
- DelayPerturbation: Sensor and actuator latency
- SensorNoisePerturbation: Sensor noise and failures
- PhysicsPerturbation: Physical property variations
- AerodynamicsPerturbation: Aerodynamic effects
- ExternalForcesPerturbation: External disturbances
"""

from .aerodynamics import (
    AerodynamicsPerturbation,
    AirDensityModel,
    VortexRingState,
    create_aggressive_flight_aero,
    create_high_altitude_aero,
    create_minimal_aero,
    create_realistic_aero,
)
from .base import (
    AerodynamicsConfig,
    BasePerturbation,
    DelayConfig,
    ExternalForcesConfig,
    PerturbationConfig,
    PerturbationManager,
    PhysicsConfig,
    SensorNoiseConfig,
    WindConfig,
)
from .delays import (
    DelayBuffer,
    DelayPerturbation,
    FirstOrderFilter,
    create_high_latency,
    create_low_latency,
    create_typical_latency,
    create_unreliable_connection,
)
from .external_forces import (
    ExternalForcesPerturbation,
    ImpulseEvent,
    ImpulseType,
    PeriodicDisturbance,
    create_calm_environment,
    create_industrial_environment,
    create_turbulent_environment,
    create_urban_environment,
)
from .physics import (
    GroundEffect,
    MotorModel,
    PhysicsPerturbation,
    create_ideal_physics,
    create_payload_variation,
    create_realistic_physics,
    create_worn_drone,
)
from .sensor_noise import (
    SensorNoisePerturbation,
    SensorType,
    create_harsh_conditions,
    create_low_noise,
    create_noisy_sensors,
    create_typical_noise,
)
from .wind import (
    WindPerturbation,
    create_gusty_conditions,
    create_light_breeze,
    create_moderate_wind,
    create_strong_wind,
)

__all__ = [
    # Base classes
    "BasePerturbation",
    "PerturbationConfig",
    "PerturbationManager",
    # Config classes
    "WindConfig",
    "DelayConfig",
    "SensorNoiseConfig",
    "PhysicsConfig",
    "AerodynamicsConfig",
    "ExternalForcesConfig",
    # Wind
    "WindPerturbation",
    "create_light_breeze",
    "create_moderate_wind",
    "create_strong_wind",
    "create_gusty_conditions",
    # Delays
    "DelayPerturbation",
    "DelayBuffer",
    "FirstOrderFilter",
    "create_low_latency",
    "create_typical_latency",
    "create_high_latency",
    "create_unreliable_connection",
    # Sensor noise
    "SensorNoisePerturbation",
    "SensorType",
    "create_low_noise",
    "create_typical_noise",
    "create_noisy_sensors",
    "create_harsh_conditions",
    # Physics
    "PhysicsPerturbation",
    "MotorModel",
    "GroundEffect",
    "create_ideal_physics",
    "create_realistic_physics",
    "create_worn_drone",
    "create_payload_variation",
    # Aerodynamics
    "AerodynamicsPerturbation",
    "AirDensityModel",
    "VortexRingState",
    "create_minimal_aero",
    "create_realistic_aero",
    "create_high_altitude_aero",
    "create_aggressive_flight_aero",
    # External forces
    "ExternalForcesPerturbation",
    "ImpulseType",
    "ImpulseEvent",
    "PeriodicDisturbance",
    "create_calm_environment",
    "create_urban_environment",
    "create_turbulent_environment",
    "create_industrial_environment",
]
