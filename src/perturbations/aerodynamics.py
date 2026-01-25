"""Aerodynamic perturbations for realistic flight dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import AerodynamicsConfig, BasePerturbation

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


class AirDensityModel:
    """Models air density variations with altitude and temperature.

    Uses International Standard Atmosphere (ISA) model with random variations.
    """

    # Standard atmosphere constants
    SEA_LEVEL_DENSITY = 1.225  # kg/m^3
    SEA_LEVEL_TEMPERATURE = 288.15  # K (15°C)
    TEMPERATURE_LAPSE_RATE = 0.0065  # K/m
    GAS_CONSTANT = 287.058  # J/(kg·K)
    GRAVITY = 9.80665  # m/s^2

    def __init__(
        self,
        base_density: float = 1.225,
        altitude_coefficient: float = 0.00012,
        temperature_coefficient: float = 0.004,
        random_variation: float = 0.02,
    ):
        """Initialize air density model.

        Args:
            base_density: Sea-level air density (kg/m^3)
            altitude_coefficient: Density decrease per meter altitude
            temperature_coefficient: Density change per degree C
            random_variation: Random variation factor (0-1)
        """
        self.base_density = base_density
        self.altitude_coefficient = altitude_coefficient
        self.temperature_coefficient = temperature_coefficient
        self.random_variation = random_variation

        self._current_variation = 0.0
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset with new random variation."""
        self._rng = rng
        self._current_variation = rng.uniform(-self.random_variation, self.random_variation)

    def get_density(
        self,
        altitude: float,
        temperature_offset: float = 0.0,
    ) -> float:
        """Calculate air density at given altitude.

        Args:
            altitude: Altitude above sea level (m)
            temperature_offset: Temperature deviation from ISA (°C)

        Returns:
            Air density (kg/m^3)
        """
        # ISA temperature at altitude
        temperature = self.SEA_LEVEL_TEMPERATURE - self.TEMPERATURE_LAPSE_RATE * altitude
        temperature += temperature_offset  # Apply offset

        # Pressure using barometric formula
        exponent = self.GRAVITY / (self.GAS_CONSTANT * self.TEMPERATURE_LAPSE_RATE)
        pressure_ratio = (temperature / self.SEA_LEVEL_TEMPERATURE) ** exponent

        # Density from ideal gas law
        density = self.base_density * pressure_ratio * (self.SEA_LEVEL_TEMPERATURE / temperature)

        # Apply random variation
        density *= 1.0 + self._current_variation

        return max(0.1, density)  # Prevent negative/zero density


class BladeFlappingModel:
    """Models blade flapping effects in forward flight.

    Rotor blades flap in response to changing airflow,
    creating additional forces and moments.
    """

    def __init__(
        self,
        flapping_coefficient: float = 0.01,
        rotor_radius: float = 0.127,  # 5 inch prop
        blade_lock_number: float = 8.0,  # Typical for small rotors
    ):
        """Initialize blade flapping model.

        Args:
            flapping_coefficient: Overall flapping response coefficient
            rotor_radius: Rotor radius (m)
            blade_lock_number: Lock number (blade inertia parameter)
        """
        self.flapping_coefficient = flapping_coefficient
        self.rotor_radius = rotor_radius
        self.blade_lock_number = blade_lock_number

    def get_forces_and_moments(
        self,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
        thrust: float,
    ) -> tuple:
        """Calculate blade flapping forces and moments.

        Args:
            velocity: Body-frame velocity [vx, vy, vz]
            angular_velocity: Body-frame angular velocity [p, q, r]
            thrust: Current total thrust (N)

        Returns:
            Tuple of (force, moment) arrays
        """
        # Advance ratio (normalized forward velocity)
        horizontal_velocity = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        # Simplified flapping response
        # In forward flight, blades flap due to asymmetric airflow

        # Force perturbation (H-force from blade flapping)
        h_force_x = -self.flapping_coefficient * thrust * velocity[0] / (horizontal_velocity + 0.1)
        h_force_y = -self.flapping_coefficient * thrust * velocity[1] / (horizontal_velocity + 0.1)

        force = np.array([h_force_x, h_force_y, 0.0])

        # Moment from flapping
        # Flapping creates pitching/rolling moments proportional to velocity
        roll_moment = -self.flapping_coefficient * velocity[1] * thrust * 0.1
        pitch_moment = self.flapping_coefficient * velocity[0] * thrust * 0.1

        moment = np.array([roll_moment, pitch_moment, 0.0])

        return force, moment


class VortexRingState:
    """Models Vortex Ring State (VRS) - dangerous condition during descent.

    VRS occurs when a rotor descends into its own downwash,
    causing loss of lift and control.
    """

    def __init__(
        self,
        descent_threshold: float = 2.0,  # m/s
        vrs_intensity: float = 0.3,
        induced_velocity: float = 4.0,  # m/s (hover induced velocity)
    ):
        """Initialize VRS model.

        Args:
            descent_threshold: Descent rate to trigger VRS (m/s)
            vrs_intensity: Intensity of VRS effects (0-1)
            induced_velocity: Hover induced velocity (m/s)
        """
        self.descent_threshold = descent_threshold
        self.vrs_intensity = vrs_intensity
        self.induced_velocity = induced_velocity

        self._in_vrs = False
        self._vrs_factor = 0.0
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset VRS state."""
        self._rng = rng
        self._in_vrs = False
        self._vrs_factor = 0.0

    def update(
        self,
        dt: float,
        vertical_velocity: float,
        horizontal_velocity: float,
    ) -> None:
        """Update VRS state.

        Args:
            dt: Time step
            vertical_velocity: Vertical velocity (positive up, m/s)
            horizontal_velocity: Horizontal velocity magnitude (m/s)
        """
        # VRS region: descending faster than threshold, low horizontal speed
        descent_rate = -vertical_velocity  # Positive when descending

        # VRS boundary (Wolkovitch/Johnson model approximation)
        vrs_boundary = self.descent_threshold * (
            1.0 + 0.3 * horizontal_velocity / self.induced_velocity
        )

        if descent_rate > vrs_boundary and horizontal_velocity < self.induced_velocity * 1.5:
            # Entering VRS
            self._in_vrs = True
            # VRS factor increases with how deep into VRS we are
            penetration = (descent_rate - vrs_boundary) / self.induced_velocity
            target_factor = min(1.0, penetration * self.vrs_intensity)

            # Gradual transition
            alpha = min(1.0, dt * 2.0)
            self._vrs_factor = (1 - alpha) * self._vrs_factor + alpha * target_factor
        else:
            # Exiting VRS
            alpha = min(1.0, dt * 0.5)  # Slower exit
            self._vrs_factor *= 1 - alpha
            if self._vrs_factor < 0.01:
                self._in_vrs = False
                self._vrs_factor = 0.0

    def get_thrust_loss(self) -> float:
        """Get thrust loss factor due to VRS.

        Returns:
            Thrust loss factor (0 = no loss, 1 = complete loss)
        """
        return self._vrs_factor

    def get_turbulence_force(self, thrust: float) -> np.ndarray:
        """Get random turbulent force from VRS.

        Args:
            thrust: Current thrust (N)

        Returns:
            Random force vector
        """
        if not self._in_vrs or self._rng is None:
            return np.zeros(3)

        # VRS causes chaotic, turbulent flow
        turbulence_magnitude = thrust * self._vrs_factor * 0.3
        random_force = self._rng.normal(0, turbulence_magnitude, 3)

        return random_force

    @property
    def in_vrs(self) -> bool:
        """Check if currently in VRS."""
        return self._in_vrs


class AerodynamicsPerturbation(BasePerturbation):
    """Comprehensive aerodynamic perturbations.

    Components:
    - Air drag: Velocity-dependent drag force
    - Blade flapping: Rotor blade dynamics in forward flight
    - Vortex ring state: Dangerous descent condition
    - Air density variation: Altitude and temperature effects
    """

    def __init__(self, config: AerodynamicsConfig | None = None):
        """Initialize aerodynamics perturbation.

        Args:
            config: Aerodynamics configuration. Uses defaults if None.
        """
        super().__init__(config or AerodynamicsConfig())
        self.aero_config: AerodynamicsConfig = self.config

        # Air density model
        self._density_model = AirDensityModel(
            base_density=self.aero_config.air_density,
            altitude_coefficient=self.aero_config.density_altitude_coeff,
            temperature_coefficient=self.aero_config.density_temperature_coeff,
            random_variation=self.aero_config.density_random_variation,
        )

        # Blade flapping model
        self._blade_flapping = BladeFlappingModel(
            flapping_coefficient=self.aero_config.flapping_coefficient,
        )

        # Vortex ring state model
        self._vrs = VortexRingState(
            descent_threshold=self.aero_config.vrs_descent_threshold,
            vrs_intensity=self.aero_config.vrs_intensity,
        )

        # Current air density
        self._current_density = self.aero_config.air_density

        # Drag force
        self._drag_force = np.zeros(3)

        # Blade flapping effects
        self._flapping_force = np.zeros(3)
        self._flapping_moment = np.zeros(3)

    def reset(self, rng: np.random.Generator) -> None:
        """Reset aerodynamics state."""
        super().reset(rng)

        self._density_model.reset(rng)
        self._vrs.reset(rng)

        self._current_density = self.aero_config.air_density
        self._drag_force = np.zeros(3)
        self._flapping_force = np.zeros(3)
        self._flapping_moment = np.zeros(3)

    def update(self, dt: float, state: QuadrotorState) -> None:
        """Update aerodynamic effects.

        Args:
            dt: Time step
            state: Current quadrotor state
        """
        if not self.enabled:
            self._current_force = np.zeros(3)
            self._current_torque = np.zeros(3)
            return

        self._time += dt
        cfg = self.aero_config

        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # Update air density
        if cfg.density_variation_enabled:
            self._current_density = self._density_model.get_density(
                altitude=state.position[2],
                temperature_offset=0.0,
            )
        else:
            self._current_density = cfg.air_density

        # Calculate drag force
        if cfg.drag_enabled:
            self._calculate_drag(state)
            total_force += self._drag_force * cfg.intensity

        # Blade flapping
        if cfg.blade_flapping_enabled:
            # Estimate current thrust
            thrust = np.mean(state.motor_speeds) * 32.0  # 4 motors * 8N

            (
                self._flapping_force,
                self._flapping_moment,
            ) = self._blade_flapping.get_forces_and_moments(
                velocity=state.velocity,
                angular_velocity=state.angular_velocity,
                thrust=thrust,
            )
            total_force += self._flapping_force * cfg.intensity
            total_torque += self._flapping_moment * cfg.intensity

        # Vortex ring state
        if cfg.vrs_enabled:
            horizontal_vel = np.sqrt(state.velocity[0] ** 2 + state.velocity[1] ** 2)
            self._vrs.update(dt, state.velocity[2], horizontal_vel)

            if self._vrs.in_vrs:
                thrust = np.mean(state.motor_speeds) * 32.0

                # Thrust loss (force downward)
                thrust_loss = thrust * self._vrs.get_thrust_loss()
                total_force[2] -= thrust_loss * cfg.intensity

                # Turbulent forces
                vrs_turbulence = self._vrs.get_turbulence_force(thrust)
                total_force += vrs_turbulence * cfg.intensity

        self._current_force = total_force
        self._current_torque = total_torque

    def _calculate_drag(self, state: QuadrotorState) -> None:
        """Calculate aerodynamic drag force.

        Args:
            state: Current state
        """
        cfg = self.aero_config

        velocity = state.velocity
        speed = np.linalg.norm(velocity)

        if speed < 0.01:
            self._drag_force = np.zeros(3)
            return

        # Drag equation: F = 0.5 * rho * Cd * A * v^2
        drag_magnitude = (
            0.5 * self._current_density * cfg.drag_coefficient * cfg.reference_area * speed**2
        )

        # Drag opposes velocity
        drag_direction = -velocity / speed

        self._drag_force = drag_magnitude * drag_direction

    def get_air_density(self) -> float:
        """Get current air density.

        Returns:
            Air density in kg/m^3
        """
        return self._current_density

    def get_drag_force(self) -> np.ndarray:
        """Get current drag force.

        Returns:
            Drag force vector
        """
        return self._drag_force.copy()

    def is_in_vrs(self) -> bool:
        """Check if in vortex ring state.

        Returns:
            True if in VRS
        """
        return self._vrs.in_vrs

    def get_vrs_severity(self) -> float:
        """Get VRS severity (0-1).

        Returns:
            VRS severity factor
        """
        return self._vrs._vrs_factor

    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply density-based thrust correction.

        Lower air density = less thrust for same motor command.

        Args:
            action: Motor commands

        Returns:
            Adjusted motor commands
        """
        if not self.enabled:
            return action

        cfg = self.aero_config

        # Thrust is proportional to air density
        # Adjust commands to compensate (or not, for realism)
        if cfg.density_variation_enabled:
            self._current_density / cfg.air_density
            # Don't compensate - let the controller deal with it
            # This simulates real thrust reduction at altitude
            # The action itself doesn't change, but the simulator should
            # account for this in force calculations

        # VRS thrust reduction
        if cfg.vrs_enabled and self._vrs.in_vrs:
            # Reduce effective thrust during VRS
            thrust_factor = 1.0 - self._vrs.get_thrust_loss() * 0.5
            return action * thrust_factor

        return action

    def get_info(self) -> dict[str, Any]:
        """Get aerodynamics perturbation information."""
        info = super().get_info()
        info.update(
            {
                "air_density": float(self._current_density),
                "drag_force": self._drag_force.tolist(),
                "drag_magnitude": float(np.linalg.norm(self._drag_force)),
                "in_vrs": self._vrs.in_vrs,
                "vrs_severity": float(self._vrs._vrs_factor),
                "flapping_force": self._flapping_force.tolist(),
                "flapping_moment": self._flapping_moment.tolist(),
            }
        )
        return info


# Convenience factory functions
def create_minimal_aero() -> AerodynamicsPerturbation:
    """Create minimal aerodynamics (only basic drag)."""
    config = AerodynamicsConfig(
        enabled=True,
        intensity=1.0,
        drag_enabled=True,
        drag_coefficient=0.3,
        blade_flapping_enabled=False,
        vrs_enabled=False,
        density_variation_enabled=False,
    )
    return AerodynamicsPerturbation(config)


def create_realistic_aero() -> AerodynamicsPerturbation:
    """Create realistic aerodynamics."""
    config = AerodynamicsConfig(
        enabled=True,
        intensity=1.0,
        drag_enabled=True,
        drag_coefficient=0.5,
        air_density=1.225,
        reference_area=0.1,
        blade_flapping_enabled=True,
        flapping_coefficient=0.01,
        vrs_enabled=True,
        vrs_descent_threshold=2.0,
        vrs_intensity=0.3,
        density_variation_enabled=True,
        density_altitude_coeff=0.00012,
        density_random_variation=0.02,
    )
    return AerodynamicsPerturbation(config)


def create_high_altitude_aero() -> AerodynamicsPerturbation:
    """Create aerodynamics for high altitude simulation."""
    config = AerodynamicsConfig(
        enabled=True,
        intensity=1.0,
        drag_enabled=True,
        drag_coefficient=0.5,
        air_density=1.0,  # Reduced for altitude
        reference_area=0.1,
        blade_flapping_enabled=True,
        flapping_coefficient=0.015,
        vrs_enabled=True,
        vrs_descent_threshold=2.5,
        vrs_intensity=0.4,
        density_variation_enabled=True,
        density_altitude_coeff=0.00015,
        density_random_variation=0.05,
    )
    return AerodynamicsPerturbation(config)


def create_aggressive_flight_aero() -> AerodynamicsPerturbation:
    """Create aerodynamics for aggressive flight maneuvers."""
    config = AerodynamicsConfig(
        enabled=True,
        intensity=1.0,
        drag_enabled=True,
        drag_coefficient=0.6,  # Higher drag at speed
        air_density=1.225,
        reference_area=0.12,
        blade_flapping_enabled=True,
        flapping_coefficient=0.02,  # More pronounced flapping
        vrs_enabled=True,
        vrs_descent_threshold=1.5,  # More sensitive
        vrs_intensity=0.4,
        density_variation_enabled=False,
    )
    return AerodynamicsPerturbation(config)
