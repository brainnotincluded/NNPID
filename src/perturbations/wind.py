"""Wind perturbations for realistic environmental simulation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from .base import BasePerturbation, WindConfig

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


class WindPerturbation(BasePerturbation):
    """Comprehensive wind simulation including steady wind, gusts, turbulence, and more.
    
    Components:
    - Steady wind: Constant wind with slowly varying direction
    - Gusts: Random bursts of increased wind
    - Turbulence: High-frequency random variations
    - Wind shear: Wind speed gradient with altitude
    - Thermals: Vertical updrafts
    """
    
    def __init__(self, config: Optional[WindConfig] = None):
        """Initialize wind perturbation.
        
        Args:
            config: Wind configuration. Uses defaults if None.
        """
        super().__init__(config or WindConfig())
        self.wind_config: WindConfig = self.config  # Type hint
        
        # Steady wind state
        self._steady_direction = 0.0
        self._steady_velocity = 0.0
        
        # Gust state
        self._gust_active = False
        self._gust_start_time = 0.0
        self._gust_duration = 0.0
        self._gust_intensity = 0.0
        self._gust_direction = np.zeros(3)
        self._gust_phase = "inactive"  # inactive, rising, sustained, falling
        self._last_gust_end_time = -10.0
        
        # Turbulence state
        self._turbulence_phase = np.zeros(3)
        self._turbulence_velocity = np.zeros(3)
        
        # Dryden turbulence model state (for more realistic turbulence)
        self._dryden_state = np.zeros((3, 2))  # State for each axis
        
        # Perlin noise state
        self._perlin_offsets = np.zeros(3)
        
        # Thermal state
        self._thermal_active = False
        self._thermal_position = np.zeros(3)
        self._thermal_strength = 0.0
        
        # Combined wind velocity
        self._wind_velocity = np.zeros(3)
    
    def reset(self, rng: np.random.Generator) -> None:
        """Reset wind state."""
        super().reset(rng)
        
        cfg = self.wind_config
        
        # Reset steady wind
        self._steady_direction = cfg.steady_wind_direction
        self._steady_velocity = cfg.steady_wind_velocity
        
        # Reset gusts
        self._gust_active = False
        self._gust_start_time = 0.0
        self._gust_duration = 0.0
        self._gust_intensity = 0.0
        self._gust_phase = "inactive"
        self._last_gust_end_time = -10.0
        
        # Reset turbulence
        self._turbulence_phase = self._rng.uniform(0, 2 * np.pi, 3)
        self._turbulence_velocity = np.zeros(3)
        self._dryden_state = np.zeros((3, 2))
        self._perlin_offsets = self._rng.uniform(0, 1000, 3)
        
        # Reset thermals
        self._thermal_active = False
        self._thermal_position = np.zeros(3)
        self._thermal_strength = 0.0
        
        self._wind_velocity = np.zeros(3)
    
    def _randomize_parameters(self) -> None:
        """Randomize wind parameters."""
        cfg = self.wind_config
        
        # Randomize steady wind
        if cfg.steady_wind_enabled:
            self._steady_direction = self._rng.uniform(0, 2 * np.pi)
            self._steady_velocity = self._rng.uniform(0, cfg.steady_wind_velocity * 1.5)
    
    def update(self, dt: float, state: "QuadrotorState") -> None:
        """Update wind state.
        
        Args:
            dt: Time step
            state: Current quadrotor state
        """
        if not self.enabled:
            self._current_force = np.zeros(3)
            return
        
        self._time += dt
        cfg = self.wind_config
        
        # Reset wind velocity
        self._wind_velocity = np.zeros(3)
        
        # 1. Steady wind component
        if cfg.steady_wind_enabled:
            self._update_steady_wind(dt)
        
        # 2. Gust component
        if cfg.gusts_enabled:
            self._update_gusts(dt)
        
        # 3. Turbulence component
        if cfg.turbulence_enabled:
            self._update_turbulence(dt, state)
        
        # 4. Wind shear (altitude dependent)
        if cfg.shear_enabled:
            self._apply_wind_shear(state)
        
        # 5. Thermals
        if cfg.thermals_enabled:
            self._update_thermals(dt, state)
        
        # Calculate force on drone
        # F = 0.5 * rho * Cd * A * v^2
        # Simplified: F proportional to relative wind velocity squared
        relative_wind = self._wind_velocity - state.velocity
        wind_speed = np.linalg.norm(relative_wind)
        
        if wind_speed > 0.01:
            # Drag-like force in wind direction
            drag_coeff = 0.5  # Simplified drag coefficient
            air_density = 1.225  # kg/m^3
            reference_area = 0.1  # m^2 (approximate drone frontal area)
            
            force_magnitude = 0.5 * air_density * drag_coeff * reference_area * wind_speed ** 2
            force_direction = relative_wind / wind_speed
            
            self._current_force = force_magnitude * force_direction * cfg.intensity
        else:
            self._current_force = np.zeros(3)
        
        # Add some torque from asymmetric wind loading
        if wind_speed > 0.1:
            # Random torque component proportional to wind
            torque_scale = 0.01 * cfg.intensity
            self._current_torque = self._rng.normal(0, torque_scale * wind_speed, 3)
        else:
            self._current_torque = np.zeros(3)
    
    def _update_steady_wind(self, dt: float) -> None:
        """Update steady wind component."""
        cfg = self.wind_config
        
        # Slowly change wind direction
        direction_change = self._rng.normal(0, cfg.direction_change_rate * dt)
        self._steady_direction += direction_change
        
        # Calculate steady wind velocity
        steady_x = self._steady_velocity * np.cos(self._steady_direction)
        steady_y = self._steady_velocity * np.sin(self._steady_direction)
        steady_z = cfg.steady_wind_vertical
        
        self._wind_velocity += np.array([steady_x, steady_y, steady_z])
    
    def _update_gusts(self, dt: float) -> None:
        """Update gust component with realistic rise/fall profile."""
        cfg = self.wind_config
        
        if not self._gust_active:
            # Check for new gust
            time_since_last = self._time - self._last_gust_end_time
            min_interval = cfg.gust_min_duration * 2  # Minimum time between gusts
            
            if time_since_last > min_interval and self._rng.random() < cfg.gust_probability:
                # Start new gust
                self._gust_active = True
                self._gust_start_time = self._time
                self._gust_duration = self._rng.uniform(
                    cfg.gust_min_duration, cfg.gust_max_duration
                )
                self._gust_intensity = self._rng.uniform(
                    cfg.gust_min_intensity, cfg.gust_max_intensity
                )
                # Random 3D direction with slight upward bias
                direction = self._rng.normal(0, 1, 3)
                direction[2] = abs(direction[2]) * 0.3  # Slight upward
                self._gust_direction = direction / np.linalg.norm(direction)
                self._gust_phase = "rising"
        
        if self._gust_active:
            elapsed = self._time - self._gust_start_time
            total_duration = cfg.gust_rise_time + self._gust_duration + cfg.gust_fall_time
            
            if elapsed < cfg.gust_rise_time:
                # Rising phase - smooth ramp up
                self._gust_phase = "rising"
                t = elapsed / cfg.gust_rise_time
                # Smooth step function
                intensity_mult = t * t * (3 - 2 * t)
            elif elapsed < cfg.gust_rise_time + self._gust_duration:
                # Sustained phase
                self._gust_phase = "sustained"
                intensity_mult = 1.0
            elif elapsed < total_duration:
                # Falling phase - smooth ramp down
                self._gust_phase = "falling"
                t = (elapsed - cfg.gust_rise_time - self._gust_duration) / cfg.gust_fall_time
                intensity_mult = 1.0 - t * t * (3 - 2 * t)
            else:
                # Gust ended
                self._gust_active = False
                self._gust_phase = "inactive"
                self._last_gust_end_time = self._time
                intensity_mult = 0.0
            
            if self._gust_active:
                gust_velocity = self._gust_intensity * intensity_mult * self._gust_direction
                self._wind_velocity += gust_velocity
    
    def _update_turbulence(self, dt: float, state: "QuadrotorState") -> None:
        """Update turbulence component."""
        cfg = self.wind_config
        
        if cfg.turbulence_type == "gaussian":
            self._update_gaussian_turbulence(dt)
        elif cfg.turbulence_type == "dryden":
            self._update_dryden_turbulence(dt, state)
        elif cfg.turbulence_type == "perlin":
            self._update_perlin_turbulence(dt, state)
        else:
            self._update_gaussian_turbulence(dt)
        
        self._wind_velocity += self._turbulence_velocity
    
    def _update_gaussian_turbulence(self, dt: float) -> None:
        """Simple Gaussian turbulence model."""
        cfg = self.wind_config
        
        # First-order filter for smooth turbulence
        tau = 1.0 / cfg.turbulence_frequency  # Time constant
        alpha = dt / (tau + dt)
        
        # Target turbulence (random)
        target = self._rng.normal(0, cfg.turbulence_intensity * cfg.steady_wind_velocity, 3)
        
        # Smooth filter
        self._turbulence_velocity = (1 - alpha) * self._turbulence_velocity + alpha * target
    
    def _update_dryden_turbulence(self, dt: float, state: "QuadrotorState") -> None:
        """Dryden turbulence model for more realistic wind.
        
        Based on MIL-F-8785C specification.
        """
        cfg = self.wind_config
        
        # Altitude affects turbulence scale
        altitude = max(0.1, state.position[2])
        
        # Scale lengths (simplified)
        L_u = L_v = 200 * (altitude / 20) ** 0.5  # meters
        L_w = 50  # meters
        
        # Turbulence intensities
        sigma_w = 0.1 * cfg.turbulence_intensity * cfg.steady_wind_velocity
        sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 * altitude) ** 0.4
        
        # Airspeed (use magnitude of velocity)
        V = max(0.1, np.linalg.norm(state.velocity))
        
        # White noise input
        noise = self._rng.normal(0, 1, 3)
        
        # First-order Markov process for each axis
        for i, (L, sigma) in enumerate([(L_u, sigma_u), (L_v, sigma_v), (L_w, sigma_w)]):
            tau = L / V
            alpha = dt / tau
            self._turbulence_velocity[i] = (
                (1 - alpha) * self._turbulence_velocity[i] +
                sigma * np.sqrt(2 * alpha) * noise[i]
            )
    
    def _update_perlin_turbulence(self, dt: float, state: "QuadrotorState") -> None:
        """Perlin-like noise turbulence (simplified)."""
        cfg = self.wind_config
        
        # Use position and time to create coherent noise
        pos = state.position
        scale = 0.1  # Spatial scale
        
        for i in range(3):
            # Simple coherent noise approximation using sin waves
            self._perlin_offsets[i] += dt * cfg.turbulence_frequency
            
            noise = (
                np.sin(pos[0] * scale + self._perlin_offsets[i]) *
                np.cos(pos[1] * scale + self._perlin_offsets[(i+1)%3]) *
                np.sin(pos[2] * scale * 0.5 + self._perlin_offsets[(i+2)%3] * 0.7)
            )
            
            self._turbulence_velocity[i] = (
                noise * cfg.turbulence_intensity * cfg.steady_wind_velocity
            )
    
    def _apply_wind_shear(self, state: "QuadrotorState") -> None:
        """Apply wind shear - wind speed increases with altitude."""
        cfg = self.wind_config
        
        altitude = max(0, state.position[2])
        
        # Power law wind profile
        if altitude < cfg.shear_reference_height:
            shear_factor = (altitude / cfg.shear_reference_height) ** 0.2
        else:
            shear_factor = 1.0 + cfg.shear_gradient * (altitude - cfg.shear_reference_height)
        
        # Apply shear to horizontal components only
        base_shear = cfg.shear_ground_velocity + cfg.shear_gradient * altitude
        shear_velocity = np.array([
            base_shear * np.cos(self._steady_direction),
            base_shear * np.sin(self._steady_direction),
            0.0
        ])
        
        self._wind_velocity += shear_velocity * shear_factor
    
    def _update_thermals(self, dt: float, state: "QuadrotorState") -> None:
        """Update thermal (vertical updraft) effects."""
        cfg = self.wind_config
        
        altitude = state.position[2]
        
        # Check altitude range for thermals
        if altitude < cfg.thermal_height_min or altitude > cfg.thermal_height_max:
            if self._thermal_active:
                self._thermal_active = False
            return
        
        if not self._thermal_active:
            # Check for entering a thermal
            if self._rng.random() < cfg.thermal_probability * dt:
                self._thermal_active = True
                self._thermal_position = state.position.copy()
                self._thermal_strength = self._rng.uniform(
                    cfg.thermal_strength * 0.5, cfg.thermal_strength * 1.5
                )
        
        if self._thermal_active:
            # Calculate distance from thermal center
            horizontal_dist = np.sqrt(
                (state.position[0] - self._thermal_position[0]) ** 2 +
                (state.position[1] - self._thermal_position[1]) ** 2
            )
            
            if horizontal_dist > cfg.thermal_radius * 2:
                # Left the thermal
                self._thermal_active = False
            else:
                # Gaussian profile for thermal
                thermal_velocity = self._thermal_strength * np.exp(
                    -(horizontal_dist / cfg.thermal_radius) ** 2
                )
                self._wind_velocity[2] += thermal_velocity
    
    def get_wind_velocity(self) -> np.ndarray:
        """Get current wind velocity vector.
        
        Returns:
            Wind velocity [vx, vy, vz] in m/s
        """
        return self._wind_velocity.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """Get wind perturbation information."""
        info = super().get_info()
        info.update({
            "wind_velocity": self._wind_velocity.tolist(),
            "wind_speed": float(np.linalg.norm(self._wind_velocity)),
            "steady_direction": float(self._steady_direction),
            "steady_velocity": float(self._steady_velocity),
            "gust_active": self._gust_active,
            "gust_phase": self._gust_phase,
            "gust_intensity": float(self._gust_intensity) if self._gust_active else 0.0,
            "thermal_active": self._thermal_active,
        })
        return info


# Convenience factory functions
def create_light_breeze() -> WindPerturbation:
    """Create a light breeze wind perturbation."""
    config = WindConfig(
        enabled=True,
        intensity=0.5,
        steady_wind_velocity=1.5,
        gusts_enabled=False,
        turbulence_intensity=0.1,
    )
    return WindPerturbation(config)


def create_moderate_wind() -> WindPerturbation:
    """Create moderate wind conditions."""
    config = WindConfig(
        enabled=True,
        intensity=1.0,
        steady_wind_velocity=4.0,
        gusts_enabled=True,
        gust_probability=0.02,
        gust_max_intensity=3.0,
        turbulence_intensity=0.3,
    )
    return WindPerturbation(config)


def create_strong_wind() -> WindPerturbation:
    """Create strong wind conditions."""
    config = WindConfig(
        enabled=True,
        intensity=1.0,
        steady_wind_velocity=8.0,
        gusts_enabled=True,
        gust_probability=0.05,
        gust_max_intensity=6.0,
        turbulence_intensity=0.5,
        turbulence_type="dryden",
    )
    return WindPerturbation(config)


def create_gusty_conditions() -> WindPerturbation:
    """Create gusty wind conditions with frequent gusts."""
    config = WindConfig(
        enabled=True,
        intensity=1.0,
        steady_wind_velocity=3.0,
        gusts_enabled=True,
        gust_probability=0.08,
        gust_min_duration=0.3,
        gust_max_duration=1.5,
        gust_min_intensity=2.0,
        gust_max_intensity=8.0,
        turbulence_intensity=0.4,
    )
    return WindPerturbation(config)
