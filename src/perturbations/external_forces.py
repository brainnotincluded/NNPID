"""External force perturbations for simulating impacts, vibrations, and disturbances."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BasePerturbation, ExternalForcesConfig

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


class ImpulseType(Enum):
    """Types of impulse disturbances."""

    RANDOM = "random"
    COLLISION = "collision"
    BIRD_STRIKE = "bird_strike"
    PROP_WASH = "prop_wash"


@dataclass
class ImpulseEvent:
    """Represents a single impulse event."""

    start_time: float
    duration: float
    force: np.ndarray
    torque: np.ndarray
    impulse_type: ImpulseType

    def is_active(self, current_time: float) -> bool:
        """Check if impulse is currently active."""
        return self.start_time <= current_time < (self.start_time + self.duration)

    def get_force_at_time(self, current_time: float) -> np.ndarray:
        """Get force at current time with smooth profile."""
        if not self.is_active(current_time):
            return np.zeros(3)

        # Smooth pulse shape (half-sine)
        t_normalized = (current_time - self.start_time) / self.duration
        envelope = np.sin(np.pi * t_normalized)

        return self.force * envelope

    def get_torque_at_time(self, current_time: float) -> np.ndarray:
        """Get torque at current time with smooth profile."""
        if not self.is_active(current_time):
            return np.zeros(3)

        t_normalized = (current_time - self.start_time) / self.duration
        envelope = np.sin(np.pi * t_normalized)

        return self.torque * envelope


class PeriodicDisturbance:
    """Periodic sinusoidal disturbance force.

    Useful for simulating regular disturbances like
    nearby machinery, structural vibrations, etc.
    """

    def __init__(
        self,
        frequency: float = 1.0,
        amplitude: float = 0.5,
        direction: np.ndarray = None,
        phase: float = 0.0,
    ):
        """Initialize periodic disturbance.

        Args:
            frequency: Oscillation frequency (Hz)
            amplitude: Force amplitude (N)
            direction: Force direction (unit vector)
            phase: Initial phase (radians)
        """
        self.frequency = frequency
        self.amplitude = amplitude
        self.direction = direction if direction is not None else np.array([1.0, 0.0, 0.0])
        self.direction = self.direction / (np.linalg.norm(self.direction) + 1e-6)
        self.phase = phase

    def get_force(self, time: float) -> np.ndarray:
        """Get force at current time.

        Args:
            time: Current time in seconds

        Returns:
            Force vector
        """
        magnitude = self.amplitude * np.sin(2 * np.pi * self.frequency * time + self.phase)
        return magnitude * self.direction


class VibrationModel:
    """Models motor-induced vibrations.

    Vibrations are coupled to motor speeds and consist of
    multiple frequency components.
    """

    def __init__(
        self,
        frequencies: list[float] = None,
        amplitudes: list[float] = None,
        motor_coupled: bool = True,
    ):
        """Initialize vibration model.

        Args:
            frequencies: Vibration frequencies (Hz)
            amplitudes: Vibration amplitudes (N)
            motor_coupled: Whether vibrations scale with motor speed
        """
        self.frequencies = frequencies if frequencies is not None else [50.0, 100.0]
        self.amplitudes = amplitudes if amplitudes is not None else [0.1, 0.05]
        self.motor_coupled = motor_coupled

        # Random phases for each frequency
        self._phases: np.ndarray | None = None
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset vibration state."""
        self._rng = rng
        self._phases = rng.uniform(0, 2 * np.pi, len(self.frequencies))

    def get_force(
        self,
        time: float,
        motor_speeds: np.ndarray,
    ) -> np.ndarray:
        """Get vibration force.

        Args:
            time: Current time
            motor_speeds: Motor speed commands (0-1)

        Returns:
            Vibration force vector
        """
        if self._phases is None:
            return np.zeros(3)

        # Motor coupling factor
        coupling = np.mean(motor_speeds) if self.motor_coupled else 1.0

        # Sum of frequency components
        force = np.zeros(3)
        for i, (freq, amp) in enumerate(zip(self.frequencies, self.amplitudes, strict=False)):
            phase = self._phases[i] if i < len(self._phases) else 0

            # Random direction for each component
            direction = self._rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction) + 1e-6

            magnitude = amp * np.sin(2 * np.pi * freq * time + phase) * coupling
            force += magnitude * direction

        return force

    def get_torque(
        self,
        time: float,
        motor_speeds: np.ndarray,
    ) -> np.ndarray:
        """Get vibration-induced torque.

        Args:
            time: Current time
            motor_speeds: Motor speed commands

        Returns:
            Vibration torque vector
        """
        if self._phases is None:
            return np.zeros(3)

        # Torque is smaller than force
        force = self.get_force(time, motor_speeds)
        return force * 0.1  # Scale down for torque


class EMIModel:
    """Electromagnetic Interference model.

    Simulates the effect of EMI on motor control signals,
    causing momentary control glitches.
    """

    def __init__(
        self,
        burst_probability: float = 0.001,
        max_duration: float = 0.1,
        max_magnitude: float = 0.2,
    ):
        """Initialize EMI model.

        Args:
            burst_probability: Probability of EMI burst per timestep
            max_duration: Maximum burst duration (s)
            max_magnitude: Maximum control signal deviation
        """
        self.burst_probability = burst_probability
        self.max_duration = max_duration
        self.max_magnitude = max_magnitude

        self._burst_active = False
        self._burst_start = 0.0
        self._burst_duration = 0.0
        self._burst_effect = np.zeros(4)
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator) -> None:
        """Reset EMI state."""
        self._rng = rng
        self._burst_active = False
        self._burst_start = 0.0
        self._burst_duration = 0.0
        self._burst_effect = np.zeros(4)

    def update(self, time: float, dt: float) -> None:
        """Update EMI state.

        Args:
            time: Current time
            dt: Time step
        """
        if self._rng is None:
            return

        if self._burst_active:
            if time > self._burst_start + self._burst_duration:
                self._burst_active = False
        elif self._rng.random() < self.burst_probability:
            # Start new burst
            self._burst_active = True
            self._burst_start = time
            self._burst_duration = self._rng.uniform(0.01, self.max_duration)

            # Random effect on each motor
            self._burst_effect = self._rng.uniform(-self.max_magnitude, self.max_magnitude, 4)

    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply EMI effect to motor commands.

        Args:
            action: Original motor commands

        Returns:
            Affected motor commands
        """
        if not self._burst_active:
            return action

        result = action + self._burst_effect
        return np.clip(result, 0.0, 1.0)

    @property
    def is_active(self) -> bool:
        """Check if EMI burst is active."""
        return self._burst_active


class ExternalForcesPerturbation(BasePerturbation):
    """Comprehensive external forces simulation.

    Components:
    - Random impulses: Sudden force/torque events
    - Periodic disturbances: Regular sinusoidal forces
    - Vibrations: Motor-coupled high-frequency disturbances
    - EMI: Electromagnetic interference effects on controls
    """

    def __init__(self, config: ExternalForcesConfig | None = None):
        """Initialize external forces perturbation.

        Args:
            config: External forces configuration. Uses defaults if None.
        """
        super().__init__(config or ExternalForcesConfig())
        self.forces_config: ExternalForcesConfig = self.config

        # Active impulse events
        self._impulse_events: list[ImpulseEvent] = []
        self._last_impulse_time = -10.0

        # Periodic disturbance
        self._periodic = PeriodicDisturbance(
            frequency=self.forces_config.periodic_frequency,
            amplitude=self.forces_config.periodic_amplitude,
            direction=np.array(self.forces_config.periodic_direction),
        )

        # Vibration model
        self._vibration = VibrationModel(
            frequencies=list(self.forces_config.vibration_frequencies),
            amplitudes=list(self.forces_config.vibration_amplitudes),
            motor_coupled=self.forces_config.vibration_motor_coupled,
        )

        # EMI model
        self._emi = EMIModel(
            burst_probability=0.001,  # Default low probability
            max_duration=0.1,
            max_magnitude=0.1,
        )

        # Current state
        self._motor_speeds = np.zeros(4)

    def reset(self, rng: np.random.Generator) -> None:
        """Reset external forces state."""
        super().reset(rng)

        self._impulse_events.clear()
        self._last_impulse_time = -10.0

        self._vibration.reset(rng)
        self._emi.reset(rng)

        # Randomize periodic disturbance direction
        cfg = self.forces_config
        if cfg.periodic_enabled:
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
            self._periodic = PeriodicDisturbance(
                frequency=cfg.periodic_frequency,
                amplitude=cfg.periodic_amplitude,
                direction=direction,
                phase=rng.uniform(0, 2 * np.pi),
            )

        self._motor_speeds = np.zeros(4)

    def update(self, dt: float, state: QuadrotorState) -> None:
        """Update external forces.

        Args:
            dt: Time step
            state: Current quadrotor state
        """
        if not self.enabled:
            self._current_force = np.zeros(3)
            self._current_torque = np.zeros(3)
            return

        self._time += dt
        cfg = self.forces_config

        # Update motor speeds from state
        self._motor_speeds = state.motor_speeds.copy()

        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # 1. Handle impulses
        if cfg.impulses_enabled:
            self._update_impulses(dt)
            for event in self._impulse_events:
                total_force += event.get_force_at_time(self._time)
                total_torque += event.get_torque_at_time(self._time)

        # 2. Periodic disturbance
        if cfg.periodic_enabled:
            periodic_force = self._periodic.get_force(self._time)
            total_force += periodic_force

        # 3. Vibrations
        if cfg.vibrations_enabled:
            vib_force = self._vibration.get_force(self._time, self._motor_speeds)
            vib_torque = self._vibration.get_torque(self._time, self._motor_speeds)
            total_force += vib_force
            total_torque += vib_torque

        # 4. Update EMI model
        self._emi.update(self._time, dt)

        self._current_force = total_force * cfg.intensity
        self._current_torque = total_torque * cfg.intensity

    def _update_impulses(self, dt: float) -> None:
        """Update impulse events."""
        cfg = self.forces_config

        # Remove expired events
        self._impulse_events = [
            e for e in self._impulse_events if e.is_active(self._time) or e.start_time > self._time
        ]

        # Check for new impulse
        time_since_last = self._time - self._last_impulse_time
        should_impulse = (
            time_since_last > cfg.impulse_min_interval
            and self._rng.random() < cfg.impulse_probability
        )
        if should_impulse:
            self._generate_impulse()

    def _generate_impulse(self) -> None:
        """Generate a new impulse event."""
        cfg = self.forces_config

        # Random force
        force_magnitude = self._rng.uniform(cfg.impulse_force_min, cfg.impulse_force_max)
        force_direction = self._rng.normal(0, 1, 3)
        force_direction /= np.linalg.norm(force_direction)
        force = force_magnitude * force_direction

        # Random torque
        torque_magnitude = self._rng.uniform(cfg.impulse_torque_min, cfg.impulse_torque_max)
        torque_direction = self._rng.normal(0, 1, 3)
        torque_direction /= np.linalg.norm(torque_direction)
        torque = torque_magnitude * torque_direction

        # Create event
        event = ImpulseEvent(
            start_time=self._time,
            duration=cfg.impulse_duration,
            force=force,
            torque=torque,
            impulse_type=ImpulseType.RANDOM,
        )

        self._impulse_events.append(event)
        self._last_impulse_time = self._time

    def apply_impulse(
        self,
        force: np.ndarray,
        torque: np.ndarray = None,
        duration: float = 0.05,
        impulse_type: ImpulseType = ImpulseType.RANDOM,
    ) -> None:
        """Manually apply an impulse.

        Args:
            force: Force vector (N)
            torque: Torque vector (Nm), default zeros
            duration: Impulse duration (s)
            impulse_type: Type of impulse
        """
        if torque is None:
            torque = np.zeros(3)

        event = ImpulseEvent(
            start_time=self._time,
            duration=duration,
            force=force.copy(),
            torque=torque.copy(),
            impulse_type=impulse_type,
        )

        self._impulse_events.append(event)

    def apply_collision(
        self,
        impact_position: np.ndarray,
        impact_velocity: np.ndarray,
        collision_mass: float = 0.1,
    ) -> None:
        """Apply a collision impulse.

        Args:
            impact_position: Position of impact relative to CoM
            impact_velocity: Velocity of impacting object
            collision_mass: Mass of impacting object (kg)
        """
        # Impulse = change in momentum
        impulse_magnitude = collision_mass * np.linalg.norm(impact_velocity)

        # Force over short duration
        duration = 0.02  # 20ms collision
        force = -impact_velocity / np.linalg.norm(impact_velocity) * impulse_magnitude / duration

        # Torque from off-center impact
        torque = np.cross(impact_position, force)

        event = ImpulseEvent(
            start_time=self._time,
            duration=duration,
            force=force,
            torque=torque,
            impulse_type=ImpulseType.COLLISION,
        )

        self._impulse_events.append(event)

    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply EMI effects to motor commands.

        Args:
            action: Original motor commands

        Returns:
            Possibly affected motor commands
        """
        if not self.enabled:
            return action

        return self._emi.apply_to_action(action)

    def get_info(self) -> dict[str, Any]:
        """Get external forces perturbation information."""
        info = super().get_info()
        info.update(
            {
                "active_impulses": len(
                    [e for e in self._impulse_events if e.is_active(self._time)]
                ),
                "total_impulses_generated": len(self._impulse_events),
                "emi_active": self._emi.is_active,
                "vibration_force": self._vibration.get_force(
                    self._time, self._motor_speeds
                ).tolist(),
            }
        )
        return info


# Convenience factory functions
def create_calm_environment() -> ExternalForcesPerturbation:
    """Create calm environment with minimal disturbances."""
    config = ExternalForcesConfig(
        enabled=True,
        intensity=0.3,
        impulses_enabled=False,
        periodic_enabled=False,
        vibrations_enabled=True,
        vibration_frequencies=[50.0],
        vibration_amplitudes=[0.02],
        vibration_motor_coupled=True,
    )
    return ExternalForcesPerturbation(config)


def create_urban_environment() -> ExternalForcesPerturbation:
    """Create urban environment with various disturbances."""
    config = ExternalForcesConfig(
        enabled=True,
        intensity=1.0,
        impulses_enabled=True,
        impulse_probability=0.005,
        impulse_min_interval=5.0,
        impulse_force_min=0.5,
        impulse_force_max=3.0,
        impulse_torque_min=0.01,
        impulse_torque_max=0.05,
        impulse_duration=0.1,
        periodic_enabled=True,
        periodic_frequency=2.0,
        periodic_amplitude=0.3,
        vibrations_enabled=True,
        vibration_frequencies=[50.0, 100.0, 150.0],
        vibration_amplitudes=[0.05, 0.03, 0.02],
    )
    return ExternalForcesPerturbation(config)


def create_turbulent_environment() -> ExternalForcesPerturbation:
    """Create turbulent environment with frequent disturbances."""
    config = ExternalForcesConfig(
        enabled=True,
        intensity=1.0,
        impulses_enabled=True,
        impulse_probability=0.02,
        impulse_min_interval=1.0,
        impulse_force_min=1.0,
        impulse_force_max=5.0,
        impulse_torque_min=0.05,
        impulse_torque_max=0.2,
        impulse_duration=0.05,
        periodic_enabled=True,
        periodic_frequency=0.5,
        periodic_amplitude=1.0,
        vibrations_enabled=True,
        vibration_frequencies=[30.0, 60.0, 120.0],
        vibration_amplitudes=[0.1, 0.08, 0.05],
    )
    return ExternalForcesPerturbation(config)


def create_industrial_environment() -> ExternalForcesPerturbation:
    """Create industrial environment with EMI and vibrations."""
    config = ExternalForcesConfig(
        enabled=True,
        intensity=1.0,
        impulses_enabled=False,
        periodic_enabled=True,
        periodic_frequency=50.0,  # Power line frequency
        periodic_amplitude=0.2,
        vibrations_enabled=True,
        vibration_frequencies=[25.0, 50.0, 100.0, 200.0],
        vibration_amplitudes=[0.08, 0.15, 0.1, 0.05],
        vibration_motor_coupled=False,
    )

    perturbation = ExternalForcesPerturbation(config)
    # Enhanced EMI for industrial setting
    perturbation._emi = EMIModel(
        burst_probability=0.005,
        max_duration=0.2,
        max_magnitude=0.15,
    )

    return perturbation
