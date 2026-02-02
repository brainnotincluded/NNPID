"""Target motion patterns for yaw tracking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class TargetPatternType(Enum):
    """Target motion pattern types."""

    CIRCULAR = "circular"
    RANDOM = "random"
    SINUSOIDAL = "sinusoidal"
    STEP = "step"
    # Advanced patterns
    FIGURE8 = "figure8"
    SPIRAL = "spiral"
    EVASIVE = "evasive"
    LISSAJOUS = "lissajous"
    MULTI_FREQUENCY = "multi_frequency"


class TargetPattern(ABC):
    """Abstract base class for target motion patterns."""

    @abstractmethod
    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        """Get target position at given time."""

    @abstractmethod
    def get_angular_velocity(self) -> float:
        """Get current angular velocity of target motion."""

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Reset target pattern with new random parameters."""


class CircularTarget(TargetPattern):
    """Target that orbits around the drone at constant angular velocity."""

    def __init__(
        self,
        radius: float = 3.0,
        angular_velocity: float = 1.0,
        height: float = 1.0,
    ):
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.height = height
        self.phase = 0.0

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        angle = self.phase + self.angular_velocity * time
        x = drone_position[0] + self.radius * np.cos(angle)
        y = drone_position[1] + self.radius * np.sin(angle)
        z = self.height
        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        return self.angular_velocity

    def reset(self, rng: np.random.Generator) -> None:
        self.phase = rng.uniform(0, 2 * np.pi)


class RandomTarget(TargetPattern):
    """Target that moves to random positions around the drone."""

    def __init__(
        self,
        radius: float = 3.0,
        change_interval: float = 2.0,
        height: float = 1.0,
    ):
        self.radius = radius
        self.change_interval = change_interval
        self.height = height
        self.current_angle = 0.0
        self.target_angle = 0.0
        self.last_change_time = 0.0
        self._rng: np.random.Generator | None = None

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        # Smoothly interpolate between angles
        t = (time - self.last_change_time) / self.change_interval
        t = np.clip(t, 0, 1)

        # Smooth step interpolation
        t = t * t * (3 - 2 * t)

        angle = self.current_angle + t * self._angle_diff(self.target_angle, self.current_angle)

        # Check if it's time to pick a new target
        if time - self.last_change_time >= self.change_interval and self._rng is not None:
            self.current_angle = self.target_angle
            self.target_angle = self._rng.uniform(0, 2 * np.pi)
            self.last_change_time = time

        x = drone_position[0] + self.radius * np.cos(angle)
        y = drone_position[1] + self.radius * np.sin(angle)
        z = self.height
        return np.array([x, y, z])

    def _angle_diff(self, a: float, b: float) -> float:
        """Compute shortest angle difference."""
        diff = a - b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def get_angular_velocity(self) -> float:
        # Approximate angular velocity based on movement
        return abs(self._angle_diff(self.target_angle, self.current_angle)) / self.change_interval

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.current_angle = rng.uniform(0, 2 * np.pi)
        self.target_angle = rng.uniform(0, 2 * np.pi)
        self.last_change_time = 0.0


class SinusoidalTarget(TargetPattern):
    """Target that oscillates back and forth."""

    def __init__(
        self,
        radius: float = 3.0,
        frequency: float = 0.5,
        amplitude: float = np.pi / 2,
        height: float = 1.0,
    ):
        self.radius = radius
        self.frequency = frequency
        self.amplitude = amplitude
        self.height = height
        self.center_angle = 0.0

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        angle = self.center_angle + self.amplitude * np.sin(2 * np.pi * self.frequency * time)
        x = drone_position[0] + self.radius * np.cos(angle)
        y = drone_position[1] + self.radius * np.sin(angle)
        z = self.height
        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        # Maximum angular velocity of sinusoidal motion
        return self.amplitude * 2 * np.pi * self.frequency

    def reset(self, rng: np.random.Generator) -> None:
        self.center_angle = rng.uniform(0, 2 * np.pi)


class StepTarget(TargetPattern):
    """Target that makes discrete jumps to new positions."""

    def __init__(
        self,
        radius: float = 3.0,
        step_interval: float = 3.0,
        step_size: float = np.pi / 2,
        height: float = 1.0,
    ):
        self.radius = radius
        self.step_interval = step_interval
        self.step_size = step_size
        self.height = height
        self.current_angle = 0.0
        self.last_step_time = 0.0
        self._rng: np.random.Generator | None = None

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        # Check if it's time for a step
        if time - self.last_step_time >= self.step_interval and self._rng is not None:
            direction = self._rng.choice([-1, 1])
            self.current_angle += direction * self.step_size
            self.last_step_time = time

        x = drone_position[0] + self.radius * np.cos(self.current_angle)
        y = drone_position[1] + self.radius * np.sin(self.current_angle)
        z = self.height
        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        return self.step_size / self.step_interval

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.current_angle = rng.uniform(0, 2 * np.pi)
        self.last_step_time = 0.0


class Figure8Target(TargetPattern):
    """Target that follows a figure-8 (lemniscate) pattern."""

    def __init__(
        self,
        radius: float = 3.0,
        angular_velocity: float = 0.5,
        height: float = 1.0,
    ):
        self.radius = radius
        self.angular_velocity = angular_velocity
        self.height = height
        self.phase = 0.0

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        t = self.phase + self.angular_velocity * time

        # Lemniscate of Bernoulli parametric equations
        scale = self.radius / (1 + np.sin(t) ** 2 + 0.1)

        x = drone_position[0] + scale * np.cos(t)
        y = drone_position[1] + scale * np.sin(t) * np.cos(t)
        z = self.height

        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        return self.angular_velocity * 1.5

    def reset(self, rng: np.random.Generator) -> None:
        self.phase = rng.uniform(0, 2 * np.pi)


class SpiralTarget(TargetPattern):
    """Target that spirals inward and outward."""

    def __init__(
        self,
        radius_min: float = 2.0,
        radius_max: float = 5.0,
        angular_velocity: float = 0.5,
        spiral_frequency: float = 0.1,
        height: float = 1.0,
    ):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.angular_velocity = angular_velocity
        self.spiral_frequency = spiral_frequency
        self.height = height
        self.phase = 0.0
        self.spiral_phase = 0.0

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        angle = self.phase + self.angular_velocity * time

        # Radius oscillates between min and max
        radius_range = (self.radius_max - self.radius_min) / 2
        radius_mid = (self.radius_max + self.radius_min) / 2
        current_radius = radius_mid + radius_range * np.sin(
            self.spiral_phase + self.spiral_frequency * time
        )

        x = drone_position[0] + current_radius * np.cos(angle)
        y = drone_position[1] + current_radius * np.sin(angle)
        z = self.height

        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        return self.angular_velocity

    def reset(self, rng: np.random.Generator) -> None:
        self.phase = rng.uniform(0, 2 * np.pi)
        self.spiral_phase = rng.uniform(0, 2 * np.pi)


class EvasiveTarget(TargetPattern):
    """Target that performs aggressive, unpredictable evasive maneuvers."""

    def __init__(
        self,
        radius: float = 3.0,
        base_angular_velocity: float = 0.5,
        jerk_probability: float = 0.02,
        max_jerk_magnitude: float = 2.0,
        height: float = 1.0,
    ):
        self.radius = radius
        self.base_angular_velocity = base_angular_velocity
        self.jerk_probability = jerk_probability
        self.max_jerk_magnitude = max_jerk_magnitude
        self.height = height

        self.current_angle = 0.0
        self.current_velocity = 0.0
        self.target_velocity = 0.0
        self.last_update_time = 0.0
        self._rng: np.random.Generator | None = None

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        dt = time - self.last_update_time
        if dt > 0 and self._rng is not None:
            # Possibly trigger a jerk (sudden direction change)
            if self._rng.random() < self.jerk_probability:
                self.target_velocity = self._rng.uniform(
                    -self.max_jerk_magnitude * self.base_angular_velocity,
                    self.max_jerk_magnitude * self.base_angular_velocity,
                )

            # Smooth velocity transition
            alpha = min(1.0, dt * 2.0)
            self.current_velocity += alpha * (self.target_velocity - self.current_velocity)

            # Update angle
            self.current_angle += self.current_velocity * dt
            self.last_update_time = time

        x = drone_position[0] + self.radius * np.cos(self.current_angle)
        y = drone_position[1] + self.radius * np.sin(self.current_angle)
        z = self.height

        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        return (
            abs(self.current_velocity) if self.current_velocity != 0 else self.base_angular_velocity
        )

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.current_angle = rng.uniform(0, 2 * np.pi)
        self.current_velocity = rng.uniform(-1, 1) * self.base_angular_velocity
        self.target_velocity = self.current_velocity
        self.last_update_time = 0.0


class LissajousTarget(TargetPattern):
    """Target following Lissajous curves."""

    def __init__(
        self,
        radius: float = 3.0,
        freq_x: float = 1.0,
        freq_y: float = 1.5,
        phase_offset: float = np.pi / 4,
        angular_velocity: float = 0.5,
        height: float = 1.0,
    ):
        self.radius = radius
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.phase_offset = phase_offset
        self.angular_velocity = angular_velocity
        self.height = height
        self.phase = 0.0

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        t = self.phase + self.angular_velocity * time

        # Lissajous parametric equations
        x = drone_position[0] + self.radius * np.sin(self.freq_x * t)
        y = drone_position[1] + self.radius * np.sin(self.freq_y * t + self.phase_offset)
        z = self.height

        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        # Average angular velocity (varies along the curve)
        return self.angular_velocity * max(self.freq_x, self.freq_y)

    def reset(self, rng: np.random.Generator) -> None:
        self.phase = rng.uniform(0, 2 * np.pi)
        ratios = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 5), (3, 5)]
        ratio = ratios[rng.integers(0, len(ratios))]
        self.freq_x = ratio[0]
        self.freq_y = ratio[1]
        self.phase_offset = rng.uniform(0, np.pi / 2)


class MultiFrequencyTarget(TargetPattern):
    """Target with motion composed of multiple frequency components."""

    def __init__(
        self,
        radius: float = 3.0,
        base_frequency: float = 0.2,
        num_harmonics: int = 3,
        height: float = 1.0,
    ):
        self.radius = radius
        self.base_frequency = base_frequency
        self.num_harmonics = num_harmonics
        self.height = height

        # Harmonic amplitudes and phases (randomized on reset)
        self.amplitudes: list[float] = []
        self.phases: list[float] = []
        self.frequencies: list[float] = []

    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        if not self.amplitudes:
            # Not yet reset, use simple circular
            angle = self.base_frequency * time
        else:
            # Sum of harmonics
            angle = 0.0
            for amp, freq, phase in zip(
                self.amplitudes, self.frequencies, self.phases, strict=True
            ):
                angle += amp * np.sin(2 * np.pi * freq * time + phase)

        x = drone_position[0] + self.radius * np.cos(angle)
        y = drone_position[1] + self.radius * np.sin(angle)
        z = self.height

        return np.array([x, y, z])

    def get_angular_velocity(self) -> float:
        if not self.amplitudes:
            return self.base_frequency
        # Approximate max angular velocity
        max_vel = sum(
            amp * 2 * np.pi * freq
            for amp, freq in zip(self.amplitudes, self.frequencies, strict=True)
        )
        return max_vel

    def reset(self, rng: np.random.Generator) -> None:
        self.amplitudes = []
        self.phases = []
        self.frequencies = []

        # Generate harmonics with decreasing amplitudes
        total_amplitude = 0.0
        for i in range(self.num_harmonics):
            # Amplitude decreases for higher harmonics
            amp = 1.0 / (i + 1) ** 0.5
            freq = self.base_frequency * (i + 1) * rng.uniform(0.8, 1.2)
            phase = rng.uniform(0, 2 * np.pi)

            self.amplitudes.append(amp)
            self.frequencies.append(freq)
            self.phases.append(phase)
            total_amplitude += amp

        # Normalize so total amplitude equals 2*pi for full rotation
        scale = 2 * np.pi / total_amplitude
        self.amplitudes = [a * scale for a in self.amplitudes]
