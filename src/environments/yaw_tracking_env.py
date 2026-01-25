"""Yaw tracking environment - NN learns to face a moving target."""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

import gymnasium as gym
from gymnasium import spaces

from ..core.mujoco_sim import MuJoCoSimulator, QuadrotorState, create_simulator
from ..utils.rotations import Rotations


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
        """Get target position at given time.
        
        Args:
            time: Simulation time in seconds
            drone_position: Current drone position
            
        Returns:
            Target position [x, y, z]
        """
        pass
    
    @abstractmethod
    def get_angular_velocity(self) -> float:
        """Get current angular velocity of target motion.
        
        Returns:
            Angular velocity in rad/s
        """
        pass
    
    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Reset target pattern with new random parameters."""
        pass


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
        self._rng: Optional[np.random.Generator] = None
    
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
        self._rng: Optional[np.random.Generator] = None
    
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


# =============================================================================
# Advanced Target Patterns
# =============================================================================


class Figure8Target(TargetPattern):
    """Target that follows a figure-8 (lemniscate) pattern.
    
    Creates smooth, continuous motion with direction reversals,
    challenging the controller to handle acceleration changes.
    """
    
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
        # Modified for angular tracking (projects to a figure-8 when viewed from above)
        scale = self.radius / (1 + np.sin(t) ** 2 + 0.1)  # Avoid singularity
        
        x = drone_position[0] + scale * np.cos(t)
        y = drone_position[1] + scale * np.sin(t) * np.cos(t)
        z = self.height
        
        return np.array([x, y, z])
    
    def get_angular_velocity(self) -> float:
        return self.angular_velocity * 1.5  # Figure-8 has varying angular velocity
    
    def reset(self, rng: np.random.Generator) -> None:
        self.phase = rng.uniform(0, 2 * np.pi)


class SpiralTarget(TargetPattern):
    """Target that spirals inward and outward.
    
    The radius changes over time, creating a dynamic tracking challenge
    where the target moves closer and farther from the drone.
    """
    
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
    """Target that performs aggressive, unpredictable evasive maneuvers.
    
    Simulates a target actively trying to avoid being tracked,
    with sudden direction changes and variable speeds.
    """
    
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
        self._rng: Optional[np.random.Generator] = None
    
    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        dt = time - self.last_update_time
        if dt > 0 and self._rng is not None:
            # Possibly trigger a jerk (sudden direction change)
            if self._rng.random() < self.jerk_probability:
                self.target_velocity = self._rng.uniform(
                    -self.max_jerk_magnitude * self.base_angular_velocity,
                    self.max_jerk_magnitude * self.base_angular_velocity
                )
            
            # Smooth velocity transition
            alpha = min(1.0, dt * 2.0)  # Smoothing factor
            self.current_velocity += alpha * (self.target_velocity - self.current_velocity)
            
            # Update angle
            self.current_angle += self.current_velocity * dt
            self.last_update_time = time
        
        x = drone_position[0] + self.radius * np.cos(self.current_angle)
        y = drone_position[1] + self.radius * np.sin(self.current_angle)
        z = self.height
        
        return np.array([x, y, z])
    
    def get_angular_velocity(self) -> float:
        return abs(self.current_velocity) if self.current_velocity != 0 else self.base_angular_velocity
    
    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.current_angle = rng.uniform(0, 2 * np.pi)
        self.current_velocity = rng.uniform(-1, 1) * self.base_angular_velocity
        self.target_velocity = self.current_velocity
        self.last_update_time = 0.0


class LissajousTarget(TargetPattern):
    """Target following Lissajous curves.
    
    Creates complex, non-repeating patterns by combining two
    sinusoidal motions at different frequencies.
    """
    
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
        # Randomize frequency ratio for variety
        ratios = [(1, 2), (2, 3), (3, 4), (1, 3), (2, 5), (3, 5)]
        ratio = ratios[rng.integers(0, len(ratios))]
        self.freq_x = ratio[0]
        self.freq_y = ratio[1]
        self.phase_offset = rng.uniform(0, np.pi / 2)


class MultiFrequencyTarget(TargetPattern):
    """Target with motion composed of multiple frequency components.
    
    Combines several sinusoidal waves to create complex, realistic
    motion patterns that are harder to predict.
    """
    
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
        self.amplitudes: List[float] = []
        self.phases: List[float] = []
        self.frequencies: List[float] = []
    
    def get_position(self, time: float, drone_position: np.ndarray) -> np.ndarray:
        if not self.amplitudes:
            # Not yet reset, use simple circular
            angle = self.base_frequency * time
        else:
            # Sum of harmonics
            angle = 0.0
            for amp, freq, phase in zip(self.amplitudes, self.frequencies, self.phases):
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
            for amp, freq in zip(self.amplitudes, self.frequencies)
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


@dataclass
class YawTrackingConfig:
    """Configuration for yaw tracking environment."""
    
    # Simulation
    model: str = "generic"
    physics_timestep: float = 0.002  # 500 Hz
    control_frequency: float = 50.0  # 50 Hz control
    max_episode_steps: int = 1000  # 20 seconds at 50Hz
    
    # Hover settings
    hover_height: float = 1.0
    hover_position: Tuple[float, float] = (0.0, 0.0)
    
    # Target settings
    target_patterns: List[str] = field(default_factory=lambda: ["circular", "random", "sinusoidal", "step"])
    target_radius: float = 3.0
    target_speed_min: float = 0.5  # rad/s
    target_speed_max: float = 2.0  # rad/s
    
    # Action scaling
    max_yaw_rate: float = 2.0  # rad/s
    
    # Observation noise
    yaw_noise: float = 0.01
    angular_velocity_noise: float = 0.01
    
    # Reward weights
    facing_reward_weight: float = 1.0
    facing_reward_scale: float = 5.0  # exp(-scale * error^2)
    yaw_rate_penalty_weight: float = 0.1
    action_rate_penalty_weight: float = 0.05
    sustained_tracking_bonus: float = 0.5
    sustained_tracking_threshold: float = 0.1  # radians (~6 degrees)
    sustained_tracking_time: float = 0.5  # seconds
    crash_penalty: float = 10.0
    alive_bonus: float = 0.1
    
    # Success criteria
    success_threshold: float = 0.1  # radians
    
    # Termination - more lenient for learning
    max_tilt_angle: float = 1.2  # radians (~69 degrees)
    max_altitude_error: float = 5.0  # meters
    
    # Stabilizer PD gains
    altitude_kp: float = 5.0
    altitude_kd: float = 3.0
    attitude_kp: float = 12.0
    attitude_kd: float = 4.0
    yaw_rate_kp: float = 1.5
    base_thrust: float = 0.62  # Hover throttle for 2kg drone with 4x8N motors


class YawTrackingEnv(gym.Env):
    """Gymnasium environment for yaw target tracking.
    
    The drone hovers in place while a neural network controls its yaw rate
    to keep facing a moving target. Roll, pitch, and altitude are stabilized
    by an internal PD controller.
    
    Observation Space (11 dimensions):
        - target_direction (2): Unit vector to target in body frame [x, y]
        - target_angular_velocity (1): Target's angular velocity
        - current_yaw_rate (1): Drone's yaw rate
        - yaw_error (1): Angle to target in [-pi, pi]
        - roll, pitch (2): Current tilt angles
        - altitude_error (1): Height deviation from hover
        - previous_action (1): Last yaw rate command
        - time_on_target (1): Normalized time spent on target
        - target_distance (1): Distance to target (normalized)
    
    Action Space (1 dimension):
        - yaw_rate_command: Normalized [-1, 1] mapped to [-max_yaw_rate, max_yaw_rate]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: Optional[YawTrackingConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.config = config or YawTrackingConfig()
        self.render_mode = render_mode
        
        # Create simulator
        self.sim = create_simulator(model=self.config.model)
        
        # Control timing
        self._physics_steps_per_control = int(
            1.0 / (self.config.control_frequency * self.config.physics_timestep)
        )
        self._dt = 1.0 / self.config.control_frequency
        
        # Create target patterns
        self._target_patterns = self._create_target_patterns()
        self._current_pattern: Optional[TargetPattern] = None
        
        # State tracking
        self._step_count = 0
        self._time = 0.0
        self._previous_action = 0.0
        self._time_on_target = 0.0
        self._episode_reward = 0.0
        
        # RNG
        self._np_random: Optional[np.random.Generator] = None
        
        # Define spaces
        self._define_spaces()
        
        # Rendering
        self._renderer = None
    
    def _create_target_patterns(self) -> Dict[str, TargetPattern]:
        """Create target pattern instances."""
        cfg = self.config
        patterns = {}
        
        # Basic patterns
        if "circular" in cfg.target_patterns:
            patterns["circular"] = CircularTarget(
                radius=cfg.target_radius,
                angular_velocity=cfg.target_speed_min,
                height=cfg.hover_height,
            )
        
        if "random" in cfg.target_patterns:
            patterns["random"] = RandomTarget(
                radius=cfg.target_radius,
                change_interval=2.0,
                height=cfg.hover_height,
            )
        
        if "sinusoidal" in cfg.target_patterns:
            patterns["sinusoidal"] = SinusoidalTarget(
                radius=cfg.target_radius,
                frequency=cfg.target_speed_min / (2 * np.pi),
                height=cfg.hover_height,
            )
        
        if "step" in cfg.target_patterns:
            patterns["step"] = StepTarget(
                radius=cfg.target_radius,
                step_interval=3.0,
                height=cfg.hover_height,
            )
        
        # Advanced patterns
        if "figure8" in cfg.target_patterns:
            patterns["figure8"] = Figure8Target(
                radius=cfg.target_radius,
                angular_velocity=cfg.target_speed_min,
                height=cfg.hover_height,
            )
        
        if "spiral" in cfg.target_patterns:
            patterns["spiral"] = SpiralTarget(
                radius_min=cfg.target_radius * 0.5,
                radius_max=cfg.target_radius * 1.5,
                angular_velocity=cfg.target_speed_min,
                spiral_frequency=0.1,
                height=cfg.hover_height,
            )
        
        if "evasive" in cfg.target_patterns:
            patterns["evasive"] = EvasiveTarget(
                radius=cfg.target_radius,
                base_angular_velocity=cfg.target_speed_min,
                jerk_probability=0.02,
                max_jerk_magnitude=2.0,
                height=cfg.hover_height,
            )
        
        if "lissajous" in cfg.target_patterns:
            patterns["lissajous"] = LissajousTarget(
                radius=cfg.target_radius,
                angular_velocity=cfg.target_speed_min,
                height=cfg.hover_height,
            )
        
        if "multi_frequency" in cfg.target_patterns:
            patterns["multi_frequency"] = MultiFrequencyTarget(
                radius=cfg.target_radius,
                base_frequency=cfg.target_speed_min / (2 * np.pi),
                num_harmonics=3,
                height=cfg.hover_height,
            )
        
        return patterns
    
    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: 11 dimensions
        self.observation_space = spaces.Box(
            low=np.array([
                -1, -1,         # target_direction (unit vector)
                -5,             # target_angular_velocity
                -5,             # current_yaw_rate
                -np.pi,         # yaw_error
                -1, -1,         # roll, pitch (normalized)
                -5,             # altitude_error
                -1,             # previous_action
                0,              # time_on_target (normalized)
                0,              # target_distance (normalized)
            ], dtype=np.float32),
            high=np.array([
                1, 1,           # target_direction
                5,              # target_angular_velocity
                5,              # current_yaw_rate
                np.pi,          # yaw_error
                1, 1,           # roll, pitch
                5,              # altitude_error
                1,              # previous_action
                1,              # time_on_target
                10,             # target_distance
            ], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Action: yaw rate command [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()
        
        # Reset simulator - start hovering at target height
        init_pos = np.array([
            self.config.hover_position[0],
            self.config.hover_position[1],
            self.config.hover_height,
        ])
        init_yaw = self._np_random.uniform(-np.pi, np.pi)
        init_quat = Rotations.euler_to_quaternion(0, 0, init_yaw)
        
        self.sim.reset(
            position=init_pos,
            velocity=np.zeros(3),
            quaternion=init_quat,
            angular_velocity=np.zeros(3),
        )
        
        # Select and reset target pattern
        if options and "pattern" in options:
            pattern_name = options["pattern"]
        else:
            pattern_name = self._np_random.choice(list(self._target_patterns.keys()))
        
        self._current_pattern = self._target_patterns[pattern_name]
        self._current_pattern.reset(self._np_random)
        
        # Randomize target speed for patterns that support it
        speed = self._np_random.uniform(
            self.config.target_speed_min,
            self.config.target_speed_max,
        )
        if hasattr(self._current_pattern, 'angular_velocity'):
            self._current_pattern.angular_velocity = speed
        if hasattr(self._current_pattern, 'base_angular_velocity'):
            self._current_pattern.base_angular_velocity = speed
        if hasattr(self._current_pattern, 'base_frequency'):
            self._current_pattern.base_frequency = speed / (2 * np.pi)
        
        # Reset state tracking
        self._step_count = 0
        self._time = 0.0
        self._previous_action = 0.0
        self._time_on_target = 0.0
        self._episode_reward = 0.0
        
        # Set initial target marker position
        state = self.sim.get_state()
        target_pos = self._current_pattern.get_position(0.0, state.position)
        self.sim.set_mocap_pos("target", target_pos)
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step.
        
        Args:
            action: Yaw rate command [-1, 1]
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Scale action to yaw rate
        action = np.clip(action, -1, 1)
        yaw_rate_cmd = float(action[0]) * self.config.max_yaw_rate
        
        # Get current state
        state = self.sim.get_state()
        
        # Compute stabilized motor commands
        motor_cmds = self._compute_stabilized_motors(state, yaw_rate_cmd)
        
        # Step physics
        for _ in range(self._physics_steps_per_control):
            self.sim.step(motor_cmds)
        
        self._step_count += 1
        self._time += self._dt
        
        # Get new state
        state = self.sim.get_state()
        
        # Update target marker position in MuJoCo
        target_pos = self._current_pattern.get_position(self._time, state.position)
        self.sim.set_mocap_pos("target", target_pos)
        
        # Compute yaw error for time-on-target tracking
        yaw_error = self._compute_yaw_error(state)
        if abs(yaw_error) < self.config.sustained_tracking_threshold:
            self._time_on_target += self._dt
        else:
            self._time_on_target = max(0, self._time_on_target - self._dt * 0.5)
        
        # Check termination
        terminated = self._check_termination(state)
        truncated = self._step_count >= self.config.max_episode_steps
        
        # Compute reward
        reward = self._compute_reward(state, action, terminated)
        self._episode_reward += reward
        
        # Store action
        action_change = abs(float(action[0]) - self._previous_action)
        self._previous_action = float(action[0])
        
        obs = self._get_observation()
        info = self._get_info()
        info["action_change"] = action_change
        
        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["episode_length"] = self._step_count
        
        return obs, reward, terminated, truncated, info
    
    def _compute_stabilized_motors(
        self,
        state: QuadrotorState,
        yaw_rate_cmd: float,
    ) -> np.ndarray:
        """Compute motor commands with internal stabilization.
        
        Uses PD control to maintain hover while applying yaw rate command.
        """
        cfg = self.config
        
        # Get current state
        roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)
        omega = state.angular_velocity
        
        # Altitude control
        alt_error = cfg.hover_height - state.position[2]
        alt_rate = -state.velocity[2]
        thrust = cfg.base_thrust + cfg.altitude_kp * alt_error + cfg.altitude_kd * alt_rate
        
        # Attitude control - match base_controller.py exactly
        # torque = kp * (desired - actual) - kd * rate
        roll_torque = cfg.attitude_kp * (0 - roll) - cfg.attitude_kd * omega[0]
        pitch_torque = cfg.attitude_kp * (0 - pitch) - cfg.attitude_kd * omega[1]
        
        # Clamp torques
        roll_torque = np.clip(roll_torque, -0.3, 0.3)
        pitch_torque = np.clip(pitch_torque, -0.3, 0.3)
        
        # Yaw rate control - reduced to minimize roll/pitch coupling
        yaw_rate_error = yaw_rate_cmd - omega[2]
        yaw_torque = np.clip(cfg.yaw_rate_kp * yaw_rate_error, -0.05, 0.05)
        
        # Motor mixing (X configuration) - from base_controller.py
        # Motor 1 (FR): + thrust + roll + pitch + yaw
        # Motor 2 (FL): + thrust - roll + pitch - yaw
        # Motor 3 (BL): + thrust - roll - pitch + yaw
        # Motor 4 (BR): + thrust + roll - pitch - yaw
        m1 = thrust + roll_torque + pitch_torque + yaw_torque
        m2 = thrust - roll_torque + pitch_torque - yaw_torque
        m3 = thrust - roll_torque - pitch_torque + yaw_torque
        m4 = thrust + roll_torque - pitch_torque - yaw_torque
        
        motors = np.array([m1, m2, m3, m4])
        return np.clip(motors, 0, 1)
    
    def _compute_yaw_error(self, state: QuadrotorState) -> float:
        """Compute angle from drone heading to target."""
        # Get target position
        target_pos = self._current_pattern.get_position(self._time, state.position)
        
        # Vector to target in world frame
        to_target = target_pos - state.position
        to_target[2] = 0  # Only horizontal
        
        if np.linalg.norm(to_target) < 0.01:
            return 0.0
        
        to_target = to_target / np.linalg.norm(to_target)
        
        # Get drone heading in world frame
        _, _, yaw = Rotations.quaternion_to_euler(state.quaternion)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])
        
        # Angle between heading and target direction
        dot = np.clip(np.dot(heading, to_target), -1, 1)
        cross = heading[0] * to_target[1] - heading[1] * to_target[0]
        
        yaw_error = np.arctan2(cross, dot)
        return yaw_error
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation."""
        state = self.sim.get_state()
        cfg = self.config
        rng = self._np_random
        
        # Get target position and direction
        target_pos = self._current_pattern.get_position(self._time, state.position)
        to_target = target_pos - state.position
        target_distance = np.linalg.norm(to_target[:2])  # Horizontal distance
        
        # Transform to body frame
        _, _, yaw = Rotations.quaternion_to_euler(state.quaternion)
        yaw += rng.normal(0, cfg.yaw_noise)
        
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        body_x = cos_yaw * to_target[0] - sin_yaw * to_target[1]
        body_y = sin_yaw * to_target[0] + cos_yaw * to_target[1]
        
        # Normalize to unit vector
        body_dist = np.sqrt(body_x**2 + body_y**2)
        if body_dist > 0.01:
            target_dir = np.array([body_x / body_dist, body_y / body_dist])
        else:
            target_dir = np.array([1.0, 0.0])
        
        # Yaw error
        yaw_error = self._compute_yaw_error(state)
        
        # Angular velocities
        omega = state.angular_velocity + rng.normal(0, cfg.angular_velocity_noise, 3)
        current_yaw_rate = omega[2]
        target_angular_velocity = self._current_pattern.get_angular_velocity()
        
        # Attitude
        roll, pitch, _ = Rotations.quaternion_to_euler(state.quaternion)
        
        # Altitude error
        altitude_error = cfg.hover_height - state.position[2]
        
        # Time on target (normalized to [0, 1])
        time_on_target_norm = min(self._time_on_target / cfg.sustained_tracking_time, 1.0)
        
        # Target distance (normalized)
        target_dist_norm = target_distance / cfg.target_radius
        
        obs = np.array([
            target_dir[0],
            target_dir[1],
            target_angular_velocity,
            current_yaw_rate,
            yaw_error,
            np.clip(roll, -1, 1),
            np.clip(pitch, -1, 1),
            np.clip(altitude_error, -5, 5),
            self._previous_action,
            time_on_target_norm,
            target_dist_norm,
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(
        self,
        state: QuadrotorState,
        action: np.ndarray,
        terminated: bool,
    ) -> float:
        """Compute reward."""
        cfg = self.config
        reward = 0.0
        
        # 1. Facing reward (exponential)
        yaw_error = self._compute_yaw_error(state)
        facing_reward = np.exp(-cfg.facing_reward_scale * yaw_error**2)
        reward += cfg.facing_reward_weight * facing_reward
        
        # 2. Yaw rate penalty (smooth control)
        yaw_rate = state.angular_velocity[2]
        reward -= cfg.yaw_rate_penalty_weight * yaw_rate**2
        
        # 3. Action rate penalty
        action_rate = (float(action[0]) - self._previous_action)**2
        reward -= cfg.action_rate_penalty_weight * action_rate
        
        # 4. Sustained tracking bonus
        if self._time_on_target >= cfg.sustained_tracking_time:
            reward += cfg.sustained_tracking_bonus
        
        # 5. Alive bonus
        reward += cfg.alive_bonus
        
        # 6. Crash penalty
        if terminated:
            reward -= cfg.crash_penalty
        
        return reward
    
    def _check_termination(self, state: QuadrotorState) -> bool:
        """Check if episode should terminate."""
        cfg = self.config
        
        # Check tilt angle
        roll, pitch, _ = Rotations.quaternion_to_euler(state.quaternion)
        tilt = np.sqrt(roll**2 + pitch**2)
        if tilt > cfg.max_tilt_angle:
            return True
        
        # Check altitude
        altitude_error = abs(cfg.hover_height - state.position[2])
        if altitude_error > cfg.max_altitude_error:
            return True
        
        # Check ground collision
        if state.position[2] < 0.05:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        state = self.sim.get_state()
        yaw_error = self._compute_yaw_error(state)
        roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)
        
        target_pos = self._current_pattern.get_position(self._time, state.position)
        
        return {
            "yaw_error": yaw_error,
            "yaw_error_deg": np.degrees(yaw_error),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "altitude": state.position[2],
            "yaw_rate": state.angular_velocity[2],
            "time_on_target": self._time_on_target,
            "target_position": target_pos.copy(),
            "drone_position": state.position.copy(),
            "step_count": self._step_count,
            "is_tracking": abs(yaw_error) < self.config.success_threshold,
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment with target visualization overlay."""
        if self.render_mode is None:
            return None
        
        if self._renderer is None:
            self._renderer = self.sim.create_renderer(width=640, height=480)
        
        pixels = self.sim.render(self._renderer)
        
        if self.render_mode == "rgb_array" and pixels is not None:
            # Add target visualization overlay
            pixels = self._add_target_overlay(pixels)
            return pixels
        
        return None
    
    def _add_target_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add target direction and status overlay to frame."""
        import cv2
        
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Get current state
        state = self.sim.get_state()
        drone_pos = state.position
        target_pos = self._current_pattern.get_position(self._time, state.position)
        
        # Calculate direction to target
        to_target = target_pos - drone_pos
        target_angle = np.arctan2(to_target[1], to_target[0])
        
        # Get drone yaw
        drone_yaw = Rotations.quaternion_to_euler(state.quaternion)[2]
        yaw_error = target_angle - drone_yaw
        # Normalize to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi
        
        is_tracking = abs(yaw_error) < self.config.success_threshold
        
        # Draw compass circle (top-right)
        cx, cy = w - 80, 80
        radius = 60
        
        # Draw compass background
        cv2.circle(frame, (cx, cy), radius + 5, (40, 40, 40), -1)
        cv2.circle(frame, (cx, cy), radius, (80, 80, 80), 2)
        
        # Draw North marker
        cv2.line(frame, (cx, cy - radius), (cx, cy - radius + 10), (200, 200, 200), 2)
        
        # Draw drone direction (green arrow)
        drone_dx = int(radius * 0.7 * np.cos(drone_yaw - np.pi/2))
        drone_dy = int(radius * 0.7 * np.sin(drone_yaw - np.pi/2))
        cv2.arrowedLine(frame, (cx, cy), (cx + drone_dx, cy + drone_dy), (0, 255, 0), 3, tipLength=0.3)
        
        # Draw target direction (red/yellow arrow)
        target_dx = int(radius * 0.9 * np.cos(target_angle - np.pi/2))
        target_dy = int(radius * 0.9 * np.sin(target_angle - np.pi/2))
        target_color = (0, 255, 255) if is_tracking else (0, 0, 255)  # Yellow if tracking, red if not
        cv2.arrowedLine(frame, (cx, cy), (cx + target_dx, cy + target_dy), target_color, 2, tipLength=0.25)
        
        # Draw target marker (circle at target direction edge)
        target_marker_x = int(cx + radius * np.cos(target_angle - np.pi/2))
        target_marker_y = int(cy + radius * np.sin(target_angle - np.pi/2))
        cv2.circle(frame, (target_marker_x, target_marker_y), 8, target_color, -1)
        cv2.circle(frame, (target_marker_x, target_marker_y), 8, (255, 255, 255), 2)
        
        # Status text
        status_color = (0, 255, 0) if is_tracking else (0, 0, 255)
        status_text = "TRACKING" if is_tracking else "ACQUIRING"
        
        # Draw text info panel (top-left)
        cv2.rectangle(frame, (10, 10), (200, 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (200, 100), (100, 100, 100), 1)
        
        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Yaw Error: {np.degrees(yaw_error):+.1f} deg", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Step: {self._step_count}", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw target distance indicator
        target_dist = np.linalg.norm(to_target[:2])
        cv2.putText(frame, f"Target: {target_dist:.1f}m", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None


# Register environment
gym.register(
    id="DroneYawTracking-v0",
    entry_point="src.environments.yaw_tracking_env:YawTrackingEnv",
)
