"""
Trajectory generation for moving target simulation.
Combines Lissajous curves with Perlin noise for realistic, unpredictable motion.

Based on research findings: smooth mathematical functions create better training
data than random waypoints, while maintaining unpredictability.

Implements 3 complexity levels for curriculum learning:
- Stationary: Fixed position (warm-up)
- Linear: Constant velocity motion
- Complex: Lissajous + Perlin noise (realistic)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum


try:
    import noise  # pip install noise
    HAS_PERLIN = True
except ImportError:
    HAS_PERLIN = False
    # Only warn once at startup, not every episode


class TrajectoryType(Enum):
    """Types of trajectories for curriculum learning"""
    STATIONARY = "stationary"
    LINEAR = "linear"
    CIRCULAR = "circular"
    LISSAJOUS = "lissajous"
    LISSAJOUS_PERLIN = "lissajous_perlin"
    RANDOM_WALK = "random_walk"
    # Advanced unpredictable patterns
    EVASIVE = "evasive"  # Sudden direction changes (evasive maneuvers)
    CHAOTIC = "chaotic"  # Multiple overlaid frequencies + noise
    SPIRAL_DIVE = "spiral_dive"  # 3D spiral with altitude changes
    ZIGZAG = "zigzag"  # Sharp direction reversals
    FIGURE_EIGHT = "figure_eight"  # Classic 8 pattern with variations
    DRUNK_WALK = "drunk_walk"  # Smooth but erratic Brownian motion
    PREDATOR = "predator"  # Actively tries to evade tracking (adversarial)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation"""
    trajectory_type: TrajectoryType
    duration: float = 60.0  # seconds
    dt: float = 0.05  # timestep (20 Hz)
    
    # Lissajous parameters (randomized per episode)
    amplitude_x: Tuple[float, float] = (1.0, 5.0)  # (min, max) meters
    amplitude_y: Tuple[float, float] = (1.0, 5.0)
    amplitude_z: Tuple[float, float] = (0.5, 2.0)
    frequency_x: Tuple[float, float] = (0.3, 1.5)  # (min, max) Hz
    frequency_y: Tuple[float, float] = (0.3, 1.5)
    frequency_z: Tuple[float, float] = (0.2, 1.0)
    
    # Linear motion
    velocity_min: float = 0.5  # m/s
    velocity_max: float = 3.0  # m/s
    
    # Perlin noise parameters
    perlin_scale: Tuple[float, float] = (0.2, 1.5)  # (min, max) meters
    perlin_octaves: int = 2
    perlin_persistence: float = 0.5
    perlin_lacunarity: float = 2.0
    
    # Spatial bounds (safety geofence)
    bounds_x: Tuple[float, float] = (-10.0, 10.0)
    bounds_y: Tuple[float, float] = (-10.0, 10.0)
    bounds_z: Tuple[float, float] = (-5.0, -0.5)  # NED: negative is up
    
    # Center position offset
    center: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -2.0]))


class TrajectoryGenerator:
    """
    Generates target trajectories for training.
    Supports multiple trajectory types with randomization.
    """
    
    def __init__(self, config: TrajectoryConfig, seed: Optional[int] = None):
        """
        Initialize trajectory generator.
        
        Args:
            config: Trajectory configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        # Precompute timesteps
        self.num_steps = int(self.config.duration / self.config.dt)
        self.time = np.arange(self.num_steps) * self.config.dt
        
        # Current trajectory cache
        self._current_trajectory: Optional[np.ndarray] = None
        self._current_velocities: Optional[np.ndarray] = None
    
    def generate(self, trajectory_type: Optional[TrajectoryType] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a new trajectory.
        
        Args:
            trajectory_type: Override config trajectory type
            
        Returns:
            positions: (N, 3) array of [x, y, z] positions in NED frame
            velocities: (N, 3) array of [vx, vy, vz] velocities
        """
        traj_type = trajectory_type or self.config.trajectory_type
        
        if traj_type == TrajectoryType.STATIONARY:
            positions, velocities = self._generate_stationary()
        elif traj_type == TrajectoryType.LINEAR:
            positions, velocities = self._generate_linear()
        elif traj_type == TrajectoryType.CIRCULAR:
            positions, velocities = self._generate_circular()
        elif traj_type == TrajectoryType.LISSAJOUS:
            positions, velocities = self._generate_lissajous()
        elif traj_type == TrajectoryType.LISSAJOUS_PERLIN:
            positions, velocities = self._generate_lissajous_perlin()
        elif traj_type == TrajectoryType.RANDOM_WALK:
            positions, velocities = self._generate_random_walk()
        elif traj_type == TrajectoryType.EVASIVE:
            positions, velocities = self._generate_evasive()
        elif traj_type == TrajectoryType.CHAOTIC:
            positions, velocities = self._generate_chaotic()
        elif traj_type == TrajectoryType.SPIRAL_DIVE:
            positions, velocities = self._generate_spiral_dive()
        elif traj_type == TrajectoryType.ZIGZAG:
            positions, velocities = self._generate_zigzag()
        elif traj_type == TrajectoryType.FIGURE_EIGHT:
            positions, velocities = self._generate_figure_eight()
        elif traj_type == TrajectoryType.DRUNK_WALK:
            positions, velocities = self._generate_drunk_walk()
        elif traj_type == TrajectoryType.PREDATOR:
            positions, velocities = self._generate_predator()
        else:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
        # Apply bounds
        positions = self._apply_bounds(positions)
        
        # Recompute velocities after bounds (may have been clamped)
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.config.dt
        
        # Cache
        self._current_trajectory = positions
        self._current_velocities = velocities
        
        return positions, velocities
    
    def _generate_stationary(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate stationary target (for warm-up training)"""
        # Random position within bounds
        position = self.rng.uniform(
            [self.config.bounds_x[0], self.config.bounds_y[0], self.config.bounds_z[0]],
            [self.config.bounds_x[1], self.config.bounds_y[1], self.config.bounds_z[1]]
        )
        
        positions = np.tile(position, (self.num_steps, 1))
        velocities = np.zeros((self.num_steps, 3))
        
        return positions, velocities
    
    def _generate_linear(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate linear motion with constant velocity"""
        # Random starting position
        start_pos = self.rng.uniform(
            [self.config.bounds_x[0], self.config.bounds_y[0], self.config.bounds_z[0]],
            [self.config.bounds_x[1], self.config.bounds_y[1], self.config.bounds_z[1]]
        )
        
        # Random velocity direction
        velocity_magnitude = self.rng.uniform(self.config.velocity_min, self.config.velocity_max)
        velocity_direction = self.rng.randn(3)
        velocity_direction[2] *= 0.3  # Less vertical motion
        velocity_direction /= np.linalg.norm(velocity_direction)
        velocity = velocity_direction * velocity_magnitude
        
        # Generate trajectory
        positions = start_pos + np.outer(self.time, velocity)
        velocities = np.tile(velocity, (self.num_steps, 1))
        
        return positions, velocities
    
    def _generate_circular(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate circular motion in horizontal plane"""
        # Random radius and angular velocity
        radius = self.rng.uniform(2.0, 5.0)
        angular_vel = self.rng.uniform(0.3, 1.0)  # rad/s
        
        # Altitude
        altitude = self.rng.uniform(self.config.bounds_z[0], self.config.bounds_z[1])
        
        # Generate circle
        theta = angular_vel * self.time
        x = self.config.center[0] + radius * np.cos(theta)
        y = self.config.center[1] + radius * np.sin(theta)
        z = np.full(self.num_steps, altitude)
        
        positions = np.column_stack([x, y, z])
        
        # Velocities (tangent to circle)
        vx = -radius * angular_vel * np.sin(theta)
        vy = radius * angular_vel * np.cos(theta)
        vz = np.zeros(self.num_steps)
        velocities = np.column_stack([vx, vy, vz])
        
        return positions, velocities
    
    def _generate_lissajous(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Lissajous curve (parametric oscillation).
        Creates beautiful figure-8 and complex patterns.
        """
        # Randomize parameters
        A_x = self.rng.uniform(*self.config.amplitude_x)
        A_y = self.rng.uniform(*self.config.amplitude_y)
        A_z = self.rng.uniform(*self.config.amplitude_z)
        
        omega_x = 2 * np.pi * self.rng.uniform(*self.config.frequency_x)
        omega_y = 2 * np.pi * self.rng.uniform(*self.config.frequency_y)
        omega_z = 2 * np.pi * self.rng.uniform(*self.config.frequency_z)
        
        phi_x = self.rng.uniform(0, 2 * np.pi)
        phi_y = self.rng.uniform(0, 2 * np.pi)
        phi_z = self.rng.uniform(0, 2 * np.pi)
        
        # Generate Lissajous curve
        x = self.config.center[0] + A_x * np.sin(omega_x * self.time + phi_x)
        y = self.config.center[1] + A_y * np.sin(omega_y * self.time + phi_y)
        z = self.config.center[2] + A_z * np.sin(omega_z * self.time + phi_z)
        
        positions = np.column_stack([x, y, z])
        
        # Analytical velocities (derivatives)
        vx = A_x * omega_x * np.cos(omega_x * self.time + phi_x)
        vy = A_y * omega_y * np.cos(omega_y * self.time + phi_y)
        vz = A_z * omega_z * np.cos(omega_z * self.time + phi_z)
        velocities = np.column_stack([vx, vy, vz])
        
        return positions, velocities
    
    def _generate_lissajous_perlin(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Lissajous curve with Perlin noise overlay.
        This is THE BEST trajectory for realistic training.
        
        Lissajous provides smooth, predictable base motion.
        Perlin adds natural, unpredictable variations.
        """
        # Start with Lissajous
        positions, velocities = self._generate_lissajous()
        
        if not HAS_PERLIN:
            # Silently fallback to pure Lissajous
            return positions, velocities
        
        # Add Perlin noise
        perlin_magnitude = self.rng.uniform(*self.config.perlin_scale)
        
        # Use random offset for unique noise each episode
        noise_offset = self.rng.uniform(0, 1000)
        
        perlin_x = np.array([
            noise.pnoise1(
                t * 0.1 + noise_offset,
                octaves=self.config.perlin_octaves,
                persistence=self.config.perlin_persistence,
                lacunarity=self.config.perlin_lacunarity
            ) for t in range(self.num_steps)
        ]) * perlin_magnitude
        
        perlin_y = np.array([
            noise.pnoise1(
                t * 0.1 + noise_offset + 100,  # Different offset for y
                octaves=self.config.perlin_octaves,
                persistence=self.config.perlin_persistence,
                lacunarity=self.config.perlin_lacunarity
            ) for t in range(self.num_steps)
        ]) * perlin_magnitude
        
        perlin_z = np.array([
            noise.pnoise1(
                t * 0.1 + noise_offset + 200,  # Different offset for z
                octaves=self.config.perlin_octaves,
                persistence=self.config.perlin_persistence,
                lacunarity=self.config.perlin_lacunarity
            ) for t in range(self.num_steps)
        ]) * perlin_magnitude * 0.5  # Less vertical variation
        
        perlin_offset = np.column_stack([perlin_x, perlin_y, perlin_z])
        
        # Add Perlin to Lissajous
        positions += perlin_offset
        
        # Recompute velocities
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.config.dt
        
        return positions, velocities
    
    def _generate_random_walk(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random walk trajectory"""
        # Starting position
        position = self.config.center.copy()
        positions = [position.copy()]
        
        # Random walk with momentum
        velocity = np.zeros(3)
        velocities = [velocity.copy()]
        
        for _ in range(1, self.num_steps):
            # Random acceleration
            acceleration = self.rng.randn(3) * 0.5
            acceleration[2] *= 0.3  # Less vertical acceleration
            
            # Update velocity with damping
            velocity = 0.9 * velocity + acceleration * self.config.dt
            
            # Clip velocity
            speed = np.linalg.norm(velocity)
            if speed > self.config.velocity_max:
                velocity = velocity / speed * self.config.velocity_max
            
            # Update position
            position = position + velocity * self.config.dt
            
            positions.append(position.copy())
            velocities.append(velocity.copy())
        
        return np.array(positions), np.array(velocities)
    
    def _generate_evasive(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate evasive maneuver trajectory.
        Sudden direction changes at random intervals - like a fighter jet evading.
        """
        position = self.config.center.copy()
        positions = [position.copy()]
        
        velocity = self.rng.randn(3) * 2.0
        velocity[2] *= 0.3
        velocities = [velocity.copy()]
        
        # Schedule random evasion events
        evasion_times = []
        t = self.rng.uniform(0.5, 2.0)
        while t < self.config.duration:
            evasion_times.append(t)
            t += self.rng.uniform(0.3, 1.5)  # Random interval between maneuvers
        
        evasion_idx = 0
        for step in range(1, self.num_steps):
            current_time = step * self.config.dt
            
            # Check for evasion event
            if evasion_idx < len(evasion_times) and current_time >= evasion_times[evasion_idx]:
                # SUDDEN direction change (90-180 degree turn)
                angle = self.rng.uniform(np.pi/2, np.pi)
                # Rotate velocity vector
                c, s = np.cos(angle), np.sin(angle)
                vx, vy = velocity[0], velocity[1]
                velocity[0] = c * vx - s * vy
                velocity[1] = s * vx + c * vy
                # Random vertical juke
                velocity[2] = self.rng.uniform(-2.0, 2.0)
                # Boost speed
                velocity *= self.rng.uniform(1.2, 1.8)
                evasion_idx += 1
            
            # Natural deceleration
            velocity *= 0.98
            
            # Clip
            speed = np.linalg.norm(velocity)
            if speed > self.config.velocity_max * 1.5:
                velocity = velocity / speed * self.config.velocity_max * 1.5
            
            position = position + velocity * self.config.dt
            positions.append(position.copy())
            velocities.append(velocity.copy())
        
        return np.array(positions), np.array(velocities)
    
    def _generate_chaotic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate chaotic trajectory - multiple overlaid sinusoids with different frequencies.
        Creates highly unpredictable motion while remaining smooth.
        """
        # Multiple frequency components
        n_components = self.rng.randint(4, 8)
        
        x = np.zeros(self.num_steps)
        y = np.zeros(self.num_steps)
        z = np.zeros(self.num_steps)
        
        for _ in range(n_components):
            # Random amplitude, frequency, phase for each component
            amp = self.rng.uniform(0.5, 2.0)
            freq = self.rng.uniform(0.1, 2.0)
            phase = self.rng.uniform(0, 2 * np.pi)
            
            x += amp * np.sin(2 * np.pi * freq * self.time + phase)
            y += amp * np.cos(2 * np.pi * freq * self.time + phase + self.rng.uniform(0, np.pi))
            z += amp * 0.3 * np.sin(2 * np.pi * freq * 0.5 * self.time + phase)
        
        # Normalize to reasonable bounds
        x = x / n_components * 3 + self.config.center[0]
        y = y / n_components * 3 + self.config.center[1]
        z = z / n_components + self.config.center[2]
        
        positions = np.column_stack([x, y, z])
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.config.dt
        
        return positions, velocities
    
    def _generate_spiral_dive(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D spiral with altitude changes.
        Like a bird diving and climbing while circling.
        """
        # Spiral parameters
        radius_start = self.rng.uniform(2.0, 4.0)
        radius_end = self.rng.uniform(1.0, radius_start)
        angular_vel = self.rng.uniform(0.5, 1.5)
        
        # Altitude oscillation
        alt_amp = self.rng.uniform(1.0, 3.0)
        alt_freq = self.rng.uniform(0.2, 0.5)
        
        # Radius changes over time
        radius = np.linspace(radius_start, radius_end, self.num_steps)
        
        theta = angular_vel * self.time
        x = self.config.center[0] + radius * np.cos(theta)
        y = self.config.center[1] + radius * np.sin(theta)
        z = self.config.center[2] + alt_amp * np.sin(2 * np.pi * alt_freq * self.time)
        
        positions = np.column_stack([x, y, z])
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.config.dt
        
        return positions, velocities
    
    def _generate_zigzag(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sharp zigzag pattern with sudden direction reversals.
        Triangle wave motion.
        """
        # Zigzag parameters
        amplitude = self.rng.uniform(2.0, 5.0)
        period = self.rng.uniform(1.0, 3.0)
        
        # Triangle wave function
        def triangle_wave(t, period, amp):
            return 2 * amp / np.pi * np.arcsin(np.sin(2 * np.pi * t / period))
        
        # Base forward motion
        forward_speed = self.rng.uniform(1.0, 3.0)
        x = self.config.center[0] + forward_speed * self.time
        
        # Zigzag lateral motion
        y = self.config.center[1] + triangle_wave(self.time, period, amplitude)
        
        # Smaller vertical zigzag
        z = self.config.center[2] + triangle_wave(self.time + period/4, period * 1.5, amplitude * 0.3)
        
        positions = np.column_stack([x, y, z])
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.config.dt
        
        return positions, velocities
    
    def _generate_figure_eight(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate figure-8 pattern with randomized parameters.
        Classic training pattern with variations.
        """
        # Figure-8 is a special Lissajous with 2:1 frequency ratio
        amplitude = self.rng.uniform(2.0, 5.0)
        base_freq = self.rng.uniform(0.3, 0.8)
        
        # Add tilt angle
        tilt = self.rng.uniform(-np.pi/6, np.pi/6)
        
        x_raw = amplitude * np.sin(2 * np.pi * base_freq * self.time)
        y_raw = amplitude * np.sin(2 * np.pi * base_freq * 2 * self.time) / 2
        
        # Apply tilt rotation
        x = x_raw * np.cos(tilt) - y_raw * np.sin(tilt) + self.config.center[0]
        y = x_raw * np.sin(tilt) + y_raw * np.cos(tilt) + self.config.center[1]
        
        # Gentle vertical oscillation
        z = self.config.center[2] + self.rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * base_freq * 0.5 * self.time)
        
        positions = np.column_stack([x, y, z])
        velocities = np.diff(positions, axis=0, prepend=positions[0:1]) / self.config.dt
        
        return positions, velocities
    
    def _generate_drunk_walk(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate smooth but erratic "drunk walk" using Ornstein-Uhlenbeck process.
        More realistic than pure random walk - has mean-reversion.
        """
        theta = 0.15  # Mean reversion rate
        sigma = 1.5   # Volatility
        
        position = self.config.center.copy()
        velocity = np.zeros(3)
        
        positions = [position.copy()]
        velocities = [velocity.copy()]
        
        for _ in range(1, self.num_steps):
            # Ornstein-Uhlenbeck update
            dW = self.rng.randn(3) * np.sqrt(self.config.dt)
            
            # Mean reversion toward center
            drift = theta * (self.config.center - position)
            diffusion = sigma * dW
            
            velocity = velocity * 0.9 + drift + diffusion
            velocity[2] *= 0.5  # Less vertical motion
            
            # Clip velocity
            speed = np.linalg.norm(velocity)
            if speed > self.config.velocity_max:
                velocity = velocity / speed * self.config.velocity_max
            
            position = position + velocity * self.config.dt
            
            positions.append(position.copy())
            velocities.append(velocity.copy())
        
        return np.array(positions), np.array(velocities)
    
    def _generate_predator(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adversarial "predator" trajectory.
        Simulates a target actively trying to break tracking lock.
        Combines multiple evasion strategies.
        """
        position = self.config.center.copy()
        positions = [position.copy()]
        
        velocity = self.rng.randn(3)
        velocity = velocity / np.linalg.norm(velocity) * 2.0
        velocities = [velocity.copy()]
        
        # State machine for behavior
        behavior = 'cruise'  # cruise, evade, feint, dive
        behavior_timer = 0
        behavior_duration = self.rng.uniform(1.0, 3.0)
        
        for step in range(1, self.num_steps):
            behavior_timer += self.config.dt
            
            # Behavior transitions
            if behavior_timer > behavior_duration:
                behavior = self.rng.choice(['cruise', 'evade', 'feint', 'dive'], 
                                          p=[0.3, 0.3, 0.2, 0.2])
                behavior_timer = 0
                behavior_duration = self.rng.uniform(0.5, 2.0)
            
            if behavior == 'cruise':
                # Gentle turns
                velocity += self.rng.randn(3) * 0.3 * self.config.dt
            
            elif behavior == 'evade':
                # Sharp random turns
                angle = self.rng.uniform(np.pi/3, np.pi)
                c, s = np.cos(angle), np.sin(angle)
                vx, vy = velocity[0], velocity[1]
                velocity[0] = c * vx - s * vy + self.rng.randn() * 0.5
                velocity[1] = s * vx + c * vy + self.rng.randn() * 0.5
                velocity *= 1.3  # Speed burst
            
            elif behavior == 'feint':
                # Fake one direction then go another
                if behavior_timer < behavior_duration / 2:
                    velocity += np.array([1, 0, 0]) * 2 * self.config.dt
                else:
                    velocity += np.array([-2, 1, 0]) * 2 * self.config.dt
            
            elif behavior == 'dive':
                # Sudden altitude change
                velocity[2] = self.rng.choice([-3, 3])
                velocity[:2] *= 0.7
            
            # Damping and limits
            velocity *= 0.97
            speed = np.linalg.norm(velocity)
            if speed > self.config.velocity_max * 1.5:
                velocity = velocity / speed * self.config.velocity_max * 1.5
            if speed < 0.5:
                velocity = velocity / (speed + 0.01) * 0.5
            
            position = position + velocity * self.config.dt
            positions.append(position.copy())
            velocities.append(velocity.copy())
        
        return np.array(positions), np.array(velocities)
    
    def _apply_bounds(self, positions: np.ndarray) -> np.ndarray:
        """Apply spatial bounds with soft clamping"""
        positions[:, 0] = np.clip(positions[:, 0], *self.config.bounds_x)
        positions[:, 1] = np.clip(positions[:, 1], *self.config.bounds_y)
        positions[:, 2] = np.clip(positions[:, 2], *self.config.bounds_z)
        return positions
    
    def get_position_at_time(self, t: float) -> np.ndarray:
        """Get target position at specific time"""
        if self._current_trajectory is None:
            raise RuntimeError("No trajectory generated yet. Call generate() first.")
        
        idx = int(t / self.config.dt)
        idx = np.clip(idx, 0, len(self._current_trajectory) - 1)
        return self._current_trajectory[idx]
    
    def get_velocity_at_time(self, t: float) -> np.ndarray:
        """Get target velocity at specific time"""
        if self._current_velocities is None:
            raise RuntimeError("No trajectory generated yet. Call generate() first.")
        
        idx = int(t / self.config.dt)
        idx = np.clip(idx, 0, len(self._current_velocities) - 1)
        return self._current_velocities[idx]


# ============================================================================
# CURRICULUM HELPER
# ============================================================================

class CurriculumTrajectoryManager:
    """
    Manages trajectory difficulty progression for curriculum learning.
    
    Stage 1 (0-100K steps): Stationary targets
    Stage 2 (100K-300K): Linear motion
    Stage 3 (300K+): Complex (Lissajous + Perlin)
    """
    
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.current_stage = 1
        self.steps_trained = 0
        
        # Stage thresholds
        self.stage_thresholds = {
            1: 0,
            2: 100_000,
            3: 300_000
        }
        
        self.stage_types = {
            1: TrajectoryType.STATIONARY,
            2: TrajectoryType.LINEAR,
            3: TrajectoryType.LISSAJOUS_PERLIN
        }
    
    def get_trajectory_type(self) -> TrajectoryType:
        """Get appropriate trajectory type for current training stage"""
        # Update stage based on steps
        for stage, threshold in sorted(self.stage_thresholds.items(), reverse=True):
            if self.steps_trained >= threshold:
                self.current_stage = stage
                break
        
        return self.stage_types[self.current_stage]
    
    def update(self, num_steps: int):
        """Update training progress"""
        self.steps_trained += num_steps
    
    def get_stage_info(self) -> Dict:
        """Get current stage information"""
        return {
            'stage': self.current_stage,
            'steps_trained': self.steps_trained,
            'trajectory_type': self.stage_types[self.current_stage].value
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print("=== Trajectory Generator Tests ===\n")
    
    # Create config
    config = TrajectoryConfig(
        trajectory_type=TrajectoryType.LISSAJOUS_PERLIN,
        duration=30.0,
        dt=0.05
    )
    
    generator = TrajectoryGenerator(config, seed=42)
    
    # Test all trajectory types
    types_to_test = [
        TrajectoryType.STATIONARY,
        TrajectoryType.LINEAR,
        TrajectoryType.LISSAJOUS,
        TrajectoryType.LISSAJOUS_PERLIN if HAS_PERLIN else TrajectoryType.CIRCULAR
    ]
    
    fig = plt.figure(figsize=(15, 10))
    
    for idx, traj_type in enumerate(types_to_test):
        positions, velocities = generator.generate(traj_type)
        
        # 3D plot
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1)
        ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='x', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{traj_type.value.capitalize()} Trajectory')
        ax.legend()
        ax.grid(True)
        
        # Stats
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
        max_speed = np.max(np.linalg.norm(velocities, axis=1))
        print(f"{traj_type.value}:")
        print(f"  Duration: {config.duration}s, Steps: {len(positions)}")
        print(f"  Avg speed: {avg_speed:.2f} m/s, Max speed: {max_speed:.2f} m/s")
        print()
    
    plt.tight_layout()
    plt.savefig('/tmp/trajectories.png', dpi=150)
    print("Trajectories saved to /tmp/trajectories.png")
    
    print("\n=== Curriculum Manager Test ===")
    curriculum = CurriculumTrajectoryManager(config)
    
    for step in [0, 50_000, 150_000, 350_000]:
        curriculum.steps_trained = step
        info = curriculum.get_stage_info()
        print(f"Steps {step:>7}: Stage {info['stage']}, Type: {info['trajectory_type']}")
