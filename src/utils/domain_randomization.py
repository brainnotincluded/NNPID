"""
Domain Randomization for Sim-to-Real transfer.

Randomizes physical parameters during training to make the policy robust
to real-world variations. This is CRITICAL for zero-shot sim-to-real transfer.

Based on research: Randomize mass, thrust, drag, sensor noise, latency.
Models trained with domain randomization transfer to real drones without fine-tuning.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class DronePhysicsParams:
    """Physical parameters of the drone"""
    mass: float = 1.0  # kg
    inertia: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.01, 0.02]))  # kg⋅m²
    arm_length: float = 0.2  # meters
    thrust_coefficient: float = 1.0  # multiplier on motor thrust
    drag_coefficient: float = 0.1  # air resistance
    motor_time_constant: float = 0.02  # seconds (actuator lag)
    
    # Sensor noise parameters
    imu_gyro_noise: float = 0.01  # rad/s std dev
    imu_accel_noise: float = 0.1  # m/s² std dev
    gps_position_noise: float = 0.1  # meters std dev
    barometer_noise: float = 0.05  # meters std dev
    
    # Communication
    mavlink_latency: float = 0.03  # seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'mass': self.mass,
            'inertia': self.inertia.tolist() if isinstance(self.inertia, np.ndarray) else self.inertia,
            'arm_length': self.arm_length,
            'thrust_coefficient': self.thrust_coefficient,
            'drag_coefficient': self.drag_coefficient,
            'motor_time_constant': self.motor_time_constant,
            'imu_gyro_noise': self.imu_gyro_noise,
            'imu_accel_noise': self.imu_accel_noise,
            'gps_position_noise': self.gps_position_noise,
            'barometer_noise': self.barometer_noise,
            'mavlink_latency': self.mavlink_latency
        }


@dataclass
class EnvironmentParams:
    """Environmental parameters"""
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s [north, east, down]
    wind_turbulence: float = 0.0  # m/s std dev
    air_density: float = 1.225  # kg/m³ (sea level)
    gravity: float = 9.81  # m/s²
    magnetic_declination: float = 0.0  # radians
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'wind_velocity': self.wind_velocity.tolist() if isinstance(self.wind_velocity, np.ndarray) else self.wind_velocity,
            'wind_turbulence': self.wind_turbulence,
            'air_density': self.air_density,
            'gravity': self.gravity,
            'magnetic_declination': self.magnetic_declination
        }


class DomainRandomizer:
    """
    Randomizes simulation parameters for each episode.
    
    Key parameters from research:
    - Mass: ±20% (battery drain, payload)
    - Thrust: ±10% (motor degradation)
    - Drag: 0.5x - 2.0x (aerodynamics variation)
    - Latency: 20-100ms (communication delays)
    - Sensor noise: Gaussian with realistic std devs
    """
    
    def __init__(
        self,
        nominal_drone_params: DronePhysicsParams,
        nominal_env_params: EnvironmentParams,
        randomization_level: float = 1.0,  # 0.0 = no randomization, 1.0 = full
        seed: Optional[int] = None
    ):
        """
        Initialize domain randomizer.
        
        Args:
            nominal_drone_params: Nominal (baseline) drone parameters
            nominal_env_params: Nominal environment parameters
            randomization_level: Strength of randomization [0.0, 1.0]
            seed: Random seed for reproducibility
        """
        self.nominal_drone = nominal_drone_params
        self.nominal_env = nominal_env_params
        self.randomization_level = randomization_level
        self.rng = np.random.RandomState(seed)
    
    def randomize_episode(self) -> Tuple[DronePhysicsParams, EnvironmentParams]:
        """
        Generate randomized parameters for a new episode.
        
        Returns:
            drone_params: Randomized drone physics
            env_params: Randomized environment
        """
        if self.randomization_level == 0.0:
            return self.nominal_drone, self.nominal_env
        
        # Randomize drone physics
        drone_params = DronePhysicsParams(
            mass=self._randomize_uniform(
                self.nominal_drone.mass,
                0.8, 1.2,
                "Mass"
            ),
            inertia=self._randomize_uniform_array(
                self.nominal_drone.inertia,
                0.85, 1.15,
                "Inertia"
            ),
            arm_length=self.nominal_drone.arm_length,  # Don't randomize geometry
            thrust_coefficient=self._randomize_uniform(
                self.nominal_drone.thrust_coefficient,
                0.9, 1.1,
                "Thrust coefficient"
            ),
            drag_coefficient=self._randomize_uniform(
                self.nominal_drone.drag_coefficient,
                0.5, 2.0,
                "Drag coefficient"
            ),
            motor_time_constant=self._randomize_uniform(
                self.nominal_drone.motor_time_constant,
                0.01, 0.05,
                "Motor time constant"
            ),
            imu_gyro_noise=self._randomize_uniform(
                self.nominal_drone.imu_gyro_noise,
                0.005, 0.02,
                "IMU gyro noise"
            ),
            imu_accel_noise=self._randomize_uniform(
                self.nominal_drone.imu_accel_noise,
                0.05, 0.2,
                "IMU accel noise"
            ),
            gps_position_noise=self._randomize_uniform(
                self.nominal_drone.gps_position_noise,
                0.05, 0.2,
                "GPS noise"
            ),
            barometer_noise=self._randomize_uniform(
                self.nominal_drone.barometer_noise,
                0.02, 0.1,
                "Barometer noise"
            ),
            mavlink_latency=self._randomize_uniform(
                self.nominal_drone.mavlink_latency,
                0.02, 0.1,
                "MAVLink latency"
            )
        )
        
        # Randomize environment
        wind_speed = self.rng.uniform(0.0, 5.0 * self.randomization_level)
        wind_direction = self.rng.uniform(0, 2 * np.pi)
        
        env_params = EnvironmentParams(
            wind_velocity=np.array([
                wind_speed * np.cos(wind_direction),
                wind_speed * np.sin(wind_direction),
                self.rng.uniform(-1.0, 1.0)  # vertical wind
            ]),
            wind_turbulence=self.rng.uniform(0.0, 1.0 * self.randomization_level),
            air_density=self._randomize_uniform(
                self.nominal_env.air_density,
                1.1, 1.3,  # From sea level to 1500m altitude
                "Air density"
            ),
            gravity=self._randomize_uniform(
                self.nominal_env.gravity,
                0.98, 1.02,  # Sensor calibration variation
                "Gravity"
            ),
            magnetic_declination=self.rng.uniform(-np.pi/6, np.pi/6)
        )
        
        return drone_params, env_params
    
    def _randomize_uniform(
        self,
        nominal: float,
        min_scale: float,
        max_scale: float,
        param_name: str = ""
    ) -> float:
        """
        Randomize a scalar parameter with uniform distribution.
        
        Args:
            nominal: Nominal value
            min_scale: Minimum scale factor
            max_scale: Maximum scale factor
            param_name: Parameter name for debugging
            
        Returns:
            Randomized value
        """
        if self.randomization_level == 0.0:
            return nominal
        
        # Scale randomization by randomization_level
        scale_range = (max_scale - min_scale) * self.randomization_level
        scale_center = (max_scale + min_scale) / 2
        scale_min = scale_center - scale_range / 2
        scale_max = scale_center + scale_range / 2
        
        scale = self.rng.uniform(scale_min, scale_max)
        value = nominal * scale
        
        return value
    
    def _randomize_uniform_array(
        self,
        nominal: np.ndarray,
        min_scale: float,
        max_scale: float,
        param_name: str = ""
    ) -> np.ndarray:
        """Randomize array with uniform distribution"""
        return np.array([
            self._randomize_uniform(v, min_scale, max_scale, param_name)
            for v in nominal
        ])
    
    def get_current_params(self) -> Dict:
        """Get current randomized parameters as dict"""
        drone_params, env_params = self.randomize_episode()
        return {
            'drone': drone_params.to_dict(),
            'environment': env_params.to_dict()
        }


class LatencySimulator:
    """
    Simulates communication and processing latency.
    
    Critical for sim-to-real: Model must learn to compensate for delays.
    Typical real-world latency: 20-100ms
    """
    
    def __init__(
        self,
        base_latency: float = 0.05,  # seconds
        latency_variation: float = 0.03,  # seconds
        buffer_size: int = 10
    ):
        """
        Initialize latency simulator.
        
        Args:
            base_latency: Base communication latency (seconds)
            latency_variation: Random variation in latency (seconds)
            buffer_size: Size of delay buffer
        """
        self.base_latency = base_latency
        self.latency_variation = latency_variation
        self.buffer_size = buffer_size
        
        # Circular buffer for delayed observations
        self.buffer = []
        self.timestamps = []
    
    def add_observation(self, obs: np.ndarray, timestamp: float):
        """Add observation to delay buffer"""
        self.buffer.append(obs.copy())
        self.timestamps.append(timestamp)
        
        # Keep buffer size limited
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
            self.timestamps.pop(0)
    
    def get_delayed_observation(
        self,
        current_time: float,
        rng: Optional[np.random.RandomState] = None
    ) -> Optional[np.ndarray]:
        """
        Get observation with simulated latency.
        
        Args:
            current_time: Current simulation time
            rng: Random number generator
            
        Returns:
            Delayed observation or None if buffer empty
        """
        if len(self.buffer) == 0:
            return None
        
        # Generate random latency
        if rng is None:
            rng = np.random.RandomState()
        
        latency = self.base_latency + rng.uniform(-self.latency_variation, self.latency_variation)
        latency = max(0.0, latency)  # No negative latency
        
        # Find observation from the past
        target_time = current_time - latency
        
        # Find closest observation before target_time
        valid_indices = [i for i, t in enumerate(self.timestamps) if t <= target_time]
        
        if len(valid_indices) == 0:
            # No observations old enough, return oldest
            return self.buffer[0]
        
        # Return most recent valid observation
        idx = valid_indices[-1]
        return self.buffer[idx]
    
    def reset(self):
        """Clear delay buffer"""
        self.buffer = []
        self.timestamps = []


class SensorNoiseSimulator:
    """
    Adds realistic sensor noise to observations.
    
    Noise characteristics based on real IMU/GPS/Barometer specs.
    """
    
    def __init__(
        self,
        imu_gyro_noise: float = 0.01,  # rad/s
        imu_accel_noise: float = 0.1,  # m/s²
        gps_noise: float = 0.1,  # meters
        barometer_noise: float = 0.05,  # meters
        seed: Optional[int] = None
    ):
        """
        Initialize sensor noise simulator.
        
        Args:
            imu_gyro_noise: Gyroscope noise std dev (rad/s)
            imu_accel_noise: Accelerometer noise std dev (m/s²)
            gps_noise: GPS position noise std dev (meters)
            barometer_noise: Barometer altitude noise std dev (meters)
            seed: Random seed
        """
        self.imu_gyro_noise = imu_gyro_noise
        self.imu_accel_noise = imu_accel_noise
        self.gps_noise = gps_noise
        self.barometer_noise = barometer_noise
        self.rng = np.random.RandomState(seed)
    
    def add_noise_to_position(self, position: np.ndarray) -> np.ndarray:
        """Add GPS noise to position"""
        noise = self.rng.normal(0, self.gps_noise, size=3)
        return position + noise
    
    def add_noise_to_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Add IMU-derived velocity noise"""
        # Velocity comes from integration of accelerometer
        noise = self.rng.normal(0, self.imu_accel_noise * 0.1, size=3)
        return velocity + noise
    
    def add_noise_to_orientation(self, orientation: np.ndarray) -> np.ndarray:
        """Add IMU noise to orientation (euler angles)"""
        noise = self.rng.normal(0, self.imu_gyro_noise * 0.1, size=3)
        return orientation + noise
    
    def add_noise_to_angular_velocity(self, angular_vel: np.ndarray) -> np.ndarray:
        """Add gyroscope noise"""
        noise = self.rng.normal(0, self.imu_gyro_noise, size=3)
        return angular_vel + noise
    
    def add_noise_to_altitude(self, altitude: float) -> float:
        """Add barometer noise to altitude"""
        noise = self.rng.normal(0, self.barometer_noise)
        return altitude + noise


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== Domain Randomization Tests ===\n")
    
    # Create nominal parameters
    nominal_drone = DronePhysicsParams(
        mass=1.0,
        thrust_coefficient=1.0,
        drag_coefficient=0.1
    )
    
    nominal_env = EnvironmentParams(
        wind_velocity=np.zeros(3),
        gravity=9.81
    )
    
    # Create randomizer
    randomizer = DomainRandomizer(
        nominal_drone,
        nominal_env,
        randomization_level=1.0,
        seed=42
    )
    
    # Generate multiple episodes
    print("Generated 5 randomized episodes:\n")
    for i in range(5):
        drone, env = randomizer.randomize_episode()
        
        print(f"Episode {i+1}:")
        print(f"  Mass: {drone.mass:.3f} kg ({(drone.mass/nominal_drone.mass - 1)*100:+.1f}%)")
        print(f"  Thrust coeff: {drone.thrust_coefficient:.3f} ({(drone.thrust_coefficient - 1)*100:+.1f}%)")
        print(f"  Drag coeff: {drone.drag_coefficient:.3f}")
        print(f"  Latency: {drone.mavlink_latency*1000:.1f} ms")
        wind_speed = np.linalg.norm(env.wind_velocity)
        print(f"  Wind: {wind_speed:.2f} m/s")
        print()
    
    print("\n=== Latency Simulator Test ===\n")
    
    latency_sim = LatencySimulator(base_latency=0.05, latency_variation=0.02)
    
    # Simulate observations over time
    for t in np.arange(0, 0.5, 0.05):
        obs = np.array([t, t**2, np.sin(t)])  # Dummy observation
        latency_sim.add_observation(obs, t)
        
        delayed_obs = latency_sim.get_delayed_observation(t)
        if delayed_obs is not None:
            delay = t - delayed_obs[0]  # delayed_obs[0] is timestamp
            print(f"t={t:.2f}s: Received obs from t={delayed_obs[0]:.2f}s (delay: {delay*1000:.1f}ms)")
    
    print("\n=== Sensor Noise Test ===\n")
    
    noise_sim = SensorNoiseSimulator(seed=42)
    
    true_position = np.array([10.0, 5.0, -2.0])
    print(f"True position: {true_position}")
    print("Noisy readings:")
    for i in range(5):
        noisy = noise_sim.add_noise_to_position(true_position)
        error = np.linalg.norm(noisy - true_position)
        print(f"  {i+1}: {noisy} (error: {error:.3f}m)")
    
    print("\n=== All tests complete ===")
