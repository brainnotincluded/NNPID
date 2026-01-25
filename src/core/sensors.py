"""Sensor simulation for quadrotor."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .mujoco_sim import QuadrotorState


@dataclass
class IMUData:
    """IMU sensor readings."""
    
    gyro: np.ndarray  # Angular velocity [rad/s], body frame (3,)
    accel: np.ndarray  # Linear acceleration [m/s²], body frame (3,)
    timestamp: float = 0.0  # Simulation time [s]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "gyro": self.gyro.tolist(),
            "accel": self.accel.tolist(),
            "timestamp": self.timestamp,
        }


@dataclass
class GPSData:
    """GPS sensor readings."""
    
    latitude: float = 0.0  # degrees
    longitude: float = 0.0  # degrees
    altitude: float = 0.0  # meters (AMSL)
    velocity_north: float = 0.0  # m/s
    velocity_east: float = 0.0  # m/s
    velocity_down: float = 0.0  # m/s
    ground_speed: float = 0.0  # m/s
    course_over_ground: float = 0.0  # degrees
    hdop: float = 1.0  # Horizontal dilution of precision
    vdop: float = 1.0  # Vertical dilution of precision
    satellites_visible: int = 10
    fix_type: int = 3  # 0-1: no fix, 2: 2D fix, 3: 3D fix
    timestamp: float = 0.0  # Simulation time [s]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "lat": self.latitude,
            "lon": self.longitude,
            "alt": self.altitude,
            "vn": self.velocity_north,
            "ve": self.velocity_east,
            "vd": self.velocity_down,
            "ground_speed": self.ground_speed,
            "cog": self.course_over_ground,
            "hdop": self.hdop,
            "vdop": self.vdop,
            "satellites": self.satellites_visible,
            "fix_type": self.fix_type,
            "timestamp": self.timestamp,
        }


@dataclass
class BarometerData:
    """Barometer sensor readings."""
    
    pressure: float = 101325.0  # Pascal (sea level)
    temperature: float = 20.0  # Celsius
    altitude: float = 0.0  # meters (pressure altitude)
    timestamp: float = 0.0


@dataclass
class MagnetometerData:
    """Magnetometer sensor readings."""
    
    field: np.ndarray = field(default_factory=lambda: np.array([0.21, 0.0, 0.42]))  # Gauss
    timestamp: float = 0.0


@dataclass
class SensorConfig:
    """Configuration for sensor simulation."""
    
    # IMU parameters
    imu_rate: float = 500.0  # Hz
    gyro_noise_std: float = 0.001  # rad/s
    accel_noise_std: float = 0.01  # m/s²
    gyro_bias_std: float = 0.0001  # rad/s per sqrt(s)
    accel_bias_std: float = 0.001  # m/s² per sqrt(s)
    
    # GPS parameters
    gps_rate: float = 10.0  # Hz
    gps_position_noise_std: float = 0.5  # meters
    gps_velocity_noise_std: float = 0.1  # m/s
    
    # Barometer parameters
    baro_rate: float = 50.0  # Hz
    baro_altitude_noise_std: float = 0.5  # meters
    
    # Magnetometer parameters
    mag_rate: float = 100.0  # Hz
    mag_noise_std: float = 0.01  # Gauss
    
    # Reference location for GPS
    reference_latitude: float = 47.397742  # CMAC (ArduPilot default)
    reference_longitude: float = 8.545594
    reference_altitude: float = 488.0  # meters AMSL


class SensorSimulator:
    """Simulates realistic sensor readings from quadrotor state.
    
    Adds noise, bias, and realistic timing to sensor data.
    Converts between coordinate frames as needed.
    """
    
    # Earth constants
    EARTH_RADIUS = 6371000.0  # meters
    
    def __init__(self, config: Optional[SensorConfig] = None, seed: Optional[int] = None):
        """Initialize sensor simulator.
        
        Args:
            config: Sensor configuration. Uses defaults if None.
            seed: Random seed for reproducibility.
        """
        self.config = config or SensorConfig()
        self.rng = np.random.default_rng(seed)
        
        # Initialize biases
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        
        # Timing
        self._last_imu_time = 0.0
        self._last_gps_time = 0.0
        self._last_baro_time = 0.0
        self._last_mag_time = 0.0
        
        # Earth's magnetic field (approximate, NED frame)
        # Typical values for mid-latitudes
        self._mag_field_ned = np.array([0.21, 0.0, 0.42])  # Gauss
    
    def reset(self) -> None:
        """Reset sensor state (biases, timing)."""
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._last_imu_time = 0.0
        self._last_gps_time = 0.0
        self._last_baro_time = 0.0
        self._last_mag_time = 0.0
    
    def get_imu(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        timestamp: float,
    ) -> IMUData:
        """Get IMU reading with noise and bias.
        
        Args:
            gyro: True angular velocity [rad/s], body frame
            accel: True acceleration [m/s²], body frame
            timestamp: Simulation time [s]
            
        Returns:
            IMU sensor data with noise
        """
        cfg = self.config
        dt = timestamp - self._last_imu_time
        self._last_imu_time = timestamp
        
        # Update biases (random walk)
        if dt > 0:
            self._gyro_bias += self.rng.normal(0, cfg.gyro_bias_std * np.sqrt(dt), 3)
            self._accel_bias += self.rng.normal(0, cfg.accel_bias_std * np.sqrt(dt), 3)
        
        # Add noise and bias
        noisy_gyro = gyro + self._gyro_bias + self.rng.normal(0, cfg.gyro_noise_std, 3)
        noisy_accel = accel + self._accel_bias + self.rng.normal(0, cfg.accel_noise_std, 3)
        
        return IMUData(
            gyro=noisy_gyro,
            accel=noisy_accel,
            timestamp=timestamp,
        )
    
    def get_gps(
        self,
        position_ned: np.ndarray,
        velocity_ned: np.ndarray,
        timestamp: float,
    ) -> GPSData:
        """Get GPS reading from NED position and velocity.
        
        Args:
            position_ned: Position in NED frame [m]
            velocity_ned: Velocity in NED frame [m/s]
            timestamp: Simulation time [s]
            
        Returns:
            GPS sensor data
        """
        cfg = self.config
        
        # Add position noise
        noisy_position = position_ned + self.rng.normal(0, cfg.gps_position_noise_std, 3)
        
        # Convert NED to lat/lon/alt
        lat = cfg.reference_latitude + np.degrees(noisy_position[0] / self.EARTH_RADIUS)
        lon = cfg.reference_longitude + np.degrees(
            noisy_position[1] / (self.EARTH_RADIUS * np.cos(np.radians(cfg.reference_latitude)))
        )
        alt = cfg.reference_altitude - noisy_position[2]  # NED Z is down
        
        # Add velocity noise
        noisy_velocity = velocity_ned + self.rng.normal(0, cfg.gps_velocity_noise_std, 3)
        
        # Compute ground speed and course
        ground_speed = np.sqrt(noisy_velocity[0]**2 + noisy_velocity[1]**2)
        if ground_speed > 0.1:
            cog = np.degrees(np.arctan2(noisy_velocity[1], noisy_velocity[0]))
            if cog < 0:
                cog += 360
        else:
            cog = 0.0
        
        return GPSData(
            latitude=lat,
            longitude=lon,
            altitude=alt,
            velocity_north=noisy_velocity[0],
            velocity_east=noisy_velocity[1],
            velocity_down=noisy_velocity[2],
            ground_speed=ground_speed,
            course_over_ground=cog,
            hdop=1.0 + self.rng.uniform(0, 0.5),
            vdop=1.5 + self.rng.uniform(0, 0.5),
            satellites_visible=10 + int(self.rng.integers(-2, 3)),
            fix_type=3,
            timestamp=timestamp,
        )
    
    def get_barometer(
        self,
        altitude: float,
        timestamp: float,
    ) -> BarometerData:
        """Get barometer reading from altitude.
        
        Args:
            altitude: Altitude above reference [m]
            timestamp: Simulation time [s]
            
        Returns:
            Barometer sensor data
        """
        cfg = self.config
        
        # Add noise
        noisy_alt = altitude + self.rng.normal(0, cfg.baro_altitude_noise_std)
        
        # Convert altitude to pressure (barometric formula)
        # P = P0 * (1 - L*h/T0)^(g*M/(R*L))
        # Simplified: P ≈ P0 * exp(-h/8500)
        sea_level_pressure = 101325.0  # Pa
        pressure = sea_level_pressure * np.exp(-noisy_alt / 8500.0)
        
        # Temperature (simple lapse rate)
        temperature = 15.0 - 0.0065 * noisy_alt
        
        return BarometerData(
            pressure=pressure,
            temperature=temperature,
            altitude=noisy_alt,
            timestamp=timestamp,
        )
    
    def get_magnetometer(
        self,
        quaternion: np.ndarray,
        timestamp: float,
    ) -> MagnetometerData:
        """Get magnetometer reading from attitude.
        
        Args:
            quaternion: Body orientation [w, x, y, z]
            timestamp: Simulation time [s]
            
        Returns:
            Magnetometer sensor data
        """
        cfg = self.config
        
        # Rotate Earth's magnetic field to body frame
        # R_body_to_world * mag_body = mag_world
        # mag_body = R_world_to_body * mag_world
        
        w, x, y, z = quaternion
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ])
        
        # R is body-to-world, we need world-to-body
        R_inv = R.T
        
        mag_body = R_inv @ self._mag_field_ned
        
        # Add noise
        noisy_mag = mag_body + self.rng.normal(0, cfg.mag_noise_std, 3)
        
        return MagnetometerData(
            field=noisy_mag,
            timestamp=timestamp,
        )
    
    def should_update_imu(self, timestamp: float) -> bool:
        """Check if IMU should be updated based on rate."""
        dt = timestamp - self._last_imu_time
        return dt >= 1.0 / self.config.imu_rate
    
    def should_update_gps(self, timestamp: float) -> bool:
        """Check if GPS should be updated based on rate."""
        dt = timestamp - self._last_gps_time
        return dt >= 1.0 / self.config.gps_rate
    
    def should_update_baro(self, timestamp: float) -> bool:
        """Check if barometer should be updated based on rate."""
        dt = timestamp - self._last_baro_time
        return dt >= 1.0 / self.config.baro_rate
    
    def should_update_mag(self, timestamp: float) -> bool:
        """Check if magnetometer should be updated based on rate."""
        dt = timestamp - self._last_mag_time
        return dt >= 1.0 / self.config.mag_rate
    
    def update_timing(self, sensor_type: str, timestamp: float) -> None:
        """Update last update time for a sensor.
        
        Args:
            sensor_type: One of "imu", "gps", "baro", "mag"
            timestamp: Current simulation time
        """
        if sensor_type == "imu":
            self._last_imu_time = timestamp
        elif sensor_type == "gps":
            self._last_gps_time = timestamp
        elif sensor_type == "baro":
            self._last_baro_time = timestamp
        elif sensor_type == "mag":
            self._last_mag_time = timestamp
