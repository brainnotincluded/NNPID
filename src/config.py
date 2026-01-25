"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import yaml


@dataclass
class SimulationConfig:
    """Simulation physics configuration."""
    
    timestep: float = 0.002
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    realtime_factor: float = 1.0
    max_episode_steps: int = 5000


@dataclass
class QuadrotorPhysicsConfig:
    """Quadrotor physical properties."""
    
    mass: float = 2.0
    arm_length: float = 0.25
    max_thrust_per_motor: float = 8.0
    min_thrust_per_motor: float = 0.0
    motor_time_constant: float = 0.02
    prop_torque_coefficient: float = 0.01
    
    # Inertia
    ixx: float = 0.029125
    iyy: float = 0.029125
    izz: float = 0.055225
    
    # Motor positions (X-config)
    motor_positions: List[List[float]] = field(default_factory=lambda: [
        [0.1768, 0.1768, 0.0],
        [-0.1768, 0.1768, 0.0],
        [-0.1768, -0.1768, 0.0],
        [0.1768, -0.1768, 0.0],
    ])
    motor_directions: List[int] = field(default_factory=lambda: [1, -1, 1, -1])


@dataclass
class SensorNoiseConfig:
    """Sensor noise parameters."""
    
    # IMU
    imu_rate: float = 500.0
    gyro_noise_std: float = 0.001
    accel_noise_std: float = 0.01
    gyro_bias_std: float = 0.0001
    accel_bias_std: float = 0.001
    
    # GPS
    gps_rate: float = 10.0
    gps_position_noise_std: float = 0.5
    gps_velocity_noise_std: float = 0.1
    
    # Barometer
    baro_rate: float = 50.0
    baro_altitude_noise_std: float = 0.5
    
    # Magnetometer
    mag_rate: float = 100.0
    mag_noise_std: float = 0.01


@dataclass
class PX4Config:
    """PX4 SITL connection configuration."""
    
    host: str = "127.0.0.1"
    port: int = 4560
    lockstep: bool = True
    
    hil_sensor_rate: int = 250
    hil_gps_rate: int = 10
    hil_state_rate: int = 50
    
    connection_timeout: float = 30.0
    heartbeat_interval: float = 1.0
    receive_timeout: float = 0.1
    
    system_id: int = 1
    component_id: int = 1


@dataclass
class TrainingConfig:
    """RL training configuration."""
    
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    seed: int = 42
    
    save_freq: int = 50_000
    checkpoint_path: str = "checkpoints/"
    log_interval: int = 10
    tensorboard_log: str = "logs/tensorboard/"
    eval_freq: int = 25_000
    n_eval_episodes: int = 10
    
    # Network architecture
    pi_layers: List[int] = field(default_factory=lambda: [256, 256])
    vf_layers: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class RewardConfig:
    """Reward function configuration."""
    
    position_weight: float = 1.0
    position_scale: float = 2.0
    velocity_weight: float = 0.1
    angular_rate_weight: float = 0.05
    action_rate_weight: float = 0.01
    orientation_weight: float = 0.1
    alive_bonus: float = 0.1
    crash_penalty: float = -100.0
    success_bonus: float = 10.0


@dataclass
class TerminationConfig:
    """Episode termination conditions."""
    
    max_position_error: float = 10.0
    max_velocity: float = 20.0
    max_tilt_angle: float = 1.57
    min_altitude: float = 0.0
    max_altitude: float = 50.0


@dataclass
class DomainRandomizationConfig:
    """Domain randomization settings."""
    
    enabled: bool = False
    mass_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    inertia_range: List[float] = field(default_factory=lambda: [0.9, 1.1])
    motor_thrust_range: List[float] = field(default_factory=lambda: [0.95, 1.05])
    motor_lag_range: List[float] = field(default_factory=lambda: [0.8, 1.2])


@dataclass
class Config:
    """Master configuration container."""
    
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    quadrotor: QuadrotorPhysicsConfig = field(default_factory=QuadrotorPhysicsConfig)
    sensors: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)
    px4: PX4Config = field(default_factory=PX4Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    domain_randomization: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        # Process quadrotor config - flatten nested inertia
        quadrotor_data = data.get("quadrotor", {}).copy()
        if "inertia" in quadrotor_data and isinstance(quadrotor_data["inertia"], dict):
            inertia = quadrotor_data.pop("inertia")
            quadrotor_data.update(inertia)
        
        # Process sensors config - flatten nested structure
        sensors_data = data.get("sensors", {}).copy()
        flat_sensors = {}
        if "imu" in sensors_data:
            imu = sensors_data.pop("imu")
            flat_sensors["imu_rate"] = imu.get("rate", 500.0)
            flat_sensors["gyro_noise_std"] = imu.get("gyro_noise_std", 0.001)
            flat_sensors["accel_noise_std"] = imu.get("accel_noise_std", 0.01)
            flat_sensors["gyro_bias_std"] = imu.get("gyro_bias_std", 0.0001)
            flat_sensors["accel_bias_std"] = imu.get("accel_bias_std", 0.001)
        if "gps" in sensors_data:
            gps = sensors_data.pop("gps")
            flat_sensors["gps_rate"] = gps.get("rate", 10.0)
            flat_sensors["gps_position_noise_std"] = gps.get("position_noise_std", 0.5)
            flat_sensors["gps_velocity_noise_std"] = gps.get("velocity_noise_std", 0.1)
        if "barometer" in sensors_data:
            baro = sensors_data.pop("barometer")
            flat_sensors["baro_rate"] = baro.get("rate", 50.0)
            flat_sensors["baro_altitude_noise_std"] = baro.get("altitude_noise_std", 0.5)
        if "magnetometer" in sensors_data:
            mag = sensors_data.pop("magnetometer")
            flat_sensors["mag_rate"] = mag.get("rate", 100.0)
            flat_sensors["mag_noise_std"] = mag.get("noise_std", 0.01)
        sensors_data.update(flat_sensors)
        
        return cls(
            simulation=SimulationConfig(**data.get("simulation", {})),
            quadrotor=QuadrotorPhysicsConfig(**quadrotor_data),
            sensors=SensorNoiseConfig(**sensors_data),
            px4=PX4Config(**data.get("px4", {})),
            training=TrainingConfig(**data.get("training", {})),
            reward=RewardConfig(**data.get("reward", {})),
            termination=TerminationConfig(**data.get("termination", {})),
            domain_randomization=DomainRandomizationConfig(
                **data.get("domain_randomization", {})
            ),
        )


def load_config(
    path: Optional[Union[str, Path]] = None,
    defaults_path: Optional[Union[str, Path]] = None,
) -> Config:
    """Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, uses defaults.
        defaults_path: Path to defaults file. Merged with path config.
        
    Returns:
        Loaded configuration
    """
    config_data = {}
    
    # Load defaults first
    if defaults_path is not None:
        defaults_path = Path(defaults_path)
        if defaults_path.exists():
            with open(defaults_path) as f:
                config_data = yaml.safe_load(f) or {}
    
    # Load and merge main config
    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
            config_data = deep_merge(config_data, user_config)
    
    # If no config files, try environment variable
    if not config_data:
        env_config = os.environ.get("NNPID_CONFIG")
        if env_config and Path(env_config).exists():
            with open(env_config) as f:
                config_data = yaml.safe_load(f) or {}
    
    # Create config object
    if config_data:
        return Config.from_dict(config_data)
    else:
        return Config()


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    return Path(__file__).parent.parent / "config" / "default.yaml"


def get_training_config_path() -> Path:
    """Get path to training configuration file."""
    return Path(__file__).parent.parent / "config" / "training.yaml"


def get_px4_config_path() -> Path:
    """Get path to PX4 configuration file."""
    return Path(__file__).parent.parent / "config" / "px4_sitl.yaml"


# Convenience function
def get_config(
    config_name: Optional[str] = None,
    custom_path: Optional[str] = None,
) -> Config:
    """Get configuration by name or path.
    
    Args:
        config_name: Preset name ("default", "training", "px4")
        custom_path: Custom config file path
        
    Returns:
        Loaded configuration
    """
    if custom_path:
        return load_config(path=custom_path)
    
    config_root = Path(__file__).parent.parent / "config"
    
    if config_name == "training":
        return load_config(
            path=config_root / "training.yaml",
            defaults_path=config_root / "default.yaml",
        )
    elif config_name == "px4":
        return load_config(
            path=config_root / "px4_sitl.yaml",
            defaults_path=config_root / "default.yaml",
        )
    else:
        return load_config(path=config_root / "default.yaml")
