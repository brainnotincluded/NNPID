"""Telemetry logging utilities."""

from __future__ import annotations

import numpy as np
import json
import time as time_module
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime


@dataclass
class TelemetryFrame:
    """Single frame of telemetry data."""
    
    timestamp: float  # Simulation time
    wall_time: float  # Real wall clock time
    
    # State
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    euler_angles: np.ndarray  # [roll, pitch, yaw]
    
    # Control
    motor_commands: np.ndarray
    
    # Sensors (optional)
    gyro: Optional[np.ndarray] = None
    accel: Optional[np.ndarray] = None
    
    # Reference/target (optional)
    target_position: Optional[np.ndarray] = None
    target_velocity: Optional[np.ndarray] = None
    
    # Reward (for RL)
    reward: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "timestamp": self.timestamp,
            "wall_time": self.wall_time,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "quaternion": self.quaternion.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
            "euler_angles": self.euler_angles.tolist(),
            "motor_commands": self.motor_commands.tolist(),
        }
        
        if self.gyro is not None:
            d["gyro"] = self.gyro.tolist()
        if self.accel is not None:
            d["accel"] = self.accel.tolist()
        if self.target_position is not None:
            d["target_position"] = self.target_position.tolist()
        if self.target_velocity is not None:
            d["target_velocity"] = self.target_velocity.tolist()
        if self.reward is not None:
            d["reward"] = self.reward
        
        return d


class TelemetryLogger:
    """Log and save telemetry data from simulation runs."""
    
    def __init__(
        self,
        log_dir: Optional[str | Path] = None,
        max_frames: int = 100000,
        auto_save: bool = True,
    ):
        """Initialize telemetry logger.
        
        Args:
            log_dir: Directory to save logs. If None, uses "logs/telemetry/"
            max_frames: Maximum frames to keep in memory
            auto_save: Whether to auto-save on episode end
        """
        self.log_dir = Path(log_dir) if log_dir else Path("logs/telemetry")
        self.max_frames = max_frames
        self.auto_save = auto_save
        
        self._frames: List[TelemetryFrame] = []
        self._episode_count = 0
        self._start_time = time_module.time()
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_frame(
        self,
        timestamp: float,
        position: np.ndarray,
        velocity: np.ndarray,
        quaternion: np.ndarray,
        angular_velocity: np.ndarray,
        motor_commands: np.ndarray,
        euler_angles: Optional[np.ndarray] = None,
        gyro: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        target_velocity: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
    ) -> None:
        """Log a single telemetry frame.
        
        Args:
            timestamp: Simulation time
            position: Current position
            velocity: Current velocity
            quaternion: Current orientation
            angular_velocity: Current angular velocity
            motor_commands: Current motor commands
            euler_angles: Euler angles (computed if not provided)
            gyro: Gyroscope reading
            accel: Accelerometer reading
            target_position: Target/reference position
            target_velocity: Target/reference velocity
            reward: Reward value (for RL)
        """
        if euler_angles is None:
            # Compute from quaternion
            w, x, y, z = quaternion
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (w * y - z * x)
            sinp = np.clip(sinp, -1.0, 1.0)
            pitch = np.arcsin(sinp)
            
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            euler_angles = np.array([roll, pitch, yaw])
        
        frame = TelemetryFrame(
            timestamp=timestamp,
            wall_time=time_module.time() - self._start_time,
            position=position.copy(),
            velocity=velocity.copy(),
            quaternion=quaternion.copy(),
            angular_velocity=angular_velocity.copy(),
            euler_angles=euler_angles.copy(),
            motor_commands=motor_commands.copy(),
            gyro=gyro.copy() if gyro is not None else None,
            accel=accel.copy() if accel is not None else None,
            target_position=target_position.copy() if target_position is not None else None,
            target_velocity=target_velocity.copy() if target_velocity is not None else None,
            reward=reward,
        )
        
        self._frames.append(frame)
        
        # Trim if too many frames
        if len(self._frames) > self.max_frames:
            self._frames = self._frames[-self.max_frames:]
    
    def get_episode_data(self) -> Dict[str, np.ndarray]:
        """Get episode data as numpy arrays.
        
        Returns:
            Dictionary with arrays for each logged quantity
        """
        if not self._frames:
            return {}
        
        n = len(self._frames)
        
        data = {
            "timestamp": np.zeros(n),
            "wall_time": np.zeros(n),
            "position": np.zeros((n, 3)),
            "velocity": np.zeros((n, 3)),
            "quaternion": np.zeros((n, 4)),
            "angular_velocity": np.zeros((n, 3)),
            "euler_angles": np.zeros((n, 3)),
            "motor_commands": np.zeros((n, 4)),
        }
        
        # Check for optional fields
        has_gyro = self._frames[0].gyro is not None
        has_accel = self._frames[0].accel is not None
        has_target_pos = self._frames[0].target_position is not None
        has_reward = self._frames[0].reward is not None
        
        if has_gyro:
            data["gyro"] = np.zeros((n, 3))
        if has_accel:
            data["accel"] = np.zeros((n, 3))
        if has_target_pos:
            data["target_position"] = np.zeros((n, 3))
        if has_reward:
            data["reward"] = np.zeros(n)
        
        for i, frame in enumerate(self._frames):
            data["timestamp"][i] = frame.timestamp
            data["wall_time"][i] = frame.wall_time
            data["position"][i] = frame.position
            data["velocity"][i] = frame.velocity
            data["quaternion"][i] = frame.quaternion
            data["angular_velocity"][i] = frame.angular_velocity
            data["euler_angles"][i] = frame.euler_angles
            data["motor_commands"][i] = frame.motor_commands
            
            if has_gyro and frame.gyro is not None:
                data["gyro"][i] = frame.gyro
            if has_accel and frame.accel is not None:
                data["accel"][i] = frame.accel
            if has_target_pos and frame.target_position is not None:
                data["target_position"][i] = frame.target_position
            if has_reward and frame.reward is not None:
                data["reward"][i] = frame.reward
        
        return data
    
    def save_episode(self, filename: Optional[str] = None) -> Path:
        """Save current episode to file.
        
        Args:
            filename: Output filename. If None, auto-generates.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{self._episode_count:04d}_{timestamp}.npz"
        
        filepath = self.log_dir / filename
        
        data = self.get_episode_data()
        np.savez_compressed(filepath, **data)
        
        return filepath
    
    def save_episode_json(self, filename: Optional[str] = None) -> Path:
        """Save current episode to JSON file.
        
        Args:
            filename: Output filename. If None, auto-generates.
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{self._episode_count:04d}_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        frames_data = [frame.to_dict() for frame in self._frames]
        
        with open(filepath, 'w') as f:
            json.dump(frames_data, f, indent=2)
        
        return filepath
    
    def end_episode(self) -> Optional[Path]:
        """End current episode and optionally save.
        
        Returns:
            Path to saved file if auto_save is True
        """
        self._episode_count += 1
        
        saved_path = None
        if self.auto_save and self._frames:
            saved_path = self.save_episode()
        
        self.reset()
        return saved_path
    
    def reset(self) -> None:
        """Clear logged frames for new episode."""
        self._frames = []
        self._start_time = time_module.time()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics from logged data.
        
        Returns:
            Dictionary of statistics
        """
        if not self._frames:
            return {}
        
        data = self.get_episode_data()
        
        stats = {
            "episode_length": len(self._frames),
            "duration_sim": data["timestamp"][-1] - data["timestamp"][0],
            "duration_wall": data["wall_time"][-1] - data["wall_time"][0],
        }
        
        # Position statistics
        pos = data["position"]
        stats["position_mean"] = pos.mean(axis=0).tolist()
        stats["position_std"] = pos.std(axis=0).tolist()
        stats["position_range"] = (pos.max(axis=0) - pos.min(axis=0)).tolist()
        
        # Velocity statistics
        vel = data["velocity"]
        stats["velocity_mean"] = vel.mean(axis=0).tolist()
        stats["velocity_max"] = np.abs(vel).max(axis=0).tolist()
        
        # Motor statistics
        motors = data["motor_commands"]
        stats["motor_mean"] = motors.mean(axis=0).tolist()
        stats["motor_std"] = motors.std(axis=0).tolist()
        
        # Reward statistics (if available)
        if "reward" in data:
            stats["total_reward"] = data["reward"].sum()
            stats["mean_reward"] = data["reward"].mean()
        
        return stats
    
    @property
    def frame_count(self) -> int:
        """Get number of logged frames."""
        return len(self._frames)
    
    @property
    def episode_number(self) -> int:
        """Get current episode number."""
        return self._episode_count
