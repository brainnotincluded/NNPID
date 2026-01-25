"""Real-time telemetry dashboard."""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Deque
from dataclasses import dataclass, field
import threading
import time

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


@dataclass
class TelemetryBuffer:
    """Ring buffer for telemetry data."""
    
    max_size: int = 500
    
    # Time series data
    timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=500))
    positions: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    velocities: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    euler_angles: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    angular_velocities: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    motor_commands: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=500))
    
    def __post_init__(self):
        """Initialize buffers with correct maxlen."""
        self.timestamps = deque(maxlen=self.max_size)
        self.positions = deque(maxlen=self.max_size)
        self.velocities = deque(maxlen=self.max_size)
        self.euler_angles = deque(maxlen=self.max_size)
        self.angular_velocities = deque(maxlen=self.max_size)
        self.motor_commands = deque(maxlen=self.max_size)
    
    def add(
        self,
        timestamp: float,
        position: np.ndarray,
        velocity: np.ndarray,
        euler_angles: np.ndarray,
        angular_velocity: np.ndarray,
        motor_commands: np.ndarray,
    ) -> None:
        """Add telemetry sample."""
        self.timestamps.append(timestamp)
        self.positions.append(position.copy())
        self.velocities.append(velocity.copy())
        self.euler_angles.append(euler_angles.copy())
        self.angular_velocities.append(angular_velocity.copy())
        self.motor_commands.append(motor_commands.copy())
    
    def clear(self) -> None:
        """Clear all buffers."""
        self.timestamps.clear()
        self.positions.clear()
        self.velocities.clear()
        self.euler_angles.clear()
        self.angular_velocities.clear()
        self.motor_commands.clear()
    
    def to_arrays(self) -> Dict[str, np.ndarray]:
        """Convert buffers to numpy arrays."""
        if len(self.timestamps) == 0:
            return {}
        
        return {
            "timestamp": np.array(self.timestamps),
            "position": np.array(list(self.positions)),
            "velocity": np.array(list(self.velocities)),
            "euler_angles": np.array(list(self.euler_angles)),
            "angular_velocity": np.array(list(self.angular_velocities)),
            "motor_commands": np.array(list(self.motor_commands)),
        }


class TelemetryDashboard:
    """Real-time telemetry visualization dashboard.
    
    Displays live plots of:
    - Position (X, Y, Z)
    - Velocity
    - Attitude (Roll, Pitch, Yaw)
    - Motor commands
    """
    
    def __init__(
        self,
        buffer_size: int = 500,
        update_interval: int = 50,  # ms
    ):
        """Initialize dashboard.
        
        Args:
            buffer_size: Number of samples to display
            update_interval: Plot update interval in milliseconds
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for dashboard")
        
        self.buffer = TelemetryBuffer(max_size=buffer_size)
        self.update_interval = update_interval
        
        self._fig = None
        self._axes = None
        self._lines = {}
        self._animation = None
        self._running = False
        self._lock = threading.Lock()
        
        # Target position for reference
        self._target_position: Optional[np.ndarray] = None
    
    def start(self) -> None:
        """Start the dashboard."""
        self._setup_figure()
        self._running = True
        
        self._animation = FuncAnimation(
            self._fig,
            self._update_plots,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False,
        )
        
        plt.show(block=False)
    
    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False
        if self._animation is not None:
            self._animation.event_source.stop()
        if self._fig is not None:
            plt.close(self._fig)
    
    def _setup_figure(self) -> None:
        """Setup matplotlib figure and axes."""
        self._fig, self._axes = plt.subplots(2, 2, figsize=(14, 10))
        self._fig.suptitle("Drone Telemetry Dashboard", fontsize=14)
        
        # Position plot
        ax = self._axes[0, 0]
        ax.set_title("Position")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        self._lines["pos_x"], = ax.plot([], [], "r-", label="X")
        self._lines["pos_y"], = ax.plot([], [], "g-", label="Y")
        self._lines["pos_z"], = ax.plot([], [], "b-", label="Z")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Velocity plot
        ax = self._axes[0, 1]
        ax.set_title("Velocity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        self._lines["vel_x"], = ax.plot([], [], "r-", label="Vx")
        self._lines["vel_y"], = ax.plot([], [], "g-", label="Vy")
        self._lines["vel_z"], = ax.plot([], [], "b-", label="Vz")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Attitude plot
        ax = self._axes[1, 0]
        ax.set_title("Attitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        self._lines["roll"], = ax.plot([], [], "r-", label="Roll")
        self._lines["pitch"], = ax.plot([], [], "g-", label="Pitch")
        self._lines["yaw"], = ax.plot([], [], "b-", label="Yaw")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Motor commands plot
        ax = self._axes[1, 1]
        ax.set_title("Motor Commands")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Command")
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        for i in range(4):
            self._lines[f"motor_{i}"], = ax.plot(
                [], [], color=colors[i], label=f"M{i+1}"
            )
        ax.legend(loc="upper right")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def _update_plots(self, frame) -> None:
        """Update plot data."""
        if not self._running:
            return
        
        with self._lock:
            data = self.buffer.to_arrays()
        
        if not data:
            return
        
        t = data["timestamp"]
        if len(t) < 2:
            return
        
        t = t - t[0]  # Relative time
        
        pos = data["position"]
        vel = data["velocity"]
        euler = np.degrees(data["euler_angles"])
        motors = data["motor_commands"]
        
        # Update position
        self._lines["pos_x"].set_data(t, pos[:, 0])
        self._lines["pos_y"].set_data(t, pos[:, 1])
        self._lines["pos_z"].set_data(t, pos[:, 2])
        
        # Update velocity
        self._lines["vel_x"].set_data(t, vel[:, 0])
        self._lines["vel_y"].set_data(t, vel[:, 1])
        self._lines["vel_z"].set_data(t, vel[:, 2])
        
        # Update attitude
        self._lines["roll"].set_data(t, euler[:, 0])
        self._lines["pitch"].set_data(t, euler[:, 1])
        self._lines["yaw"].set_data(t, euler[:, 2])
        
        # Update motors
        for i in range(4):
            self._lines[f"motor_{i}"].set_data(t, motors[:, i])
        
        # Adjust axis limits
        for ax in self._axes.flat:
            ax.relim()
            ax.autoscale_view()
        
        self._fig.canvas.draw_idle()
    
    def update(
        self,
        timestamp: float,
        position: np.ndarray,
        velocity: np.ndarray,
        euler_angles: np.ndarray,
        angular_velocity: np.ndarray,
        motor_commands: np.ndarray,
    ) -> None:
        """Add new telemetry data.
        
        Thread-safe method to update dashboard data.
        """
        with self._lock:
            self.buffer.add(
                timestamp=timestamp,
                position=position,
                velocity=velocity,
                euler_angles=euler_angles,
                angular_velocity=angular_velocity,
                motor_commands=motor_commands,
            )
    
    def set_target(self, position: np.ndarray) -> None:
        """Set target position for reference display."""
        self._target_position = position.copy()
    
    def clear(self) -> None:
        """Clear telemetry buffer."""
        with self._lock:
            self.buffer.clear()


def create_summary_plot(
    data: Dict[str, np.ndarray],
    title: str = "Flight Summary",
    save_path: Optional[str] = None,
) -> None:
    """Create summary plot from telemetry data.
    
    Args:
        data: Dictionary with telemetry arrays
        title: Plot title
        save_path: Optional path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14)
    
    t = data.get("timestamp", np.arange(len(data.get("position", []))))
    if len(t) > 0:
        t = t - t[0]
    
    # Position
    if "position" in data:
        ax = axes[0, 0]
        pos = data["position"]
        ax.plot(t, pos[:, 0], "r-", label="X")
        ax.plot(t, pos[:, 1], "g-", label="Y")
        ax.plot(t, pos[:, 2], "b-", label="Z")
        ax.set_title("Position")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (m)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Velocity
    if "velocity" in data:
        ax = axes[0, 1]
        vel = data["velocity"]
        ax.plot(t, vel[:, 0], "r-", label="Vx")
        ax.plot(t, vel[:, 1], "g-", label="Vy")
        ax.plot(t, vel[:, 2], "b-", label="Vz")
        ax.set_title("Velocity")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Attitude
    if "euler_angles" in data:
        ax = axes[1, 0]
        euler = np.degrees(data["euler_angles"])
        ax.plot(t, euler[:, 0], "r-", label="Roll")
        ax.plot(t, euler[:, 1], "g-", label="Pitch")
        ax.plot(t, euler[:, 2], "b-", label="Yaw")
        ax.set_title("Attitude")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Motors
    if "motor_commands" in data:
        ax = axes[1, 1]
        motors = data["motor_commands"]
        for i in range(min(4, motors.shape[1])):
            ax.plot(t, motors[:, i], label=f"M{i+1}")
        ax.set_title("Motor Commands")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Command")
        ax.legend()
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    
    plt.show()
