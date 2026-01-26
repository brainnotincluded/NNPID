"""Real-time telemetry HUD (Heads-Up Display) for drone visualization.

This module provides real-time graphs, gauges, and indicators for
displaying drone telemetry data on rendered frames.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


@dataclass
class TelemetryHUDConfig:
    """Configuration for telemetry HUD."""

    # Layout
    graphs_x: int = 10
    graphs_y: int = 10
    graph_width: int = 180
    graph_height: int = 60
    graph_spacing: int = 10

    # Motor indicator position (bottom right)
    motors_x: int = -200  # Negative = from right edge
    motors_y: int = -150  # Negative = from bottom edge
    motor_bar_width: int = 30
    motor_bar_height: int = 100
    motor_spacing: int = 10

    # Attitude indicator position
    attitude_x: int = -120  # From right
    attitude_y: int = 80  # From top
    attitude_radius: int = 50

    # History length
    history_seconds: float = 5.0
    sample_rate: float = 50.0  # Hz

    # Colors (BGR)
    background_color: tuple = (30, 30, 30)
    border_color: tuple = (80, 80, 80)
    text_color: tuple = (200, 200, 200)
    grid_color: tuple = (50, 50, 50)

    # Graph colors
    roll_color: tuple = (0, 100, 255)  # Orange
    pitch_color: tuple = (0, 255, 100)  # Green
    yaw_color: tuple = (255, 100, 0)  # Blue
    yaw_rate_color: tuple = (255, 0, 255)  # Magenta
    altitude_color: tuple = (0, 255, 255)  # Yellow

    # Motor colors
    motor_colors: tuple = (
        (0, 0, 255),  # Red - M1
        (0, 255, 0),  # Green - M2
        (255, 0, 0),  # Blue - M3
        (0, 255, 255),  # Yellow - M4
    )


class RingBuffer:
    """Efficient ring buffer for time series data."""

    def __init__(self, max_size: int):
        """Initialize ring buffer.

        Args:
            max_size: Maximum number of elements
        """
        self._data = deque(maxlen=max_size)

    def push(self, value: float) -> None:
        """Add value to buffer.

        Args:
            value: Value to add
        """
        self._data.append(value)

    def get_array(self) -> np.ndarray:
        """Get buffer contents as numpy array.

        Returns:
            Array of values
        """
        return np.array(list(self._data))

    def clear(self) -> None:
        """Clear the buffer."""
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class TelemetryGraph:
    """Real-time scrolling graph for a single value."""

    def __init__(
        self,
        name: str,
        color: tuple,
        min_val: float = -1.0,
        max_val: float = 1.0,
        unit: str = "",
        history_size: int = 250,
    ):
        """Initialize telemetry graph.

        Args:
            name: Graph title
            color: Line color (BGR)
            min_val: Minimum Y value
            max_val: Maximum Y value
            unit: Unit string for display
            history_size: Number of samples to keep
        """
        self.name = name
        self.color = color
        self.min_val = min_val
        self.max_val = max_val
        self.unit = unit
        self._buffer = RingBuffer(history_size)
        self._current_value = 0.0

    def push(self, value: float) -> None:
        """Add new value.

        Args:
            value: New data point
        """
        self._current_value = value
        self._buffer.push(value)

    def clear(self) -> None:
        """Clear history."""
        self._buffer.clear()
        self._current_value = 0.0

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        cfg: TelemetryHUDConfig,
    ) -> np.ndarray:
        """Render graph on frame.

        Args:
            frame: Frame to draw on
            x: X position
            y: Y position
            width: Graph width
            height: Graph height
            cfg: HUD configuration

        Returns:
            Modified frame
        """
        if not CV2_AVAILABLE:
            return frame

        # Background
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            cfg.background_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            cfg.border_color,
            1,
        )

        # Grid lines
        mid_y = y + height // 2
        cv2.line(frame, (x, mid_y), (x + width, mid_y), cfg.grid_color, 1)

        # Title and current value
        title_text = f"{self.name}: {self._current_value:+.2f}{self.unit}"
        cv2.putText(
            frame,
            title_text,
            (x + 5, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            cfg.text_color,
            1,
        )

        # Plot data
        data = self._buffer.get_array()
        if len(data) < 2:
            return frame

        # Scale data to pixel coordinates
        plot_height = height - 20  # Leave room for title
        plot_y_start = y + 18

        # Normalize data
        data_range = self.max_val - self.min_val
        if data_range < 0.001:
            data_range = 1.0

        normalized = (data - self.min_val) / data_range
        normalized = np.clip(normalized, 0, 1)

        # Convert to pixel positions
        num_points = len(data)
        x_positions = np.linspace(x + 2, x + width - 2, num_points).astype(int)
        y_positions = (plot_y_start + plot_height * (1 - normalized)).astype(int)

        # Draw line
        points = np.column_stack([x_positions, y_positions]).reshape(-1, 1, 2)
        cv2.polylines(frame, [points], False, self.color, 1, cv2.LINE_AA)

        return frame


class AttitudeIndicator:
    """Circular attitude indicator showing roll/pitch/yaw."""

    def __init__(self):
        """Initialize attitude indicator."""
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0

    def update(self, roll: float, pitch: float, yaw: float) -> None:
        """Update attitude values.

        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            yaw: Yaw angle (radians)
        """
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        radius: int,
        cfg: TelemetryHUDConfig,
    ) -> np.ndarray:
        """Render attitude indicator on frame.

        Args:
            frame: Frame to draw on
            x: Center X position
            y: Center Y position
            radius: Indicator radius
            cfg: HUD configuration

        Returns:
            Modified frame
        """
        if not CV2_AVAILABLE:
            return frame

        # Outer circle
        cv2.circle(frame, (x, y), radius, cfg.border_color, 2)
        cv2.circle(frame, (x, y), radius - 1, cfg.background_color, -1)

        # Horizon line (affected by roll and pitch)
        np.degrees(self._roll)
        pitch_offset = int(self._pitch * radius * 0.5)

        # Calculate horizon line endpoints
        cos_r = np.cos(self._roll)
        sin_r = np.sin(self._roll)

        line_half = radius - 5
        x1 = int(x - line_half * cos_r)
        y1 = int(y + line_half * sin_r + pitch_offset)
        x2 = int(x + line_half * cos_r)
        y2 = int(y - line_half * sin_r + pitch_offset)

        # Sky (above horizon) - blue
        # Ground (below horizon) - brown
        cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 100), 2)

        # Center crosshair (fixed)
        cv2.line(frame, (x - 10, y), (x - 3, y), (0, 255, 255), 2)
        cv2.line(frame, (x + 3, y), (x + 10, y), (0, 255, 255), 2)
        cv2.circle(frame, (x, y), 3, (0, 255, 255), 1)

        # Yaw indicator (arrow at top)
        int(x + (radius + 10) * np.sin(self._yaw))
        cv2.circle(frame, (x, y - radius - 8), 4, cfg.yaw_color, -1)

        # Labels
        cv2.putText(
            frame,
            f"R:{np.degrees(self._roll):+.0f}",
            (x - radius, y + radius + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            cfg.roll_color,
            1,
        )
        cv2.putText(
            frame,
            f"P:{np.degrees(self._pitch):+.0f}",
            (x - radius, y + radius + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            cfg.pitch_color,
            1,
        )
        cv2.putText(
            frame,
            f"Y:{np.degrees(self._yaw):+.0f}",
            (x - radius, y + radius + 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            cfg.yaw_color,
            1,
        )

        return frame


class MotorIndicator:
    """Vertical bar indicators for motor outputs."""

    def __init__(self):
        """Initialize motor indicator."""
        self._motor_values = np.zeros(4)

    def update(self, motor_values: np.ndarray) -> None:
        """Update motor values.

        Args:
            motor_values: Array of 4 motor values (0-1)
        """
        self._motor_values = np.clip(motor_values, 0, 1)

    def render(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        cfg: TelemetryHUDConfig,
    ) -> np.ndarray:
        """Render motor indicators on frame.

        Args:
            frame: Frame to draw on
            x: X position (left edge of indicator group)
            y: Y position (top edge)
            cfg: HUD configuration

        Returns:
            Modified frame
        """
        if not CV2_AVAILABLE:
            return frame

        bar_w = cfg.motor_bar_width
        bar_h = cfg.motor_bar_height
        spacing = cfg.motor_spacing

        4 * bar_w + 3 * spacing

        # Title
        cv2.putText(
            frame,
            "Motors",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            cfg.text_color,
            1,
        )

        for i in range(4):
            bar_x = x + i * (bar_w + spacing)
            color = cfg.motor_colors[i]

            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, y),
                (bar_x + bar_w, y + bar_h),
                cfg.background_color,
                -1,
            )
            cv2.rectangle(
                frame,
                (bar_x, y),
                (bar_x + bar_w, y + bar_h),
                cfg.border_color,
                1,
            )

            # Fill based on value
            fill_height = int(self._motor_values[i] * bar_h)
            if fill_height > 0:
                cv2.rectangle(
                    frame,
                    (bar_x + 1, y + bar_h - fill_height),
                    (bar_x + bar_w - 1, y + bar_h - 1),
                    color,
                    -1,
                )

            # Value label
            cv2.putText(
                frame,
                f"M{i + 1}",
                (bar_x + 2, y + bar_h + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
            )

            # Percentage
            pct = int(self._motor_values[i] * 100)
            cv2.putText(
                frame,
                f"{pct}%",
                (bar_x + 2, y + bar_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                cfg.text_color,
                1,
            )

        return frame


class TelemetryHUD:
    """Complete telemetry HUD combining all indicators."""

    def __init__(self, config: TelemetryHUDConfig | None = None):
        """Initialize telemetry HUD.

        Args:
            config: HUD configuration
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for TelemetryHUD")

        self.config = config or TelemetryHUDConfig()
        cfg = self.config

        history_size = int(cfg.history_seconds * cfg.sample_rate)

        # Create graphs
        self._graphs = {
            "roll": TelemetryGraph("Roll", cfg.roll_color, -45, 45, "°", history_size),
            "pitch": TelemetryGraph("Pitch", cfg.pitch_color, -45, 45, "°", history_size),
            "yaw_rate": TelemetryGraph(
                "Yaw Rate", cfg.yaw_rate_color, -180, 180, "°/s", history_size
            ),
            "altitude": TelemetryGraph("Alt", cfg.altitude_color, 0, 3, "m", history_size),
        }

        # Create indicators
        self._attitude = AttitudeIndicator()
        self._motors = MotorIndicator()

        # Frame dimensions (set on first render)
        self._frame_width = 0
        self._frame_height = 0

    def reset(self) -> None:
        """Reset all graphs and indicators."""
        for graph in self._graphs.values():
            graph.clear()

    def update(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        yaw_rate: float,
        altitude: float,
        motor_values: np.ndarray,
    ) -> None:
        """Update all telemetry values.

        Args:
            roll: Roll angle (radians)
            pitch: Pitch angle (radians)
            yaw: Yaw angle (radians)
            yaw_rate: Yaw rate (radians/s)
            altitude: Altitude (meters)
            motor_values: Motor outputs (0-1)
        """
        # Update graphs (convert to degrees)
        self._graphs["roll"].push(np.degrees(roll))
        self._graphs["pitch"].push(np.degrees(pitch))
        self._graphs["yaw_rate"].push(np.degrees(yaw_rate))
        self._graphs["altitude"].push(altitude)

        # Update indicators
        self._attitude.update(roll, pitch, yaw)
        self._motors.update(motor_values)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render complete HUD on frame.

        Args:
            frame: Input frame

        Returns:
            Frame with HUD overlay
        """
        cfg = self.config
        height, width = frame.shape[:2]
        self._frame_width = width
        self._frame_height = height

        result = frame.copy()

        # Render graphs (top left)
        graph_y = cfg.graphs_y
        for name in ["roll", "pitch", "yaw_rate", "altitude"]:
            graph = self._graphs[name]
            result = graph.render(
                result,
                cfg.graphs_x,
                graph_y,
                cfg.graph_width,
                cfg.graph_height,
                cfg,
            )
            graph_y += cfg.graph_height + cfg.graph_spacing

        # Render attitude indicator (top right)
        att_x = width + cfg.attitude_x
        att_y = cfg.attitude_y
        result = self._attitude.render(result, att_x, att_y, cfg.attitude_radius, cfg)

        # Render motor indicators (bottom right)
        motors_x = width + cfg.motors_x
        motors_y = height + cfg.motors_y
        result = self._motors.render(result, motors_x, motors_y, cfg)

        return result


def create_default_hud() -> TelemetryHUD:
    """Create telemetry HUD with default settings."""
    return TelemetryHUD()


def create_compact_hud() -> TelemetryHUD:
    """Create compact telemetry HUD."""
    config = TelemetryHUDConfig(
        graph_width=140,
        graph_height=45,
        motor_bar_width=20,
        motor_bar_height=70,
        attitude_radius=35,
    )
    return TelemetryHUD(config)
