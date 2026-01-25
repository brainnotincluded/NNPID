"""Visualization for perturbation effects in MuJoCo simulation.

This module provides visual overlays and effects to show active perturbations
during simulation rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

if TYPE_CHECKING:
    from .base import PerturbationManager


@dataclass
class VisualizationConfig:
    """Configuration for perturbation visualization."""

    # Enable/disable components
    show_wind_arrow: bool = True
    show_force_arrow: bool = True
    show_status_panel: bool = True
    show_wind_particles: bool = True
    show_ground_effect: bool = True
    show_vrs_warning: bool = True

    # Colors (BGR format for OpenCV)
    wind_color: tuple = (255, 200, 100)  # Light blue
    force_color: tuple = (0, 100, 255)  # Orange-red
    gust_color: tuple = (0, 255, 255)  # Yellow
    vrs_color: tuple = (0, 0, 255)  # Red
    ground_effect_color: tuple = (100, 255, 100)  # Light green
    panel_bg_color: tuple = (40, 40, 40)  # Dark gray
    text_color: tuple = (255, 255, 255)  # White

    # Sizes
    arrow_thickness: int = 2
    arrow_scale: float = 15.0  # Pixels per m/s for wind
    force_arrow_scale: float = 10.0  # Pixels per Newton
    particle_count: int = 20
    particle_size: int = 2
    panel_width: int = 220
    panel_padding: int = 10
    font_scale: float = 0.5
    line_height: int = 20


class WindParticle:
    """A single wind particle for visualization."""

    def __init__(self, bounds: tuple, rng: np.random.Generator):
        """Initialize particle at random position.

        Args:
            bounds: (width, height) of the frame
            rng: Random number generator
        """
        self.x = rng.uniform(0, bounds[0])
        self.y = rng.uniform(0, bounds[1])
        self.lifetime = rng.uniform(0.5, 2.0)
        self.age = 0.0
        self.size = rng.integers(1, 4)
        self.alpha = rng.uniform(0.3, 0.8)

    def update(self, dt: float, wind_velocity: np.ndarray, frame_size: tuple) -> bool:
        """Update particle position.

        Args:
            dt: Time step
            wind_velocity: Wind velocity [vx, vy, vz]
            frame_size: (width, height) of frame

        Returns:
            True if particle is still alive
        """
        # Move with wind (project to screen space)
        speed_scale = 50.0  # Pixels per m/s
        self.x += wind_velocity[0] * speed_scale * dt
        self.y -= wind_velocity[1] * speed_scale * dt  # Y is inverted in screen space

        self.age += dt

        # Wrap around screen
        width, height = frame_size
        if self.x < 0:
            self.x += width
        elif self.x > width:
            self.x -= width
        if self.y < 0:
            self.y += height
        elif self.y > height:
            self.y -= height

        return self.age < self.lifetime


class PerturbationVisualizer:
    """Visualizes perturbation effects on rendered frames.

    Adds overlays showing:
    - Wind direction and speed (arrow + particles)
    - External forces (arrow)
    - Status panel with active perturbations
    - Ground effect indicator
    - VRS warning
    """

    def __init__(self, config: VisualizationConfig | None = None):
        """Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for perturbation visualization")

        self.config = config or VisualizationConfig()
        self._rng = np.random.default_rng()

        # Wind particles
        self._particles: list[WindParticle] = []
        self._last_frame_size: tuple = (0, 0)

        # Animation state
        self._time = 0.0
        self._gust_flash_timer = 0.0

    def reset(self) -> None:
        """Reset visualization state."""
        self._particles.clear()
        self._time = 0.0
        self._gust_flash_timer = 0.0

    def render_overlay(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
        drone_position: np.ndarray,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Render perturbation effects overlay on frame.

        Args:
            frame: Input RGB frame
            manager: PerturbationManager with current state
            drone_position: Current drone position [x, y, z]
            dt: Time step for animations

        Returns:
            Frame with overlays added
        """
        self._time += dt
        height, width = frame.shape[:2]
        frame_size = (width, height)

        # Update frame size for particles
        if frame_size != self._last_frame_size:
            self._particles.clear()
            self._last_frame_size = frame_size

        # Make a copy to draw on
        result = frame.copy()

        # Get perturbation info
        info = manager.get_info()

        # Draw components based on config
        cfg = self.config

        if cfg.show_wind_particles:
            result = self._draw_wind_particles(result, manager, dt)

        if cfg.show_wind_arrow:
            result = self._draw_wind_arrow(result, manager)

        if cfg.show_force_arrow:
            result = self._draw_force_arrow(result, manager)

        if cfg.show_ground_effect:
            result = self._draw_ground_effect(result, manager, drone_position)

        if cfg.show_vrs_warning:
            result = self._draw_vrs_warning(result, manager)

        if cfg.show_status_panel:
            result = self._draw_status_panel(result, manager, info)

        return result

    def _draw_wind_arrow(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
    ) -> np.ndarray:
        """Draw wind direction arrow."""
        cfg = self.config
        height, width = frame.shape[:2]

        # Get wind info
        wind = manager.get_perturbation("wind")
        if wind is None or not wind.enabled:
            return frame

        wind_info = wind.get_info()
        wind_vel = np.array(wind_info.get("wind_velocity", [0, 0, 0]))
        wind_speed = wind_info.get("wind_speed", 0)

        if wind_speed < 0.1:
            return frame

        # Arrow position (top-right area)
        center_x = width - 80
        center_y = 80

        # Draw compass circle
        radius = 50
        cv2.circle(frame, (center_x, center_y), radius, (100, 100, 100), 1)
        cv2.circle(frame, (center_x, center_y), 3, cfg.wind_color, -1)

        # Draw N/S/E/W labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            frame, "N", (center_x - 5, center_y - radius - 5), font, 0.4, (150, 150, 150), 1
        )
        cv2.putText(
            frame, "E", (center_x + radius + 5, center_y + 4), font, 0.4, (150, 150, 150), 1
        )

        # Calculate arrow direction (project XY to screen)
        arrow_len = min(radius - 5, wind_speed * cfg.arrow_scale)
        angle = np.arctan2(wind_vel[1], wind_vel[0])

        end_x = int(center_x + arrow_len * np.cos(angle))
        end_y = int(center_y - arrow_len * np.sin(angle))  # Y inverted

        # Draw arrow with gust flash effect
        gust_active = wind_info.get("gust_active", False)
        if gust_active:
            self._gust_flash_timer = 0.3
            color = cfg.gust_color
        elif self._gust_flash_timer > 0:
            self._gust_flash_timer -= 0.02
            # Blend between gust and normal color
            t = self._gust_flash_timer / 0.3
            color = tuple(
                int(c1 * t + c2 * (1 - t))
                for c1, c2 in zip(cfg.gust_color, cfg.wind_color, strict=True)
            )
        else:
            color = cfg.wind_color

        cv2.arrowedLine(
            frame,
            (center_x, center_y),
            (end_x, end_y),
            color,
            cfg.arrow_thickness,
            tipLength=0.3,
        )

        # Speed label
        speed_text = f"{wind_speed:.1f} m/s"
        cv2.putText(
            frame,
            speed_text,
            (center_x - 30, center_y + radius + 20),
            font,
            0.5,
            color,
            1,
        )

        # Gust indicator
        if gust_active:
            cv2.putText(
                frame,
                "GUST",
                (center_x - 20, center_y + radius + 40),
                font,
                0.5,
                cfg.gust_color,
                2,
            )

        return frame

    def _draw_wind_particles(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
        dt: float,
    ) -> np.ndarray:
        """Draw wind particles floating across screen."""
        cfg = self.config
        height, width = frame.shape[:2]

        wind = manager.get_perturbation("wind")
        if wind is None or not wind.enabled:
            return frame

        wind_info = wind.get_info()
        wind_vel = np.array(wind_info.get("wind_velocity", [0, 0, 0]))
        wind_speed = wind_info.get("wind_speed", 0)

        if wind_speed < 0.5:
            # Clear particles in calm conditions
            self._particles = [
                p for p in self._particles if p.update(dt, wind_vel, (width, height))
            ]
            return frame

        # Spawn new particles
        spawn_rate = min(cfg.particle_count, int(wind_speed * 5))
        while len(self._particles) < spawn_rate:
            self._particles.append(WindParticle((width, height), self._rng))

        # Update and draw particles
        alive_particles = []
        for particle in self._particles:
            if particle.update(dt, wind_vel, (width, height)):
                alive_particles.append(particle)

                # Draw particle as small dot or line
                x, y = int(particle.x), int(particle.y)
                alpha = particle.alpha * (1 - particle.age / particle.lifetime)

                # Streak based on wind speed
                streak_len = int(wind_speed * 3)
                angle = np.arctan2(wind_vel[1], wind_vel[0])
                x2 = int(x - streak_len * np.cos(angle))
                y2 = int(y + streak_len * np.sin(angle))

                # Color with alpha
                color = tuple(int(c * alpha) for c in cfg.wind_color)
                cv2.line(frame, (x, y), (x2, y2), color, particle.size)

        self._particles = alive_particles

        return frame

    def _draw_force_arrow(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
    ) -> np.ndarray:
        """Draw total external force arrow at drone center."""
        cfg = self.config
        height, width = frame.shape[:2]

        total_force = manager.get_total_force()
        force_magnitude = np.linalg.norm(total_force)

        if force_magnitude < 0.1:
            return frame

        # Force arrow in bottom-left corner
        center_x = 80
        center_y = height - 80

        # Draw reference circle
        cv2.circle(frame, (center_x, center_y), 40, (80, 80, 80), 1)
        cv2.circle(frame, (center_x, center_y), 3, cfg.force_color, -1)

        # Calculate arrow (project XY force)
        arrow_len = min(35, force_magnitude * cfg.force_arrow_scale)
        angle = np.arctan2(total_force[1], total_force[0])

        end_x = int(center_x + arrow_len * np.cos(angle))
        end_y = int(center_y - arrow_len * np.sin(angle))

        # Color intensity based on force magnitude
        intensity = min(1.0, force_magnitude / 5.0)
        color = tuple(int(c * (0.5 + 0.5 * intensity)) for c in cfg.force_color)

        cv2.arrowedLine(
            frame,
            (center_x, center_y),
            (end_x, end_y),
            color,
            cfg.arrow_thickness + 1,
            tipLength=0.3,
        )

        # Vertical force indicator (bar)
        bar_x = center_x + 55
        bar_height = 60
        bar_y_center = center_y

        # Background bar
        cv2.rectangle(
            frame,
            (bar_x, bar_y_center - bar_height // 2),
            (bar_x + 10, bar_y_center + bar_height // 2),
            (60, 60, 60),
            -1,
        )

        # Force indicator
        z_force = total_force[2]
        z_normalized = np.clip(z_force / 5.0, -1, 1)
        z_bar_height = int(abs(z_normalized) * bar_height // 2)

        if z_force > 0:
            # Upward force - green
            cv2.rectangle(
                frame,
                (bar_x, bar_y_center - z_bar_height),
                (bar_x + 10, bar_y_center),
                (100, 255, 100),
                -1,
            )
        else:
            # Downward force - red
            cv2.rectangle(
                frame,
                (bar_x, bar_y_center),
                (bar_x + 10, bar_y_center + z_bar_height),
                (100, 100, 255),
                -1,
            )

        # Center line
        cv2.line(
            frame,
            (bar_x - 2, bar_y_center),
            (bar_x + 12, bar_y_center),
            (200, 200, 200),
            1,
        )

        # Label
        cv2.putText(
            frame,
            f"F: {force_magnitude:.1f}N",
            (center_x - 25, center_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            cfg.force_color,
            1,
        )

        return frame

    def _draw_ground_effect(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
        drone_position: np.ndarray,
    ) -> np.ndarray:
        """Draw ground effect indicator."""
        cfg = self.config
        height, width = frame.shape[:2]

        physics = manager.get_perturbation("physics")
        if physics is None or not physics.enabled:
            return frame

        physics_info = physics.get_info()
        if not physics_info.get("ground_effect_active", False):
            return frame

        # Draw ground effect indicator at bottom center
        altitude = drone_position[2]
        indicator_y = height - 30

        # Pulsing effect
        pulse = 0.7 + 0.3 * np.sin(self._time * 5)
        color = tuple(int(c * pulse) for c in cfg.ground_effect_color)

        # Draw wave-like effect
        wave_width = 200
        wave_x_start = width // 2 - wave_width // 2

        points = []
        for i in range(wave_width):
            x = wave_x_start + i
            wave = 5 * np.sin((i / 20.0) + self._time * 3) * (1 - altitude / 0.5)
            y = int(indicator_y + wave)
            points.append((x, y))

        if len(points) > 1:
            points_arr = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [points_arr], False, color, 2)

        # Text
        cv2.putText(
            frame,
            f"GROUND EFFECT ({altitude:.2f}m)",
            (width // 2 - 80, indicator_y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        return frame

    def _draw_vrs_warning(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
    ) -> np.ndarray:
        """Draw Vortex Ring State warning."""
        cfg = self.config
        height, width = frame.shape[:2]

        aero = manager.get_perturbation("aerodynamics")
        if aero is None or not aero.enabled:
            return frame

        aero_info = aero.get_info()
        if not aero_info.get("in_vrs", False):
            return frame

        severity = aero_info.get("vrs_severity", 0)

        # Flashing red warning
        flash = 0.5 + 0.5 * np.sin(self._time * 10)
        color = tuple(int(c * flash) for c in cfg.vrs_color)

        # Warning box
        box_width = 300
        box_height = 60
        box_x = (width - box_width) // 2
        box_y = 50

        # Red border (pulsing)
        border_thickness = int(2 + 3 * severity)
        cv2.rectangle(
            frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            color,
            border_thickness,
        )

        # Background
        cv2.rectangle(
            frame,
            (box_x + border_thickness, box_y + border_thickness),
            (box_x + box_width - border_thickness, box_y + box_height - border_thickness),
            (20, 20, 40),
            -1,
        )

        # Warning text
        cv2.putText(
            frame,
            "! VORTEX RING STATE !",
            (box_x + 40, box_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"Severity: {severity * 100:.0f}% - INCREASE FORWARD SPEED",
            (box_x + 25, box_y + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        return frame

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        manager: PerturbationManager,
        info: dict[str, Any],
    ) -> np.ndarray:
        """Draw status panel showing active perturbations."""
        cfg = self.config
        height, width = frame.shape[:2]

        # Panel position (left side)
        panel_x = cfg.panel_padding
        panel_y = cfg.panel_padding

        # Count active perturbations and prepare lines
        lines = []
        lines.append(("PERTURBATIONS", cfg.text_color, True))

        if not info.get("enabled", True):
            lines.append(("  [DISABLED]", (100, 100, 100), False))
        else:
            intensity = info.get("global_intensity", 1.0)
            lines.append((f"  Intensity: {intensity * 100:.0f}%", (150, 150, 150), False))

            perturbations = info.get("perturbations", {})

            for name, p_info in perturbations.items():
                if p_info.get("enabled", False):
                    # Color code by type
                    if name == "wind":
                        color = cfg.wind_color
                        speed = p_info.get("wind_speed", 0)
                        status = f"{speed:.1f}m/s"
                        if p_info.get("gust_active", False):
                            status += " GUST"
                    elif name == "delays":
                        color = (255, 200, 150)  # Light orange
                        imu_delay = p_info.get("effective_imu_delay_ms", 0)
                        status = f"{imu_delay:.0f}ms"
                    elif name == "sensor_noise":
                        color = (200, 150, 255)  # Light purple
                        status = "active"
                        if p_info.get("gps_lost", False):
                            status = "GPS LOST"
                            color = (0, 0, 255)
                    elif name == "physics":
                        color = (150, 255, 200)  # Light cyan
                        status = "active"
                    elif name == "aerodynamics":
                        color = (255, 150, 150)  # Light red
                        if p_info.get("in_vrs", False):
                            status = "VRS!"
                            color = cfg.vrs_color
                        else:
                            drag = p_info.get("drag_magnitude", 0)
                            status = f"drag {drag:.1f}N"
                    elif name == "external_forces":
                        color = (200, 200, 150)  # Light yellow
                        impulses = p_info.get("active_impulses", 0)
                        status = f"{impulses} impulses"
                    else:
                        color = cfg.text_color
                        status = "active"

                    # Icon based on perturbation type
                    icon = self._get_perturbation_icon(name)
                    lines.append((f"  {icon} {name}: {status}", color, False))

        # Calculate panel size
        panel_height = len(lines) * cfg.line_height + cfg.panel_padding * 2

        # Draw panel background
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + cfg.panel_width, panel_y + panel_height),
            cfg.panel_bg_color,
            -1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + cfg.panel_width, panel_y + panel_height),
            (100, 100, 100),
            1,
        )

        # Draw lines
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = panel_y + cfg.panel_padding + 15

        for text, color, is_header in lines:
            thickness = 2 if is_header else 1
            scale = cfg.font_scale * 1.1 if is_header else cfg.font_scale

            cv2.putText(
                frame,
                text,
                (panel_x + cfg.panel_padding, y),
                font,
                scale,
                color,
                thickness,
            )
            y += cfg.line_height

        return frame

    def _get_perturbation_icon(self, name: str) -> str:
        """Get text icon for perturbation type."""
        icons = {
            "wind": "~",
            "delays": "t",
            "sensor_noise": "*",
            "physics": "#",
            "aerodynamics": "^",
            "external_forces": "!",
        }
        return icons.get(name, ">")


def create_default_visualizer() -> PerturbationVisualizer:
    """Create visualizer with default settings."""
    return PerturbationVisualizer()


def create_minimal_visualizer() -> PerturbationVisualizer:
    """Create visualizer showing only essential info."""
    config = VisualizationConfig(
        show_wind_particles=False,
        show_ground_effect=False,
        show_force_arrow=False,
    )
    return PerturbationVisualizer(config)


def create_full_visualizer() -> PerturbationVisualizer:
    """Create visualizer with all effects enabled."""
    config = VisualizationConfig(
        show_wind_arrow=True,
        show_force_arrow=True,
        show_status_panel=True,
        show_wind_particles=True,
        show_ground_effect=True,
        show_vrs_warning=True,
        particle_count=30,
    )
    return PerturbationVisualizer(config)
