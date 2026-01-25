"""Combined MuJoCo overlay system integrating all visualization components.

This module provides a unified interface for rendering all visualization
overlays on MuJoCo frames, including 3D scene objects, neural network
visualization, telemetry HUD, and perturbation effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

from .nn_visualizer import NNVisualizer, NNVisualizerConfig
from .scene_objects import SceneObjectConfig, SceneObjectManager
from .telemetry_hud import TelemetryHUD, TelemetryHUDConfig

if TYPE_CHECKING:
    from ..perturbations import PerturbationManager


@dataclass
class MegaVisualizerConfig:
    """Master configuration for all visualization components."""

    # Component enables
    scene_objects_enabled: bool = True
    nn_visualizer_enabled: bool = True
    telemetry_hud_enabled: bool = True
    perturbation_overlay_enabled: bool = True

    # Frame overlay settings
    status_bar_enabled: bool = True
    timestamp_enabled: bool = True
    fps_counter_enabled: bool = True
    recording_indicator_enabled: bool = True

    # Component configurations
    scene_config: SceneObjectConfig = field(default_factory=SceneObjectConfig)
    nn_config: NNVisualizerConfig = field(default_factory=NNVisualizerConfig)
    hud_config: TelemetryHUDConfig = field(default_factory=TelemetryHUDConfig)

    # Status bar settings
    status_bar_height: int = 25
    status_bar_color: tuple = (40, 40, 40)
    status_text_color: tuple = (200, 200, 200)

    # Recording indicator
    record_dot_color: tuple = (0, 0, 255)  # Red


class FrameAnnotator:
    """Adds status bar and annotations to frames."""

    def __init__(self, config: MegaVisualizerConfig):
        """Initialize frame annotator.

        Args:
            config: Master configuration
        """
        self.config = config
        self._frame_count = 0
        self._fps = 0.0
        self._last_time = 0.0
        self._fps_buffer: list[float] = []
        self._is_recording = False

    def set_recording(self, recording: bool) -> None:
        """Set recording state.

        Args:
            recording: Whether currently recording
        """
        self._is_recording = recording

    def update_fps(self, dt: float) -> None:
        """Update FPS calculation.

        Args:
            dt: Time since last frame
        """
        if dt > 0:
            self._fps_buffer.append(1.0 / dt)
            if len(self._fps_buffer) > 30:
                self._fps_buffer.pop(0)
            self._fps = np.mean(self._fps_buffer)
        self._frame_count += 1

    def render(
        self,
        frame: np.ndarray,
        sim_time: float,
        episode: int = 0,
        step: int = 0,
        reward: float = 0.0,
    ) -> np.ndarray:
        """Render status bar and annotations.

        Args:
            frame: Input frame
            sim_time: Simulation time
            episode: Current episode number
            step: Current step number
            reward: Current reward

        Returns:
            Annotated frame
        """
        if not CV2_AVAILABLE:
            return frame

        cfg = self.config
        height, width = frame.shape[:2]
        result = frame.copy()

        if not cfg.status_bar_enabled:
            return result

        # Draw status bar at bottom
        bar_y = height - cfg.status_bar_height
        cv2.rectangle(
            result,
            (0, bar_y),
            (width, height),
            cfg.status_bar_color,
            -1,
        )

        # Status text
        text_parts = []

        if cfg.timestamp_enabled:
            text_parts.append(f"T:{sim_time:.2f}s")

        text_parts.append(f"Ep:{episode}")
        text_parts.append(f"Step:{step}")
        text_parts.append(f"R:{reward:+.2f}")

        if cfg.fps_counter_enabled:
            text_parts.append(f"FPS:{self._fps:.0f}")

        status_text = " | ".join(text_parts)
        cv2.putText(
            result,
            status_text,
            (10, bar_y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cfg.status_text_color,
            1,
        )

        # Recording indicator
        if cfg.recording_indicator_enabled and self._is_recording:
            # Flashing red dot
            if (self._frame_count // 15) % 2 == 0:
                cv2.circle(
                    result,
                    (width - 20, bar_y + 12),
                    8,
                    cfg.record_dot_color,
                    -1,
                )
            cv2.putText(
                result,
                "REC",
                (width - 55, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                cfg.record_dot_color,
                1,
            )

        return result


class MegaVisualizer:
    """Combined visualizer integrating all visualization components."""

    def __init__(self, config: MegaVisualizerConfig | None = None):
        """Initialize mega visualizer.

        Args:
            config: Master configuration
        """
        self.config = config or MegaVisualizerConfig()
        cfg = self.config

        # Initialize components
        self._scene_objects: SceneObjectManager | None = None
        self._nn_visualizer: NNVisualizer | None = None
        self._telemetry_hud: TelemetryHUD | None = None
        self._annotator = FrameAnnotator(self.config)

        # Create enabled components
        if cfg.scene_objects_enabled:
            self._scene_objects = SceneObjectManager(cfg.scene_config)

        if cfg.nn_visualizer_enabled and CV2_AVAILABLE:
            self._nn_visualizer = NNVisualizer(cfg.nn_config)

        if cfg.telemetry_hud_enabled and CV2_AVAILABLE:
            self._telemetry_hud = TelemetryHUD(cfg.hud_config)

        # Perturbation visualizer (optional import)
        self._perturbation_viz = None
        if cfg.perturbation_overlay_enabled:
            try:
                from ..perturbations.visualization import PerturbationVisualizer

                self._perturbation_viz = PerturbationVisualizer()
            except ImportError:
                pass

        # State
        self._model = None
        self._perturbation_manager = None
        self._sim_time = 0.0
        self._episode = 0
        self._step = 0
        self._last_reward = 0.0

    def set_model(self, model) -> None:
        """Set the neural network model.

        Args:
            model: SB3 model for NN visualization
        """
        self._model = model
        if self._nn_visualizer is not None:
            self._nn_visualizer.set_model(model)

    def set_perturbation_manager(self, manager: PerturbationManager) -> None:
        """Set the perturbation manager.

        Args:
            manager: PerturbationManager instance
        """
        self._perturbation_manager = manager

    def set_recording(self, recording: bool) -> None:
        """Set recording state.

        Args:
            recording: Whether currently recording
        """
        self._annotator.set_recording(recording)

    def reset(self, episode: int = 0) -> None:
        """Reset visualizer for new episode.

        Args:
            episode: Episode number
        """
        self._episode = episode
        self._step = 0
        self._sim_time = 0.0

        if self._scene_objects is not None:
            self._scene_objects.reset()

        if self._telemetry_hud is not None:
            self._telemetry_hud.reset()

    def update_scene(
        self,
        scene,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        yaw_error: float,
        forces: dict[str, np.ndarray] | None = None,
        dt: float = 0.02,
    ) -> int:
        """Update 3D scene objects.

        Args:
            scene: mjvScene to add geometries to
            drone_pos: Drone position
            target_pos: Target position
            yaw_error: Yaw tracking error
            forces: Optional force vectors to display
            dt: Time step

        Returns:
            Number of geometries added
        """
        if self._scene_objects is None:
            return 0

        return self._scene_objects.update(
            scene=scene,
            drone_pos=drone_pos,
            target_pos=target_pos,
            yaw_error=yaw_error,
            perturbation_manager=self._perturbation_manager,
            forces=forces,
            dt=dt,
        )

    def update_telemetry(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        yaw_rate: float,
        altitude: float,
        motor_values: np.ndarray,
    ) -> None:
        """Update telemetry data.

        Args:
            roll: Roll angle
            pitch: Pitch angle
            yaw: Yaw angle
            yaw_rate: Yaw rate
            altitude: Altitude
            motor_values: Motor outputs
        """
        if self._telemetry_hud is not None:
            self._telemetry_hud.update(
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                yaw_rate=yaw_rate,
                altitude=altitude,
                motor_values=motor_values,
            )

    def update_nn(self, observation: np.ndarray, action: np.ndarray | None = None) -> None:
        """Update neural network visualization.

        Args:
            observation: Current observation
            action: Current action
        """
        if self._nn_visualizer is not None:
            self._nn_visualizer.update(observation, action)

    def step(self, dt: float, reward: float = 0.0) -> None:
        """Advance visualization state by one step.

        Args:
            dt: Time step
            reward: Current reward
        """
        self._sim_time += dt
        self._step += 1
        self._last_reward = reward
        self._annotator.update_fps(dt)

    def render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render all 2D overlays on frame.

        Args:
            frame: Input frame (from MuJoCo render)

        Returns:
            Frame with all overlays
        """
        if not CV2_AVAILABLE:
            return frame

        result = frame.copy()

        # 1. Perturbation effects (background layer)
        if self._perturbation_viz is not None and self._perturbation_manager is not None:
            info = self._perturbation_manager.get_info()
            result = self._perturbation_viz.render_overlay(result, info)

        # 2. Telemetry HUD
        if self._telemetry_hud is not None:
            result = self._telemetry_hud.render(result)

        # 3. Neural network visualization
        if self._nn_visualizer is not None:
            result = self._nn_visualizer.render(result)

        # 4. Status bar and annotations (top layer)
        result = self._annotator.render(
            result,
            sim_time=self._sim_time,
            episode=self._episode,
            step=self._step,
            reward=self._last_reward,
        )

        return result

    def render_full(
        self,
        frame: np.ndarray,
        observation: np.ndarray,
        action: np.ndarray | None,
        roll: float,
        pitch: float,
        yaw: float,
        yaw_rate: float,
        altitude: float,
        motor_values: np.ndarray,
        reward: float = 0.0,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Convenience method: update and render everything.

        Args:
            frame: Input frame
            observation: Current observation
            action: Current action
            roll: Roll angle
            pitch: Pitch angle
            yaw: Yaw angle
            yaw_rate: Yaw rate
            altitude: Altitude
            motor_values: Motor outputs
            reward: Current reward
            dt: Time step

        Returns:
            Fully rendered frame
        """
        # Update all components
        self.update_telemetry(roll, pitch, yaw, yaw_rate, altitude, motor_values)
        self.update_nn(observation, action)
        self.step(dt, reward)

        # Render overlay
        return self.render_overlay(frame)


def create_default_visualizer() -> MegaVisualizer:
    """Create mega visualizer with default settings."""
    return MegaVisualizer()


def create_minimal_visualizer() -> MegaVisualizer:
    """Create minimal mega visualizer (HUD only)."""
    config = MegaVisualizerConfig(
        scene_objects_enabled=True,
        nn_visualizer_enabled=False,
        telemetry_hud_enabled=True,
        perturbation_overlay_enabled=False,
    )
    return MegaVisualizer(config)


def create_full_visualizer() -> MegaVisualizer:
    """Create mega visualizer with all features enabled."""
    config = MegaVisualizerConfig(
        scene_objects_enabled=True,
        nn_visualizer_enabled=True,
        telemetry_hud_enabled=True,
        perturbation_overlay_enabled=True,
        status_bar_enabled=True,
        timestamp_enabled=True,
        fps_counter_enabled=True,
        recording_indicator_enabled=True,
    )
    return MegaVisualizer(config)


def create_recording_visualizer() -> MegaVisualizer:
    """Create mega visualizer optimized for video recording."""
    hud_config = TelemetryHUDConfig(
        graph_width=160,
        graph_height=50,
    )
    nn_config = NNVisualizerConfig(
        width=250,
        height=200,
    )
    config = MegaVisualizerConfig(
        scene_objects_enabled=True,
        nn_visualizer_enabled=True,
        telemetry_hud_enabled=True,
        perturbation_overlay_enabled=True,
        status_bar_enabled=True,
        timestamp_enabled=True,
        fps_counter_enabled=False,
        recording_indicator_enabled=True,
        hud_config=hud_config,
        nn_config=nn_config,
    )
    viz = MegaVisualizer(config)
    viz.set_recording(True)
    return viz
