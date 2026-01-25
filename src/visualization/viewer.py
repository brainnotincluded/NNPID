"""MuJoCo visualization utilities.

This module provides wrappers for MuJoCo visualization including:
- Interactive viewing with camera controls
- Offscreen rendering for video creation
- Integration with MegaVisualizer for overlay effects
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import mujoco
    import mujoco.viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

if TYPE_CHECKING:
    from .mujoco_overlay import MegaVisualizer
    from .scene_objects import SceneObjectManager


class MuJoCoViewer:
    """Wrapper for MuJoCo visualization.

    Provides easy-to-use interface for:
    - Interactive viewing
    - Offscreen rendering
    - Video recording
    """

    def __init__(
        self,
        model,
        data,
        width: int = 1280,
        height: int = 720,
    ):
        """Initialize viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Render width
            height: Render height
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo is required for visualization")

        self.model = model
        self.data = data
        self.width = width
        self.height = height

        self._viewer = None
        self._renderer: mujoco.Renderer | None = None
        self._frames: list[np.ndarray] = []
        self._recording = False

        # Mega visualization integration
        self._mega_visualizer: MegaVisualizer | None = None
        self._scene_objects: SceneObjectManager | None = None
        self._overlay_enabled = True

    def launch_interactive(
        self,
        step_callback: Callable[[], None] | None = None,
        key_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Launch interactive viewer.

        Args:
            step_callback: Optional callback called each frame
            key_callback: Optional callback for key presses
        """
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self._viewer = viewer

            while viewer.is_running():
                if step_callback is not None:
                    step_callback()

                if self._recording:
                    frame = self.render_frame()
                    self._frames.append(frame)

                viewer.sync()

            self._viewer = None

    def launch_with_overlay(
        self,
        step_callback: Callable[[], dict[str, Any]] | None = None,
        fps: int = 30,
    ) -> None:
        """Launch viewer with OpenCV overlay window.

        This provides a separate window with MegaVisualizer overlays
        rendered on top of the MuJoCo scene.

        Args:
            step_callback: Callback that returns state dict for visualization
            fps: Target frame rate
        """
        import time

        if not CV2_AVAILABLE:
            print("Warning: OpenCV not available, using standard viewer")
            self.launch_interactive(step_callback)
            return

        window_name = "MuJoCo + Overlay"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.width, self.height)

        running = True
        frame_time = 1.0 / fps

        while running:
            start = time.time()

            # Call step callback
            step_callback() if step_callback is not None else {}

            # Render frame with overlay
            frame = self.render_frame(apply_overlay=True)

            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, frame_bgr)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running = False
            elif key == ord("r") and self._recording:
                self._frames.append(frame)
            elif key == ord("s"):
                # Screenshot
                cv2.imwrite(f"screenshot_{int(time.time())}.png", frame_bgr)

            # Frame rate limiting
            elapsed = time.time() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        cv2.destroyWindow(window_name)

    def set_mega_visualizer(self, visualizer: MegaVisualizer) -> None:
        """Set mega visualizer for overlay effects.

        Args:
            visualizer: MegaVisualizer instance
        """
        self._mega_visualizer = visualizer

    def set_scene_objects(self, scene_objects: SceneObjectManager) -> None:
        """Set scene object manager for 3D objects.

        Args:
            scene_objects: SceneObjectManager instance
        """
        self._scene_objects = scene_objects

    def enable_overlay(self, enabled: bool = True) -> None:
        """Enable or disable overlay rendering.

        Args:
            enabled: Whether to render overlays
        """
        self._overlay_enabled = enabled

    def create_renderer(self) -> mujoco.Renderer:
        """Create offscreen renderer."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.height,
                width=self.width,
            )
        return self._renderer

    def render_frame(
        self,
        apply_overlay: bool = True,
    ) -> np.ndarray:
        """Render current frame with optional overlays.

        Args:
            apply_overlay: Whether to apply MegaVisualizer overlay

        Returns:
            RGB image as numpy array
        """
        renderer = self.create_renderer()
        renderer.update_scene(self.data)
        frame = renderer.render()

        # Apply overlay if enabled
        if apply_overlay and self._overlay_enabled and self._mega_visualizer is not None:
            frame = self._mega_visualizer.render_overlay(frame)

        return frame

    def render_frame_with_scene_objects(
        self,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        yaw_error: float,
        forces: dict[str, np.ndarray] | None = None,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Render frame with 3D scene objects added.

        This method injects 3D geometries into the scene before rendering.

        Args:
            drone_pos: Current drone position
            target_pos: Current target position
            yaw_error: Yaw tracking error
            forces: Optional force vectors to display
            dt: Time step

        Returns:
            RGB image as numpy array
        """
        renderer = self.create_renderer()

        # Get scene for geometry injection
        scene = renderer._scene

        # Add scene objects if available
        if self._scene_objects is not None:
            self._scene_objects.update(
                scene=scene,
                drone_pos=drone_pos,
                target_pos=target_pos,
                yaw_error=yaw_error,
                perturbation_manager=(
                    self._mega_visualizer._perturbation_manager if self._mega_visualizer else None
                ),
                forces=forces,
                dt=dt,
            )

        renderer.update_scene(self.data)
        frame = renderer.render()

        # Apply 2D overlay
        if self._overlay_enabled and self._mega_visualizer is not None:
            frame = self._mega_visualizer.render_overlay(frame)

        return frame

    def start_recording(self) -> None:
        """Start recording frames."""
        self._recording = True
        self._frames = []

    def stop_recording(self) -> list:
        """Stop recording and return frames.

        Returns:
            List of recorded frames
        """
        self._recording = False
        frames = self._frames
        self._frames = []
        return frames

    def save_video(
        self,
        output_path: str | Path,
        fps: int = 30,
    ) -> bool:
        """Save recorded frames as video.

        Args:
            output_path: Output video path
            fps: Frames per second

        Returns:
            True if saved successfully
        """
        if not self._frames:
            print("No frames to save")
            return False

        try:
            import imageio

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            imageio.mimsave(str(output_path), self._frames, fps=fps)
            print(f"Saved video to {output_path}")
            return True

        except ImportError:
            print("imageio required for video saving")
            return False

    def save_screenshot(
        self,
        output_path: str | Path,
    ) -> bool:
        """Save current frame as image.

        Args:
            output_path: Output image path

        Returns:
            True if saved successfully
        """
        try:
            import imageio

            frame = self.render_frame()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            imageio.imwrite(str(output_path), frame)
            print(f"Saved screenshot to {output_path}")
            return True

        except ImportError:
            print("imageio required for image saving")
            return False

    def set_camera(
        self,
        distance: float = 5.0,
        azimuth: float = 45.0,
        elevation: float = -30.0,
        lookat: np.ndarray | None = None,
    ) -> None:
        """Set camera position.

        Args:
            distance: Distance from target
            azimuth: Horizontal angle (degrees)
            elevation: Vertical angle (degrees)
            lookat: Camera target point
        """
        if self._viewer is not None:
            cam = self._viewer.cam
            cam.distance = distance
            cam.azimuth = azimuth
            cam.elevation = elevation

            if lookat is not None:
                cam.lookat[:] = lookat

    def track_body(self, body_name: str) -> None:
        """Set camera to track a body.

        Args:
            body_name: Name of body to track
        """
        if self._viewer is not None:
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if body_id >= 0:
                self._viewer.cam.trackbodyid = body_id
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None


class TrajectoryVisualizer:
    """Visualize trajectories in MuJoCo."""

    def __init__(self, model, data, max_points: int = 1000):
        """Initialize trajectory visualizer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            max_points: Maximum trajectory points to store
        """
        self.model = model
        self.data = data
        self.max_points = max_points

        self._trajectory_points: list = []
        self._target_point: np.ndarray | None = None

    def add_point(self, position: np.ndarray) -> None:
        """Add point to trajectory.

        Args:
            position: 3D position
        """
        self._trajectory_points.append(position.copy())

        # Trim if too long
        if len(self._trajectory_points) > self.max_points:
            self._trajectory_points = self._trajectory_points[-self.max_points :]

    def set_target(self, position: np.ndarray) -> None:
        """Set target visualization point.

        Args:
            position: 3D target position
        """
        self._target_point = position.copy()

    def clear(self) -> None:
        """Clear trajectory."""
        self._trajectory_points = []
        self._target_point = None

    def get_trajectory_array(self) -> np.ndarray:
        """Get trajectory as numpy array.

        Returns:
            Array of shape (N, 3)
        """
        if not self._trajectory_points:
            return np.zeros((0, 3))
        return np.array(self._trajectory_points)
