"""MuJoCo visualization utilities."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None


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
        self._renderer: Optional[mujoco.Renderer] = None
        self._frames = []
        self._recording = False
    
    def launch_interactive(
        self,
        step_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        """Launch interactive viewer.
        
        Args:
            step_callback: Optional callback called each frame
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
    
    def create_renderer(self) -> mujoco.Renderer:
        """Create offscreen renderer."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.height,
                width=self.width,
            )
        return self._renderer
    
    def render_frame(self) -> np.ndarray:
        """Render current frame.
        
        Returns:
            RGB image as numpy array
        """
        renderer = self.create_renderer()
        renderer.update_scene(self.data)
        return renderer.render()
    
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
        lookat: Optional[np.ndarray] = None,
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
        self._target_point: Optional[np.ndarray] = None
    
    def add_point(self, position: np.ndarray) -> None:
        """Add point to trajectory.
        
        Args:
            position: 3D position
        """
        self._trajectory_points.append(position.copy())
        
        # Trim if too long
        if len(self._trajectory_points) > self.max_points:
            self._trajectory_points = self._trajectory_points[-self.max_points:]
    
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
