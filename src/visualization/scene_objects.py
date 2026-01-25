"""3D scene objects for MuJoCo visualization.

This module provides classes for adding dynamic 3D geometries to MuJoCo scenes,
including wind arrows, force vectors, trajectory trails, and danger zones.

Uses MuJoCo's mjvGeom system for runtime geometry injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

try:
    import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None

if TYPE_CHECKING:
    from ..perturbations import PerturbationManager


@dataclass
class SceneObjectConfig:
    """Configuration for 3D scene objects."""

    # Wind arrow
    wind_arrow_enabled: bool = True
    wind_arrow_scale: float = 0.3  # Length per m/s
    wind_arrow_offset: tuple = (0.0, 0.0, 0.5)  # Offset above drone
    wind_arrow_radius: float = 0.02

    # Force vectors
    force_vectors_enabled: bool = True
    force_scale: float = 0.1  # Length per Newton
    force_line_width: float = 0.01

    # Trajectory trail
    trail_enabled: bool = True
    trail_max_points: int = 200
    trail_sphere_radius: float = 0.015
    trail_fade: bool = True

    # Target beam
    target_beam_enabled: bool = True
    target_beam_width: float = 0.005

    # VRS danger zone
    vrs_zone_enabled: bool = True
    vrs_zone_radius: float = 0.5
    vrs_zone_height: float = 1.0

    # Colors (RGBA)
    wind_color_calm: tuple = (0.3, 0.6, 1.0, 0.8)
    wind_color_strong: tuple = (1.0, 1.0, 0.0, 0.9)
    wind_color_gust: tuple = (1.0, 0.3, 0.0, 1.0)

    force_color_thrust: tuple = (0.2, 1.0, 0.2, 0.7)
    force_color_drag: tuple = (1.0, 0.5, 0.0, 0.7)
    force_color_wind: tuple = (0.3, 0.7, 1.0, 0.7)
    force_color_external: tuple = (1.0, 0.0, 1.0, 0.7)

    trail_color_old: tuple = (0.3, 0.3, 0.3, 0.3)
    trail_color_new: tuple = (0.0, 1.0, 0.5, 0.8)

    target_color_on: tuple = (0.0, 1.0, 0.0, 0.6)
    target_color_off: tuple = (1.0, 0.0, 0.0, 0.6)

    vrs_color: tuple = (1.0, 0.0, 0.0, 0.3)


class GeomBuilder:
    """Helper class to build MuJoCo geometries."""

    # MuJoCo geometry type constants
    GEOM_PLANE = 0
    GEOM_HFIELD = 1
    GEOM_SPHERE = 2
    GEOM_CAPSULE = 3
    GEOM_ELLIPSOID = 4
    GEOM_CYLINDER = 5
    GEOM_BOX = 6
    GEOM_MESH = 7
    GEOM_ARROW = 100  # Custom for arrow (capsule + cone)
    GEOM_LINE = 103  # Connector line

    @staticmethod
    def init_geom(
        geom,
        geom_type: int,
        size: tuple,
        pos: np.ndarray,
        mat: np.ndarray,
        rgba: tuple,
    ) -> None:
        """Initialize a mjvGeom structure.

        Args:
            geom: mjvGeom to initialize
            geom_type: MuJoCo geometry type
            size: Size parameters (depends on type)
            pos: Position [x, y, z]
            mat: 3x3 rotation matrix (flattened to 9)
            rgba: Color [r, g, b, a]
        """
        if not MUJOCO_AVAILABLE:
            return

        mujoco.mjv_initGeom(
            geom,
            geom_type,
            np.array(size, dtype=np.float64),
            np.array(pos, dtype=np.float64),
            np.array(mat, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32),
        )

    @staticmethod
    def make_connector(
        geom,
        connector_type: int,
        width: float,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        rgba: tuple,
    ) -> None:
        """Create a connector geometry (line/arrow).

        Args:
            geom: mjvGeom to initialize
            connector_type: Connector type (100=line, 101=arrow, etc.)
            width: Line width
            from_pos: Start position
            to_pos: End position
            rgba: Color
        """
        if not MUJOCO_AVAILABLE:
            return

        mujoco.mjv_makeConnector(
            geom,
            connector_type,
            width,
            from_pos[0],
            from_pos[1],
            from_pos[2],
            to_pos[0],
            to_pos[1],
            to_pos[2],
        )
        geom.rgba[:] = rgba

    @staticmethod
    def rotation_matrix_from_direction(direction: np.ndarray) -> np.ndarray:
        """Create rotation matrix that points Z-axis along direction.

        Args:
            direction: Target direction vector (will be normalized)

        Returns:
            3x3 rotation matrix
        """
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.eye(3)

        z_axis = direction / norm

        # Find perpendicular vector
        up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        return np.column_stack([x_axis, y_axis, z_axis])


class WindArrow3D:
    """3D wind direction arrow that hovers above the drone."""

    def __init__(self, config: SceneObjectConfig):
        """Initialize wind arrow.

        Args:
            config: Scene object configuration
        """
        self.config = config
        self._time = 0.0

    def add_to_scene(
        self,
        scene,
        drone_pos: np.ndarray,
        wind_velocity: np.ndarray,
        gust_active: bool = False,
    ) -> int:
        """Add wind arrow geometries to scene.

        Args:
            scene: mjvScene to add to
            drone_pos: Drone position
            wind_velocity: Wind velocity vector [vx, vy, vz]
            gust_active: Whether a gust is currently active

        Returns:
            Number of geometries added
        """
        if not self.config.wind_arrow_enabled or not MUJOCO_AVAILABLE:
            return 0

        wind_speed = np.linalg.norm(wind_velocity)
        if wind_speed < 0.1:
            return 0

        cfg = self.config
        num_added = 0

        # Arrow position (above drone)
        arrow_base = drone_pos + np.array(cfg.wind_arrow_offset)

        # Arrow direction and length
        wind_dir = wind_velocity / wind_speed
        arrow_length = wind_speed * cfg.wind_arrow_scale
        arrow_base + wind_dir * arrow_length

        # Color based on wind intensity
        if gust_active:
            color = cfg.wind_color_gust
        elif wind_speed > 5.0:
            color = cfg.wind_color_strong
        else:
            # Interpolate between calm and strong
            t = min(1.0, wind_speed / 5.0)
            color = tuple(
                c1 * (1 - t) + c2 * t
                for c1, c2 in zip(cfg.wind_color_calm, cfg.wind_color_strong, strict=True)
            )

        # Add arrow shaft (capsule)
        if scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            shaft_end = arrow_base + wind_dir * (arrow_length * 0.7)

            GeomBuilder.make_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                cfg.wind_arrow_radius,
                arrow_base,
                shaft_end,
                color,
            )
            scene.ngeom += 1
            num_added += 1

        # Add arrow head (cone approximated as sphere)
        if scene.ngeom < scene.maxgeom:
            geom = scene.geoms[scene.ngeom]
            head_pos = arrow_base + wind_dir * (arrow_length * 0.85)

            GeomBuilder.init_geom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (cfg.wind_arrow_radius * 2.5, 0, 0),
                head_pos,
                np.eye(3),
                color,
            )
            scene.ngeom += 1
            num_added += 1

        return num_added


class ForceVectors3D:
    """3D force vectors emanating from the drone."""

    def __init__(self, config: SceneObjectConfig):
        """Initialize force vectors.

        Args:
            config: Scene object configuration
        """
        self.config = config

    def add_to_scene(
        self,
        scene,
        drone_pos: np.ndarray,
        forces: dict[str, np.ndarray],
    ) -> int:
        """Add force vector geometries to scene.

        Args:
            scene: mjvScene to add to
            drone_pos: Drone center of mass position
            forces: Dictionary of force name -> force vector

        Returns:
            Number of geometries added
        """
        if not self.config.force_vectors_enabled or not MUJOCO_AVAILABLE:
            return 0

        cfg = self.config
        num_added = 0

        # Color mapping
        color_map = {
            "thrust": cfg.force_color_thrust,
            "drag": cfg.force_color_drag,
            "wind": cfg.force_color_wind,
            "external": cfg.force_color_external,
        }

        for force_name, force_vec in forces.items():
            force_mag = np.linalg.norm(force_vec)
            if force_mag < 0.1:
                continue

            if scene.ngeom >= scene.maxgeom:
                break

            # Scale force for visualization
            end_pos = drone_pos + force_vec * cfg.force_scale

            # Get color
            color = color_map.get(force_name, (1.0, 1.0, 1.0, 0.7))

            geom = scene.geoms[scene.ngeom]
            GeomBuilder.make_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                cfg.force_line_width,
                drone_pos,
                end_pos,
                color,
            )
            scene.ngeom += 1
            num_added += 1

        return num_added


class TrajectoryTrail3D:
    """3D trajectory trail showing recent drone path."""

    def __init__(self, config: SceneObjectConfig):
        """Initialize trajectory trail.

        Args:
            config: Scene object configuration
        """
        self.config = config
        self._points: list[np.ndarray] = []
        self._max_points = config.trail_max_points

    def update(self, position: np.ndarray) -> None:
        """Add a new point to the trail.

        Args:
            position: Current drone position
        """
        self._points.append(position.copy())
        if len(self._points) > self._max_points:
            self._points.pop(0)

    def clear(self) -> None:
        """Clear the trail."""
        self._points.clear()

    def add_to_scene(self, scene) -> int:
        """Add trail geometries to scene.

        Args:
            scene: mjvScene to add to

        Returns:
            Number of geometries added
        """
        if not self.config.trail_enabled or not MUJOCO_AVAILABLE:
            return 0

        if len(self._points) < 2:
            return 0

        cfg = self.config
        num_added = 0
        num_points = len(self._points)

        # Add spheres at each point with color gradient
        step = max(1, num_points // 50)  # Limit to ~50 spheres
        for i in range(0, num_points, step):
            if scene.ngeom >= scene.maxgeom:
                break

            point = self._points[i]
            t = i / max(1, num_points - 1)  # 0 = oldest, 1 = newest

            # Interpolate color
            if cfg.trail_fade:
                color = tuple(
                    c1 * (1 - t) + c2 * t
                    for c1, c2 in zip(cfg.trail_color_old, cfg.trail_color_new, strict=True)
                )
            else:
                color = cfg.trail_color_new

            # Vary size slightly
            radius = cfg.trail_sphere_radius * (0.5 + 0.5 * t)

            geom = scene.geoms[scene.ngeom]
            GeomBuilder.init_geom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                (radius, 0, 0),
                point,
                np.eye(3),
                color,
            )
            scene.ngeom += 1
            num_added += 1

        return num_added


class TargetBeam3D:
    """3D beam from drone to target showing tracking status."""

    def __init__(self, config: SceneObjectConfig):
        """Initialize target beam.

        Args:
            config: Scene object configuration
        """
        self.config = config
        self._pulse_phase = 0.0

    def add_to_scene(
        self,
        scene,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        yaw_error: float,
        dt: float = 0.02,
    ) -> int:
        """Add target beam geometry to scene.

        Args:
            scene: mjvScene to add to
            drone_pos: Drone position
            target_pos: Target position
            yaw_error: Yaw error in radians (for color)
            dt: Time step for animation

        Returns:
            Number of geometries added
        """
        if not self.config.target_beam_enabled or not MUJOCO_AVAILABLE:
            return 0

        if scene.ngeom >= scene.maxgeom:
            return 0

        cfg = self.config

        # Update pulse animation
        self._pulse_phase += dt * 5.0
        pulse = 0.7 + 0.3 * np.sin(self._pulse_phase)

        # Color based on tracking accuracy
        error_normalized = min(1.0, abs(yaw_error) / np.pi)
        color = tuple(
            (c1 * (1 - error_normalized) + c2 * error_normalized) * pulse
            if i < 3
            else c1 * (1 - error_normalized) + c2 * error_normalized
            for i, (c1, c2) in enumerate(
                zip(cfg.target_color_on, cfg.target_color_off, strict=True)
            )
        )

        # Draw beam from drone to target
        geom = scene.geoms[scene.ngeom]
        GeomBuilder.make_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            cfg.target_beam_width * (1 + 0.5 * (1 - error_normalized)),
            drone_pos,
            target_pos,
            color,
        )
        scene.ngeom += 1

        return 1


class VRSDangerZone3D:
    """3D danger zone indicator for Vortex Ring State."""

    def __init__(self, config: SceneObjectConfig):
        """Initialize VRS danger zone.

        Args:
            config: Scene object configuration
        """
        self.config = config
        self._flash_phase = 0.0

    def add_to_scene(
        self,
        scene,
        drone_pos: np.ndarray,
        vrs_severity: float,
        dt: float = 0.02,
    ) -> int:
        """Add VRS danger zone geometry to scene.

        Args:
            scene: mjvScene to add to
            drone_pos: Drone position
            vrs_severity: VRS severity (0-1)
            dt: Time step for animation

        Returns:
            Number of geometries added
        """
        if not self.config.vrs_zone_enabled or not MUJOCO_AVAILABLE:
            return 0

        if vrs_severity < 0.01:
            return 0

        if scene.ngeom >= scene.maxgeom:
            return 0

        cfg = self.config

        # Flash animation based on severity
        self._flash_phase += dt * (5 + 10 * vrs_severity)
        flash = 0.5 + 0.5 * np.sin(self._flash_phase)

        # Zone below drone
        zone_center = drone_pos - np.array([0, 0, cfg.vrs_zone_height / 2])

        # Color with flashing alpha
        color = (
            cfg.vrs_color[0],
            cfg.vrs_color[1],
            cfg.vrs_color[2],
            cfg.vrs_color[3] * vrs_severity * flash,
        )

        # Draw as cylinder
        geom = scene.geoms[scene.ngeom]
        GeomBuilder.init_geom(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            (cfg.vrs_zone_radius * vrs_severity, cfg.vrs_zone_height / 2, 0),
            zone_center,
            np.eye(3),
            color,
        )
        scene.ngeom += 1

        return 1


class SceneObjectManager:
    """Manages all 3D scene objects for visualization."""

    def __init__(self, config: SceneObjectConfig | None = None):
        """Initialize scene object manager.

        Args:
            config: Configuration for all scene objects
        """
        self.config = config or SceneObjectConfig()

        self.wind_arrow = WindArrow3D(self.config)
        self.force_vectors = ForceVectors3D(self.config)
        self.trajectory_trail = TrajectoryTrail3D(self.config)
        self.target_beam = TargetBeam3D(self.config)
        self.vrs_zone = VRSDangerZone3D(self.config)

        self._time = 0.0

    def reset(self) -> None:
        """Reset all scene objects."""
        self.trajectory_trail.clear()
        self._time = 0.0

    def update(
        self,
        scene,
        drone_pos: np.ndarray,
        target_pos: np.ndarray,
        yaw_error: float,
        perturbation_manager: PerturbationManager | None = None,
        forces: dict[str, np.ndarray] | None = None,
        dt: float = 0.02,
    ) -> int:
        """Update and add all scene objects.

        Args:
            scene: mjvScene to add geometries to
            drone_pos: Current drone position
            target_pos: Current target position
            yaw_error: Current yaw error
            perturbation_manager: Optional perturbation manager for wind/VRS info
            forces: Optional dictionary of force vectors
            dt: Time step

        Returns:
            Total number of geometries added
        """
        self._time += dt
        num_added = 0

        # Update trajectory
        self.trajectory_trail.update(drone_pos)

        # Add trajectory trail
        num_added += self.trajectory_trail.add_to_scene(scene)

        # Add target beam
        num_added += self.target_beam.add_to_scene(scene, drone_pos, target_pos, yaw_error, dt)

        # Add force vectors
        if forces:
            num_added += self.force_vectors.add_to_scene(scene, drone_pos, forces)

        # Add perturbation-related objects
        if perturbation_manager is not None:
            info = perturbation_manager.get_info()
            perturbations = info.get("perturbations", {})

            # Wind arrow
            wind_info = perturbations.get("wind", {})
            if wind_info.get("enabled", False):
                wind_vel = np.array(wind_info.get("wind_velocity", [0, 0, 0]))
                gust_active = wind_info.get("gust_active", False)
                num_added += self.wind_arrow.add_to_scene(scene, drone_pos, wind_vel, gust_active)

            # VRS zone
            aero_info = perturbations.get("aerodynamics", {})
            if aero_info.get("in_vrs", False):
                vrs_severity = aero_info.get("vrs_severity", 0)
                num_added += self.vrs_zone.add_to_scene(scene, drone_pos, vrs_severity, dt)

        return num_added


def create_default_scene_objects() -> SceneObjectManager:
    """Create scene object manager with default settings."""
    return SceneObjectManager()


def create_minimal_scene_objects() -> SceneObjectManager:
    """Create scene object manager with minimal objects."""
    config = SceneObjectConfig(
        force_vectors_enabled=False,
        trail_enabled=True,
        trail_max_points=50,
        vrs_zone_enabled=False,
    )
    return SceneObjectManager(config)


def create_full_scene_objects() -> SceneObjectManager:
    """Create scene object manager with all objects enabled."""
    config = SceneObjectConfig(
        wind_arrow_enabled=True,
        force_vectors_enabled=True,
        trail_enabled=True,
        trail_max_points=300,
        target_beam_enabled=True,
        vrs_zone_enabled=True,
    )
    return SceneObjectManager(config)
