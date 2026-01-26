"""MuJoCo simulation wrapper for quadrotor dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class QuadrotorState:
    """Complete state of the quadrotor."""

    position: np.ndarray  # (3,) world frame position [x, y, z]
    velocity: np.ndarray  # (3,) world frame velocity [vx, vy, vz]
    quaternion: np.ndarray  # (4,) orientation [w, x, y, z]
    angular_velocity: np.ndarray  # (3,) body frame angular velocity [p, q, r]
    motor_speeds: np.ndarray  # (4,) normalized motor commands [0, 1]

    def to_array(self) -> np.ndarray:
        """Flatten state to a single array."""
        return np.concatenate(
            [
                self.position,
                self.velocity,
                self.quaternion,
                self.angular_velocity,
                self.motor_speeds,
            ]
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> QuadrotorState:
        """Create state from flattened array."""
        return cls(
            position=arr[0:3],
            velocity=arr[3:6],
            quaternion=arr[6:10],
            angular_velocity=arr[10:13],
            motor_speeds=arr[13:17],
        )


class MuJoCoSimulator:
    """MuJoCo-based quadrotor physics simulator.

    Wraps MuJoCo model and data structures, providing a clean interface
    for stepping physics, applying motor commands, and reading state.

    Attributes:
        model: MuJoCo model (read-only physics parameters)
        data: MuJoCo data (mutable simulation state)
        timestep: Physics timestep in seconds
    """

    # Default model path relative to package
    DEFAULT_MODEL = "models/quadrotor_x500.xml"

    def __init__(
        self,
        model_path: str | Path | None = None,
        timestep: float | None = None,
    ):
        """Initialize the simulator.

        Args:
            model_path: Path to MuJoCo XML model. If None, uses default X500 model.
            timestep: Override model timestep. If None, uses model default.
        """
        # Resolve model path
        if model_path is None:
            # Use default model from package
            package_root = Path(__file__).parent.parent.parent
            model_path = package_root / self.DEFAULT_MODEL
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # Override timestep if specified
        if timestep is not None:
            self.model.opt.timestep = timestep

        self.timestep = self.model.opt.timestep

        # Cache body and actuator indices
        self._body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor")
        self._motor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor{i}")
            for i in range(1, 5)
        ]

        # Sensor indices
        self._gyro_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        ]
        self._accel_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "accel")
        ]
        self._pos_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "pos")
        ]
        self._quat_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        ]
        self._linvel_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "linvel")
        ]
        self._angvel_adr = self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angvel")
        ]

        # Initial state backup for reset
        self._initial_qpos = self.data.qpos.copy()
        self._initial_qvel = self.data.qvel.copy()

        # Motor command state
        self._motor_commands = np.zeros(4)

        # External force/torque for perturbations
        self._external_force = np.zeros(3)
        self._external_torque = np.zeros(3)

    def reset(
        self,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
        quaternion: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
    ) -> QuadrotorState:
        """Reset simulation to initial or specified state.

        Args:
            position: Initial position [x, y, z]. Default: [0, 0, 0.5]
            velocity: Initial velocity [vx, vy, vz]. Default: [0, 0, 0]
            quaternion: Initial orientation [w, x, y, z]. Default: [1, 0, 0, 0]
            angular_velocity: Initial angular velocity [p, q, r]. Default: [0, 0, 0]

        Returns:
            Initial quadrotor state
        """
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)

        # Set position (qpos[0:3] for freejoint)
        if position is not None:
            self.data.qpos[0:3] = position
        else:
            self.data.qpos[0:3] = [0.0, 0.0, 0.5]

        # Set orientation (qpos[3:7] for freejoint quaternion)
        if quaternion is not None:
            self.data.qpos[3:7] = quaternion
        else:
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        # Set velocity (qvel[0:3] for linear, qvel[3:6] for angular)
        if velocity is not None:
            self.data.qvel[0:3] = velocity

        if angular_velocity is not None:
            self.data.qvel[3:6] = angular_velocity

        # Reset motor commands
        self._motor_commands = np.zeros(4)
        self.data.ctrl[:] = 0.0

        # Reset external forces
        self._external_force = np.zeros(3)
        self._external_torque = np.zeros(3)
        self.data.xfrc_applied[self._body_id] = np.zeros(6)

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.model, self.data)

        return self.get_state()

    def step(self, motor_commands: np.ndarray) -> QuadrotorState:
        """Step simulation with given motor commands.

        Args:
            motor_commands: Normalized motor commands [0, 1] for each motor.
                           Shape: (4,)

        Returns:
            Updated quadrotor state
        """
        # Clip commands to valid range
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        self._motor_commands = motor_commands.copy()

        # Apply motor commands to actuators
        self.data.ctrl[:4] = motor_commands

        # Apply external forces/torques (from perturbations)
        # xfrc_applied format: [fx, fy, fz, tx, ty, tz]
        self.data.xfrc_applied[self._body_id, :3] = self._external_force
        self.data.xfrc_applied[self._body_id, 3:6] = self._external_torque

        # Step physics
        mujoco.mj_step(self.model, self.data)

        return self.get_state()

    def step_multiple(self, motor_commands: np.ndarray, n_steps: int) -> QuadrotorState:
        """Step simulation multiple times with same motor commands.

        Useful for lower-frequency control loops.

        Args:
            motor_commands: Normalized motor commands [0, 1]
            n_steps: Number of physics steps to take

        Returns:
            Final quadrotor state
        """
        motor_commands = np.clip(motor_commands, 0.0, 1.0)
        self._motor_commands = motor_commands.copy()
        self.data.ctrl[:4] = motor_commands

        # Apply external forces during stepping
        self.data.xfrc_applied[self._body_id, :3] = self._external_force
        self.data.xfrc_applied[self._body_id, 3:6] = self._external_torque

        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

        return self.get_state()

    def set_external_force(self, force: np.ndarray) -> None:
        """Set external force to apply to quadrotor.

        The force is applied in world frame at the body's center of mass.
        This is useful for simulating wind, collisions, etc.

        Args:
            force: Force vector [fx, fy, fz] in Newtons, world frame
        """
        self._external_force = np.asarray(force).flatten()[:3].copy()

    def set_external_torque(self, torque: np.ndarray) -> None:
        """Set external torque to apply to quadrotor.

        The torque is applied in body frame.

        Args:
            torque: Torque vector [tx, ty, tz] in Newton-meters
        """
        self._external_torque = np.asarray(torque).flatten()[:3].copy()

    def set_external_wrench(self, force: np.ndarray, torque: np.ndarray) -> None:
        """Set external force and torque simultaneously.

        Args:
            force: Force vector [fx, fy, fz] in Newtons, world frame
            torque: Torque vector [tx, ty, tz] in Newton-meters, body frame
        """
        self.set_external_force(force)
        self.set_external_torque(torque)

    def clear_external_forces(self) -> None:
        """Clear all external forces and torques."""
        self._external_force = np.zeros(3)
        self._external_torque = np.zeros(3)
        self.data.xfrc_applied[self._body_id] = np.zeros(6)

    def get_external_force(self) -> np.ndarray:
        """Get current external force being applied.

        Returns:
            Force vector [fx, fy, fz] in Newtons
        """
        return self._external_force.copy()

    def get_external_torque(self) -> np.ndarray:
        """Get current external torque being applied.

        Returns:
            Torque vector [tx, ty, tz] in Newton-meters
        """
        return self._external_torque.copy()

    def get_state(self) -> QuadrotorState:
        """Get current quadrotor state.

        Returns:
            Current state including position, velocity, orientation, etc.
        """
        return QuadrotorState(
            position=self.data.sensordata[self._pos_adr : self._pos_adr + 3].copy(),
            velocity=self.data.sensordata[self._linvel_adr : self._linvel_adr + 3].copy(),
            quaternion=self.data.sensordata[self._quat_adr : self._quat_adr + 4].copy(),
            angular_velocity=self.data.sensordata[self._angvel_adr : self._angvel_adr + 3].copy(),
            motor_speeds=self._motor_commands.copy(),
        )

    def get_imu_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get IMU sensor readings.

        Returns:
            Tuple of (gyroscope, accelerometer) readings in body frame.
            - gyro: Angular velocity [rad/s], shape (3,)
            - accel: Linear acceleration [m/s²], shape (3,)
        """
        gyro = self.data.sensordata[self._gyro_adr : self._gyro_adr + 3].copy()
        accel = self.data.sensordata[self._accel_adr : self._accel_adr + 3].copy()
        return gyro, accel

    def get_body_position(self) -> np.ndarray:
        """Get quadrotor body position in world frame."""
        return self.data.xpos[self._body_id].copy()

    def get_body_orientation(self) -> np.ndarray:
        """Get quadrotor body orientation as rotation matrix."""
        return self.data.xmat[self._body_id].reshape(3, 3).copy()

    def get_body_quaternion(self) -> np.ndarray:
        """Get quadrotor body orientation as quaternion [w, x, y, z]."""
        return self.data.xquat[self._body_id].copy()

    def get_time(self) -> float:
        """Get current simulation time in seconds."""
        return self.data.time

    def has_contact(self) -> bool:
        """Check if quadrotor is in contact with anything."""
        return self.data.ncon > 0

    def get_contact_force(self) -> np.ndarray:
        """Get total contact force on quadrotor."""
        total_force = np.zeros(3)
        for i in range(self.data.ncon):
            self.data.contact[i]
            # Get contact force
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            total_force += force[:3]
        return total_force

    def get_motor_forces(self) -> np.ndarray:
        """Get actual motor forces being applied."""
        # Actuator forces are stored in qfrc_actuator
        # For our model, these are the thrust forces
        return self.data.actuator_force[:4].copy()

    @property
    def gravity(self) -> np.ndarray:
        """Get gravity vector."""
        return self.model.opt.gravity.copy()

    @property
    def mass(self) -> float:
        """Get quadrotor mass."""
        return self.model.body_mass[self._body_id]

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Set gravity vector (for domain randomization)."""
        self.model.opt.gravity[:] = gravity

    def set_mocap_pos(self, name: str, pos: np.ndarray) -> None:
        """Set position of a mocap body.

        Args:
            name: Name of the mocap body
            pos: Position [x, y, z]
        """
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            mocap_id = self.model.body_mocapid[body_id]
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = pos
        except (KeyError, ValueError, IndexError):
            pass  # Silently ignore if body doesn't exist

    def has_mocap_body(self, name: str) -> bool:
        """Check if a mocap body exists."""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            return self.model.body_mocapid[body_id] >= 0
        except (KeyError, ValueError, IndexError):
            return False

    def create_renderer(self, width: int = 640, height: int = 480) -> mujoco.Renderer:
        """Create an offscreen renderer.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            MuJoCo Renderer instance
        """
        return mujoco.Renderer(self.model, height=height, width=width)

    def render(self, renderer: mujoco.Renderer) -> np.ndarray:
        """Render current frame.

        Args:
            renderer: MuJoCo Renderer instance

        Returns:
            RGB image as numpy array, shape (height, width, 3)
        """
        renderer.update_scene(self.data)
        return renderer.render()


def create_simulator(
    model: str = "x500",
    timestep: float | None = None,
) -> MuJoCoSimulator:
    """Factory function to create a simulator.

    Args:
        model: Model name ("x500" or "generic") or path to XML file.
        timestep: Override physics timestep.

    Returns:
        Configured MuJoCoSimulator instance
    """
    package_root = Path(__file__).parent.parent.parent

    if model == "x500":
        model_path = package_root / "models" / "quadrotor_x500.xml"
    elif model == "generic":
        model_path = package_root / "models" / "quadrotor_generic.xml"
    else:
        model_path = Path(model)

    return MuJoCoSimulator(model_path=model_path, timestep=timestep)
