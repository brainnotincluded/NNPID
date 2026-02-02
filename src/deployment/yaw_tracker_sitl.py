"""Yaw tracker deployment to ArduPilot SITL.

This module deploys a trained yaw tracking neural network to ArduPilot SITL,
sending SET_ATTITUDE_TARGET commands to control yaw rate while ArduPilot
handles roll/pitch/thrust stabilization.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pymavlink import mavutil

    PYMAVLINK_AVAILABLE = True
except ImportError:
    PYMAVLINK_AVAILABLE = False
    mavutil = None

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.base_class import BaseAlgorithm

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.rotations import Rotations

logger = get_logger(__name__)


@dataclass
class YawTrackerSITLConfig:
    """Configuration for yaw tracker SITL deployment."""

    # MAVLink connection
    connection_string: str = "tcp:127.0.0.1:5760"
    baudrate: int = 57600

    # System IDs
    system_id: int = 1
    component_id: int = 1

    # Control settings
    control_rate: float = 50.0  # Hz
    max_yaw_rate: float = 2.0  # rad/s

    # Target settings
    target_radius: float = 5.0  # meters
    target_height: float = 0.0  # meters (relative to drone)
    target_angular_velocity: float = 0.5  # rad/s
    target_pattern: str = "circular"  # circular, random, sinusoidal

    # Timeouts
    connection_timeout: float = 30.0
    heartbeat_timeout: float = 5.0

    # Failsafe
    failsafe_timeout: float = 5.0
    failsafe_action: str = "hold"  # hold, land, rtl


@dataclass
class TargetState:
    """State of the tracking target."""

    # Position in world frame
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Velocity in world frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Angular position/velocity (for circular motion)
    angle: float = 0.0
    angular_velocity: float = 0.5


class YawTrackerSITL:
    """Deploy yaw tracking NN to ArduPilot SITL.

    The neural network outputs yaw rate commands, which are sent to
    ArduPilot using SET_ATTITUDE_TARGET with a type mask that only
    controls yaw rate. ArduPilot handles all other stabilization.

    Usage:
        tracker = YawTrackerSITL(
            model_path="runs/yaw_tracking/best_model",
            config=YawTrackerSITLConfig()
        )

        if tracker.connect():
            tracker.arm_and_takeoff()
            tracker.run_loop()
    """

    def __init__(
        self,
        model_path: Path | None = None,
        config: YawTrackerSITLConfig | None = None,
    ):
        """Initialize yaw tracker.

        Args:
            model_path: Path to trained model (optional for testing)
            config: Configuration
        """
        if not PYMAVLINK_AVAILABLE:
            raise ImportError("pymavlink is required for SITL deployment")

        self.config = config or YawTrackerSITLConfig()
        self.model_path = model_path

        # Load model if provided
        self.model: BaseAlgorithm | None = None
        self._vec_normalize = None  # VecNormalize for observation normalization
        if model_path is not None:
            self._load_model(model_path)

        # MAVLink connection
        self._mav = None
        self._connected = False

        # Drone state
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._attitude = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion [w,x,y,z]
        self._angular_velocity = np.zeros(3)
        self._armed = False
        self._mode = "UNKNOWN"

        # Target state
        self._target = TargetState()
        self._target.angular_velocity = config.target_angular_velocity if config else 0.5

        # Control state
        self._running = False
        self._previous_action = 0.0
        self._time_on_target = 0.0
        self._start_time = 0.0

        # Threading
        self._receive_thread = None
        self._state_lock = threading.Lock()

        # Metrics
        self._metrics = {
            "yaw_errors": [],
            "actions": [],
            "tracking_time": 0.0,
        }

    def _load_model(self, model_path: Path) -> None:
        """Load trained model and VecNormalize if available."""
        if not SB3_AVAILABLE:
            logger.warning("stable-baselines3 not available, model not loaded")
            return

        model_path = Path(model_path)
        original_path = model_path

        # Handle directory or file
        if model_path.is_dir():
            for name in ["best_model.zip", "final_model.zip", "best_model", "final_model"]:
                candidate = model_path / name
                if candidate.exists():
                    model_path = candidate
                    break

        if not model_path.suffix:
            model_path = model_path.with_suffix(".zip")

        logger.info("Loading model from %s", model_path)

        try:
            self.model = PPO.load(str(model_path))
        except Exception:
            try:
                self.model = SAC.load(str(model_path))
            except Exception as e:
                logger.error("Error loading model: %s", e)
                return

        # Try to load VecNormalize (CRITICAL for correct inference!)
        self._vec_normalize = None
        vec_norm_paths = [
            model_path.parent.parent / "vec_normalize.pkl",  # runs/xxx/vec_normalize.pkl
            model_path.parent / "vec_normalize.pkl",  # runs/xxx/best_model/vec_normalize.pkl
            original_path / "vec_normalize.pkl",  # If original was directory
        ]

        for vec_norm_path in vec_norm_paths:
            if vec_norm_path.exists():
                try:
                    import pickle

                    with open(vec_norm_path, "rb") as f:
                        self._vec_normalize = pickle.load(f)
                    self._vec_normalize.training = False
                    logger.info("Loaded VecNormalize from %s", vec_norm_path)
                    break
                except Exception as e:
                    logger.warning("Could not load VecNormalize: %s", e)

        if self._vec_normalize is None:
            logger.warning("VecNormalize not found - model may produce incorrect results")

    def connect(self) -> bool:
        """Connect to ArduPilot SITL.

        Returns:
            True if connection successful
        """
        logger.info("Connecting to %s...", self.config.connection_string)

        try:
            self._mav = mavutil.mavlink_connection(
                self.config.connection_string,
                baud=self.config.baudrate,
            )

            # Wait for heartbeat
            logger.info("Waiting for heartbeat...")
            msg = self._mav.wait_heartbeat(timeout=self.config.connection_timeout)

            if msg is None:
                logger.error("No heartbeat received")
                return False

            logger.info("Connected to system %s", self._mav.target_system)

            # Start receive thread
            self._connected = True
            self._running = True
            self._receive_thread = threading.Thread(
                target=self._receive_loop,
                daemon=True,
            )
            self._receive_thread.start()

            # Request data streams
            self._request_data_streams()

            return True

        except Exception as e:
            logger.error("Connection failed: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from SITL."""
        self._running = False
        self._connected = False

        if self._receive_thread is not None:
            self._receive_thread.join(timeout=2.0)

        if self._mav is not None:
            self._mav.close()
            self._mav = None

    def _request_data_streams(self) -> None:
        """Request telemetry data streams."""
        if self._mav is None:
            return

        # Request all streams at high rate
        self._mav.mav.request_data_stream_send(
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            50,  # Hz
            1,  # Start
        )

    def _receive_loop(self) -> None:
        """Background loop to receive MAVLink messages."""
        while self._running:
            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.1)
                if msg is not None:
                    self._handle_message(msg)
            except Exception as e:
                if self._running:
                    logger.warning("Receive error: %s", e)

    def _handle_message(self, msg) -> None:
        """Handle received MAVLink message."""
        msg_type = msg.get_type()

        def _safe_array(values: list[float]) -> np.ndarray | None:
            try:
                arr = np.array(values, dtype=float)
            except (TypeError, ValueError):
                return None
            if not np.all(np.isfinite(arr)):
                return None
            return arr

        if msg_type == "LOCAL_POSITION_NED":
            position = _safe_array([msg.x, msg.y, msg.z])
            velocity = _safe_array([msg.vx, msg.vy, msg.vz])
            if position is not None and velocity is not None:
                with self._state_lock:
                    self._position = position
                    self._velocity = velocity

        elif msg_type == "ATTITUDE":
            euler = _safe_array([msg.roll, msg.pitch, msg.yaw])
            rates = _safe_array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
            if euler is not None and rates is not None:
                attitude = Rotations.euler_to_quaternion(*euler.tolist())
                with self._state_lock:
                    # Convert Euler to quaternion
                    self._attitude = attitude
                    self._angular_velocity = rates

        elif msg_type == "ATTITUDE_QUATERNION":
            quat = _safe_array([msg.q1, msg.q2, msg.q3, msg.q4])
            rates = _safe_array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
            if quat is not None and rates is not None:
                with self._state_lock:
                    self._attitude = quat
                    self._angular_velocity = rates

        elif msg_type == "HEARTBEAT":
            with self._state_lock:
                self._armed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

                # Decode ArduPilot mode
                mode_mapping = {
                    0: "STABILIZE",
                    2: "ALT_HOLD",
                    3: "AUTO",
                    4: "GUIDED",
                    5: "LOITER",
                    6: "RTL",
                    9: "LAND",
                }
                self._mode = mode_mapping.get(msg.custom_mode, f"MODE_{msg.custom_mode}")

    def get_state(self) -> dict[str, Any]:
        """Get current drone state (thread-safe).

        Returns:
            Dictionary of state values
        """
        with self._state_lock:
            roll, pitch, yaw = Rotations.quaternion_to_euler(self._attitude)
            return {
                "position": self._position.copy(),
                "velocity": self._velocity.copy(),
                "quaternion": self._attitude.copy(),
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                "angular_velocity": self._angular_velocity.copy(),
                "armed": self._armed,
                "mode": self._mode,
            }

    def set_mode(self, mode: str) -> bool:
        """Set flight mode.

        Args:
            mode: Mode name (GUIDED, LOITER, etc.)

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        mode_mapping = {
            "STABILIZE": 0,
            "ALT_HOLD": 2,
            "AUTO": 3,
            "GUIDED": 4,
            "LOITER": 5,
            "RTL": 6,
            "LAND": 9,
        }

        mode_id = mode_mapping.get(mode.upper())
        if mode_id is None:
            logger.error("Unknown mode: %s", mode)
            return False

        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id,
                0,
                0,
                0,
                0,
                0,
            )
            logger.info("Set mode: %s", mode)
            return True
        except Exception as e:
            logger.error("Set mode failed: %s", e)
            return False

    def arm(self) -> bool:
        """Arm the drone.

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                1,  # Arm
                0,
                0,
                0,
                0,
                0,
                0,
            )
            logger.info("Arm command sent")
            return True
        except Exception as e:
            logger.error("Arm failed: %s", e)
            return False

    def disarm(self, force: bool = False) -> bool:
        """Disarm the drone.

        Args:
            force: Force disarm even if flying

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                0,  # Disarm
                21196.0 if force else 0,  # Force magic number
                0,
                0,
                0,
                0,
                0,
            )
            logger.info("Disarm command sent")
            return True
        except Exception as e:
            logger.error("Disarm failed: %s", e)
            return False

    def takeoff(self, altitude: float = 2.0) -> bool:
        """Command takeoff.

        Args:
            altitude: Target altitude in meters

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                altitude,
            )
            logger.info("Takeoff to %sm", altitude)
            return True
        except Exception as e:
            logger.error("Takeoff failed: %s", e)
            return False

    def land(self) -> bool:
        """Command landing.

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        try:
            self._mav.mav.command_long_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )
            logger.info("Land command sent")
            return True
        except Exception as e:
            logger.error("Land failed: %s", e)
            return False

    def send_yaw_rate(self, yaw_rate: float) -> bool:
        """Send yaw rate command via SET_ATTITUDE_TARGET.

        Only controls yaw rate - ArduPilot handles roll/pitch/thrust.

        Args:
            yaw_rate: Desired yaw rate in rad/s

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        # Clamp yaw rate
        yaw_rate = np.clip(yaw_rate, -self.config.max_yaw_rate, self.config.max_yaw_rate)

        # Type mask: only control yaw rate (bit 2), ignore everything else
        # Bits: 0=roll rate, 1=pitch rate, 2=yaw rate, 6=thrust
        # Set bits to ignore: 0b00111011 = 0x3B (ignore roll rate, pitch rate, attitude, thrust)
        # We want to control ONLY body_yaw_rate
        type_mask = 0b01111011  # 0x7B - only body_yaw_rate active

        # Note: For ArduPilot, we may need to use a different approach
        # since GUIDED mode typically uses position/velocity commands
        # We'll use SET_ATTITUDE_TARGET for yaw rate control

        try:
            # Quaternion for no rotation request (will be ignored due to mask)
            q = [1.0, 0.0, 0.0, 0.0]

            self._mav.mav.set_attitude_target_send(
                int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
                self._mav.target_system,
                self._mav.target_component,
                type_mask,
                q,  # Attitude quaternion (ignored)
                0.0,  # Roll rate (ignored)
                0.0,  # Pitch rate (ignored)
                yaw_rate,  # Yaw rate (active)
                0.0,  # Thrust (ignored - ArduPilot maintains altitude)
            )
            return True

        except Exception as e:
            logger.error("Send yaw rate failed: %s", e)
            return False

    def send_guided_velocity(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yaw_rate: float = 0.0,
    ) -> bool:
        """Send velocity command with yaw rate in GUIDED mode.

        For ArduPilot GUIDED mode, this is often more reliable than
        SET_ATTITUDE_TARGET for controlling yaw rate while hovering.

        Args:
            vx, vy, vz: Velocity in NED frame (m/s)
            yaw_rate: Yaw rate in rad/s

        Returns:
            True if command sent
        """
        if self._mav is None:
            return False

        try:
            # Type mask for velocity + yaw rate
            # Position ignored, acceleration ignored
            type_mask = 0b0000_0001_1100_0111  # Control velocity and yaw rate

            self._mav.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0,
                0,
                0,  # Position (ignored)
                vx,
                vy,
                vz,  # Velocity
                0,
                0,
                0,  # Acceleration (ignored)
                0,  # Yaw (ignored)
                yaw_rate,  # Yaw rate
            )
            return True

        except Exception as e:
            logger.error("Send velocity failed: %s", e)
            return False

    def update_target(self, dt: float) -> np.ndarray:
        """Update target position based on pattern.

        Args:
            dt: Time step

        Returns:
            Target position in world frame
        """
        cfg = self.config

        if cfg.target_pattern == "circular":
            # Circular motion around drone's starting position
            self._target.angle += self._target.angular_velocity * dt

            x = cfg.target_radius * np.cos(self._target.angle)
            y = cfg.target_radius * np.sin(self._target.angle)
            z = cfg.target_height

            self._target.position = np.array([x, y, z])

        elif cfg.target_pattern == "sinusoidal":
            # Oscillating motion
            t = time.time() - self._start_time
            angle = np.pi / 2 * np.sin(self._target.angular_velocity * t)

            x = cfg.target_radius * np.cos(angle)
            y = cfg.target_radius * np.sin(angle)
            z = cfg.target_height

            self._target.position = np.array([x, y, z])

        return self._target.position

    def compute_observation(self) -> np.ndarray:
        """Compute observation for NN.

        Returns:
            11-dimensional observation matching YawTrackingEnv
        """
        state = self.get_state()

        # Target direction in body frame
        to_target = self._target.position - state["position"]
        to_target[2] = 0  # Only horizontal

        target_distance = np.linalg.norm(to_target[:2])

        yaw = state["yaw"]
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        body_x = cos_yaw * to_target[0] - sin_yaw * to_target[1]
        body_y = sin_yaw * to_target[0] + cos_yaw * to_target[1]

        body_dist = np.sqrt(body_x**2 + body_y**2)
        if body_dist > 0.01:
            target_dir = np.array([body_x / body_dist, body_y / body_dist])
        else:
            target_dir = np.array([1.0, 0.0])

        # Yaw error
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        to_target_norm = to_target[:2] / (target_distance + 1e-6)

        dot = np.clip(np.dot(heading, to_target_norm), -1, 1)
        cross = heading[0] * to_target_norm[1] - heading[1] * to_target_norm[0]
        yaw_error = np.arctan2(cross, dot)

        # Build observation
        obs = np.array(
            [
                target_dir[0],
                target_dir[1],
                self._target.angular_velocity,
                state["angular_velocity"][2],  # Current yaw rate
                yaw_error,
                np.clip(state["roll"], -1, 1),
                np.clip(state["pitch"], -1, 1),
                0.0,  # Altitude error (assuming stable hover)
                self._previous_action,
                min(self._time_on_target / 0.5, 1.0),  # Normalized time on target
                target_distance / self.config.target_radius,
            ],
            dtype=np.float32,
        )

        return obs

    def step(self, dt: float) -> tuple[float, float]:
        """Run one control step.

        Args:
            dt: Time step

        Returns:
            Tuple of (yaw_error, action)
        """
        # Update target
        self.update_target(dt)

        # Compute observation
        obs = self.compute_observation()
        yaw_error = obs[4]

        # Get action from NN or use simple controller
        if self.model is not None:
            # Normalize observation if VecNormalize is available
            obs_for_model = obs
            if self._vec_normalize is not None:
                obs_normalized = self._vec_normalize.normalize_obs(obs.reshape(1, -1))
                obs_for_model = np.array(obs_normalized[0], dtype=np.float32)

            action, _ = self.model.predict(obs_for_model, deterministic=True)
            yaw_rate_cmd = float(action[0]) * self.config.max_yaw_rate
        else:
            # Simple proportional controller as fallback
            kp = 2.0
            yaw_rate_cmd = kp * yaw_error
            yaw_rate_cmd = np.clip(
                yaw_rate_cmd, -self.config.max_yaw_rate, self.config.max_yaw_rate
            )
            action = np.array([yaw_rate_cmd / self.config.max_yaw_rate])

        # Send command
        self.send_guided_velocity(0, 0, 0, yaw_rate_cmd)

        # Update metrics
        self._metrics["yaw_errors"].append(yaw_error)
        self._metrics["actions"].append(float(action[0]))

        # Track time on target
        if abs(yaw_error) < 0.1:  # ~6 degrees
            self._time_on_target += dt
        else:
            self._time_on_target = max(0, self._time_on_target - dt * 0.5)

        self._previous_action = float(action[0])

        return yaw_error, float(action[0])

    def arm_and_takeoff(self, altitude: float = 2.0, timeout: float = 30.0) -> bool:
        """Arm and takeoff to specified altitude.

        Args:
            altitude: Target altitude
            timeout: Timeout in seconds

        Returns:
            True if successful
        """
        logger.info("=== Arm and Takeoff ===")

        # Set GUIDED mode
        logger.info("Setting GUIDED mode...")
        self.set_mode("GUIDED")
        time.sleep(1)

        # Arm
        logger.info("Arming...")
        self.arm()

        # Wait for arm
        start = time.time()
        while not self._armed and time.time() - start < 10:
            time.sleep(0.1)

        if not self._armed:
            logger.error("Failed to arm")
            return False

        logger.info("Armed!")

        # Takeoff
        logger.info("Taking off to %sm...", altitude)
        self.takeoff(altitude)

        # Wait for altitude
        start = time.time()
        while time.time() - start < timeout:
            state = self.get_state()
            current_alt = -state["position"][2]

            if current_alt >= altitude * 0.9:
                logger.info("Reached altitude: %.1fm", current_alt)
                return True

            time.sleep(0.5)
            logger.info("Altitude: %.1fm", current_alt)

        logger.error("Takeoff timeout")
        return False

    def run_loop(
        self,
        duration: float = 60.0,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run yaw tracking control loop.

        Args:
            duration: Duration in seconds
            verbose: Print status updates

        Returns:
            Dictionary of metrics
        """
        logger.info("=== Running Yaw Tracking for %ss ===", duration)
        logger.info("Press Ctrl+C to stop")

        self._start_time = time.time()
        self._metrics = {"yaw_errors": [], "actions": [], "tracking_time": 0.0}

        dt = 1.0 / self.config.control_rate

        try:
            while time.time() - self._start_time < duration:
                loop_start = time.time()

                yaw_error, action = self.step(dt)

                if verbose and int((time.time() - self._start_time) * 10) % 10 == 0:
                    state = self.get_state()
                    logger.info(
                        f"  t={time.time() - self._start_time:.1f}s "
                        f"yaw_err={np.degrees(yaw_error):+6.1f}° "
                        f"action={action:+.2f} "
                        f"mode={state['mode']}"
                    )

                # Maintain rate
                elapsed = time.time() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        except KeyboardInterrupt:
            logger.info("Interrupted")

        # Compute final metrics
        errors = self._metrics["yaw_errors"]
        if errors:
            self._metrics["mean_yaw_error"] = np.mean(np.abs(errors))
            self._metrics["max_yaw_error"] = np.max(np.abs(errors))
            self._metrics["tracking_percentage"] = 100 * np.mean(
                [1 if abs(e) < 0.1 else 0 for e in errors]
            )

        logger.info("=== Results ===")
        logger.info(
            "Mean yaw error: %.1f°",
            np.degrees(self._metrics.get("mean_yaw_error", 0)),
        )
        logger.info(
            "Tracking %%: %.1f%%",
            self._metrics.get("tracking_percentage", 0),
        )

        return self._metrics

    def shutdown(self) -> None:
        """Safe shutdown."""
        logger.info("=== Shutting down ===")

        # Land if armed
        if self._armed:
            logger.info("Landing...")
            self.land()

            # Wait for landing
            start = time.time()
            while self._armed and time.time() - start < 30:
                time.sleep(0.5)

        self.disconnect()
        logger.info("Shutdown complete")


def main():
    """Main entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Yaw tracker SITL deployment")
    parser.add_argument("--model", type=Path, default=None, help="Path to trained model")
    parser.add_argument("--connection", type=str, default="tcp:127.0.0.1:5760")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--altitude", type=float, default=2.0)
    parser.add_argument("--target-speed", type=float, default=0.5)

    args = parser.parse_args()

    config = YawTrackerSITLConfig(
        connection_string=args.connection,
        target_angular_velocity=args.target_speed,
    )

    tracker = YawTrackerSITL(model_path=args.model, config=config)

    try:
        if tracker.connect() and tracker.arm_and_takeoff(altitude=args.altitude):
            time.sleep(2)  # Stabilize
            tracker.run_loop(duration=args.duration)
    finally:
        tracker.shutdown()


if __name__ == "__main__":
    main()
