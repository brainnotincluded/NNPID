"""Yaw tracking environment - NN learns to face a moving target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..controllers.hover_stabilizer import HoverStabilizer, HoverStabilizerConfig
from ..core.mujoco_sim import QuadrotorState, create_simulator
from ..perturbations import PerturbationManager
from ..utils.rotations import Rotations
from .target_patterns import (
    CircularTarget,
    EvasiveTarget,
    Figure8Target,
    LissajousTarget,
    MultiFrequencyTarget,
    RandomTarget,
    SinusoidalTarget,
    SpiralTarget,
    StepTarget,
    TargetPattern,
)


@dataclass
class YawTrackingConfig:
    """Configuration for yaw tracking environment."""

    # Simulation
    model: str = "generic"
    physics_timestep: float = 0.002  # 500 Hz
    control_frequency: float = 100.0  # 100 Hz control (was 50Hz, increased for stability)
    max_episode_steps: int = 2000  # 20 seconds at 100Hz

    # Hover settings
    hover_height: float = 1.0
    hover_position: tuple[float, float] = (0.0, 0.0)

    # Target settings
    target_patterns: list[str] = field(
        default_factory=lambda: [
            "circular",
            "random",
            "sinusoidal",
            "step",
            "figure8",
            "spiral",
            "evasive",
            "lissajous",
            "multi_frequency",
        ]
    )
    target_radius: float = 3.0
    target_speed_min: float = 0.1  # rad/s
    target_speed_max: float = 0.3  # rad/s

    # Action scaling
    max_yaw_rate: float = 1.0  # rad/s
    action_smoothing: float = 0.3  # Low-pass filter: 0=no smoothing, 1=full smoothing
    max_action_change: float = 0.5  # Max allowed action change per step
    action_dead_zone: float = 0.08  # Dead zone: actions below this threshold are zeroed

    # Observation noise
    yaw_noise: float = 0.01
    angular_velocity_noise: float = 0.01

    # Reward weights (v2 - zone-based with shaping)
    facing_reward_weight: float = 1.5  # Increased for zone-based rewards
    facing_reward_scale: float = 5.0  # exp(-scale * error^2) - kept for v1 compat
    error_reduction_weight: float = 0.5  # NEW: reward for reducing error
    velocity_match_weight: float = 0.2  # NEW: reward for matching target velocity
    direction_alignment_bonus: float = 0.1  # NEW: bonus for turning toward target
    excess_yaw_rate_penalty: float = 0.05  # NEW: replaces yaw_rate_penalty_weight
    yaw_rate_penalty_weight: float = 0.1  # Kept for v1 compat
    action_rate_penalty_weight: float = 0.03  # Reduced from 0.05
    sustained_tracking_bonus: float = 0.3  # Reduced from 0.5 (now continuous)
    sustained_tracking_threshold: float = 0.1  # radians (~6 degrees)
    sustained_tracking_time: float = 0.5  # seconds
    crash_penalty: float = 50.0  # Increased to strongly discourage crashes
    alive_bonus: float = 0.1  # Increased to reward survival

    # Zone thresholds for facing reward
    on_target_threshold: float = 0.1  # 6° - high reward zone
    close_tracking_threshold: float = 0.35  # 20° - positive reward zone
    searching_threshold: float = 1.57  # 90° - negative reward zone

    # Success criteria
    success_threshold: float = 0.1  # radians

    # Termination - lenient for learning
    max_tilt_angle: float = 0.6  # radians (~34 degrees)
    max_altitude_error: float = 2.0  # meters

    # Stabilizer PID gains (tuned for stability at 100Hz+)
    # See docs/issues/003-hover-pid-instability.md for details
    altitude_kp: float = 18.0
    altitude_ki: float = 2.0
    altitude_kd: float = 10.0
    attitude_kp: float = 12.0  # reduced for stability
    attitude_ki: float = 0.3  # reduced to prevent windup
    attitude_kd: float = 8.0  # increased for damping
    yaw_rate_kp: float = 1.5
    base_thrust: float = 0.62  # Hover throttle for 2kg drone with 4x8N motors

    # Safety settings (SITL-style)
    safety_tilt_threshold: float = 0.3  # radians (~17 degrees) - ignore yaw if exceeded
    yaw_authority: float = 0.10  # Yaw torque authority (very low for stability)
    max_integral: float = 0.2  # Anti-windup limit for integral terms

    # Perturbation settings
    perturbations_enabled: bool = False
    perturbations_config_path: str | None = None
    perturbation_intensity: float = 1.0  # Global intensity multiplier (0-1)


class YawTrackingEnv(gym.Env):
    """Gymnasium environment for yaw target tracking.

    The drone hovers in place while a neural network controls its yaw rate
    to keep facing a moving target. Roll, pitch, and altitude are stabilized
    by an internal PD controller.

    Observation Space (11 dimensions):
        - target_direction (2): Unit vector to target in body frame [x, y]
        - target_angular_velocity (1): Target's angular velocity
        - current_yaw_rate (1): Drone's yaw rate
        - yaw_error (1): Angle to target in [-pi, pi]
        - roll, pitch (2): Current tilt angles
        - altitude_error (1): Height deviation from hover
        - previous_action (1): Last yaw rate command
        - time_on_target (1): Normalized time spent on target
        - target_distance (1): Distance to target (normalized)

    Action Space (1 dimension):
        - yaw_rate_command: Normalized [-1, 1] mapped to [-max_yaw_rate, max_yaw_rate]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        config: YawTrackingConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.config = config or YawTrackingConfig()
        self.render_mode = render_mode

        # Create simulator
        self.sim = create_simulator(model=self.config.model)

        # Control timing
        self._physics_steps_per_control = int(
            1.0 / (self.config.control_frequency * self.config.physics_timestep)
        )
        self._dt = 1.0 / self.config.control_frequency

        # Create target patterns
        self._target_patterns = self._create_target_patterns()
        self._current_pattern: TargetPattern | None = None

        # State tracking
        self._step_count = 0
        self._time = 0.0
        self._previous_action = 0.0
        self._time_on_target = 0.0
        self._episode_reward = 0.0

        # SITL-style hover stabilizer (guarantees drone cannot crash)
        stabilizer_config = HoverStabilizerConfig(
            hover_height=self.config.hover_height,
            altitude_kp=self.config.altitude_kp,
            altitude_ki=self.config.altitude_ki,
            altitude_kd=self.config.altitude_kd,
            attitude_kp=self.config.attitude_kp,
            attitude_ki=self.config.attitude_ki,
            attitude_kd=self.config.attitude_kd,
            yaw_rate_kp=self.config.yaw_rate_kp,
            yaw_authority=self.config.yaw_authority,
            safety_tilt_threshold=self.config.safety_tilt_threshold,
            base_thrust=self.config.base_thrust,
            max_integral=self.config.max_integral,
        )
        self._stabilizer = HoverStabilizer(stabilizer_config)

        # Perturbation manager
        self._perturbation_manager: PerturbationManager | None = None
        if self.config.perturbations_enabled:
            self._init_perturbations()

        # RNG
        self._np_random: np.random.Generator | None = None

        # Define spaces
        self._define_spaces()

        # Rendering
        self._renderer = None

    def _create_target_patterns(self) -> dict[str, TargetPattern]:
        """Create target pattern instances."""
        cfg = self.config
        patterns: dict[str, TargetPattern] = {}

        # Basic patterns
        if "circular" in cfg.target_patterns:
            patterns["circular"] = CircularTarget(
                radius=cfg.target_radius,
                angular_velocity=cfg.target_speed_min,
                height=cfg.hover_height,
            )

        if "random" in cfg.target_patterns:
            patterns["random"] = RandomTarget(
                radius=cfg.target_radius,
                change_interval=2.0,
                height=cfg.hover_height,
            )

        if "sinusoidal" in cfg.target_patterns:
            patterns["sinusoidal"] = SinusoidalTarget(
                radius=cfg.target_radius,
                frequency=cfg.target_speed_min / (2 * np.pi),
                height=cfg.hover_height,
            )

        if "step" in cfg.target_patterns:
            patterns["step"] = StepTarget(
                radius=cfg.target_radius,
                step_interval=3.0,
                height=cfg.hover_height,
            )

        # Advanced patterns
        if "figure8" in cfg.target_patterns:
            patterns["figure8"] = Figure8Target(
                radius=cfg.target_radius,
                angular_velocity=cfg.target_speed_min,
                height=cfg.hover_height,
            )

        if "spiral" in cfg.target_patterns:
            patterns["spiral"] = SpiralTarget(
                radius_min=cfg.target_radius * 0.5,
                radius_max=cfg.target_radius * 1.5,
                angular_velocity=cfg.target_speed_min,
                spiral_frequency=0.1,
                height=cfg.hover_height,
            )

        if "evasive" in cfg.target_patterns:
            patterns["evasive"] = EvasiveTarget(
                radius=cfg.target_radius,
                base_angular_velocity=cfg.target_speed_min,
                jerk_probability=0.02,
                max_jerk_magnitude=2.0,
                height=cfg.hover_height,
            )

        if "lissajous" in cfg.target_patterns:
            patterns["lissajous"] = LissajousTarget(
                radius=cfg.target_radius,
                angular_velocity=cfg.target_speed_min,
                height=cfg.hover_height,
            )

        if "multi_frequency" in cfg.target_patterns:
            patterns["multi_frequency"] = MultiFrequencyTarget(
                radius=cfg.target_radius,
                base_frequency=cfg.target_speed_min / (2 * np.pi),
                num_harmonics=3,
                height=cfg.hover_height,
            )

        return patterns

    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: 11 dimensions
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -1,
                    -1,  # target_direction (unit vector)
                    -5,  # target_angular_velocity
                    -5,  # current_yaw_rate
                    -np.pi,  # yaw_error
                    -1,
                    -1,  # roll, pitch (normalized)
                    -5,  # altitude_error
                    -1,  # previous_action
                    0,  # time_on_target (normalized)
                    0,  # target_distance (normalized)
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1,
                    1,  # target_direction
                    5,  # target_angular_velocity
                    5,  # current_yaw_rate
                    np.pi,  # yaw_error
                    1,
                    1,  # roll, pitch
                    5,  # altitude_error
                    1,  # previous_action
                    1,  # time_on_target
                    10,  # target_distance
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Action: yaw rate command [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def _init_perturbations(self) -> None:
        """Initialize perturbation manager from config."""
        cfg = self.config

        if cfg.perturbations_config_path is not None:
            self._perturbation_manager = PerturbationManager(
                config_path=cfg.perturbations_config_path
            )
            self._perturbation_manager.global_intensity = cfg.perturbation_intensity
        else:
            # Create default manager with minimal perturbations
            self._perturbation_manager = PerturbationManager()
            self._perturbation_manager.global_intensity = cfg.perturbation_intensity

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        elif self._np_random is None:
            self._np_random = np.random.default_rng()

        # Reset simulator - start hovering at target height
        init_pos = np.array(
            [
                self.config.hover_position[0],
                self.config.hover_position[1],
                self.config.hover_height,
            ]
        )
        # Limit initial yaw to ±120° to avoid gimbal lock instability near ±180°
        # See docs/issues/003-hover-pid-instability.md for details
        init_yaw = self._np_random.uniform(-2.1, 2.1)  # ±120°
        init_quat = Rotations.euler_to_quaternion(0, 0, init_yaw)

        self.sim.reset(
            position=init_pos,
            velocity=np.zeros(3),
            quaternion=init_quat,
            angular_velocity=np.zeros(3),
        )

        # Select and reset target pattern
        if options and "pattern" in options:
            pattern_name = options["pattern"]
        else:
            pattern_name = self._np_random.choice(list(self._target_patterns.keys()))

        self._current_pattern = self._target_patterns[pattern_name]
        self._current_pattern.reset(self._np_random)

        # Randomize target speed for patterns that support it
        speed = self._np_random.uniform(
            self.config.target_speed_min,
            self.config.target_speed_max,
        )
        if hasattr(self._current_pattern, "angular_velocity"):
            self._current_pattern.angular_velocity = speed
        if hasattr(self._current_pattern, "base_angular_velocity"):
            self._current_pattern.base_angular_velocity = speed
        if hasattr(self._current_pattern, "base_frequency"):
            self._current_pattern.base_frequency = speed / (2 * np.pi)

        # Reset state tracking
        self._step_count = 0
        self._time = 0.0
        self._previous_action = 0.0
        self._time_on_target = 0.0
        self._episode_reward = 0.0
        self._prev_yaw_error: float | None = None  # For error reduction reward shaping

        # Reset hover stabilizer
        self._stabilizer.reset()

        # Reset perturbation manager if enabled
        if self._perturbation_manager is not None:
            self._perturbation_manager.reset()

        # Set initial target marker position
        state = self.sim.get_state()
        target_pos = self._current_pattern.get_position(0.0, state.position)
        self.sim.set_mocap_pos("target", target_pos)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step.

        Args:
            action: Yaw rate command [-1, 1]

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Scale action to yaw rate with smoothing
        raw_action = float(np.clip(action[0], -1, 1))

        # Apply action smoothing (low-pass filter) to prevent jerky control
        # This helps during exploration when PPO tries random actions
        if self.config.action_smoothing > 0:
            # Low-pass filter toward previous action
            alpha = self.config.action_smoothing
            smoothed_action = (1.0 - alpha) * raw_action + alpha * self._previous_action

            # Limit max change per step
            max_change = self.config.max_action_change
            action_change = smoothed_action - self._previous_action
            action_change = np.clip(action_change, -max_change, max_change)
            smoothed_action = self._previous_action + action_change
        else:
            smoothed_action = raw_action

        # Apply dead zone - zero out small actions to prevent jitter
        if self.config.action_dead_zone > 0 and abs(smoothed_action) < self.config.action_dead_zone:
            smoothed_action = 0.0

        yaw_rate_cmd = smoothed_action * self.config.max_yaw_rate

        # Get current state
        state = self.sim.get_state()

        # Update and apply perturbations if enabled
        if self._perturbation_manager is not None and self._perturbation_manager.enabled:
            self._perturbation_manager.update(self._dt, state)

        # Compute stabilized motor commands using SITL-style stabilizer
        motor_cmds = self._stabilizer.compute_motors(state, yaw_rate_cmd, self._dt)

        # Apply perturbations to motor commands (actuator effects)
        if self._perturbation_manager is not None and self._perturbation_manager.enabled:
            motor_cmds = self._perturbation_manager.apply_to_action(motor_cmds)

        # Apply external forces from perturbations before physics step
        if self._perturbation_manager is not None and self._perturbation_manager.enabled:
            ext_force = self._perturbation_manager.get_total_force()
            ext_torque = self._perturbation_manager.get_total_torque()
            self.sim.set_external_wrench(ext_force, ext_torque)
        else:
            self.sim.clear_external_forces()

        # Step physics
        for _ in range(self._physics_steps_per_control):
            self.sim.step(motor_cmds)

        self._step_count += 1
        self._time += self._dt

        # Get new state
        state = self.sim.get_state()

        # Update target marker position in MuJoCo
        target_pos = self._current_pattern.get_position(self._time, state.position)
        self.sim.set_mocap_pos("target", target_pos)

        # Compute yaw error for time-on-target tracking
        yaw_error = self._compute_yaw_error(state)
        if abs(yaw_error) < self.config.sustained_tracking_threshold:
            self._time_on_target += self._dt
        else:
            self._time_on_target = max(0, self._time_on_target - self._dt * 0.5)

        # Check termination
        terminated = self._check_termination(state)
        truncated = self._step_count >= self.config.max_episode_steps

        # Compute reward
        reward = self._compute_reward(state, action, terminated)
        self._episode_reward += reward

        # Store smoothed action (used in observation and reward)
        action_change = abs(smoothed_action - self._previous_action)
        self._previous_action = smoothed_action

        obs = self._get_observation()
        info = self._get_info()
        info["action_change"] = action_change

        if terminated or truncated:
            info["episode_reward"] = self._episode_reward
            info["episode_length"] = self._step_count

        return obs, reward, terminated, truncated, info

    def _compute_yaw_error(self, state: QuadrotorState) -> float:
        """Compute angle from drone heading to target."""
        # Get target position
        target_pos = self._current_pattern.get_position(self._time, state.position)

        # Vector to target in world frame
        to_target = target_pos - state.position
        to_target[2] = 0  # Only horizontal

        if np.linalg.norm(to_target) < 0.01:
            return 0.0

        to_target = to_target / np.linalg.norm(to_target)

        # Get drone heading in world frame
        _, _, yaw = Rotations.quaternion_to_euler(state.quaternion)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0])

        # Angle between heading and target direction
        dot = np.clip(np.dot(heading, to_target), -1, 1)
        cross = heading[0] * to_target[1] - heading[1] * to_target[0]

        yaw_error = np.arctan2(cross, dot)
        return yaw_error

    def _get_observation(self) -> np.ndarray:
        """Construct observation."""
        state = self.sim.get_state()
        cfg = self.config
        rng = self._np_random

        # Get target position and direction
        target_pos = self._current_pattern.get_position(self._time, state.position)
        to_target = target_pos - state.position
        target_distance = np.linalg.norm(to_target[:2])  # Horizontal distance

        # Transform to body frame
        _, _, yaw = Rotations.quaternion_to_euler(state.quaternion)
        yaw += rng.normal(0, cfg.yaw_noise)

        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        body_x = cos_yaw * to_target[0] - sin_yaw * to_target[1]
        body_y = sin_yaw * to_target[0] + cos_yaw * to_target[1]

        # Normalize to unit vector
        body_dist = np.sqrt(body_x**2 + body_y**2)
        if body_dist > 0.01:
            target_dir = np.array([body_x / body_dist, body_y / body_dist])
        else:
            target_dir = np.array([1.0, 0.0])

        # Yaw error
        yaw_error = self._compute_yaw_error(state)

        # Angular velocities
        omega = state.angular_velocity + rng.normal(0, cfg.angular_velocity_noise, 3)
        current_yaw_rate = omega[2]
        target_angular_velocity = self._current_pattern.get_angular_velocity()

        # Attitude
        roll, pitch, _ = Rotations.quaternion_to_euler(state.quaternion)

        # Altitude error
        altitude_error = cfg.hover_height - state.position[2]

        # Time on target (normalized to [0, 1])
        time_on_target_norm = min(self._time_on_target / cfg.sustained_tracking_time, 1.0)

        # Target distance (normalized)
        target_dist_norm = target_distance / cfg.target_radius

        obs = np.array(
            [
                target_dir[0],
                target_dir[1],
                target_angular_velocity,
                current_yaw_rate,
                yaw_error,
                np.clip(roll, -1, 1),
                np.clip(pitch, -1, 1),
                np.clip(altitude_error, -5, 5),
                self._previous_action,
                time_on_target_norm,
                target_dist_norm,
            ],
            dtype=np.float32,
        )

        return obs

    def _compute_reward(
        self,
        state: QuadrotorState,
        action: np.ndarray,
        terminated: bool,
    ) -> float:
        """Compute improved reward with zone-based facing and shaping.

        Reward components (v2):
        1. Zone-based facing reward (negative when far from target)
        2. Error reduction shaping (immediate feedback)
        3. Velocity matching (encourage matching target speed)
        4. Direction alignment bonus (turning toward target)
        5. Excess yaw rate penalty (only penalize excessive rotation)
        6. Action smoothness penalty
        7. Continuous tracking bonus (progressive, not binary)
        8. Conditional alive bonus (only when tracking reasonably)
        9. Crash penalty
        """
        cfg = self.config
        reward = 0.0

        yaw_error = self._compute_yaw_error(state)
        yaw_rate = state.angular_velocity[2]
        target_vel = self._current_pattern.get_angular_velocity()
        abs_error = abs(yaw_error)

        # 1. Zone-based facing reward (replaces exponential)
        # Creates clear gradient with NEGATIVE rewards for bad tracking
        if abs_error < cfg.on_target_threshold:
            # On target zone (<6°): High reward, peaks at 0
            facing = 1.0 - 5.0 * abs_error
        elif abs_error < cfg.close_tracking_threshold:
            # Close tracking zone (6-20°): Positive but decreasing
            facing = 0.5 * (
                1.0
                - (abs_error - cfg.on_target_threshold)
                / (cfg.close_tracking_threshold - cfg.on_target_threshold)
            )
        elif abs_error < cfg.searching_threshold:
            # Searching zone (20-90°): NEGATIVE reward - must improve!
            facing = -0.5 * (abs_error - cfg.close_tracking_threshold)
        else:
            # Lost zone (>90°): Strong negative
            facing = -1.0 - 0.3 * (abs_error - cfg.searching_threshold)
        reward += cfg.facing_reward_weight * facing

        # 2. Error reduction shaping (immediate feedback for correct actions)
        if self._prev_yaw_error is not None:
            error_reduction = abs(self._prev_yaw_error) - abs_error
            reward += cfg.error_reduction_weight * error_reduction
        self._prev_yaw_error = yaw_error

        # 3. Velocity matching (encourage matching target speed)
        velocity_error = abs(target_vel - yaw_rate)
        velocity_match = np.exp(-3.0 * velocity_error**2)
        reward += cfg.velocity_match_weight * velocity_match

        # 4. Direction alignment bonus (encourage turning toward target)
        # Only give bonus when not already on target and turning correctly
        if abs_error > 0.05 and np.sign(yaw_error) * np.sign(yaw_rate) > 0:
            reward += cfg.direction_alignment_bonus

        # 5. Excess yaw rate penalty (only penalize EXCESSIVE yaw rate)
        # Required rate = target velocity + correction for error
        required_rate = target_vel + 2.0 * yaw_error  # P-gain for error
        excess = max(0, abs(yaw_rate) - abs(required_rate) - 0.3)  # 0.3 margin
        reward -= cfg.excess_yaw_rate_penalty * excess**2

        # 6. Action smoothness penalty
        action_change = abs(float(action[0]) - self._previous_action)
        reward -= cfg.action_rate_penalty_weight * action_change**2

        # 7. Continuous tracking bonus (progressive instead of binary)
        tracking_progress = min(1.0, self._time_on_target / cfg.sustained_tracking_time)
        reward += cfg.sustained_tracking_bonus * np.sqrt(tracking_progress)

        # 8. Conditional alive bonus (only when tracking reasonably)
        if abs_error < 0.5:  # < 30°
            reward += cfg.alive_bonus * (1.0 - abs_error / 0.5)

        # 9. Crash penalty
        if terminated:
            reward -= cfg.crash_penalty

        return reward

    def _compute_reward_v1(
        self,
        state: QuadrotorState,
        action: np.ndarray,
        terminated: bool,
    ) -> float:
        """Original reward function (v1) for comparison.

        Kept for A/B testing and backward compatibility.
        """
        cfg = self.config
        reward = 0.0

        # 1. Facing reward (exponential)
        yaw_error = self._compute_yaw_error(state)
        facing_reward = np.exp(-cfg.facing_reward_scale * yaw_error**2)
        reward += cfg.facing_reward_weight * facing_reward

        # 2. Yaw rate penalty (smooth control)
        yaw_rate = state.angular_velocity[2]
        reward -= cfg.yaw_rate_penalty_weight * yaw_rate**2

        # 3. Action rate penalty
        action_rate = (float(action[0]) - self._previous_action) ** 2
        reward -= cfg.action_rate_penalty_weight * action_rate

        # 4. Sustained tracking bonus
        if self._time_on_target >= cfg.sustained_tracking_time:
            reward += cfg.sustained_tracking_bonus

        # 5. Alive bonus
        reward += cfg.alive_bonus

        # 6. Crash penalty
        if terminated:
            reward -= cfg.crash_penalty

        return reward

    def _check_termination(self, state: QuadrotorState) -> bool:
        """Check if episode should terminate.

        SITL-style: only terminate on actual crashes, not on tilt.
        The PID stabilizer should recover from any tilt within limits.
        """
        # Ground collision - actual crash
        if state.position[2] < 0.02:
            return True

        # Runaway altitude (something went very wrong)
        if state.position[2] > 10.0:
            return True

        # Extreme tilt (> 80 degrees) - unrecoverable flip
        roll, pitch, _ = Rotations.quaternion_to_euler(state.quaternion)
        tilt = np.sqrt(roll**2 + pitch**2)
        # NO termination for normal tilt - stabilizer should recover
        # NO termination for altitude error - stabilizer should recover
        # Only terminate for truly unrecoverable flip (> 80 degrees)
        return tilt > 1.4

    def _get_info(self) -> dict[str, Any]:
        """Get info dictionary."""
        state = self.sim.get_state()
        yaw_error = self._compute_yaw_error(state)
        roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)

        target_pos = self._current_pattern.get_position(self._time, state.position)

        return {
            "yaw_error": yaw_error,
            "yaw_error_deg": np.degrees(yaw_error),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "altitude": state.position[2],
            "yaw_rate": state.angular_velocity[2],
            "time_on_target": self._time_on_target,
            "target_position": target_pos.copy(),
            "drone_position": state.position.copy(),
            "step_count": self._step_count,
            "is_tracking": abs(yaw_error) < self.config.success_threshold,
        }

    def get_state(self):
        """Get current quadrotor state from simulator.

        Returns:
            QuadrotorState with position, velocity, orientation, etc.
        """
        return self.sim.get_state()

    @property
    def dt(self) -> float:
        """Get control timestep."""
        return self._dt

    def render(self) -> np.ndarray | None:
        """Render the environment with target visualization overlay."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = self.sim.create_renderer(width=640, height=480)

        pixels = self.sim.render(self._renderer)

        if self.render_mode == "rgb_array" and pixels is not None:
            # Add target visualization overlay
            pixels = self._add_target_overlay(pixels)
            return pixels

        return None

    def _add_target_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add target direction and status overlay to frame."""
        import cv2

        frame = frame.copy()
        h, w = frame.shape[:2]

        # Get current state
        state = self.sim.get_state()
        drone_pos = state.position
        target_pos = self._current_pattern.get_position(self._time, state.position)

        # Calculate direction to target
        to_target = target_pos - drone_pos
        target_angle = np.arctan2(to_target[1], to_target[0])

        # Get drone yaw
        drone_yaw = Rotations.quaternion_to_euler(state.quaternion)[2]
        yaw_error = target_angle - drone_yaw
        # Normalize to [-pi, pi]
        while yaw_error > np.pi:
            yaw_error -= 2 * np.pi
        while yaw_error < -np.pi:
            yaw_error += 2 * np.pi

        is_tracking = abs(yaw_error) < self.config.success_threshold

        # Draw compass circle (top-right)
        cx, cy = w - 80, 80
        radius = 60

        # Draw compass background
        cv2.circle(frame, (cx, cy), radius + 5, (40, 40, 40), -1)
        cv2.circle(frame, (cx, cy), radius, (80, 80, 80), 2)

        # Draw North marker
        cv2.line(frame, (cx, cy - radius), (cx, cy - radius + 10), (200, 200, 200), 2)

        # Draw drone direction (green arrow)
        drone_dx = int(radius * 0.7 * np.cos(drone_yaw - np.pi / 2))
        drone_dy = int(radius * 0.7 * np.sin(drone_yaw - np.pi / 2))
        cv2.arrowedLine(
            frame, (cx, cy), (cx + drone_dx, cy + drone_dy), (0, 255, 0), 3, tipLength=0.3
        )

        # Draw target direction (red/yellow arrow)
        target_dx = int(radius * 0.9 * np.cos(target_angle - np.pi / 2))
        target_dy = int(radius * 0.9 * np.sin(target_angle - np.pi / 2))
        target_color = (
            (0, 255, 255) if is_tracking else (0, 0, 255)
        )  # Yellow if tracking, red if not
        cv2.arrowedLine(
            frame, (cx, cy), (cx + target_dx, cy + target_dy), target_color, 2, tipLength=0.25
        )

        # Draw target marker (circle at target direction edge)
        target_marker_x = int(cx + radius * np.cos(target_angle - np.pi / 2))
        target_marker_y = int(cy + radius * np.sin(target_angle - np.pi / 2))
        cv2.circle(frame, (target_marker_x, target_marker_y), 8, target_color, -1)
        cv2.circle(frame, (target_marker_x, target_marker_y), 8, (255, 255, 255), 2)

        # Status text
        status_color = (0, 255, 0) if is_tracking else (0, 0, 255)
        status_text = "TRACKING" if is_tracking else "ACQUIRING"

        # Draw text info panel (top-left)
        cv2.rectangle(frame, (10, 10), (200, 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (200, 100), (100, 100, 100), 1)

        cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(
            frame,
            f"Yaw Error: {np.degrees(yaw_error):+.1f} deg",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"Step: {self._step_count}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # Draw target distance indicator
        target_dist = np.linalg.norm(to_target[:2])
        cv2.putText(
            frame,
            f"Target: {target_dist:.1f}m",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            del self._renderer
            self._renderer = None


# Register environment
gym.register(
    id="DroneYawTracking-v0",
    entry_point="src.environments.yaw_tracking_env:YawTrackingEnv",
)
