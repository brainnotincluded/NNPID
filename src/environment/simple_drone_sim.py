"""
Simple Python Drone Simulator for RL Training.

Pure Python implementation - no Webots, no ArduPilot SITL.
Uses artificial data from trajectory generator + physics simulation.

Perfect for:
- Fast training iterations
- Testing algorithms
- Curriculum learning
- Later: Transfer to real drone
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from src.utils.trajectory_generator import TrajectoryGenerator, TrajectoryConfig, TrajectoryType
from src.utils.coordinate_transforms import CoordinateTransform
from src.utils.domain_randomization import (
    DomainRandomizer, DronePhysicsParams, EnvironmentParams
)
from src.utils.safety import SafetyMonitor, SafetyLimits
from src.training.reward_shaper import RewardShaper, RewardType


@dataclass
class DroneState:
    """Complete drone state"""
    position_ned: np.ndarray  # [north, east, down] in meters
    velocity_ned: np.ndarray  # [vn, ve, vd] in m/s
    velocity_body: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # [roll, pitch, yaw] in radians
    angular_velocity: np.ndarray  # [p, q, r] in rad/s
    
    def copy(self):
        """Create a copy of the state"""
        return DroneState(
            position_ned=self.position_ned.copy(),
            velocity_ned=self.velocity_ned.copy(),
            velocity_body=self.velocity_body.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy()
        )


class SimpleDroneSimulator:
    """
    Simplified drone physics simulator.
    
    Assumptions (for speed):
    - Small angle approximation (no complex aerodynamics)
    - Velocity control (not PWM/thrust)
    - 2D tracking (yaw fixed, can be extended to 3D)
    - Simple first-order dynamics
    """
    
    def __init__(
        self,
        dt: float = 0.05,  # 20 Hz
        max_episode_steps: int = 1000,
        use_domain_randomization: bool = True,
        use_safety: bool = True,
        reward_type: RewardType = RewardType.DENSE_TO_SPARSE,
        trajectory_type: TrajectoryType = TrajectoryType.LISSAJOUS_PERLIN,
        seed: Optional[int] = None,
        # Latency simulation (for sim-to-real transfer)
        action_latency_range: Tuple[float, float] = (0.02, 0.08),  # 20-80ms
        # Sensor noise
        position_noise_std: float = 0.02,  # meters
        velocity_noise_std: float = 0.05,  # m/s
    ):
        """
        Initialize simulator.
        
        Args:
            dt: Timestep in seconds
            max_episode_steps: Maximum steps per episode
            use_domain_randomization: Enable physics randomization
            use_safety: Enable safety checks
            reward_type: Type of reward function
            trajectory_type: Type of target trajectory
            seed: Random seed
        """
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.use_domain_randomization = use_domain_randomization
        self.use_safety = use_safety
        
        # Latency simulation
        self.action_latency_range = action_latency_range
        self.action_buffer = []  # Queue of (time, action) tuples
        self.current_time = 0.0
        self.current_latency = 0.0  # Randomized per episode
        
        # Sensor noise
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        
        # Random state
        self.rng = np.random.RandomState(seed)
        
        # Trajectory generator
        traj_config = TrajectoryConfig(
            trajectory_type=trajectory_type,
            duration=max_episode_steps * dt,
            dt=dt
        )
        self.trajectory_generator = TrajectoryGenerator(traj_config, seed=seed)
        
        # Domain randomization
        nominal_drone = DronePhysicsParams()
        nominal_env = EnvironmentParams()
        self.domain_randomizer = DomainRandomizer(
            nominal_drone, nominal_env,
            randomization_level=1.0 if use_domain_randomization else 0.0,
            seed=seed
        )
        
        # Safety monitor
        home_position = np.array([0.0, 0.0, -2.0])  # 2m altitude
        safety_limits = SafetyLimits(
            max_distance_from_home=20.0,
            min_altitude=0.5,
            max_altitude=10.0,
            max_horizontal_velocity=5.0,
            max_vertical_velocity=3.0
        )
        self.safety_monitor = SafetyMonitor(
            safety_limits, home_position, enable_fallback=use_safety
        ) if use_safety else None
        
        # Reward shaper
        self.reward_shaper = RewardShaper(reward_type)
        
        # State
        self.drone_state: Optional[DroneState] = None
        self.target_trajectory: Optional[np.ndarray] = None
        self.target_velocities: Optional[np.ndarray] = None
        self.current_step = 0
        self.prev_action = np.zeros(3)
        
        # Episode randomization
        self.current_drone_params: Optional[DronePhysicsParams] = None
        self.current_env_params: Optional[EnvironmentParams] = None
        
        # Observation/action dimensions
        self.obs_dim = 12  # [target_xyz_body, drone_vel_body, prev_action_xyz, error_xy, distance]
        self.action_dim = 3  # [vx, vy, vz] velocity commands
    
    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Returns:
            observation: Initial observation
        """
        # Randomize physics for this episode
        if self.use_domain_randomization:
            self.current_drone_params, self.current_env_params = \
                self.domain_randomizer.randomize_episode()
        else:
            self.current_drone_params = DronePhysicsParams()
            self.current_env_params = EnvironmentParams()
        
        # Generate new target trajectory
        self.target_trajectory, self.target_velocities = \
            self.trajectory_generator.generate()
        
        # Initialize drone state (start at origin, hovering)
        self.drone_state = DroneState(
            position_ned=np.array([0.0, 0.0, -2.0]),  # 2m altitude
            velocity_ned=np.zeros(3),
            velocity_body=np.zeros(3),
            orientation=np.array([0.0, 0.0, 0.0]),  # Level, facing north
            angular_velocity=np.zeros(3)
        )
        
        # Reset step counter
        self.current_step = 0
        self.prev_action = np.zeros(3)
        self.current_time = 0.0
        self.action_buffer = []
        
        # Randomize latency for this episode
        self.current_latency = self.rng.uniform(
            self.action_latency_range[0],
            self.action_latency_range[1]
        )
        
        # Reset reward shaper
        self.reward_shaper.reset()
        if self.safety_monitor:
            self.safety_monitor.reset()
        
        # Get initial observation
        obs = self._compute_observation()
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one simulation step.
        
        Args:
            action: Velocity command [vx, vy, vz] in body frame (m/s)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Episode done flag
            info: Additional information
        """
        # Clip action to reasonable range (before safety layer)
        action = np.clip(action, -5.0, 5.0)
        
        # Add action to latency buffer with timestamp
        self.action_buffer.append((self.current_time + self.current_latency, action.copy()))
        
        # Get delayed action (actions that have completed their latency)
        delayed_action = np.zeros(3)
        while self.action_buffer and self.action_buffer[0][0] <= self.current_time:
            _, delayed_action = self.action_buffer.pop(0)
        
        # If no action is ready yet, use previous action or zero
        if np.all(delayed_action == 0) and self.current_step > 0:
            delayed_action = self.prev_action.copy()
        
        action = delayed_action
        
        # Safety layer filtering
        info = {'geofence_ok': True, 'action_filtered': False, 'used_fallback': False}
        
        if self.safety_monitor:
            # Get target position in body frame for fallback PID
            target_pos_body = self._get_target_position_body()
            
            # Safety check and filter
            safe_action, status, safety_info = self.safety_monitor.check_and_filter(
                position=self.drone_state.position_ned,
                velocity_command=action,
                current_velocity=self.drone_state.velocity_body,
                target_position=target_pos_body,
                inference_time=0.01,  # Dummy value (we're not measuring actual NN time)
                dt=self.dt
            )
            action = safe_action
            info.update(safety_info)
        
        # Simulate drone physics
        self._simulate_physics(action)
        
        # Increment step and time
        self.current_step += 1
        self.current_time += self.dt
        self.prev_action = action.copy()
        
        # Compute observation
        obs = self._compute_observation()
        
        # Compute reward
        target_pos_body = self._get_target_position_body()
        reward = self.reward_shaper.compute_reward(
            target_position=target_pos_body,
            drone_position=self.drone_state.position_ned,
            drone_velocity=self.drone_state.velocity_body,
            action=action,
            done=False,
            info=info
        )
        self.reward_shaper.step()
        
        # Check done condition
        done = self._check_done()
        
        # Additional info
        info.update({
            'step': self.current_step,
            'target_distance': np.linalg.norm(target_pos_body),
            'drone_speed': np.linalg.norm(self.drone_state.velocity_body)
        })
        
        return obs, reward, done, info
    
    def _simulate_physics(self, velocity_command: np.ndarray):
        """
        Simulate drone physics for one timestep.
        
        Simple first-order dynamics:
        - Velocity tracking with time constant
        - Position integration
        - Wind disturbance
        """
        # Current state
        pos = self.drone_state.position_ned
        vel = self.drone_state.velocity_ned
        
        # Convert velocity command from body to NED frame
        # (For simplicity, assume level flight: roll=pitch=0)
        yaw = self.drone_state.orientation[2]
        vel_command_ned = np.array([
            velocity_command[0] * np.cos(yaw) - velocity_command[1] * np.sin(yaw),
            velocity_command[0] * np.sin(yaw) + velocity_command[1] * np.cos(yaw),
            velocity_command[2]
        ])
        
        # First-order dynamics: velocity tracking
        # tau * dv/dt = (v_cmd - v) → v_new = v + dt/tau * (v_cmd - v)
        tau = self.current_drone_params.motor_time_constant
        vel_error = vel_command_ned - vel
        vel_new = vel + (self.dt / tau) * vel_error
        
        # Add drag (velocity damping)
        drag_coeff = self.current_drone_params.drag_coefficient
        vel_new *= (1.0 - drag_coeff * self.dt)
        
        # Add wind disturbance
        wind = self.current_env_params.wind_velocity
        if self.current_env_params.wind_turbulence > 0:
            wind_turbulence = self.rng.randn(3) * self.current_env_params.wind_turbulence
            wind = wind + wind_turbulence
        vel_new += wind * self.dt * 0.1  # Wind effect (simplified)
        
        # Integrate position
        pos_new = pos + vel_new * self.dt
        
        # Update state
        self.drone_state.position_ned = pos_new
        self.drone_state.velocity_ned = vel_new
        
        # Update velocity in body frame (for observation)
        self.drone_state.velocity_body = np.array([
            vel_new[0] * np.cos(yaw) + vel_new[1] * np.sin(yaw),
            -vel_new[0] * np.sin(yaw) + vel_new[1] * np.cos(yaw),
            vel_new[2]
        ])
    
    def _get_target_position_body(self) -> np.ndarray:
        """Get target position in drone body frame"""
        # Get target position from trajectory (clamp to valid index)
        step_idx = min(self.current_step, len(self.target_trajectory) - 1)
        target_pos_ned = self.target_trajectory[step_idx]
        
        # Convert to body frame
        target_pos_body = CoordinateTransform.ned_to_body(
            target_pos_ned,
            self.drone_state.position_ned,
            self.drone_state.orientation,
            orientation_type="euler"
        )
        
        return target_pos_body
    
    def _compute_observation(self) -> np.ndarray:
        """
        Compute observation vector.
        
        Observation space (12D):
        - target_xyz_body: Target position in body frame [3]
        - drone_vel_body: Drone velocity in body frame [3]
        - prev_action_xyz: Previous action [3]
        - error_xy: Horizontal error [2]
        - distance: Distance to target [1]
        """
        # Target position in body frame
        target_pos_body = self._get_target_position_body()
        
        # Drone velocity in body frame
        drone_vel_body = self.drone_state.velocity_body
        
        # Previous action
        prev_action = self.prev_action
        
        # Compute error metrics
        error_xy = target_pos_body[:2]  # Horizontal error
        distance = np.linalg.norm(target_pos_body)
        
        # Add sensor noise
        target_pos_noisy = target_pos_body + self.rng.randn(3) * self.position_noise_std
        drone_vel_noisy = drone_vel_body + self.rng.randn(3) * self.velocity_noise_std
        
        # Recompute error metrics with noise
        error_xy_noisy = target_pos_noisy[:2]
        distance_noisy = np.linalg.norm(target_pos_noisy)
        
        # Assemble observation (with sensor noise)
        obs = np.concatenate([
            target_pos_noisy,     # [3] - target in body frame (noisy)
            drone_vel_noisy,      # [3] - current velocity (noisy)
            prev_action,          # [3] - previous action
            error_xy_noisy,       # [2] - horizontal error (noisy)
            [distance_noisy]      # [1] - distance to target (noisy)
        ])
        
        return obs.astype(np.float32)
    
    def _check_done(self) -> bool:
        """Check if episode is done"""
        # Max steps reached (use len-1 to avoid trajectory overflow)
        if self.current_step >= min(self.max_episode_steps, len(self.target_trajectory) - 1):
            return True
        
        # Target lost (too far)
        target_pos_body = self._get_target_position_body()
        if np.linalg.norm(target_pos_body) > 15.0:
            return True
        
        # Crashed (too low)
        altitude = -self.drone_state.position_ned[2]
        if altitude < 0.2:
            return True
        
        return False
    
    def render(self):
        """Simple text rendering (for debugging)"""
        target_pos_body = self._get_target_position_body()
        distance = np.linalg.norm(target_pos_body)
        
        print(f"Step {self.current_step:4d} | "
              f"Distance: {distance:6.2f}m | "
              f"Altitude: {-self.drone_state.position_ned[2]:5.2f}m | "
              f"Speed: {np.linalg.norm(self.drone_state.velocity_body):5.2f}m/s")
    
    def get_state_dict(self) -> Dict:
        """Get complete state for logging/visualization"""
        step_idx = min(self.current_step, len(self.target_trajectory) - 1)
        return {
            'step': self.current_step,
            'drone_pos_ned': self.drone_state.position_ned.copy(),
            'drone_vel_ned': self.drone_state.velocity_ned.copy(),
            'target_pos_ned': self.target_trajectory[step_idx].copy(),
            'target_vel_ned': self.target_velocities[step_idx].copy(),
            'distance': np.linalg.norm(self._get_target_position_body()),
            'current_latency_ms': self.current_latency * 1000,
            'time': self.current_time
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=== Simple Drone Simulator Test ===\n")
    
    # Create simulator
    sim = SimpleDroneSimulator(
        dt=0.05,
        max_episode_steps=200,  # 10 seconds
        use_domain_randomization=True,
        use_safety=True,
        trajectory_type=TrajectoryType.LISSAJOUS
    )
    
    print(f"Observation dim: {sim.obs_dim}")
    print(f"Action dim: {sim.action_dim}")
    print()
    
    # Run one episode with random policy
    print("Running episode with random policy:")
    obs = sim.reset()
    
    total_reward = 0.0
    for step in range(200):
        # Random action
        action = np.random.randn(3) * 0.5
        
        # Step
        obs, reward, done, info = sim.step(action)
        total_reward += reward
        
        # Render every 20 steps
        if step % 20 == 0:
            sim.render()
        
        if done:
            print(f"\nEpisode done at step {step}")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Final distance: {info['target_distance']:.2f}m")
    
    # Test multiple episodes
    print("\n=== Testing 5 episodes ===")
    rewards = []
    for ep in range(5):
        obs = sim.reset()
        ep_reward = 0.0
        
        for step in range(200):
            action = np.random.randn(3) * 0.5
            obs, reward, done, info = sim.step(action)
            ep_reward += reward
            
            if done:
                break
        
        rewards.append(ep_reward)
        print(f"Episode {ep+1}: reward={ep_reward:7.2f}, steps={step+1}")
    
    print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print("\n=== Test complete ===")
