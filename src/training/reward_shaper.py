"""
Reward shaping for drone target tracking.

Based on research findings:
- Dense rewards: Better for early training
- Sparse rewards: Better for final performance
- Jerk penalty: Critical for smooth flight
- Dense-to-Sparse transition: Best of both worlds

Reward components:
1. Distance to target (primary)
2. Jerk penalty (action smoothness)
3. Velocity penalty (energy efficiency)
4. Bonus for proximity
5. Penalty for losing target
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class RewardType(Enum):
    """Types of reward functions"""
    DENSE = "dense"
    SPARSE = "sparse"
    DENSE_TO_SPARSE = "dense_to_sparse"
    SHAPED = "shaped"


@dataclass
class RewardWeights:
    """Weights for reward components"""
    # Primary objective
    distance_weight: float = 1.0
    
    # Smoothness (jerk penalty)
    jerk_weight: float = 0.1
    
    # Energy efficiency
    velocity_weight: float = 0.05
    acceleration_weight: float = 0.01
    
    # Bonus/penalties
    proximity_bonus: float = 1.0  # Bonus when close to target
    proximity_threshold: float = 0.5  # meters
    lost_target_penalty: float = 10.0  # Penalty when target too far
    lost_target_threshold: float = 10.0  # meters
    
    # Geofence violation
    geofence_penalty: float = 5.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'distance_weight': self.distance_weight,
            'jerk_weight': self.jerk_weight,
            'velocity_weight': self.velocity_weight,
            'acceleration_weight': self.acceleration_weight,
            'proximity_bonus': self.proximity_bonus,
            'proximity_threshold': self.proximity_threshold,
            'lost_target_penalty': self.lost_target_penalty,
            'lost_target_threshold': self.lost_target_threshold,
            'geofence_penalty': self.geofence_penalty
        }


class RewardShaper:
    """
    Computes rewards for drone tracking task.
    Supports multiple reward formulations and dense-to-sparse transition.
    """
    
    def __init__(
        self,
        reward_type: RewardType = RewardType.DENSE,
        weights: Optional[RewardWeights] = None,
        transition_steps: int = 200_000  # Steps to transition from dense to sparse
    ):
        """
        Initialize reward shaper.
        
        Args:
            reward_type: Type of reward function
            weights: Reward component weights
            transition_steps: Steps for dense-to-sparse transition
        """
        self.reward_type = reward_type
        self.weights = weights or RewardWeights()
        self.transition_steps = transition_steps
        
        # State for tracking
        self.current_step = 0
        self.prev_action = None
        self.prev_velocity = None
        
        # Statistics
        self.total_reward = 0.0
        self.episode_rewards = []
    
    def compute_reward(
        self,
        target_position: np.ndarray,
        drone_position: np.ndarray,
        drone_velocity: np.ndarray,
        action: np.ndarray,
        done: bool = False,
        info: Optional[Dict] = None
    ) -> float:
        """
        Compute reward for current step.
        
        Args:
            target_position: Target position in body frame [x, y, z]
            drone_position: Drone position (for geofence check)
            drone_velocity: Current velocity [vx, vy, vz]
            action: Action taken [vx, vy, vz]
            done: Episode done flag
            info: Additional info (geofence violations, etc.)
            
        Returns:
            reward: Scalar reward
        """
        if self.reward_type == RewardType.DENSE:
            reward = self._compute_dense_reward(
                target_position, drone_velocity, action
            )
        elif self.reward_type == RewardType.SPARSE:
            reward = self._compute_sparse_reward(
                target_position, done
            )
        elif self.reward_type == RewardType.DENSE_TO_SPARSE:
            reward = self._compute_dense_to_sparse_reward(
                target_position, drone_velocity, action, done
            )
        else:  # SHAPED
            reward = self._compute_shaped_reward(
                target_position, drone_velocity, action
            )
        
        # Apply penalties from info
        if info is not None:
            if not info.get('geofence_ok', True):
                reward -= self.weights.geofence_penalty
        
        # Update state
        self.prev_action = action.copy()
        self.prev_velocity = drone_velocity.copy()
        self.total_reward += reward
        
        return reward
    
    def _compute_dense_reward(
        self,
        target_position: np.ndarray,
        drone_velocity: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Dense reward for continuous feedback.
        
        Reward = -distance - jerk - velocity
        
        Good for: Early training, fast convergence
        Bad for: Can lead to local optima
        """
        # Distance to target (primary objective)
        distance = np.linalg.norm(target_position)
        distance_reward = -self.weights.distance_weight * distance
        
        # Jerk penalty (smoothness)
        jerk_penalty = 0.0
        if self.prev_action is not None:
            jerk = np.linalg.norm(action - self.prev_action)
            jerk_penalty = -self.weights.jerk_weight * jerk
        
        # Velocity penalty (energy efficiency)
        velocity_magnitude = np.linalg.norm(drone_velocity)
        velocity_penalty = -self.weights.velocity_weight * velocity_magnitude
        
        # Proximity bonus
        proximity_bonus = 0.0
        if distance < self.weights.proximity_threshold:
            proximity_bonus = self.weights.proximity_bonus
        
        # Lost target penalty
        lost_penalty = 0.0
        if distance > self.weights.lost_target_threshold:
            lost_penalty = -self.weights.lost_target_penalty
        
        reward = (
            distance_reward +
            jerk_penalty +
            velocity_penalty +
            proximity_bonus +
            lost_penalty
        )
        
        return reward
    
    def _compute_sparse_reward(
        self,
        target_position: np.ndarray,
        done: bool
    ) -> float:
        """
        Sparse reward for goal achievement.
        
        Reward = +10 if close to target, -1 for timeout
        
        Good for: Final performance, generalization
        Bad for: Slow convergence, exploration issues
        """
        distance = np.linalg.norm(target_position)
        
        # Success: Close to target
        if distance < self.weights.proximity_threshold:
            return 10.0
        
        # Failure: Episode timeout or lost target
        if done or distance > self.weights.lost_target_threshold:
            return -1.0
        
        # Neutral otherwise
        return 0.0
    
    def _compute_dense_to_sparse_reward(
        self,
        target_position: np.ndarray,
        drone_velocity: np.ndarray,
        action: np.ndarray,
        done: bool
    ) -> float:
        """
        Interpolate between dense and sparse rewards.
        
        Progress: 0.0 (start) -> 1.0 (end of transition)
        Reward = (1 - progress) * dense + progress * sparse
        
        Best of both worlds: Fast convergence + good final performance
        """
        progress = min(1.0, self.current_step / self.transition_steps)
        
        dense = self._compute_dense_reward(target_position, drone_velocity, action)
        sparse = self._compute_sparse_reward(target_position, done)
        
        # Linear interpolation
        reward = (1.0 - progress) * dense + progress * sparse
        
        return reward
    
    def _compute_shaped_reward(
        self,
        target_position: np.ndarray,
        drone_velocity: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Custom shaped reward with advanced features.
        
        Includes:
        - Velocity alignment (moving toward target)
        - Acceleration penalty
        - Progressive distance scaling
        """
        distance = np.linalg.norm(target_position)
        
        # Distance with diminishing returns
        if distance < 1.0:
            distance_reward = -distance ** 0.5  # Sqrt for diminishing returns
        else:
            distance_reward = -distance
        distance_reward *= self.weights.distance_weight
        
        # Velocity alignment (are we moving toward target?)
        if distance > 0.1:
            target_direction = target_position / distance
            velocity_alignment = np.dot(drone_velocity, target_direction)
            alignment_reward = 0.1 * velocity_alignment
        else:
            alignment_reward = 0.0
        
        # Jerk penalty
        jerk_penalty = 0.0
        if self.prev_action is not None:
            jerk = np.linalg.norm(action - self.prev_action)
            jerk_penalty = -self.weights.jerk_weight * jerk
        
        # Acceleration penalty (energy cost)
        accel_penalty = 0.0
        if self.prev_velocity is not None:
            acceleration = np.linalg.norm(drone_velocity - self.prev_velocity)
            accel_penalty = -self.weights.acceleration_weight * acceleration
        
        # Proximity bonus (exponential near target)
        if distance < self.weights.proximity_threshold:
            proximity_bonus = self.weights.proximity_bonus * np.exp(-distance)
        else:
            proximity_bonus = 0.0
        
        reward = (
            distance_reward +
            alignment_reward +
            jerk_penalty +
            accel_penalty +
            proximity_bonus
        )
        
        return reward
    
    def step(self):
        """Increment step counter (for dense-to-sparse transition)"""
        self.current_step += 1
    
    def reset(self):
        """Reset episode state"""
        self.prev_action = None
        self.prev_velocity = None
        
        if self.total_reward != 0.0:
            self.episode_rewards.append(self.total_reward)
        self.total_reward = 0.0
    
    def get_statistics(self) -> Dict:
        """Get reward statistics"""
        if len(self.episode_rewards) == 0:
            return {
                'total_episodes': 0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'min_reward': 0.0,
                'max_reward': 0.0,
                'current_step': self.current_step,
                'transition_progress': min(1.0, self.current_step / self.transition_steps)
            }
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'current_step': self.current_step,
            'transition_progress': min(1.0, self.current_step / self.transition_steps)
        }
    
    def set_weights(self, weights: RewardWeights):
        """Update reward weights (for curriculum learning)"""
        self.weights = weights
    
    def get_transition_progress(self) -> float:
        """Get progress through dense-to-sparse transition [0, 1]"""
        return min(1.0, self.current_step / self.transition_steps)


# ============================================================================
# Pre-configured reward shapers for different training stages
# ============================================================================

def get_curriculum_reward_shaper(stage: int) -> RewardShaper:
    """
    Get reward shaper configured for curriculum learning stage.
    
    Stage 1: Dense rewards, high smoothness penalty
    Stage 2: Dense rewards, balanced
    Stage 3: Dense-to-sparse transition
    
    Args:
        stage: Curriculum stage (1, 2, or 3)
        
    Returns:
        Configured reward shaper
    """
    if stage == 1:
        # Stage 1: Dense, emphasize smoothness
        weights = RewardWeights(
            distance_weight=0.5,  # Lower weight (learning to move)
            jerk_weight=0.2,  # High smoothness
            velocity_weight=0.05,
            proximity_threshold=1.0,  # Larger threshold
            proximity_bonus=0.5
        )
        return RewardShaper(RewardType.DENSE, weights)
    
    elif stage == 2:
        # Stage 2: Dense, balanced
        weights = RewardWeights(
            distance_weight=1.0,  # Normal weight
            jerk_weight=0.1,  # Standard smoothness
            velocity_weight=0.05,
            proximity_threshold=0.5,  # Standard threshold
            proximity_bonus=1.0
        )
        return RewardShaper(RewardType.DENSE, weights)
    
    elif stage == 3:
        # Stage 3: Dense-to-sparse transition
        weights = RewardWeights(
            distance_weight=1.0,
            jerk_weight=0.1,
            velocity_weight=0.05,
            proximity_threshold=0.3,  # Tighter threshold
            proximity_bonus=2.0  # Higher bonus
        )
        return RewardShaper(RewardType.DENSE_TO_SPARSE, weights, transition_steps=100_000)
    
    else:
        raise ValueError(f"Unknown curriculum stage: {stage}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== Reward Shaper Tests ===\n")
    
    # Test 1: Dense reward
    print("Test 1: Dense Reward")
    dense_shaper = RewardShaper(RewardType.DENSE)
    
    # Simulate tracking episode
    target = np.array([5.0, 2.0, -1.0])  # 5m forward, 2m right, 1m up
    drone_pos = np.array([0.0, 0.0, 0.0])
    drone_vel = np.array([1.0, 0.4, -0.2])  # Moving toward target
    action = np.array([1.2, 0.4, -0.2])  # Smooth action
    
    reward = dense_shaper.compute_reward(target, drone_pos, drone_vel, action)
    print(f"  Target distance: {np.linalg.norm(target):.2f}m")
    print(f"  Dense reward: {reward:.3f}")
    print()
    
    # Test 2: Sparse reward
    print("Test 2: Sparse Reward")
    sparse_shaper = RewardShaper(RewardType.SPARSE)
    
    # Far from target
    reward_far = sparse_shaper.compute_reward(target, drone_pos, drone_vel, action)
    print(f"  Reward when far: {reward_far:.3f}")
    
    # Close to target
    target_close = np.array([0.3, 0.2, -0.1])  # Within 0.5m
    reward_close = sparse_shaper.compute_reward(target_close, drone_pos, drone_vel, action)
    print(f"  Reward when close: {reward_close:.3f}")
    print()
    
    # Test 3: Dense-to-Sparse transition
    print("Test 3: Dense-to-Sparse Transition")
    transition_shaper = RewardShaper(RewardType.DENSE_TO_SPARSE, transition_steps=1000)
    
    print("  Progress through transition:")
    for step in [0, 250, 500, 750, 1000]:
        transition_shaper.current_step = step
        reward = transition_shaper.compute_reward(target, drone_pos, drone_vel, action)
        progress = transition_shaper.get_transition_progress()
        print(f"    Step {step:4d}: progress={progress:.2f}, reward={reward:+.3f}")
    print()
    
    # Test 4: Shaped reward with alignment
    print("Test 4: Shaped Reward (velocity alignment)")
    shaped_shaper = RewardShaper(RewardType.SHAPED)
    
    # Good: Moving toward target
    drone_vel_good = np.array([1.0, 0.4, -0.2])  # Aligned with target direction
    reward_good = shaped_shaper.compute_reward(target, drone_pos, drone_vel_good, action)
    
    # Bad: Moving away from target
    shaped_shaper.reset()
    drone_vel_bad = np.array([-1.0, -0.4, 0.2])  # Opposite direction
    reward_bad = shaped_shaper.compute_reward(target, drone_pos, drone_vel_bad, action)
    
    print(f"  Reward (moving toward): {reward_good:+.3f}")
    print(f"  Reward (moving away):   {reward_bad:+.3f}")
    print()
    
    # Test 5: Jerk penalty
    print("Test 5: Jerk Penalty (smoothness)")
    jerk_shaper = RewardShaper(RewardType.DENSE)
    
    # First action
    action1 = np.array([1.0, 0.5, -0.2])
    reward1 = jerk_shaper.compute_reward(target, drone_pos, drone_vel, action1)
    
    # Smooth action (low jerk)
    action2_smooth = np.array([1.1, 0.5, -0.2])
    reward2_smooth = jerk_shaper.compute_reward(target, drone_pos, drone_vel, action2_smooth)
    
    # Jerky action (high jerk)
    jerk_shaper.reset()
    jerk_shaper.compute_reward(target, drone_pos, drone_vel, action1)
    action2_jerky = np.array([3.0, -1.0, 0.5])
    reward2_jerky = jerk_shaper.compute_reward(target, drone_pos, drone_vel, action2_jerky)
    
    print(f"  Smooth action: {reward2_smooth:+.3f}")
    print(f"  Jerky action:  {reward2_jerky:+.3f}")
    print()
    
    # Test 6: Curriculum stages
    print("Test 6: Curriculum Stages")
    for stage in [1, 2, 3]:
        shaper = get_curriculum_reward_shaper(stage)
        reward = shaper.compute_reward(target, drone_pos, drone_vel, action)
        print(f"  Stage {stage}: reward={reward:+.3f}, type={shaper.reward_type.value}")
    print()
    
    # Test 7: Statistics
    print("Test 7: Episode Statistics")
    stats_shaper = RewardShaper(RewardType.DENSE)
    
    # Simulate 3 episodes
    for ep in range(3):
        for _ in range(10):
            reward = stats_shaper.compute_reward(target, drone_pos, drone_vel, action)
            stats_shaper.step()
        stats_shaper.reset()
    
    stats = stats_shaper.get_statistics()
    print(f"  Episodes: {stats['total_episodes']}")
    print(f"  Mean reward: {stats['mean_reward']:.3f}")
    print(f"  Std reward: {stats['std_reward']:.3f}")
    print(f"  Current step: {stats['current_step']}")
    
    print("\n=== All tests complete ===")
