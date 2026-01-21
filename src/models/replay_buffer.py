"""
Recurrent Replay Buffer for RSAC.

Key differences from standard replay buffer:
- Stores entire episodes (not individual transitions)
- Preserves temporal ordering for BPTT through GRU
- Supports variable-length episodes with padding/masking
- Efficient sampling of episode chunks

Critical for recurrent policies: You cannot shuffle transitions randomly.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Episode:
    """Container for a single episode's data"""
    observations: np.ndarray  # [seq_len, obs_dim]
    actions: np.ndarray  # [seq_len, action_dim]
    rewards: np.ndarray  # [seq_len]
    dones: np.ndarray  # [seq_len] (terminal flags)
    next_observations: np.ndarray  # [seq_len, obs_dim]
    
    # Metadata
    length: int
    total_reward: float
    
    def __post_init__(self):
        """Validate episode data"""
        assert len(self.observations) == len(self.actions) == len(self.rewards)
        assert len(self.observations) == self.length
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'next_observations': self.next_observations,
            'length': self.length,
            'total_reward': self.total_reward
        }


class RecurrentReplayBuffer:
    """
    Replay buffer for recurrent policies.
    Stores complete episodes and samples episode chunks for training.
    """
    
    def __init__(
        self,
        capacity: int = 1000,  # Number of episodes
        obs_dim: int = 12,
        action_dim: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of episodes to store
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: Device for torch tensors
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Episode storage
        self.episodes: deque = deque(maxlen=capacity)
        
        # Statistics
        self.total_steps = 0
        self.num_episodes = 0
    
    def add_episode(self, episode: Episode):
        """
        Add complete episode to buffer.
        
        Args:
            episode: Episode data
        """
        self.episodes.append(episode)
        self.total_steps += episode.length
        self.num_episodes += 1
    
    def add_trajectory(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray
    ):
        """
        Add trajectory as episode.
        
        Args:
            observations: Observations [seq_len, obs_dim]
            actions: Actions [seq_len, action_dim]
            rewards: Rewards [seq_len]
            dones: Done flags [seq_len]
            next_observations: Next observations [seq_len, obs_dim]
        """
        episode = Episode(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_observations=next_observations,
            length=len(observations),
            total_reward=np.sum(rewards)
        )
        self.add_episode(episode)
    
    def sample_episodes(
        self,
        batch_size: int,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Sample random complete episodes.
        
        Args:
            batch_size: Number of episodes to sample
            return_tensors: If True, return torch tensors; else numpy arrays
            
        Returns:
            Dictionary with batched episode data (padded to max length)
        """
        if len(self.episodes) < batch_size:
            raise ValueError(f"Not enough episodes in buffer ({len(self.episodes)} < {batch_size})")
        
        # Sample random episodes
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=False)
        sampled_episodes = [self.episodes[i] for i in indices]
        
        # Get max length for padding
        max_len = max(ep.length for ep in sampled_episodes)
        
        # Pad and stack episodes
        obs_batch = np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32)
        action_batch = np.zeros((batch_size, max_len, self.action_dim), dtype=np.float32)
        reward_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        done_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32)
        mask_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        
        for i, episode in enumerate(sampled_episodes):
            length = episode.length
            obs_batch[i, :length] = episode.observations
            action_batch[i, :length] = episode.actions
            reward_batch[i, :length] = episode.rewards
            done_batch[i, :length] = episode.dones
            next_obs_batch[i, :length] = episode.next_observations
            mask_batch[i, :length] = 1.0  # 1 = valid, 0 = padding
        
        if return_tensors:
            return {
                'observations': torch.FloatTensor(obs_batch).to(self.device),
                'actions': torch.FloatTensor(action_batch).to(self.device),
                'rewards': torch.FloatTensor(reward_batch).to(self.device),
                'dones': torch.FloatTensor(done_batch).to(self.device),
                'next_observations': torch.FloatTensor(next_obs_batch).to(self.device),
                'masks': torch.FloatTensor(mask_batch).to(self.device)
            }
        else:
            return {
                'observations': obs_batch,
                'actions': action_batch,
                'rewards': reward_batch,
                'dones': done_batch,
                'next_observations': next_obs_batch,
                'masks': mask_batch
            }
    
    def sample_chunks(
        self,
        batch_size: int,
        chunk_length: int = 50,
        overlap: int = 10,
        return_tensors: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Sample fixed-length chunks from episodes (for efficient BPTT).
        
        From research: Chunking episodes improves training stability and speed.
        Overlap between chunks maintains some temporal context.
        
        Args:
            batch_size: Number of chunks to sample
            chunk_length: Length of each chunk
            overlap: Overlap between consecutive chunks
            return_tensors: If True, return torch tensors
            
        Returns:
            Dictionary with batched chunk data
        """
        if len(self.episodes) == 0:
            raise ValueError("Buffer is empty")
        
        # Build list of valid chunk starts
        chunk_starts = []
        for ep_idx, episode in enumerate(self.episodes):
            if episode.length >= chunk_length:
                # Full chunks
                for start in range(0, episode.length - chunk_length + 1, chunk_length - overlap):
                    chunk_starts.append((ep_idx, start))
            elif episode.length > overlap:
                # Short episodes (one chunk with padding)
                chunk_starts.append((ep_idx, 0))
        
        if len(chunk_starts) < batch_size:
            # Not enough chunks, sample with replacement
            indices = np.random.choice(len(chunk_starts), size=batch_size, replace=True)
        else:
            indices = np.random.choice(len(chunk_starts), size=batch_size, replace=False)
        
        # Extract chunks
        obs_batch = np.zeros((batch_size, chunk_length, self.obs_dim), dtype=np.float32)
        action_batch = np.zeros((batch_size, chunk_length, self.action_dim), dtype=np.float32)
        reward_batch = np.zeros((batch_size, chunk_length), dtype=np.float32)
        done_batch = np.zeros((batch_size, chunk_length), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, chunk_length, self.obs_dim), dtype=np.float32)
        mask_batch = np.zeros((batch_size, chunk_length), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            ep_idx, start = chunk_starts[idx]
            episode = self.episodes[ep_idx]
            
            end = min(start + chunk_length, episode.length)
            length = end - start
            
            obs_batch[i, :length] = episode.observations[start:end]
            action_batch[i, :length] = episode.actions[start:end]
            reward_batch[i, :length] = episode.rewards[start:end]
            done_batch[i, :length] = episode.dones[start:end]
            next_obs_batch[i, :length] = episode.next_observations[start:end]
            mask_batch[i, :length] = 1.0
        
        if return_tensors:
            return {
                'observations': torch.FloatTensor(obs_batch).to(self.device),
                'actions': torch.FloatTensor(action_batch).to(self.device),
                'rewards': torch.FloatTensor(reward_batch).to(self.device),
                'dones': torch.FloatTensor(done_batch).to(self.device),
                'next_observations': torch.FloatTensor(next_obs_batch).to(self.device),
                'masks': torch.FloatTensor(mask_batch).to(self.device)
            }
        else:
            return {
                'observations': obs_batch,
                'actions': action_batch,
                'rewards': reward_batch,
                'dones': done_batch,
                'next_observations': next_obs_batch,
                'masks': mask_batch
            }
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics"""
        if len(self.episodes) == 0:
            return {
                'num_episodes': 0,
                'total_steps': 0,
                'avg_episode_length': 0.0,
                'avg_episode_reward': 0.0,
                'buffer_capacity': self.capacity
            }
        
        episode_lengths = [ep.length for ep in self.episodes]
        episode_rewards = [ep.total_reward for ep in self.episodes]
        
        return {
            'num_episodes': len(self.episodes),
            'total_steps': self.total_steps,
            'avg_episode_length': np.mean(episode_lengths),
            'avg_episode_reward': np.mean(episode_rewards),
            'min_episode_reward': np.min(episode_rewards),
            'max_episode_reward': np.max(episode_rewards),
            'buffer_capacity': self.capacity,
            'buffer_usage': len(self.episodes) / self.capacity
        }
    
    def clear(self):
        """Clear buffer"""
        self.episodes.clear()
        self.total_steps = 0
        self.num_episodes = 0
    
    def __len__(self) -> int:
        """Number of episodes in buffer"""
        return len(self.episodes)


# ============================================================================
# Prioritized Replay (Optional - for advanced usage)
# ============================================================================

class PrioritizedRecurrentReplayBuffer(RecurrentReplayBuffer):
    """
    Prioritized replay buffer for recurrent policies.
    Samples episodes based on TD error (harder experiences = higher priority).
    
    Optional: Use only if you observe training instability.
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        obs_dim: int = 12,
        action_dim: int = 3,
        device: str = 'cpu',
        alpha: float = 0.6,  # Prioritization exponent
        beta: float = 0.4,  # Importance sampling exponent
        beta_increment: float = 0.001
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            alpha: Prioritization strength (0 = uniform, 1 = full priority)
            beta: Importance sampling correction (0 = no correction, 1 = full)
            beta_increment: Increment beta per sample
        """
        super().__init__(capacity, obs_dim, action_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Priority storage (parallel to episodes deque)
        self.priorities = deque(maxlen=capacity)
    
    def add_episode(self, episode: Episode, priority: Optional[float] = None):
        """Add episode with priority"""
        super().add_episode(episode)
        
        if priority is None:
            priority = self.max_priority
        
        self.priorities.append(priority ** self.alpha)
    
    def sample_episodes(
        self,
        batch_size: int,
        return_tensors: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample episodes with prioritization.
        
        Returns:
            batch: Episode data
            indices: Sampled episode indices
            weights: Importance sampling weights
        """
        if len(self.episodes) < batch_size:
            raise ValueError(f"Not enough episodes ({len(self.episodes)} < {batch_size})")
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample with replacement based on priorities
        indices = np.random.choice(
            len(self.episodes),
            size=batch_size,
            replace=False,
            p=probs
        )
        
        # Importance sampling weights
        weights = (len(self.episodes) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get episodes
        sampled_episodes = [self.episodes[i] for i in indices]
        
        # Pad and stack (same as parent class)
        max_len = max(ep.length for ep in sampled_episodes)
        
        obs_batch = np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32)
        action_batch = np.zeros((batch_size, max_len, self.action_dim), dtype=np.float32)
        reward_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        done_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, max_len, self.obs_dim), dtype=np.float32)
        mask_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        
        for i, episode in enumerate(sampled_episodes):
            length = episode.length
            obs_batch[i, :length] = episode.observations
            action_batch[i, :length] = episode.actions
            reward_batch[i, :length] = episode.rewards
            done_batch[i, :length] = episode.dones
            next_obs_batch[i, :length] = episode.next_observations
            mask_batch[i, :length] = 1.0
        
        if return_tensors:
            batch = {
                'observations': torch.FloatTensor(obs_batch).to(self.device),
                'actions': torch.FloatTensor(action_batch).to(self.device),
                'rewards': torch.FloatTensor(reward_batch).to(self.device),
                'dones': torch.FloatTensor(done_batch).to(self.device),
                'next_observations': torch.FloatTensor(next_obs_batch).to(self.device),
                'masks': torch.FloatTensor(mask_batch).to(self.device)
            }
        else:
            batch = {
                'observations': obs_batch,
                'actions': action_batch,
                'rewards': reward_batch,
                'dones': done_batch,
                'next_observations': next_obs_batch,
                'masks': mask_batch
            }
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled episodes"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + 1e-6) ** self.alpha
            self.max_priority = max(self.max_priority, priority)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== Recurrent Replay Buffer Tests ===\n")
    
    # Parameters
    obs_dim = 12
    action_dim = 3
    capacity = 100
    
    # Test 1: Basic episode storage
    print("Test 1: Episode Storage")
    buffer = RecurrentReplayBuffer(capacity, obs_dim, action_dim)
    
    # Create dummy episode
    episode_length = 50
    episode = Episode(
        observations=np.random.randn(episode_length, obs_dim).astype(np.float32),
        actions=np.random.randn(episode_length, action_dim).astype(np.float32),
        rewards=np.random.randn(episode_length).astype(np.float32),
        dones=np.zeros(episode_length, dtype=np.float32),
        next_observations=np.random.randn(episode_length, obs_dim).astype(np.float32),
        length=episode_length,
        total_reward=10.5
    )
    
    buffer.add_episode(episode)
    print(f"  Added episode of length {episode_length}")
    print(f"  Buffer size: {len(buffer)} episodes")
    print()
    
    # Test 2: Sample episodes with padding
    print("Test 2: Sample Episodes (with padding)")
    
    # Add more episodes of varying lengths
    for length in [30, 60, 45, 70, 40]:
        ep = Episode(
            observations=np.random.randn(length, obs_dim).astype(np.float32),
            actions=np.random.randn(length, action_dim).astype(np.float32),
            rewards=np.random.randn(length).astype(np.float32),
            dones=np.zeros(length, dtype=np.float32),
            next_observations=np.random.randn(length, obs_dim).astype(np.float32),
            length=length,
            total_reward=np.random.randn() * 10
        )
        buffer.add_episode(ep)
    
    batch = buffer.sample_episodes(batch_size=4)
    print(f"  Sampled batch shapes:")
    print(f"    Observations: {batch['observations'].shape}")
    print(f"    Actions: {batch['actions'].shape}")
    print(f"    Rewards: {batch['rewards'].shape}")
    print(f"    Masks: {batch['masks'].shape}")
    print(f"    Mask sum per episode: {batch['masks'].sum(dim=1)}")
    print()
    
    # Test 3: Sample chunks
    print("Test 3: Sample Chunks (fixed length)")
    
    chunks = buffer.sample_chunks(batch_size=8, chunk_length=50, overlap=10)
    print(f"  Chunk shapes:")
    print(f"    Observations: {chunks['observations'].shape}")
    print(f"    Actions: {chunks['actions'].shape}")
    print(f"    Masks sum: {chunks['masks'].sum(dim=1)}")
    print()
    
    # Test 4: Buffer statistics
    print("Test 4: Buffer Statistics")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 5: Prioritized replay
    print("Test 5: Prioritized Replay Buffer")
    pri_buffer = PrioritizedRecurrentReplayBuffer(capacity, obs_dim, action_dim)
    
    # Add episodes with different priorities
    for i in range(5):
        ep = Episode(
            observations=np.random.randn(40, obs_dim).astype(np.float32),
            actions=np.random.randn(40, action_dim).astype(np.float32),
            rewards=np.random.randn(40).astype(np.float32),
            dones=np.zeros(40, dtype=np.float32),
            next_observations=np.random.randn(40, obs_dim).astype(np.float32),
            length=40,
            total_reward=float(i)
        )
        pri_buffer.add_episode(ep, priority=float(i + 1))  # Higher priority for later episodes
    
    batch, indices, weights = pri_buffer.sample_episodes(batch_size=3)
    print(f"  Sampled indices: {indices}")
    print(f"  Importance weights: {weights}")
    print(f"  Beta: {pri_buffer.beta:.3f}")
    
    print("\n=== All tests complete ===")
