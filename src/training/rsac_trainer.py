"""
RSAC (Recurrent Soft Actor-Critic) Trainer.

Complete training loop for drone tracking with:
- GRU-based actor-critic
- Automatic entropy tuning
- Recurrent replay buffer
- Curriculum learning
- Checkpointing and logging

Based on research: RSAC-Share architecture for 2x speedup.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
from collections import deque

from src.models.gru_networks import RSACSharedEncoder
from src.models.replay_buffer import RecurrentReplayBuffer
from src.environment.simple_drone_sim import SimpleDroneSimulator
from src.utils.trajectory_generator import TrajectoryType
from src.training.reward_shaper import RewardType


class RSACTrainer:
    """
    RSAC trainer with shared GRU encoder.
    
    Key features:
    - RSAC-Share: Single shared GRU for actor + critics
    - Automatic entropy tuning
    - Gradient clipping for stability
    - Target network soft updates
    - Curriculum learning support
    """
    
    def __init__(
        self,
        obs_dim: int = 12,
        action_dim: int = 3,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        # SAC hyperparameters
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_tune_alpha: bool = True,
        target_entropy: Optional[float] = None,
        # Learning rates
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        # Replay buffer
        buffer_capacity: int = 1000,
        batch_size: int = 16,
        chunk_length: int = 50,
        burn_in_length: int = 10,  # GRU warmup steps (no gradient)
        # Training
        gradient_clip: float = 0.5,
        device: str = 'cuda',
        # Logging
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints',
        use_tensorboard: bool = True
    ):
        """Initialize RSAC trainer."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.burn_in_length = burn_in_length
        self.gradient_clip = gradient_clip
        # Device selection: CUDA > MPS > CPU
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' or (device == 'cuda' and torch.backends.mps.is_available()):
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"🎯 RSAC Trainer initialized on device: {self.device}")
        
        # Networks (using RSAC-Share architecture)
        self.policy = RSACSharedEncoder(
            obs_dim, action_dim, hidden_dim, gru_layers
        ).to(self.device)
        
        # Target network for critics
        self.policy_target = RSACSharedEncoder(
            obs_dim, action_dim, hidden_dim, gru_layers
        ).to(self.device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        
        # Freeze target network
        for param in self.policy_target.parameters():
            param.requires_grad = False
        
# Optimizers (avoid double-stepping shared encoder)
        self.actor_params = (
            list(self.policy.actor_mlp.parameters()) +
            list(self.policy.actor_mean.parameters()) +
            list(self.policy.actor_log_std.parameters())
        )
        self.critic_params = (
            list(self.policy.shared_encoder.parameters()) +
            list(self.policy.critic1_mlp.parameters()) +
            list(self.policy.critic2_mlp.parameters())
        )
        self.actor_optimizer = optim.Adam(self.actor_params, lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_params, lr=critic_lr)
        
        # Automatic entropy tuning
        self.auto_tune_alpha = auto_tune_alpha
        if auto_tune_alpha:
            self.target_entropy = target_entropy or -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
        
        # Replay buffer
        self.replay_buffer = RecurrentReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(self.device)
        )
        
        # Logging
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if use_tensorboard else None
        
        # CSV Logging
        self.csv_log_path = self.log_dir / 'training_log.csv'
        with open(self.csv_log_path, 'w') as f:
            f.write('episode,step,reward,length,critic_loss,actor_loss,alpha\n')
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.policy.parameters())
        print(f"📊 Total parameters: {total_params:,}")
    
    def select_action(
        self,
        obs: np.ndarray,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Select action from policy.
        
        Args:
            obs: Observation [obs_dim]
            hidden_state: GRU hidden state
            deterministic: If True, return mean action
            
        Returns:
            action: Action [action_dim]
            new_hidden_state: Updated hidden state
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Initialize hidden state if needed
            if hidden_state is None:
                hidden_state = self.policy.init_hidden(1, self.device)
            
            # Encode
            encoded, new_hidden_state = self.policy.encode(obs_tensor, hidden_state)
            
            # Get action
            action, _ = self.policy.actor_forward(encoded, deterministic)
            action = action.cpu().numpy()[0]
        
        return action, new_hidden_state
    
    def collect_episode(
        self,
        env: SimpleDroneSimulator,
        deterministic: bool = False,
        random_policy: bool = False
    ) -> Dict:
        """
        Collect one complete episode.
        
        Returns:
            episode_data: Dictionary with episode info
        """
        obs = env.reset()
        done = False
        
        # Storage
        observations = []
        actions = []
        rewards = []
        dones = []
        next_observations = []
        
        # GRU hidden state
        hidden_state = None
        
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # Select action
            if random_policy:
                action = np.random.randn(self.action_dim).astype(np.float32) * 0.5
                # keep a dummy hidden update to keep interface consistent
                if hidden_state is None:
                    hidden_state = self.policy.init_hidden(1, self.device)
            else:
                action, hidden_state = self.select_action(obs, hidden_state, deterministic)
            
            # Environment step
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            next_observations.append(next_obs)
            
            # Update
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
        
        # Convert to numpy arrays
        episode_data = {
            'observations': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'dones': np.array(dones, dtype=np.float32),
            'next_observations': np.array(next_observations, dtype=np.float32),
            'episode_reward': episode_reward,
            'episode_length': episode_length
        }
        
        return episode_data
    
    def update_networks(self) -> Dict[str, float]:
        """
        Update actor and critic networks.
        
        Returns:
            losses: Dictionary of loss values
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch of episodes/chunks
        batch = self.replay_buffer.sample_chunks(
            batch_size=self.batch_size,
            chunk_length=self.chunk_length,
            overlap=10
        )
        
        obs = batch['observations']  # [batch, seq_len, obs_dim]
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        next_obs = batch['next_observations']
        masks = batch['masks']  # For variable-length episodes
        
        # Get sequence length
        batch_size, seq_len, _ = obs.shape
        
        # Burn-in: process first N steps without gradient to warm up GRU
        burn_in = min(self.burn_in_length, seq_len - 1)
        train_start = burn_in  # Start computing loss from this index
        
        # ============================================================
        # Critic Update
        # ============================================================
        
        with torch.no_grad():
            # Encode next observations with target network
            hidden_target = self.policy_target.init_hidden(batch_size, self.device)
            
            # Process sequence (including burn-in)
            next_encoded_seq = []
            for t in range(seq_len):
                encoded, hidden_target = self.policy_target.encode(
                    next_obs[:, t, :], hidden_target
                )
                next_encoded_seq.append(encoded)
            next_encoded_seq = torch.stack(next_encoded_seq, dim=1)  # [batch, seq, hidden]
            
            # Get next actions and log probs from current policy
            hidden_policy = self.policy.init_hidden(batch_size, self.device)
            next_actions_seq = []
            next_log_probs_seq = []
            for t in range(seq_len):
                encoded, hidden_policy = self.policy.encode(next_obs[:, t, :], hidden_policy)
                action, log_prob = self.policy.actor_forward(encoded, deterministic=False)
                next_actions_seq.append(action)
                next_log_probs_seq.append(log_prob)
            next_actions_seq = torch.stack(next_actions_seq, dim=1)
            next_log_probs_seq = torch.stack(next_log_probs_seq, dim=1)
            
            # Compute target Q values
            next_q1_seq = []
            next_q2_seq = []
            for t in range(seq_len):
                q1, q2 = self.policy_target.critics_forward(
                    next_encoded_seq[:, t, :], next_actions_seq[:, t, :]
                )
                next_q1_seq.append(q1)
                next_q2_seq.append(q2)
            next_q1_seq = torch.stack(next_q1_seq, dim=1).squeeze(-1)
            next_q2_seq = torch.stack(next_q2_seq, dim=1).squeeze(-1)
            
            # Min of two Q-networks
            next_q_seq = torch.min(next_q1_seq, next_q2_seq)
            
            # Target: r + γ(1-done) * (Q(s',a') - α*log_prob(a'))
            target_q_seq = rewards + self.gamma * (1 - dones) * (
                next_q_seq - self.alpha * next_log_probs_seq.squeeze(-1)
            )
        
        # Current Q values
        hidden = self.policy.init_hidden(batch_size, self.device)
        
        # Burn-in phase: warm up GRU without gradient
        with torch.no_grad():
            for t in range(burn_in):
                _, hidden = self.policy.encode(obs[:, t, :], hidden)
        
        # Training phase: encode with gradients
        encoded_seq = []
        for t in range(burn_in, seq_len):
            encoded, hidden = self.policy.encode(obs[:, t, :], hidden)
            encoded_seq.append(encoded)
        encoded_seq = torch.stack(encoded_seq, dim=1)  # [batch, seq_len - burn_in, hidden]
        
        # Adjust other sequences to match burn-in offset
        actions_train = actions[:, train_start:, :]
        target_q_train = target_q_seq[:, train_start:]
        masks_train = masks[:, train_start:]
        
        q1_seq = []
        q2_seq = []
        for t in range(encoded_seq.shape[1]):
            q1, q2 = self.policy.critics_forward(
                encoded_seq[:, t, :], actions_train[:, t, :]
            )
            q1_seq.append(q1)
            q2_seq.append(q2)
        q1_seq = torch.stack(q1_seq, dim=1).squeeze(-1)
        q2_seq = torch.stack(q2_seq, dim=1).squeeze(-1)
        
        # Critic loss (MSE, masked by valid timesteps, after burn-in)
        q1_loss = ((q1_seq - target_q_train) ** 2 * masks_train).sum() / masks_train.sum()
        q2_loss = ((q2_seq - target_q_train) ** 2 * masks_train).sum() / masks_train.sum()
        critic_loss = q1_loss + q2_loss
        
# Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_params, self.gradient_clip)
        self.critic_optimizer.step()
        
        # ============================================================
        # Actor Update
        # ============================================================
        
        # IMPORTANT: Reuse encoded features from critic update, but DETACH
        # to prevent actor gradients from flowing through shared encoder.
        # Research: "RSAC-Share gradients only from critic losses"
        encoded_seq_detached = encoded_seq.detach()
        
        # Sample new actions from detached encoded features (already burn-in adjusted)
        train_seq_len = encoded_seq_detached.shape[1]
        new_actions_seq = []
        log_probs_seq = []
        for t in range(train_seq_len):
            action, log_prob = self.policy.actor_forward(
                encoded_seq_detached[:, t, :], deterministic=False
            )
            new_actions_seq.append(action)
            log_probs_seq.append(log_prob)
        new_actions_seq = torch.stack(new_actions_seq, dim=1)
        log_probs_seq = torch.stack(log_probs_seq, dim=1).squeeze(-1)
        
        # Q values for new actions (use critic1 only for policy update)
        # Note: encoded_seq_detached prevents actor→encoder gradients
        q_new_seq = []
        for t in range(train_seq_len):
            q1, _ = self.policy.critics_forward(
                encoded_seq_detached[:, t, :],
                new_actions_seq[:, t, :]
            )
            q_new_seq.append(q1)
        q_new_seq = torch.stack(q_new_seq, dim=1).squeeze(-1)
        
        # Actor loss: E[α*log_prob - Q] (using burn-in adjusted mask)
        actor_loss = ((self.alpha * log_probs_seq - q_new_seq) * masks_train).sum() / masks_train.sum()
        
# Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_params, self.gradient_clip)
        self.actor_optimizer.step()
        
        # ============================================================
        # Alpha (entropy) Update
        # ============================================================
        
        alpha_loss = torch.tensor(0.0)
        if self.auto_tune_alpha:
            with torch.no_grad():
                # Use log_probs from actor update
                pass
            
            # Alpha loss: -log(α) * (log_prob + target_entropy)
            alpha_loss = (
                -self.log_alpha * (log_probs_seq.detach() + self.target_entropy) * masks_train
            ).sum() / masks_train.sum()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # ============================================================
        # Soft update target network (only encoder and critics)
        # ============================================================
        
        # Only update shared encoder and critic heads, not actor heads
        for param, target_param in zip(
            self.policy.shared_encoder.parameters(),
            self.policy_target.shared_encoder.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.policy.critic1_mlp.parameters(),
            self.policy_target.critic1_mlp.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
            self.policy.critic2_mlp.parameters(),
            self.policy_target.critic2_mlp.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_tune_alpha else 0.0,
            'alpha': self.alpha.item(),
            'q1_mean': q1_seq.mean().item(),
            'q2_mean': q2_seq.mean().item()
        }
    
    def train(
        self,
        env: SimpleDroneSimulator,
        total_steps: int = 1_000_000,
        warmup_steps: int = 10_000,
        updates_per_step: int = 1,
        log_interval: int = 1000,
        eval_interval: int = 10_000,
        eval_episodes: int = 10,
        save_interval: int = 50_000
    ):
        """
        Main training loop.
        
        Args:
            env: Environment
            total_steps: Total training steps
            warmup_steps: Random policy warmup
            updates_per_step: Gradient updates per env step
            log_interval: Logging frequency
            eval_interval: Evaluation frequency
            eval_episodes: Number of eval episodes
            save_interval: Checkpoint save frequency
        """
        print("\n🚀 Starting RSAC training...")
        print(f"Total steps: {total_steps:,}")
        print(f"Warmup steps: {warmup_steps:,}\n")
        
        start_time = time.time()
        
        while self.total_steps < total_steps:
            # Collect episode
            is_warmup = self.total_steps < warmup_steps
            episode_data = self.collect_episode(env, deterministic=False, random_policy=is_warmup)
            
            # Add to replay buffer
            self.replay_buffer.add_trajectory(
                episode_data['observations'],
                episode_data['actions'],
                episode_data['rewards'],
                episode_data['dones'],
                episode_data['next_observations']
            )
            
            # Update statistics
            self.total_episodes += 1
            self.episode_rewards.append(episode_data['episode_reward'])
            self.episode_lengths.append(episode_data['episode_length'])
            
            # Network updates (skip during warmup)
            if not is_warmup and len(self.replay_buffer) >= self.batch_size:
                for _ in range(updates_per_step):
                    losses = self.update_networks()
            else:
                losses = {}
            
            # Logging
            if self.total_episodes % 10 == 0:
                mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                print(f"Episode {self.total_episodes:5d} | "
                      f"Steps {self.total_steps:7d} | "
                      f"Reward: {mean_reward:7.2f} | "
                      f"Length: {mean_length:5.1f} | "
                      f"Buffer: {len(self.replay_buffer):4d}")
                
                if self.writer and losses:
                    self.writer.add_scalar('train/episode_reward', episode_data['episode_reward'], self.total_episodes)
                    self.writer.add_scalar('train/mean_reward_100', mean_reward, self.total_episodes)
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}', value, self.total_steps)
            
            # CSV Logging
            if self.total_episodes % 10 == 0:
                with open(self.csv_log_path, 'a') as f:
                    c_loss = losses.get('critic_loss', 0.0)
                    a_loss = losses.get('actor_loss', 0.0)
                    alp = losses.get('alpha', self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha)
                    f.write(f"{self.total_episodes},{self.total_steps},{episode_data['episode_reward']},{episode_data['episode_length']},{c_loss},{a_loss},{alp}\n")
            
            # Evaluation
            if self.total_steps % eval_interval == 0 and self.total_steps > 0:
                eval_stats = self.evaluate(env, num_episodes=eval_episodes)
                print(f"\n📊 Evaluation @ {self.total_steps:,} steps:")
                print(f"   Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                print(f"   Success rate: {eval_stats['success_rate']:.1%}\n")
                
                if self.writer:
                    self.writer.add_scalar('eval/mean_reward', eval_stats['mean_reward'], self.total_steps)
                    self.writer.add_scalar('eval/success_rate', eval_stats['success_rate'], self.total_steps)
            
            # Save checkpoint
            if self.total_steps % save_interval == 0 and self.total_steps > 0:
                self.save_checkpoint(f'checkpoint_{self.total_steps}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training complete!")
        print(f"⏱️  Total time: {elapsed_time/3600:.2f} hours")
        print(f"📈 Final mean reward: {np.mean(self.episode_rewards):.2f}")
    
    def evaluate(self, env: SimpleDroneSimulator, num_episodes: int = 10) -> Dict:
        """Evaluate policy"""
        rewards = []
        successes = []
        
        for _ in range(num_episodes):
            episode_data = self.collect_episode(env, deterministic=True)
            rewards.append(episode_data['episode_reward'])
            
            # Success: stayed close to target most of the time
            successes.append(1.0 if episode_data['episode_reward'] > -50 else 0.0)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'success_rate': np.mean(successes)
        }
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'policy_target_state_dict': self.policy_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        }
        # Save alpha optimizer state for proper resume
        if self.auto_tune_alpha:
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            checkpoint['log_alpha'] = self.log_alpha.item()
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_target.load_state_dict(checkpoint['policy_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.total_episodes = checkpoint['total_episodes']
        
        # Restore alpha optimizer state
        if self.auto_tune_alpha and 'alpha_optimizer_state_dict' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            if 'log_alpha' in checkpoint:
                self.log_alpha.data.fill_(checkpoint['log_alpha'])
                self.alpha = self.log_alpha.exp()
        
        print(f"📂 Checkpoint loaded: {path}")


# ============================================================================
# MAIN - Quick test
# ============================================================================

if __name__ == "__main__":
    print("=== RSAC Trainer Test ===\n")
    
    # Create environment
    env = SimpleDroneSimulator(
        dt=0.05,
        max_episode_steps=200,
        use_domain_randomization=True,
        trajectory_type=TrajectoryType.LISSAJOUS
    )
    
    # Create trainer
    trainer = RSACTrainer(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device='cpu',  # Use CPU for test
        use_tensorboard=False
    )
    
    # Train for a few episodes
    print("\nTraining for 5 episodes...")
    trainer.train(
        env=env,
        total_steps=1000,  # Just 1000 steps for test
        warmup_steps=500,
        log_interval=100,
        eval_interval=1000,
        save_interval=10000
    )
    
    print("\n=== Test complete ===")
