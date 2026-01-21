"""
GRU-based Actor and Critic networks for RSAC (Recurrent Soft Actor-Critic).

Key architectural choices from research:
- GRU (not LSTM): 30% faster, less memory, same performance
- Shared GRU encoder: 2x faster training, -40% memory
- 2 layers, 64 hidden units: Optimal for embedded deployment
- Tanh activation for bounded outputs

This is optimized for both training efficiency and real-time inference on drones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class GRUEncoder(nn.Module):
    """
    Shared GRU encoder for processing temporal sequences.
    
    This is THE KEY to adaptive behavior without backprop during inference.
    The GRU hidden state acts as "memory" of target dynamics and drone characteristics.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        """
        Initialize GRU encoder.
        
        Args:
            input_dim: Observation dimension
            hidden_dim: GRU hidden dimension (64 recommended for embedded)
            num_layers: Number of GRU layers (2 optimal)
            dropout: Dropout rate between layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU.
        
        Args:
            obs: Observations [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            hidden_state: Previous hidden state [num_layers, batch_size, hidden_dim]
            
        Returns:
            output: GRU output [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
            hidden_state: New hidden state [num_layers, batch_size, hidden_dim]
        """
        # Handle single timestep (not sequence)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # [batch_size, 1, input_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        # GRU forward
        output, hidden_state = self.gru(obs, hidden_state)
        
        # Layer norm
        output = self.layer_norm(output)
        
        # Squeeze if single timestep
        if squeeze_output:
            output = output.squeeze(1)  # [batch_size, hidden_dim]
        
        return output, hidden_state
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state to zeros"""
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device
        )


class Actor(nn.Module):
    """
    Actor network (policy) for SAC.
    Outputs mean and log_std for Gaussian policy.
    
    Architecture: GRU encoder → MLP → Gaussian distribution
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        mlp_hidden_dims: Tuple[int, int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """
        Initialize actor network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (3 for velocity control)
            hidden_dim: GRU hidden dimension
            gru_layers: Number of GRU layers
            mlp_hidden_dims: MLP layer dimensions after GRU
            log_std_min: Minimum log std (for numerical stability)
            log_std_max: Maximum log std (for bounded exploration)
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared GRU encoder
        self.encoder = GRUEncoder(obs_dim, hidden_dim, gru_layers)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU()
        )
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(mlp_hidden_dims[1], action_dim)
        self.log_std_layer = nn.Linear(mlp_hidden_dims[1], action_dim)
    
    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action.
        
        Args:
            obs: Observation [batch_size, obs_dim] or [batch_size, seq_len, obs_dim]
            hidden_state: GRU hidden state
            deterministic: If True, return mean action (no noise)
            
        Returns:
            action: Sampled action [batch_size, action_dim]
            log_prob: Log probability of action
            hidden_state: New GRU hidden state
        """
        # Encode with GRU
        encoded, hidden_state = self.encoder(obs, hidden_state)
        
        # MLP forward
        features = self.mlp(encoded)
        
        # Get mean and log_std
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        # Sample action
        if deterministic:
            action = mean
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()  # Reparameterized sample
            action = torch.tanh(x)  # Squash to [-1, 1]
            
            # Compute log probability with change of variables
            log_prob = normal.log_prob(x).sum(dim=-1, keepdim=True)
            # Correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob, hidden_state
    
    def get_action(
        self,
        obs: np.ndarray,
        hidden_state: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Get action for deployment (numpy interface).
        
        Args:
            obs: Observation numpy array
            hidden_state: GRU hidden state
            deterministic: If True, return mean action
            
        Returns:
            action: Action numpy array
            hidden_state: New hidden state
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, _, hidden_state = self.forward(obs_tensor, hidden_state, deterministic)
            return action.squeeze(0).cpu().numpy(), hidden_state


class Critic(nn.Module):
    """
    Critic network (Q-function) for SAC.
    Estimates Q(s, a) - expected return from state-action pair.
    
    Architecture: GRU encoder + action → MLP → Q-value
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        mlp_hidden_dims: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize critic network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: GRU hidden dimension
            gru_layers: Number of GRU layers
            mlp_hidden_dims: MLP layer dimensions
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Shared GRU encoder
        self.encoder = GRUEncoder(obs_dim, hidden_dim, gru_layers)
        
        # MLP head (state-action → Q-value)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1)
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get Q-value.
        
        Args:
            obs: Observation [batch_size, obs_dim] or [batch_size, seq_len, obs_dim]
            action: Action [batch_size, action_dim]
            hidden_state: GRU hidden state
            
        Returns:
            q_value: Q-value [batch_size, 1]
            hidden_state: New GRU hidden state
        """
        # Encode observation with GRU
        encoded, hidden_state = self.encoder(obs, hidden_state)
        
        # Concatenate encoded state with action
        state_action = torch.cat([encoded, action], dim=-1)
        
        # MLP forward
        q_value = self.mlp(state_action)
        
        return q_value, hidden_state


class DoubleCritic(nn.Module):
    """
    Twin Q-networks for SAC (reduces overestimation bias).
    Standard practice: use minimum of two Q-values.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        mlp_hidden_dims: Tuple[int, int] = (256, 256)
    ):
        """Initialize twin critics"""
        super().__init__()
        
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim, gru_layers, mlp_hidden_dims)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim, gru_layers, mlp_hidden_dims)
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state1: Optional[torch.Tensor] = None,
        hidden_state2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward through both critics.
        
        Returns:
            q1: Q-value from critic 1
            q2: Q-value from critic 2
            hidden_state1: New hidden state for critic 1
            hidden_state2: New hidden state for critic 2
        """
        q1, hidden_state1 = self.critic1(obs, action, hidden_state1)
        q2, hidden_state2 = self.critic2(obs, action, hidden_state2)
        
        return q1, q2, hidden_state1, hidden_state2
    
    def q1_forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through critic 1 only (for policy update)"""
        return self.critic1(obs, action, hidden_state)


# ============================================================================
# RSAC-Share Architecture (Research-backed optimization)
# ============================================================================

class RSACSharedEncoder(nn.Module):
    """
    RSAC with shared GRU encoder (RSAC-Share).
    
    From research: Sharing GRU between actor and critics gives:
    - 2x faster training
    - 40% less memory
    - Same asymptotic performance
    
    This is the RECOMMENDED architecture for your project.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        gru_layers: int = 2,
        mlp_hidden_dims: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize RSAC-Share.
        
        Args:
            obs_dim: Observation dimension (12 from research)
            action_dim: Action dimension (3 for velocity control)
            hidden_dim: GRU hidden dimension (64 recommended)
            gru_layers: Number of GRU layers (2 optimal)
            mlp_hidden_dims: MLP dimensions after GRU
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Shared GRU encoder
        self.shared_encoder = GRUEncoder(obs_dim, hidden_dim, gru_layers)
        
        # Actor head
        self.actor_mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU()
        )
        self.actor_mean = nn.Linear(mlp_hidden_dims[1], action_dim)
        self.actor_log_std = nn.Linear(mlp_hidden_dims[1], action_dim)
        
        # Twin critic heads
        self.critic1_mlp = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1)
        )
        
        self.critic2_mlp = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1)
        )
        
        # Log std bounds
        self.log_std_min = -20.0
        self.log_std_max = 2.0
    
    def encode(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode observation with shared GRU"""
        return self.shared_encoder(obs, hidden_state)
    
    def actor_forward(
        self,
        encoded: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Actor forward from encoded state"""
        features = self.actor_mlp(encoded)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)
            log_prob = normal.log_prob(x).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def critics_forward(
        self,
        encoded: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Critics forward from encoded state"""
        state_action = torch.cat([encoded, action], dim=-1)
        q1 = self.critic1_mlp(state_action)
        q2 = self.critic2_mlp(state_action)
        return q1, q2
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state"""
        return self.shared_encoder.init_hidden(batch_size, device)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== GRU Networks Tests ===\n")
    
    # Parameters from research
    obs_dim = 12  # [target_xyz, drone_vel_xyz, prev_action_xyz, error_xyz]
    action_dim = 3  # [vx, vy, vz]
    hidden_dim = 64
    batch_size = 32
    seq_len = 50
    
    # Test 1: Standalone Actor
    print("Test 1: Standalone Actor")
    actor = Actor(obs_dim, action_dim, hidden_dim)
    
    obs = torch.randn(batch_size, obs_dim)
    hidden = actor.encoder.init_hidden(batch_size, torch.device('cpu'))
    
    action, log_prob, hidden_new = actor(obs, hidden)
    print(f"  Input shape: {obs.shape}")
    print(f"  Action shape: {action.shape}")
    print(f"  Log prob shape: {log_prob.shape}")
    print(f"  Hidden state shape: {hidden_new.shape}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    print()
    
    # Test 2: Standalone Critic
    print("Test 2: Double Critic")
    critic = DoubleCritic(obs_dim, action_dim, hidden_dim)
    
    q1, q2, h1, h2 = critic(obs, action)
    print(f"  Q1 shape: {q1.shape}, mean: {q1.mean():.3f}")
    print(f"  Q2 shape: {q2.shape}, mean: {q2.mean():.3f}")
    print()
    
    # Test 3: RSAC-Share (RECOMMENDED)
    print("Test 3: RSAC-Share Architecture")
    rsac_share = RSACSharedEncoder(obs_dim, action_dim, hidden_dim)
    
    # Encode
    encoded, hidden = rsac_share.encode(obs)
    print(f"  Encoded shape: {encoded.shape}")
    
    # Actor
    action, log_prob = rsac_share.actor_forward(encoded)
    print(f"  Action shape: {action.shape}")
    
    # Critics
    q1, q2 = rsac_share.critics_forward(encoded, action)
    print(f"  Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in rsac_share.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()
    
    # Test 4: Sequence processing
    print("Test 4: Sequence Processing")
    obs_seq = torch.randn(batch_size, seq_len, obs_dim)
    encoded_seq, hidden = rsac_share.encode(obs_seq)
    print(f"  Input sequence: {obs_seq.shape}")
    print(f"  Encoded sequence: {encoded_seq.shape}")
    print()
    
    # Test 5: Deployment (numpy interface)
    print("Test 5: Deployment Interface")
    actor_standalone = Actor(obs_dim, action_dim, hidden_dim)
    actor_standalone.eval()
    
    obs_numpy = np.random.randn(obs_dim).astype(np.float32)
    action_numpy, hidden = actor_standalone.get_action(obs_numpy, deterministic=True)
    print(f"  Numpy input shape: {obs_numpy.shape}")
    print(f"  Numpy action shape: {action_numpy.shape}")
    print(f"  Action: {action_numpy}")
    
    print("\n=== All tests complete ===")
