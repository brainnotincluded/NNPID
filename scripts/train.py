#!/usr/bin/env python3
"""
Main training script for NNPID.

Usage:
    python scripts/train.py                    # Train with defaults
    python scripts/train.py --steps 500000     # Custom training steps
    python scripts/train.py --device cuda      # Use GPU
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from src.environment.simple_drone_sim import SimpleDroneSimulator
from src.training.rsac_trainer import RSACTrainer
from src.utils.trajectory_generator import TrajectoryType
from src.training.reward_shaper import RewardType


def load_config(config_path: str = 'config/training_config.yaml') -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train RSAC agent for drone tracking')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to config file')
    parser.add_argument('--steps', type=int, default=None,
                        help='Total training steps (overrides config)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Logging directory')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("🚁 NNPID - Neural Network PID Replacement Training")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Device: {args.device}")
    print(f"   Random seed: {args.seed}")
    print(f"   TensorBoard: {'Disabled' if args.no_tensorboard else 'Enabled'}")
    
    # ========================================================================
    # Create Environment
    # ========================================================================
    
    env_config = config['environment']
    traj_config = config['trajectory']
    dr_config = config['domain_randomization']
    reward_config = config['reward']
    
    dt = traj_config.get('dt', 0.05)
    max_steps = env_config.get('max_episode_steps', 1000)
    use_dr = dr_config.get('enabled', True)
    traj_type = traj_config.get('type', 'LISSAJOUS_PERLIN').upper()
    reward_type = reward_config.get('type', 'DENSE_TO_SPARSE').upper()
    
    print(f"\n🌍 Environment Setup:")
    print(f"   dt: {dt}s")
    print(f"   Max episode steps: {max_steps}")
    print(f"   Domain randomization: {use_dr}")
    
    env = SimpleDroneSimulator(
        dt=dt,
        max_episode_steps=max_steps,
        use_domain_randomization=use_dr,
        trajectory_type=TrajectoryType[traj_type],
        reward_type=RewardType[reward_type]
    )
    
    print(f"   Observation dim: {env.obs_dim}")
    print(f"   Action dim: {env.action_dim}")
    
    # ========================================================================
    # Create Trainer
    # ========================================================================
    
    network_config = config['network']
    rsac_config = config['rsac']
    logging_config = config['logging']
    
    print(f"\n🧠 Network Setup:")
    print(f"   Hidden dim: {network_config['hidden_dim']}")
    print(f"   GRU layers: {network_config['gru_layers']}")
    print(f"   Architecture: RSAC-Share (2x faster)")
    
    trainer = RSACTrainer(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=network_config['hidden_dim'],
        gru_layers=network_config['gru_layers'],
        # SAC parameters
        gamma=rsac_config['gamma'],
        tau=rsac_config['tau'],
        alpha=rsac_config['alpha'],
        auto_tune_alpha=rsac_config['auto_tune_alpha'],
        # Learning rates
        actor_lr=rsac_config['actor_lr'],
        critic_lr=rsac_config['critic_lr'],
        alpha_lr=rsac_config['alpha_lr'],
        # Replay buffer
        buffer_capacity=rsac_config['buffer_capacity'],
        batch_size=rsac_config['batch_size'],
        chunk_length=rsac_config['chunk_length'],
        # Training
        gradient_clip=rsac_config['gradient_clip'],
        device=args.device,
        # Logging
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_tensorboard=not args.no_tensorboard
    )
    
    # ========================================================================
    # Train
    # ========================================================================
    
    total_steps = args.steps if args.steps else rsac_config['total_steps']
    
    print(f"\n🎯 Training Setup:")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {rsac_config['warmup_steps']:,}")
    print(f"   Batch size: {rsac_config['batch_size']}")
    print(f"   Chunk length: {rsac_config['chunk_length']}")
    print(f"   Updates per step: {rsac_config['updates_per_step']}")
    
    try:
        trainer.train(
            env=env,
            total_steps=total_steps,
            warmup_steps=rsac_config['warmup_steps'],
            updates_per_step=rsac_config['updates_per_step'],
            log_interval=logging_config['log_interval'],
            eval_interval=logging_config['eval_interval'],
            eval_episodes=logging_config['eval_episodes'],
            save_interval=config['experiment']['save_frequency']
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("💾 Saving checkpoint...")
        trainer.save_checkpoint('interrupted_model.pt')
        print("✅ Checkpoint saved!")
    
    print("\n" + "=" * 70)
    print("🎉 Training session complete!")
    print("=" * 70)
    
    # Print final stats
    if trainer.episode_rewards:
        import numpy as np
        print(f"\n📊 Final Statistics:")
        print(f"   Total episodes: {trainer.total_episodes}")
        print(f"   Total steps: {trainer.total_steps:,}")
        print(f"   Mean reward (last 100): {np.mean(trainer.episode_rewards):.2f}")
        print(f"   Mean length (last 100): {np.mean(trainer.episode_lengths):.1f}")
    
    if not args.no_tensorboard:
        print(f"\n📈 View training curves:")
        print(f"   tensorboard --logdir={args.log_dir}")
    
    print(f"\n💾 Checkpoints saved in: {args.checkpoint_dir}/")
    print()


if __name__ == '__main__':
    main()
