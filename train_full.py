#!/usr/bin/env python3
"""
Full RSAC training with 1M steps.
Uses research-backed hyperparameters and curriculum learning.
"""

import argparse
from pathlib import Path

from src.training.rsac_trainer import RSACTrainer
from src.environment.simple_drone_sim import SimpleDroneSimulator
from src.utils.trajectory_generator import TrajectoryType
from src.training.reward_shaper import RewardType


def main():
    parser = argparse.ArgumentParser(description='Train RSAC for 1M steps')
    parser.add_argument('--steps', type=int, default=1_000_000,
                        help='Total training steps')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning (stage-based trajectory difficulty)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Device selection
    if args.device == 'auto':
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print("=" * 70)
    print("🚀 RSAC TRAINING - 1M STEPS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Total steps: {args.steps:,}")
    print(f"Curriculum learning: {args.curriculum}")
    print("=" * 70)
    print()
    
    # Create environment with research-backed settings
    trajectory_type = TrajectoryType.LISSAJOUS_PERLIN  # Start with moderate difficulty
    
    env = SimpleDroneSimulator(
        dt=0.05,  # 20 Hz
        max_episode_steps=400,  # 20 seconds
        use_domain_randomization=True,
        use_safety=True,
        reward_type=RewardType.DENSE_TO_SPARSE,  # Best of both worlds
        trajectory_type=trajectory_type,
        action_latency_range=(0.02, 0.08),  # 20-80ms latency
        position_noise_std=0.02,  # 2cm position noise
        velocity_noise_std=0.05,  # 5cm/s velocity noise
    )
    
    print(f"🌍 Environment:")
    print(f"   Trajectory: {trajectory_type.name}")
    print(f"   Episode length: {env.max_episode_steps} steps ({env.max_episode_steps * env.dt:.1f}s)")
    print(f"   Obs dim: {env.obs_dim}, Action dim: {env.action_dim}")
    print(f"   Latency: {env.action_latency_range[0]*1000:.0f}-{env.action_latency_range[1]*1000:.0f}ms")
    print(f"   Sensor noise: {env.position_noise_std*100:.1f}cm pos, {env.velocity_noise_std*100:.1f}cm/s vel")
    print()
    
    # Create trainer with research-backed hyperparameters
    trainer = RSACTrainer(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=64,  # Optimal for embedded
        gru_layers=2,  # Research recommendation
        # SAC hyperparameters (research-backed)
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_tune_alpha=True,
        target_entropy=None,  # Will default to -action_dim
        # Learning rates
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        # Replay buffer
        buffer_capacity=1000,  # 1000 episodes
        batch_size=32,  # Increased from 16
        chunk_length=50,
        burn_in_length=10,  # GRU warmup
        # Training
        gradient_clip=0.5,
        device=device,
        # Logging
        log_dir='logs/full_training',
        checkpoint_dir='checkpoints',
        use_tensorboard=True
    )
    
    print(f"🧠 Trainer:")
    print(f"   Parameters: {sum(p.numel() for p in trainer.policy.parameters()):,}")
    print(f"   Hidden dim: 64, GRU layers: 2")
    print(f"   Batch size: 32, Chunk length: 50, Burn-in: 10")
    print(f"   Gamma: 0.99, Tau: 0.005")
    print(f"   Auto-tune alpha: True (target: -3)")
    print()
    
    # Resume if requested
    if args.resume:
        print(f"📂 Resuming from: {args.resume}")
        trainer.load_checkpoint(Path(args.resume).name)
        print()
    
    # Training configuration
    warmup_steps = 10_000  # 10k warmup steps
    updates_per_step = 1
    log_interval = 1000
    eval_interval = 50_000  # Eval every 50k steps
    save_interval = 100_000  # Save every 100k steps
    
    print(f"⚙️  Training config:")
    print(f"   Warmup: {warmup_steps:,} steps")
    print(f"   Updates per step: {updates_per_step}")
    print(f"   Log interval: {log_interval:,}")
    print(f"   Eval interval: {eval_interval:,}")
    print(f"   Save interval: {save_interval:,}")
    print()
    
    if args.curriculum:
        print(f"📚 Curriculum stages:")
        print(f"   Stage 1 (0-333k):   CIRCULAR → LINEAR")
        print(f"   Stage 2 (333k-667k): LISSAJOUS → ZIGZAG")
        print(f"   Stage 3 (667k-1M):   EVASIVE → CHAOTIC")
        print()
    
    print("=" * 70)
    print("🎯 STARTING TRAINING")
    print("=" * 70)
    print()
    
    # Curriculum learning callback (optional)
    if args.curriculum:
        def update_curriculum(trainer, env):
            """Update environment difficulty based on training progress."""
            steps = trainer.total_steps
            
            if steps < 333_000:
                # Stage 1: Easy
                if steps % 50_000 == 0 and steps > 0:
                    env.trajectory_generator.config.trajectory_type = TrajectoryType.LINEAR
            elif steps < 667_000:
                # Stage 2: Medium
                if steps == 333_000:
                    env.trajectory_generator.config.trajectory_type = TrajectoryType.LISSAJOUS
                elif steps % 50_000 == 0:
                    env.trajectory_generator.config.trajectory_type = TrajectoryType.ZIGZAG
            else:
                # Stage 3: Hard
                if steps == 667_000:
                    env.trajectory_generator.config.trajectory_type = TrajectoryType.EVASIVE
                elif steps % 50_000 == 0:
                    env.trajectory_generator.config.trajectory_type = TrajectoryType.CHAOTIC
    
    # Start training
    try:
        trainer.train(
            env=env,
            total_steps=args.steps,
            warmup_steps=warmup_steps,
            updates_per_step=updates_per_step,
            log_interval=log_interval,
            eval_interval=eval_interval,
            eval_episodes=10,
            save_interval=save_interval
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("💾 Saving interrupted model...")
        trainer.save_checkpoint('interrupted_model.pt')
        print("✅ Model saved. You can resume with --resume interrupted_model.pt")
        return
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final checkpoint: checkpoints/final_model.pt")
    print(f"Total episodes: {trainer.total_episodes}")
    print(f"Final mean reward: {trainer.episode_rewards[-1] if trainer.episode_rewards else 0:.2f}")
    print()
    print("📊 View training curves:")
    print(f"   tensorboard --logdir={trainer.log_dir}")
    print()
    print("🧪 Test the model:")
    print(f"   uv run python test_model.py --checkpoint checkpoints/final_model.pt --trajectory ALL")
    print("=" * 70)


if __name__ == '__main__':
    main()
