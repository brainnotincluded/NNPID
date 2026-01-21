#!/usr/bin/env python3
"""
Test trained RSAC model on various trajectory types.
"""

import numpy as np
import torch
from pathlib import Path
import argparse

from src.training.rsac_trainer import RSACTrainer
from src.environment.simple_drone_sim import SimpleDroneSimulator
from src.utils.trajectory_generator import TrajectoryType


def test_model(
    checkpoint_path: str,
    trajectory_type: TrajectoryType,
    num_episodes: int = 5,
    render: bool = True,
    max_steps: int = 400
):
    """Test model on specific trajectory type."""
    
    # Create environment
    env = SimpleDroneSimulator(
        dt=0.05,
        max_episode_steps=max_steps,
        use_domain_randomization=True,
        trajectory_type=trajectory_type,
        action_latency_range=(0.02, 0.08),
        position_noise_std=0.02,
        velocity_noise_std=0.05
    )
    
    # Create trainer and load checkpoint
    trainer = RSACTrainer(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        use_tensorboard=False
    )
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    # Extract just filename if full path given
    checkpoint_name = Path(checkpoint_path).name
    trainer.load_checkpoint(checkpoint_name)
    
    # Test episodes
    print(f"\n🎯 Testing on {trajectory_type.name} trajectory")
    print(f"   Episodes: {num_episodes}")
    print(f"   Max steps: {max_steps}")
    print(f"   Latency: {env.action_latency_range[0]*1000:.0f}-{env.action_latency_range[1]*1000:.0f}ms")
    print()
    
    rewards = []
    distances = []
    lengths = []
    success_count = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        episode_distances = []
        step = 0
        
        # Reset hidden state for new episode
        hidden_state = None
        
        if render:
            print(f"\nEpisode {ep + 1}/{num_episodes}:")
        
        while True:
            # Select action (deterministic for evaluation)
            action, hidden_state = trainer.select_action(
                obs, hidden_state, deterministic=True
            )
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_distances.append(info['target_distance'])
            step += 1
            
            # Render progress
            if render and step % 40 == 0:
                mean_dist = np.mean(episode_distances[-40:])
                print(f"  Step {step:3d}: distance={info['target_distance']:5.2f}m, "
                      f"avg_40={mean_dist:5.2f}m, speed={info['drone_speed']:4.2f}m/s")
            
            if done:
                break
        
        # Episode stats
        mean_distance = np.mean(episode_distances)
        min_distance = np.min(episode_distances)
        success = mean_distance < 2.0 and min_distance < 5.0  # Reasonable thresholds
        
        rewards.append(episode_reward)
        distances.append(mean_distance)
        lengths.append(step)
        if success:
            success_count += 1
        
        if render:
            status = "✅ SUCCESS" if success else "❌ FAIL"
            print(f"  Result: {status} | Reward: {episode_reward:7.2f} | "
                  f"Avg dist: {mean_distance:5.2f}m | Min: {min_distance:5.2f}m")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY - {trajectory_type.name}")
    print(f"{'='*60}")
    print(f"Episodes:      {num_episodes}")
    print(f"Success rate:  {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Avg reward:    {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
    print(f"Avg distance:  {np.mean(distances):6.2f}m ± {np.std(distances):5.2f}m")
    print(f"Avg length:    {np.mean(lengths):6.1f} steps")
    print(f"Best reward:   {np.max(rewards):7.2f}")
    print(f"Worst reward:  {np.min(rewards):7.2f}")
    print(f"{'='*60}\n")
    
    return {
        'trajectory_type': trajectory_type.name,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_distance': np.mean(distances),
        'success_rate': success_count / num_episodes,
        'rewards': rewards,
        'distances': distances
    }


def main():
    parser = argparse.ArgumentParser(description='Test trained RSAC model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--trajectory', type=str, default='ALL',
                        choices=['ALL'] + [t.name for t in TrajectoryType],
                        help='Trajectory type to test')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes per trajectory')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable progress rendering')
    parser.add_argument('--max-steps', type=int, default=400,
                        help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("\nAvailable checkpoints:")
        for ckpt in Path('checkpoints').glob('*.pt'):
            print(f"  - {ckpt}")
        return
    
    # Select trajectories to test
    if args.trajectory == 'ALL':
        # Test on curriculum: easy → medium → hard
        trajectories = [
            TrajectoryType.STATIONARY,
            TrajectoryType.LINEAR,
            TrajectoryType.CIRCULAR,
            TrajectoryType.LISSAJOUS,
            TrajectoryType.ZIGZAG,
            TrajectoryType.FIGURE_EIGHT,
            TrajectoryType.EVASIVE,
            TrajectoryType.CHAOTIC
        ]
    else:
        trajectories = [TrajectoryType[args.trajectory]]
    
    print(f"\n{'='*60}")
    print(f"🧪 RSAC MODEL EVALUATION")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Episodes per trajectory: {args.episodes}")
    print(f"{'='*60}\n")
    
    # Test on each trajectory
    results = []
    for traj_type in trajectories:
        result = test_model(
            checkpoint_path=args.checkpoint,
            trajectory_type=traj_type,
            num_episodes=args.episodes,
            render=not args.no_render,
            max_steps=args.max_steps
        )
        results.append(result)
    
    # Overall summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"🏆 OVERALL PERFORMANCE")
        print(f"{'='*60}")
        
        for result in results:
            success_emoji = "✅" if result['success_rate'] >= 0.6 else "⚠️" if result['success_rate'] >= 0.4 else "❌"
            print(f"{success_emoji} {result['trajectory_type']:20s} | "
                  f"Success: {result['success_rate']*100:5.1f}% | "
                  f"Reward: {result['mean_reward']:7.1f} | "
                  f"Dist: {result['mean_distance']:5.2f}m")
        
        overall_success = np.mean([r['success_rate'] for r in results])
        overall_reward = np.mean([r['mean_reward'] for r in results])
        
        print(f"{'='*60}")
        print(f"Overall success rate: {overall_success*100:.1f}%")
        print(f"Overall mean reward:  {overall_reward:.2f}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
