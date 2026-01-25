#!/usr/bin/env python3
"""Evaluate trained controller policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

from src.environments.hover_env import HoverEnv, HoverEnvConfig
from src.environments.waypoint_env import WaypointEnv, WaypointEnvConfig
from src.environments.trajectory_env import TrajectoryEnv, TrajectoryEnvConfig
from src.controllers.nn_controller import NNController
from src.utils.logger import TelemetryLogger


def create_env(env_type: str, render_mode: Optional[str] = None):
    """Create evaluation environment."""
    if env_type == "hover":
        config = HoverEnvConfig(randomize_hover_position=True)
        return HoverEnv(config=config, render_mode=render_mode)
    elif env_type == "waypoint":
        config = WaypointEnvConfig()
        return WaypointEnv(config=config, render_mode=render_mode)
    elif env_type == "trajectory":
        config = TrajectoryEnvConfig()
        return TrajectoryEnv(config=config, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown environment: {env_type}")


def evaluate_policy(
    model_path: Path,
    env_type: str = "hover",
    n_episodes: int = 10,
    render: bool = False,
    save_video: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained policy.
    
    Args:
        model_path: Path to saved model
        env_type: Environment type
        n_episodes: Number of evaluation episodes
        render: Whether to render
        save_video: Whether to save video
        verbose: Print episode details
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 required for evaluation")
        return {}
    
    print(f"Evaluating {model_path} on {env_type} environment")
    print(f"Episodes: {n_episodes}")
    print()
    
    # Load model
    model = None
    for algo_class in [PPO, SAC]:
        try:
            model = algo_class.load(str(model_path))
            print(f"Loaded {algo_class.__name__} model")
            break
        except Exception:
            continue
    
    if model is None:
        print(f"Failed to load model from {model_path}")
        return {}
    
    # Create environment
    render_mode = "rgb_array" if (render or save_video) else None
    env = create_env(env_type, render_mode=render_mode)
    
    # Video recording setup
    frames = []
    
    # Metrics storage
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    success_count = 0
    tracking_errors: List[float] = []
    
    # Run episodes
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_errors = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Track position error
            if "position_error" in info:
                episode_errors.append(info["position_error"])
            
            # Render
            if render or save_video:
                frame = env.render()
                if save_video and frame is not None:
                    frames.append(frame)
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if info.get("is_success", False):
            success_count += 1
        
        if episode_errors:
            tracking_errors.append(np.mean(episode_errors))
        
        if verbose:
            print(
                f"Episode {episode + 1:3d}: "
                f"reward={episode_reward:8.2f}, "
                f"length={episode_length:4d}, "
                f"success={info.get('is_success', False)}"
            )
    
    env.close()
    
    # Compute statistics
    metrics = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / n_episodes,
        "mean_tracking_error": np.mean(tracking_errors) if tracking_errors else None,
    }
    
    # Print summary
    print()
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mean Reward:     {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length:     {metrics['mean_length']:.1f}")
    print(f"Success Rate:    {metrics['success_rate'] * 100:.1f}%")
    if metrics['mean_tracking_error'] is not None:
        print(f"Tracking Error:  {metrics['mean_tracking_error']:.3f} m")
    
    # Save video
    if save_video and frames:
        try:
            import imageio
            video_path = model_path.parent / f"eval_{env_type}.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)
            print(f"\nVideo saved to: {video_path}")
        except ImportError:
            print("imageio not available for video saving")
    
    return metrics


def compare_policies(
    model_paths: List[Path],
    env_type: str = "hover",
    n_episodes: int = 10,
) -> None:
    """Compare multiple policies.
    
    Args:
        model_paths: List of model paths
        env_type: Environment type
        n_episodes: Episodes per policy
    """
    results = []
    
    for model_path in model_paths:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_path.name}")
        print("=" * 60)
        
        metrics = evaluate_policy(
            model_path=model_path,
            env_type=env_type,
            n_episodes=n_episodes,
            render=False,
            verbose=False,
        )
        
        if metrics:
            metrics["model"] = model_path.name
            results.append(metrics)
    
    # Print comparison table
    if results:
        print("\n" + "=" * 80)
        print("Comparison Results")
        print("=" * 80)
        print(f"{'Model':<30} {'Reward':>12} {'Success':>10} {'Length':>10}")
        print("-" * 80)
        
        for r in sorted(results, key=lambda x: x["mean_reward"], reverse=True):
            print(
                f"{r['model']:<30} "
                f"{r['mean_reward']:>12.2f} "
                f"{r['success_rate']*100:>9.1f}% "
                f"{r['mean_length']:>10.1f}"
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument(
        "model",
        type=Path,
        nargs="+",
        help="Path(s) to model checkpoint(s)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="hover",
        choices=["hover", "waypoint", "trajectory"],
        help="Environment type",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Save evaluation video",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models",
    )
    
    args = parser.parse_args()
    
    if args.compare and len(args.model) > 1:
        compare_policies(
            model_paths=args.model,
            env_type=args.env,
            n_episodes=args.episodes,
        )
    else:
        for model_path in args.model:
            evaluate_policy(
                model_path=model_path,
                env_type=args.env,
                n_episodes=args.episodes,
                render=args.render,
                save_video=args.video,
            )


if __name__ == "__main__":
    main()
