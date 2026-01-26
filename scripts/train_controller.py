#!/usr/bin/env python3
"""Train neural network controller using Stable-Baselines3."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed")

import yaml

from src.environments.hover_env import HoverEnv, HoverEnvConfig
from src.environments.trajectory_env import TrajectoryEnv, TrajectoryEnvConfig
from src.environments.waypoint_env import WaypointEnv, WaypointEnvConfig


def load_config(config_path: Path | None) -> dict[str, Any]:
    """Load training configuration from YAML file."""
    if config_path is None or not config_path.exists():
        # Default configuration
        return {
            "algorithm": {
                "name": "PPO",
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "policy": "MlpPolicy",
                "policy_kwargs": {
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                },
            },
            "training": {
                "total_timesteps": 1_000_000,
                "n_envs": 8,
                "seed": 42,
                "save_freq": 50_000,
                "checkpoint_path": "checkpoints/",
                "log_interval": 10,
                "tensorboard_log": "logs/tensorboard/",
                "eval_freq": 25_000,
                "n_eval_episodes": 10,
            },
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


def create_env(env_type: str, config: dict[str, Any], seed: int = 0):
    """Create training environment."""
    if env_type == "hover":
        env_config = HoverEnvConfig()
        return HoverEnv(config=env_config)
    elif env_type == "waypoint":
        env_config = WaypointEnvConfig()
        return WaypointEnv(config=env_config)
    elif env_type == "trajectory":
        env_config = TrajectoryEnvConfig()
        return TrajectoryEnv(config=env_config)
    else:
        raise ValueError(f"Unknown environment: {env_type}")


def make_env(env_type: str, rank: int, seed: int = 0):
    """Create environment factory for vectorized envs."""

    def _init():
        env = create_env(env_type, {}, seed + rank)
        env = Monitor(env)
        return env

    return _init


def train(
    env_type: str = "hover",
    config_path: Path | None = None,
    output_dir: Path | None = None,
    resume_from: Path | None = None,
    device: str = "auto",
    verbose: int = 1,
):
    """Train neural network controller.

    Args:
        env_type: Environment type (hover, waypoint, trajectory)
        config_path: Path to training config YAML
        output_dir: Output directory for checkpoints and logs
        resume_from: Path to checkpoint to resume from
        device: Device for training (auto, cpu, cuda)
        verbose: Verbosity level
    """
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 is required for training")
        print("Install with: pip install stable-baselines3")
        return

    # Load configuration
    config = load_config(config_path)
    algo_config = config.get("algorithm", {})
    train_config = config.get("training", {})

    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/{env_type}_{timestamp}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    print(f"Training {algo_config.get('name', 'PPO')} on {env_type} environment")
    print(f"Output directory: {output_dir}")

    # Create vectorized environments
    n_envs = train_config.get("n_envs", 8)
    seed = train_config.get("seed", 42)

    print(f"Creating {n_envs} parallel environments...")

    # Use SubprocVecEnv for true parallelism (slower to start, faster training)
    # Use DummyVecEnv for debugging (faster to start, slower training)
    use_subprocess = n_envs > 1

    if use_subprocess:
        env = SubprocVecEnv([make_env(env_type, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(env_type, 0, seed)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(env_type, 0, seed + 1000)])

    # Setup algorithm
    algo_name = algo_config.get("name", "PPO").upper()

    policy_kwargs = algo_config.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        # Convert dict to list format expected by SB3
        net_arch = policy_kwargs["net_arch"]
        if isinstance(net_arch, dict):
            policy_kwargs["net_arch"] = {
                "pi": net_arch.get("pi", [256, 256]),
                "vf": net_arch.get("vf", [256, 256]),
            }

    common_params = {
        "policy": algo_config.get("policy", "MlpPolicy"),
        "env": env,
        "learning_rate": algo_config.get("learning_rate", 3e-4),
        "gamma": algo_config.get("gamma", 0.99),
        "verbose": verbose,
        "tensorboard_log": str(output_dir / "tensorboard"),
        "seed": seed,
        "device": device,
        "policy_kwargs": policy_kwargs if policy_kwargs else None,
    }

    if algo_name == "PPO":
        model = PPO(
            **common_params,
            n_steps=algo_config.get("n_steps", 2048),
            batch_size=algo_config.get("batch_size", 64),
            n_epochs=algo_config.get("n_epochs", 10),
            gae_lambda=algo_config.get("gae_lambda", 0.95),
            clip_range=algo_config.get("clip_range", 0.2),
            ent_coef=algo_config.get("ent_coef", 0.01),
            vf_coef=algo_config.get("vf_coef", 0.5),
            max_grad_norm=algo_config.get("max_grad_norm", 0.5),
        )
    elif algo_name == "SAC":
        model = SAC(
            **common_params,
            buffer_size=algo_config.get("buffer_size", 1_000_000),
            batch_size=algo_config.get("batch_size", 256),
            tau=algo_config.get("tau", 0.005),
            ent_coef=algo_config.get("ent_coef", "auto"),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Resume from checkpoint if specified
    if resume_from is not None and resume_from.exists():
        print(f"Resuming from {resume_from}")
        model.set_parameters(resume_from)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config.get("save_freq", 50_000) // n_envs, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="model",
        save_replay_buffer=algo_name == "SAC",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(train_config.get("eval_freq", 25_000) // n_envs, 1),
        n_eval_episodes=train_config.get("n_eval_episodes", 10),
        deterministic=True,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train
    total_timesteps = train_config.get("total_timesteps", 1_000_000)
    print(f"Starting training for {total_timesteps:,} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=train_config.get("log_interval", 10),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()

    print("Training complete!")
    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train drone controller")
    parser.add_argument(
        "--env",
        type=str,
        default="hover",
        choices=["hover", "waypoint", "trajectory"],
        help="Environment type",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )

    args = parser.parse_args()

    train(
        env_type=args.env,
        config_path=args.config,
        output_dir=args.output,
        resume_from=args.resume,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
