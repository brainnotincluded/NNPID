#!/usr/bin/env python3
"""Train neural network controllers using setpoint environments.

This script provides two training modes:
1. Fast training with simulated position controller (SetpointHoverEnv)
2. SITL-in-loop training with real PX4 controllers (SITLEnv)

Usage:
    # Fast training (no PX4 required)
    python scripts/train_setpoint.py --env hover --steps 500000

    # SITL training (requires PX4 SITL running)
    python scripts/train_setpoint.py --env sitl --steps 100000

    # Export trained model to ONNX
    python scripts/train_setpoint.py --export checkpoints/best_model.zip
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_env(env_name: str, seed: int = None):
    """Create training environment.

    Args:
        env_name: Environment name (hover, waypoint, tracking, sitl)
        seed: Random seed

    Returns:
        Gymnasium environment
    """
    from src.environments import (
        SetpointHoverEnv,
        SetpointTrackingEnv,
        SetpointWaypointEnv,
        SITLEnv,
    )

    env_map = {
        "hover": SetpointHoverEnv,
        "waypoint": SetpointWaypointEnv,
        "tracking": SetpointTrackingEnv,
        "sitl": SITLEnv,
    }

    if env_name not in env_map:
        raise ValueError(f"Unknown environment: {env_name}. Options: {list(env_map.keys())}")

    env = env_map[env_name]()

    if seed is not None and hasattr(env, "reset"):
        env.reset(seed=seed)

    return env


def train(
    env_name: str = "hover",
    algorithm: str = "PPO",
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    eval_freq: int = 10_000,
):
    """Train a neural network controller.

    Args:
        env_name: Environment name
        algorithm: RL algorithm (PPO, SAC)
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        seed: Random seed
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for TensorBoard logs
        eval_freq: Evaluation frequency
    """
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError:
        print("Error: stable-baselines3 is required for training")
        print("Install with: pip install stable-baselines3")
        sys.exit(1)

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{env_name}_{algorithm}_{timestamp}"

    checkpoint_path = Path(checkpoint_dir) / run_name
    log_path = Path(log_dir) / run_name

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print("Training configuration:")
    print(f"  Environment: {env_name}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Checkpoints: {checkpoint_path}")
    print(f"  Logs: {log_path}")
    print()

    # Create vectorized environment
    def make_env(rank: int):
        def _init():
            env = create_env(env_name, seed=seed + rank)
            env = Monitor(env)
            return env

        return _init

    if n_envs > 1 and env_name != "sitl":
        # SITL env can't be parallelized easily
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(n_envs)])

    # Create model
    algo_map = {"PPO": PPO, "SAC": SAC}
    if algorithm not in algo_map:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Algorithm-specific hyperparameters
    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_path),
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            },
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_path),
            seed=seed,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "qf": [256, 256]},
            },
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_path),
        log_path=str(log_path),
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(checkpoint_path),
        name_prefix="model",
    )

    callbacks = CallbackList([eval_callback, checkpoint_callback])

    # Train
    print("Starting training...")
    print("Monitor with: tensorboard --logdir", log_path)
    print()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_path = checkpoint_path / "final_model"
    model.save(str(final_path))
    print(f"\nFinal model saved to: {final_path}.zip")

    # Cleanup
    env.close()
    eval_env.close()

    return str(final_path) + ".zip"


def export_model(model_path: str, output_path: str = None):
    """Export trained model to ONNX.

    Args:
        model_path: Path to trained SB3 model
        output_path: Path for ONNX output
    """
    from src.deployment import export_to_onnx

    if output_path is None:
        output_path = Path(model_path).with_suffix(".onnx")

    print(f"Exporting {model_path} to ONNX...")

    # Detect algorithm from model file
    algorithm = "PPO"  # Default
    if "sac" in model_path.lower():
        algorithm = "SAC"

    export_to_onnx(
        model_path=model_path,
        output_path=output_path,
        algorithm=algorithm,
        observation_dim=19,  # SetpointBaseEnv observation
        action_dim=4,
    )

    print(f"Exported to: {output_path}")


def evaluate(model_path: str, env_name: str = "hover", n_episodes: int = 10):
    """Evaluate a trained model.

    Args:
        model_path: Path to trained model
        env_name: Environment name
        n_episodes: Number of evaluation episodes
    """
    import numpy as np
    from stable_baselines3 import PPO, SAC

    # Load model
    algorithm = "PPO"
    if "sac" in model_path.lower():
        algorithm = "SAC"

    algo_map = {"PPO": PPO, "SAC": SAC}
    model = algo_map[algorithm].load(model_path)

    # Create environment
    env = create_env(env_name)

    print(f"Evaluating {model_path}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {n_episodes}")
    print()

    rewards = []
    lengths = []
    successes = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(info.get("is_success", False))

        print(
            f"  Episode {ep + 1}: reward={total_reward:.1f}, steps={steps}, success={info.get('is_success', 'N/A')}"
        )

    print()
    print("Results:")
    print(f"  Mean reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Mean length: {np.mean(lengths):.0f}")
    print(f"  Success rate: {100 * np.mean(successes):.1f}%")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train neural network drone controllers")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--env",
        type=str,
        default="hover",
        choices=["hover", "waypoint", "tracking", "sitl"],
        help="Environment name",
    )
    train_parser.add_argument(
        "--algo",
        type=str,
        default="PPO",
        choices=["PPO", "SAC"],
        help="RL algorithm",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=500_000,
        help="Total training timesteps",
    )
    train_parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("model", type=str, help="Path to model")
    export_parser.add_argument("--output", type=str, help="Output path")

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("model", type=str, help="Path to model")
    eval_parser.add_argument(
        "--env",
        type=str,
        default="hover",
        help="Environment name",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes",
    )

    args = parser.parse_args()

    if args.command == "train":
        train(
            env_name=args.env,
            algorithm=args.algo,
            total_timesteps=args.steps,
            n_envs=args.n_envs,
            seed=args.seed,
        )
    elif args.command == "export":
        export_model(args.model, args.output)
    elif args.command == "eval":
        evaluate(args.model, args.env, args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
