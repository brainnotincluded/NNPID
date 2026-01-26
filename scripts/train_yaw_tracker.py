#!/usr/bin/env python3
"""Train yaw tracking neural network using Stable-Baselines3.

This script trains a neural network to control drone yaw rate to
track a moving target. It supports curriculum learning to gradually
increase task difficulty.

Usage:
    python scripts/train_yaw_tracker.py --config config/yaw_tracking.yaml
    python scripts/train_yaw_tracker.py --timesteps 100000 --n-envs 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3")

import yaml

from src.environments.yaw_tracking_env import YawTrackingEnv, YawTrackingConfig


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning.
    
    Updates environment parameters based on training progress.
    """
    
    def __init__(
        self,
        stages: List[Dict[str, Any]],
        env_update_fn: Callable[[VecEnv, Dict[str, Any]], None],
        verbose: int = 1,
    ):
        """Initialize curriculum callback.
        
        Args:
            stages: List of curriculum stages with timestep thresholds
            env_update_fn: Function to update environment parameters
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.stages = sorted(stages, key=lambda x: x["timesteps"])
        self.env_update_fn = env_update_fn
        self.current_stage = 0
    
    def _on_step(self) -> bool:
        """Check if curriculum should advance."""
        # Check if we should advance to next stage
        while (
            self.current_stage < len(self.stages) - 1
            and self.num_timesteps >= self.stages[self.current_stage + 1]["timesteps"]
        ):
            self.current_stage += 1
            stage = self.stages[self.current_stage]
            
            if self.verbose > 0:
                desc = stage.get("description", f"Stage {self.current_stage + 1}")
                print(f"\n[Curriculum] Advancing to: {desc}")
                print(f"  Target speed max: {stage.get('target_speed_max', 'unchanged')}")
                print(f"  Patterns: {stage.get('target_patterns', 'unchanged')}")
            
            # Update environment parameters
            self.env_update_fn(self.training_env, stage)
        
        return True
    
    def _on_training_start(self) -> None:
        """Apply initial curriculum stage."""
        if len(self.stages) > 0:
            stage = self.stages[0]
            if self.verbose > 0:
                desc = stage.get("description", "Stage 1")
                print(f"\n[Curriculum] Starting with: {desc}")
            self.env_update_fn(self.training_env, stage)


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None or not config_path.exists():
        # Default configuration
        return {
            "environment": {
                "target_patterns": ["circular", "random"],
                "target_speed_min": 0.5,
                "target_speed_max": 2.0,
                "hover_height": 1.0,
                "max_yaw_rate": 2.0,
            },
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
                    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                },
            },
            "training": {
                "total_timesteps": 500_000,
                "n_envs": 8,
                "seed": 42,
                "save_freq": 25_000,
                "eval_freq": 10_000,
                "n_eval_episodes": 10,
                "curriculum": {
                    "enabled": True,
                    "stages": [
                        {"timesteps": 0, "target_speed_max": 0.5, "target_patterns": ["circular"]},
                        {"timesteps": 100_000, "target_speed_max": 1.0},
                        {"timesteps": 250_000, "target_speed_max": 1.5},
                        {"timesteps": 400_000, "target_speed_max": 2.0},
                    ],
                },
            },
        }
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_env(
    env_config: Dict[str, Any],
    rank: int,
    seed: int = 0,
) -> Callable[[], YawTrackingEnv]:
    """Create environment factory function.
    
    Args:
        env_config: Environment configuration
        rank: Environment rank for seed offset
        seed: Base random seed
        
    Returns:
        Factory function for creating environment
    """
    def _init() -> YawTrackingEnv:
        # Get stabilizer gains from config
        stabilizer = env_config.get("stabilizer", {})
        
        # Get perturbation settings
        perturbations = env_config.get("perturbations", {})
        
        # Build config from dict
        config = YawTrackingConfig(
            model=env_config.get("model", "generic"),
            physics_timestep=env_config.get("physics_timestep", 0.002),
            control_frequency=env_config.get("control_frequency", 50.0),
            max_episode_steps=env_config.get("max_episode_steps", 1000),
            hover_height=env_config.get("hover_height", 1.0),
            target_patterns=env_config.get("target_patterns", ["circular", "random"]),
            target_radius=env_config.get("target_radius", 3.0),
            target_speed_min=env_config.get("target_speed_min", 0.5),
            target_speed_max=env_config.get("target_speed_max", 2.0),
            max_yaw_rate=env_config.get("max_yaw_rate", 2.0),
            # SITL-style PID stabilizer gains
            altitude_kp=stabilizer.get("altitude_kp", 10.0),
            altitude_ki=stabilizer.get("altitude_ki", 2.0),
            altitude_kd=stabilizer.get("altitude_kd", 5.0),
            attitude_kp=stabilizer.get("attitude_kp", 20.0),
            attitude_ki=stabilizer.get("attitude_ki", 1.0),
            attitude_kd=stabilizer.get("attitude_kd", 8.0),
            yaw_rate_kp=stabilizer.get("yaw_rate_kp", 3.0),
            base_thrust=stabilizer.get("base_thrust", 0.62),
            # Safety settings
            safety_tilt_threshold=stabilizer.get("safety_tilt_threshold", 0.5),
            yaw_authority=stabilizer.get("yaw_authority", 0.03),
            max_integral=stabilizer.get("max_integral", 0.5),
            # Perturbation settings
            perturbations_enabled=perturbations.get("enabled", False),
            perturbation_intensity=perturbations.get("intensity", 1.0) if isinstance(perturbations.get("intensity"), (int, float)) else 1.0,
        )
        
        env = YawTrackingEnv(config=config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    
    return _init


def update_env_curriculum(env: VecEnv, stage: Dict[str, Any]) -> None:
    """Update environment parameters for curriculum stage.
    
    Note: This updates the config that will be used on next reset.
    For VecEnv, we need to set attributes on each sub-environment.
    """
    # Get attributes to update
    target_speed_max = stage.get("target_speed_max")
    target_patterns = stage.get("target_patterns")
    perturbations = stage.get("perturbations")
    
    # Update each sub-environment
    # For SubprocVecEnv, we'd need to use env_method
    # For DummyVecEnv, we can access envs directly
    if hasattr(env, 'envs'):
        for sub_env in env.envs:
            # Get the actual environment (unwrap Monitor)
            actual_env = sub_env.env if hasattr(sub_env, 'env') else sub_env
            
            if target_speed_max is not None:
                actual_env.config.target_speed_max = target_speed_max
            
            if target_patterns is not None:
                actual_env.config.target_patterns = target_patterns
                # Recreate target patterns with new config
                actual_env._target_patterns = actual_env._create_target_patterns()
            
            # Update perturbations if specified
            if perturbations is not None and hasattr(actual_env, 'perturbation_manager'):
                if 'enabled' in perturbations:
                    actual_env.perturbation_manager.enabled = perturbations['enabled']
                if 'intensity' in perturbations:
                    actual_env.perturbation_manager.global_intensity = perturbations['intensity']


def train(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    total_timesteps: Optional[int] = None,
    n_envs: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = "auto",
    verbose: int = 1,
) -> None:
    """Train yaw tracking neural network.
    
    Args:
        config_path: Path to config YAML
        output_dir: Output directory for checkpoints and logs
        total_timesteps: Override total timesteps
        n_envs: Override number of environments
        seed: Override random seed
        device: Training device (auto, cpu, cuda)
        verbose: Verbosity level
    """
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 is required for training")
        print("Install with: pip install stable-baselines3")
        return
    
    # Load configuration
    config = load_config(config_path)
    env_config = config.get("environment", {})
    algo_config = config.get("algorithm", {})
    train_config = config.get("training", {})
    
    # Override from command line
    if total_timesteps is not None:
        train_config["total_timesteps"] = total_timesteps
    if n_envs is not None:
        train_config["n_envs"] = n_envs
    if seed is not None:
        train_config["seed"] = seed
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"runs/yaw_tracking_{timestamp}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("=" * 60)
    print("  Yaw Tracking Neural Network Training")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Algorithm: {algo_config.get('name', 'PPO')}")
    print(f"Total timesteps: {train_config.get('total_timesteps', 500_000):,}")
    print(f"Parallel environments: {train_config.get('n_envs', 8)}")
    print()
    
    # Set seed
    seed_value = train_config.get("seed", 42)
    set_random_seed(seed_value)
    
    # Create environments
    n_envs_actual = train_config.get("n_envs", 8)
    
    print(f"Creating {n_envs_actual} parallel environments...")
    
    # Use DummyVecEnv for easier curriculum updates
    # (SubprocVecEnv would require more complex IPC for curriculum)
    env = DummyVecEnv([
        make_env(env_config, i, seed_value) 
        for i in range(n_envs_actual)
    ])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(env_config, 0, seed_value + 1000)])
    
    # Setup algorithm
    algo_name = algo_config.get("name", "PPO").upper()
    
    policy_kwargs = algo_config.get("policy_kwargs", {})
    if "net_arch" in policy_kwargs:
        net_arch = policy_kwargs["net_arch"]
        if isinstance(net_arch, dict):
            policy_kwargs["net_arch"] = dict(
                pi=net_arch.get("pi", [64, 64]),
                vf=net_arch.get("vf", [64, 64]),
            )
    
    # Convert activation_fn string to actual function
    if "activation_fn" in policy_kwargs:
        activation_name = policy_kwargs["activation_fn"]
        if isinstance(activation_name, str):
            import torch.nn as nn
            activation_map = {
                "tanh": nn.Tanh,
                "relu": nn.ReLU,
                "elu": nn.ELU,
                "leaky_relu": nn.LeakyReLU,
                "gelu": nn.GELU,
                "silu": nn.SiLU,
            }
            policy_kwargs["activation_fn"] = activation_map.get(
                activation_name.lower(), nn.Tanh
            )
    
    common_params = {
        "policy": algo_config.get("policy", "MlpPolicy"),
        "env": env,
        "learning_rate": algo_config.get("learning_rate", 3e-4),
        "gamma": algo_config.get("gamma", 0.99),
        "verbose": verbose,
        "tensorboard_log": str(output_dir / "tensorboard"),
        "seed": seed_value,
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
    
    print(f"Model created with {sum(p.numel() for p in model.policy.parameters()):,} parameters")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(train_config.get("save_freq", 25_000) // n_envs_actual, 1),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="yaw_tracker",
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(train_config.get("eval_freq", 10_000) // n_envs_actual, 1),
        n_eval_episodes=train_config.get("n_eval_episodes", 10),
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    # Curriculum callback
    curriculum_config = train_config.get("curriculum", {})
    if curriculum_config.get("enabled", False):
        stages = curriculum_config.get("stages", [])
        if stages:
            curriculum_callback = CurriculumCallback(
                stages=stages,
                env_update_fn=update_env_curriculum,
                verbose=verbose,
            )
            callbacks.append(curriculum_callback)
            print(f"Curriculum learning enabled with {len(stages)} stages")
    
    callback_list = CallbackList(callbacks)
    
    # Train
    total_steps = train_config.get("total_timesteps", 500_000)
    print(f"\nStarting training for {total_steps:,} timesteps...")
    print("=" * 60)
    
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callback_list,
            log_interval=train_config.get("log_interval", 10),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_path = output_dir / "final_model"
    model.save(str(final_path))
    print(f"\nSaved final model to {final_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"Best model: {output_dir / 'best_model'}")
    print(f"Final model: {final_path}")
    print(f"TensorBoard: tensorboard --logdir {output_dir / 'tensorboard'}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train yaw tracking neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/yaw_tracking.yaml"),
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
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
        config_path=args.config,
        output_dir=args.output,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
