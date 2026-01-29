#!/usr/bin/env python3
"""Run trained yaw tracker model in MuJoCo with detailed logging.

This script runs the best trained model in MuJoCo simulation and logs:
1. Observations sent to the neural network
2. Actions output by the neural network
3. Drone state (position, attitude, yaw_rate)
4. Target state
5. Rewards and tracking metrics

This is useful for:
- Verifying model works locally
- Preparing for Webots-SITL integration (understanding the data flow)
- Debugging model behavior

Usage:
    python scripts/run_model_mujoco.py --model runs/analysis_20260126_150455/best_model
    python scripts/run_model_mujoco.py --model runs/best_model --episodes 5 --steps 1000
    python scripts/run_model_mujoco.py --model runs/best_model --render --output flight_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

from src.environments.yaw_tracking_env import (
    YawTrackingConfig,
    YawTrackingEnv,
)

logger = logging.getLogger(__name__)


@dataclass
class StepData:
    """Data from a single step for logging."""

    step: int
    timestamp: float
    observation: list
    action: float
    reward: float
    yaw_error_rad: float
    yaw_error_deg: float
    drone_yaw_rad: float
    drone_position: list
    drone_velocity: list
    target_angle_rad: float
    tracking: bool


def run_episode(
    env: YawTrackingEnv,
    model: Any,
    episode_num: int,
    max_steps: int = 1000,
    render: bool = False,
    vec_normalize: Any = None,
) -> tuple[dict, list]:
    """Run single episode and return metrics and step data.

    Args:
        env: Gymnasium environment
        model: Loaded trained model
        episode_num: Episode number
        max_steps: Max steps per episode
        render: Whether to capture frames
        vec_normalize: VecNormalize wrapper if available

    Returns:
        Tuple of (episode_metrics, step_data_list)
    """
    obs, info = env.reset()
    episode_reward = 0.0
    steps_data = []
    tracking_time = 0
    crash = False

    start_time = time.time()

    for step in range(max_steps):
        # Normalize observation if VecNormalize available
        obs_normalized = obs
        if vec_normalize is not None:
            obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
            obs_normalized = np.array(obs_vec[0], dtype=np.float32)

        # Get action from model
        action, _ = model.predict(obs_normalized, deterministic=True)
        action = float(action[0])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(np.array([action]))
        episode_reward += reward
        done = terminated or truncated

        # Extract state
        yaw_error_rad = abs(info.get("yaw_error", 0.0))
        yaw_error_deg = float(np.degrees(yaw_error_rad))
        tracking = yaw_error_rad < 0.1  # ~5.7 degrees

        if tracking:
            tracking_time += 1

        step_data = StepData(
            step=step,
            timestamp=time.time() - start_time,
            observation=obs.tolist() if isinstance(obs, np.ndarray) else obs,
            action=action,
            reward=float(reward),
            yaw_error_rad=float(yaw_error_rad),
            yaw_error_deg=yaw_error_deg,
            drone_yaw_rad=info.get("drone_yaw", 0.0),
            drone_position=info.get("drone_position", [0, 0, 0]),
            drone_velocity=info.get("drone_velocity", [0, 0, 0]),
            target_angle_rad=info.get("target_angle", 0.0),
            tracking=tracking,
        )
        steps_data.append(step_data)

        if render:
            env.render()

        if done:
            if terminated:
                crash = True
            break

    # Compute metrics
    metrics = {
        "episode": episode_num,
        "total_reward": float(episode_reward),
        "episode_length": len(steps_data),
        "tracking_percentage": (tracking_time / len(steps_data) * 100) if steps_data else 0,
        "mean_yaw_error_deg": float(np.mean([s.yaw_error_deg for s in steps_data])),
        "crashed": crash,
        "duration": time.time() - start_time,
    }

    return metrics, steps_data


def load_model_and_vecnorm(model_path: Path) -> tuple[Any, Any]:
    """Load model and VecNormalize.

    Returns:
        Tuple of (model, vec_normalize)
    """
    import pickle

    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import VecNormalize

    model_path = Path(model_path)

    # Handle directory or file
    if model_path.is_dir():
        for name in ["best_model.zip", "final_model.zip"]:
            candidate = model_path / name
            if candidate.exists():
                model_path = candidate
                break

    if not model_path.suffix:
        model_path = model_path.with_suffix(".zip")

    logger.info(f"Loading model from {model_path}")

    # Load model
    model = None
    try:
        model = PPO.load(str(model_path))
    except Exception:
        try:
            model = SAC.load(str(model_path))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None, None

    # Load VecNormalize if exists
    vec_normalize = None
    vec_norm_path = model_path.parent.parent / "vec_normalize.pkl"
    if vec_norm_path.exists():
        try:
            with open(vec_norm_path, "rb") as f:
                vec_normalize = pickle.load(f)
            logger.info(f"Loaded VecNormalize from {vec_norm_path}")
        except Exception as e:
            logger.warning(f"Failed to load VecNormalize: {e}")

    return model, vec_normalize


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run trained model in MuJoCo with detailed logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render MuJoCo visualization",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save detailed log to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Load model
    model, vec_normalize = load_model_and_vecnorm(args.model)
    if model is None:
        logger.error("Failed to load model")
        sys.exit(1)

    # Create environment
    config = YawTrackingConfig()
    env = YawTrackingEnv(config=config, render_mode="rgb_array" if args.render else None)

    print("=" * 70)
    print("  Running Trained Model in MuJoCo")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.steps}")
    print(f"VecNormalize: {'Yes' if vec_normalize else 'No'}")
    print()

    # Run episodes
    all_metrics = []
    all_steps = []

    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}...", end=" ", flush=True)

        metrics, steps_data = run_episode(
            env=env,
            model=model,
            episode_num=ep,
            max_steps=args.steps,
            render=args.render,
            vec_normalize=vec_normalize,
        )

        all_metrics.append(metrics)
        all_steps.append([asdict(s) for s in steps_data])

        print(
            f"Reward: {metrics['total_reward']:.1f}, "
            f"Tracking: {metrics['tracking_percentage']:.1f}%, "
            f"Yaw Error: {metrics['mean_yaw_error_deg']:.1f}°"
        )

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    mean_reward = np.mean([m["total_reward"] for m in all_metrics])
    mean_tracking = np.mean([m["tracking_percentage"] for m in all_metrics])
    mean_error = np.mean([m["mean_yaw_error_deg"] for m in all_metrics])
    crash_rate = sum(1 for m in all_metrics if m["crashed"]) / len(all_metrics) * 100

    print(f"Mean Reward: {mean_reward:.1f}")
    print(f"Mean Tracking: {mean_tracking:.1f}%")
    print(f"Mean Yaw Error: {mean_error:.1f}°")
    print(f"Crash Rate: {crash_rate:.1f}%")
    print()

    # Save detailed log
    if args.output:
        log_data = {
            "summary": {
                "model_path": str(args.model),
                "episodes": args.episodes,
                "max_steps_per_episode": args.steps,
                "mean_reward": float(mean_reward),
                "mean_tracking_percentage": float(mean_tracking),
                "mean_yaw_error_deg": float(mean_error),
                "crash_rate_percent": float(crash_rate),
            },
            "episode_metrics": all_metrics,
            "episode_steps": all_steps,
        }

        with open(args.output, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Detailed log saved to {args.output}")


if __name__ == "__main__":
    main()
