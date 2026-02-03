#!/usr/bin/env python3
"""Visualize trained yaw tracking model in MuJoCo viewer.

Usage:
    mjpython scripts/view_trained_model.py --model path/to/model.zip --pattern circular

Note: This script uses the native MuJoCo viewer. For a unified CLI (interactive + video),
prefer:
    python scripts/visualize_mujoco.py --mode interactive --model runs/<run_name>/best_model
"""

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.model_loading import load_model_and_vecnormalize
from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv


def main():
    parser = argparse.ArgumentParser(description="View trained yaw tracking model")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--pattern",
        default="circular",
        choices=[
            "circular",
            "random",
            "sinusoidal",
            "step",
            "figure8",
            "spiral",
            "evasive",
            "lissajous",
            "multi_frequency",
        ],
        help="Target pattern to test",
    )
    parser.add_argument("--speed", type=float, default=0.3, help="Target angular speed")
    args = parser.parse_args()

    print("=" * 60)
    print("  Trained Model Visualization")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Pattern: {args.pattern}")
    print(f"Speed: {args.speed} rad/s")
    print()

    # Create environment
    config = YawTrackingConfig(
        model="generic",
        target_patterns=[args.pattern],
        target_speed_min=args.speed,
        target_speed_max=args.speed,
        hover_height=1.0,
        max_episode_steps=3000,  # 60 seconds
    )
    env = YawTrackingEnv(config=config)

    # Load model + VecNormalize
    print("Loading trained model...")
    model, vec_normalize, resolved = load_model_and_vecnormalize(
        args.model,
        env_factory=lambda: YawTrackingEnv(config=config),
    )
    print(f"✓ Model loaded from {resolved}")

    print("\nStarting visualization...")
    print("Controls:")
    print("  - Drag to rotate view")
    print("  - Scroll to zoom")
    print("  - Double-click to reset")
    print("  - ESC to quit")
    print()

    # Reset environment
    obs, info = env.reset()

    # Get simulator
    sim = env.sim

    # Create viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        step_count = 0
        episode_reward = 0.0

        while viewer.is_running():
            # Normalize observation if needed
            obs_for_model = obs
            if vec_normalize is not None:
                obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
                obs_for_model = obs_vec[0]

            # Get action from policy
            action, _states = model.predict(obs_for_model, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Update viewer
            viewer.sync()

            # Print status every 50 steps
            if step_count % 50 == 0:
                yaw_error_deg = np.rad2deg(info.get("yaw_error", 0.0))
                print(
                    f"Step {step_count:4d} | Reward: {episode_reward:7.1f} | Yaw Error: {yaw_error_deg:5.1f}°"
                )

            # Reset if done
            if terminated or truncated:
                print("\nEpisode finished!")
                print(f"  Total steps: {step_count}")
                print(f"  Total reward: {episode_reward:.2f}")
                print(f"  Reason: {'terminated' if terminated else 'truncated'}")
                print("\nRestarting episode...\n")

                obs, info = env.reset()
                step_count = 0
                episode_reward = 0.0

    env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
