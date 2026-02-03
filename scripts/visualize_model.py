#!/usr/bin/env python3
"""Interactive visualization of trained yaw tracking model in MuJoCo viewer.

Note: This script is kept for backward compatibility. Prefer:
    python scripts/visualize_mujoco.py --mode interactive --model runs/<run_name>/best_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import stable_baselines3  # noqa: F401
except ImportError:
    print("Error: stable-baselines3 required")
    sys.exit(1)

from src.deployment.model_loading import load_model_and_vecnormalize
from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv

# Try to import mujoco viewer (optional, we'll use gymnasium render if not available)
try:
    import mujoco.viewer

    MUJOCO_VIEWER_AVAILABLE = True
except ImportError:
    MUJOCO_VIEWER_AVAILABLE = False
    print("Note: MuJoCo viewer not available, using gymnasium render mode")


def load_model_and_env(model_path: Path):
    """Load trained model and environment.

    Args:
        model_path: Path to model directory or file

    Returns:
        Tuple of (model, env, vec_normalize)
    """
    # Create environment (same config as training)
    config = YawTrackingConfig(
        target_patterns=["circular"],  # Start with simple pattern
        target_speed_min=0.05,
        target_speed_max=0.1,
        control_frequency=100.0,
        action_smoothing=0.3,
        max_action_change=0.5,
        yaw_authority=0.20,
        yaw_rate_kp=5.0,
    )

    env = YawTrackingEnv(config=config, render_mode="rgb_array")

    try:
        model, vec_normalize, resolved = load_model_and_vecnormalize(
            model_path,
            env_factory=lambda: YawTrackingEnv(config=config),
        )
        print(f"Loaded model from {resolved}")
        if vec_normalize:
            print("Loaded VecNormalize")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    return model, env, vec_normalize


def visualize_interactive(
    model_path: Path,
    pattern: str = "circular",
    seed: int = 42,
    duration: float = 30.0,
):
    """Run interactive MuJoCo viewer with trained model.

    Args:
        model_path: Path to trained model
        pattern: Target pattern to visualize
        seed: Random seed
        duration: Duration in seconds
    """
    # Load model and env
    model, env, vec_normalize = load_model_and_env(model_path)
    if model is None:
        return

    print(f"\n{'=' * 60}")
    print("Interactive Visualization")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Pattern: {pattern}")
    print(f"Duration: {duration}s")
    print("\nControls:")
    print("  - Drag to rotate camera")
    print("  - Scroll to zoom")
    print("  - Double-click to reset camera")
    print("  - ESC to quit")
    print(f"{'=' * 60}\n")

    # Reset environment
    options = {"pattern": pattern} if pattern else None
    obs, info = env.reset(seed=seed, options=options)

    if vec_normalize:
        obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
        obs = obs_vec[0]

    # Use matplotlib for visualization (works on all platforms)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Error: matplotlib required for visualization")
        env.close()
        return

    print("Starting visualization with matplotlib...")

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    img = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    plt.tight_layout()

    step_count = 0
    max_steps = int(duration * env.config.control_frequency)
    frames = []

    def animate(frame):
        nonlocal obs, step_count, frames

        if step_count >= max_steps:
            return img

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Normalize observation if needed
        if vec_normalize:
            # VecNormalize expects vectorized obs
            obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
            obs = obs_vec[0]

        # Render frame
        frame_img = env.render()
        if frame_img is not None:
            img.set_array(frame_img)
            frames.append(frame_img.copy())

        step_count += 1

        # Reset if episode ended
        if terminated or truncated:
            obs, info = env.reset(seed=seed + step_count, options=options)
            if vec_normalize:
                obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
                obs = obs_vec[0]
            print(f"Episode ended at step {step_count}, resetting...")

        return img

    # Create animation (keep reference to prevent garbage collection)
    _ = FuncAnimation(fig, animate, interval=33, blit=False, cache_frame_data=False)

    plt.show()

    env.close()

    env.close()
    print("\nVisualization complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive visualization of trained yaw tracking model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model (directory or .zip file)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="circular",
        choices=["circular", "sinusoidal", "step", "random"],
        help="Target pattern to visualize",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Visualization duration in seconds",
    )

    args = parser.parse_args()

    visualize_interactive(
        model_path=args.model,
        pattern=args.pattern,
        seed=args.seed,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
