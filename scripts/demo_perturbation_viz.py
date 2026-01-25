#!/usr/bin/env python3
"""Demo script showing perturbation visualization effects.

This script runs the yaw tracking environment with various perturbations
enabled and visualizes them on the rendered frames.

Usage:
    python scripts/demo_perturbation_viz.py [--save-video]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv
from src.perturbations import (
    create_full_visualizer,
    create_gusty_conditions,
    create_realistic_aero,
    create_realistic_physics,
    create_typical_latency,
    create_typical_noise,
)


def run_demo(save_video: bool = False, num_steps: int = 500) -> None:
    """Run the perturbation visualization demo.

    Args:
        save_video: Whether to save the demo as a video file
        num_steps: Number of simulation steps to run
    """
    print("=" * 60)
    print("Perturbation Visualization Demo")
    print("=" * 60)

    # Create environment
    config = YawTrackingConfig(
        model="x500",
        max_episode_steps=num_steps,
        perturbations_enabled=True,
    )
    env = YawTrackingEnv(config=config, render_mode="rgb_array")

    # Add perturbations with interesting effects
    print("\nAdding perturbations:")

    # Wind with gusts
    wind = create_gusty_conditions()
    wind.wind_config.steady_wind_velocity = 4.0
    wind.wind_config.gust_probability = 0.05
    env.add_perturbation("wind", wind)
    print("  - Wind with gusts (4 m/s base, frequent gusts)")

    # Delays
    env.add_perturbation("delays", create_typical_latency())
    print("  - Typical sensor/actuator delays")

    # Sensor noise
    env.add_perturbation("sensor_noise", create_typical_noise())
    print("  - Typical sensor noise")

    # Physics (ground effect)
    physics = create_realistic_physics()
    physics.physics_config.ground_effect_enabled = True
    physics.physics_config.ground_effect_height = 0.8
    env.add_perturbation("physics", physics)
    print("  - Physics with ground effect")

    # Aerodynamics (VRS enabled)
    aero = create_realistic_aero()
    aero.aero_config.vrs_enabled = True
    aero.aero_config.vrs_descent_threshold = 1.5
    env.add_perturbation("aerodynamics", aero)
    print("  - Aerodynamics with VRS detection")

    # Create visualizer
    visualizer = create_full_visualizer()
    print("\nVisualizer created with all effects enabled")

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nStarting simulation for {num_steps} steps...")

    # Storage for video frames
    frames = []

    # Run simulation
    for step in range(num_steps):
        # Simple policy: oscillating yaw command
        t = step * 0.02  # Time in seconds
        action = np.array([0.3 * np.sin(t * 0.5)])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Get the raw frame
        frame = env.render()

        if frame is not None:
            # Get drone position from state
            state = env.sim.get_state()

            # Apply visualization overlay
            manager = env.get_perturbation_manager()
            if manager is not None:
                frame = visualizer.render_overlay(
                    frame,
                    manager,
                    state.position,
                    dt=0.02,
                )

            frames.append(frame)

        # Print progress
        if (step + 1) % 100 == 0:
            wind_info = info.get("perturbations", {}).get("perturbations", {}).get("wind", {})
            wind_speed = wind_info.get("wind_speed", 0)
            gust = "GUST" if wind_info.get("gust_active", False) else ""
            print(f"  Step {step + 1}/{num_steps} - Wind: {wind_speed:.1f} m/s {gust}")

        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break

    env.close()

    print(f"\nRecorded {len(frames)} frames")

    # Save or display
    if save_video and frames:
        try:
            import imageio

            output_path = Path("runs/perturbation_demo.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Saving video to {output_path}...")
            imageio.mimsave(str(output_path), frames, fps=30)
            print(f"Video saved: {output_path}")

        except ImportError:
            print("Warning: imageio not available, cannot save video")
            print("Install with: pip install imageio imageio-ffmpeg")

    elif frames:
        # Display last frame if not saving video
        try:
            import cv2

            print("\nDisplaying sample frames (press any key to continue)...")

            # Show a few frames
            for i in [0, len(frames) // 4, len(frames) // 2, -1]:
                frame = frames[i]
                # Convert RGB to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Perturbation Visualization Demo", frame_bgr)
                cv2.waitKey(1000)  # Show for 1 second

            cv2.destroyAllWindows()

        except ImportError:
            print("Warning: OpenCV display not available")

    print("\nDemo complete!")


def main() -> None:
    """Parse arguments and run demo."""
    parser = argparse.ArgumentParser(description="Demo perturbation visualization effects")
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save demo as video file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of simulation steps (default: 500)",
    )

    args = parser.parse_args()

    run_demo(save_video=args.save_video, num_steps=args.steps)


if __name__ == "__main__":
    main()
