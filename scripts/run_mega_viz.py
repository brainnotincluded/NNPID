#!/usr/bin/env python3
"""Run full mega visualization with all effects.

This script launches the yaw tracking environment with a trained model
and displays all visualization components: 3D scene objects, neural network
visualization, telemetry HUD, and perturbation effects.

Usage:
    python scripts/run_mega_viz.py --model runs/best_model.zip
    python scripts/run_mega_viz.py --model runs/best_model.zip --perturbations config/perturbations.yaml
    python scripts/run_mega_viz.py --model runs/best_model.zip --record output.mp4
    python scripts/run_mega_viz.py --no-model --viz-mode minimal
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, recording disabled")

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False


def load_model(model_path: str | None):
    """Load stable-baselines3 model.

    Args:
        model_path: Path to model, or None to use random actions

    Returns:
        Model or None
    """
    if model_path is None:
        return None

    path = Path(model_path)
    if not path.exists():
        print(f"Warning: Model not found: {model_path}")
        return None

    try:
        from stable_baselines3 import PPO, SAC

        try:
            model = PPO.load(model_path)
            print(f"Loaded PPO model from {model_path}")
        except Exception:
            model = SAC.load(model_path)
            print(f"Loaded SAC model from {model_path}")

        return model

    except ImportError:
        print("Warning: stable-baselines3 not available")
        return None


def load_perturbations(config_path: str | None):
    """Load perturbation configuration.

    Args:
        config_path: Path to perturbations config

    Returns:
        PerturbationManager or None
    """
    if config_path is None:
        return None

    try:
        from src.perturbations import PerturbationManager

        manager = PerturbationManager()
        manager.load_config(config_path)
        print(f"Loaded perturbations from {config_path}")
        return manager

    except ImportError:
        print("Warning: Perturbation module not available")
        return None
    except Exception as e:
        print(f"Warning: Could not load perturbations: {e}")
        return None


def create_visualizer(viz_mode: str):
    """Create appropriate visualizer based on mode.

    Args:
        viz_mode: Visualization mode (full, minimal, hud, nn)

    Returns:
        MegaVisualizer instance
    """
    from src.visualization.mujoco_overlay import (
        MegaVisualizer,
        MegaVisualizerConfig,
        create_full_visualizer,
        create_minimal_visualizer,
        create_recording_visualizer,
    )

    if viz_mode == "full":
        return create_full_visualizer()
    elif viz_mode == "minimal":
        return create_minimal_visualizer()
    elif viz_mode == "record":
        return create_recording_visualizer()
    elif viz_mode == "hud":
        config = MegaVisualizerConfig(
            scene_objects_enabled=True,
            nn_visualizer_enabled=False,
            telemetry_hud_enabled=True,
            perturbation_overlay_enabled=True,
        )
        return MegaVisualizer(config)
    elif viz_mode == "nn":
        config = MegaVisualizerConfig(
            scene_objects_enabled=False,
            nn_visualizer_enabled=True,
            telemetry_hud_enabled=False,
            perturbation_overlay_enabled=False,
        )
        return MegaVisualizer(config)
    else:
        return create_full_visualizer()


def run_visualization(args):
    """Run the main visualization loop.

    Args:
        args: Command line arguments
    """
    # Import environment
    from src.environments import YawTrackingConfig, YawTrackingEnv
    from src.utils.rotations import Rotations

    # Load model
    model = load_model(args.model)

    # Load perturbations
    perturbation_manager = load_perturbations(args.perturbations)

    # Create visualizer
    visualizer = create_visualizer(args.viz_mode)

    if model is not None:
        visualizer.set_model(model)

    if perturbation_manager is not None:
        visualizer.set_perturbation_manager(perturbation_manager)

    # Create environment config
    env_config = YawTrackingConfig(
        hover_height=args.hover_height,
        max_episode_steps=args.max_steps,
        target_speed_min=args.target_speed,
        target_speed_max=args.target_speed * 1.5,
    )

    # Create environment
    env = YawTrackingEnv(
        config=env_config,
        render_mode="rgb_array",
        perturbation_manager=perturbation_manager,
    )

    print("\nStarting visualization...")
    print(f"  Model: {args.model if args.model else 'Random policy'}")
    print(f"  Viz mode: {args.viz_mode}")
    print(f"  Recording: {args.record if args.record else 'None'}")
    print(f"  Episodes: {args.episodes}")
    print("  Press 'q' to quit, 'r' to reset, 'p' to pause")
    print()

    # Video writer
    writer = None
    if args.record and IMAGEIO_AVAILABLE:
        writer = imageio.get_writer(
            args.record,
            fps=args.fps,
            quality=8,
        )
        visualizer.set_recording(True)
        print(f"Recording to {args.record}")

    # Window name
    window_name = "NNPID Mega Visualization"
    if CV2_AVAILABLE:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    # Main loop
    episode = 0
    total_steps = 0
    paused = False
    quit_requested = False

    try:
        while episode < args.episodes and not quit_requested:
            obs, info = env.reset()
            visualizer.reset(episode)

            done = False
            step = 0
            episode_reward = 0.0

            print(f"Episode {episode + 1}/{args.episodes}")

            while not done and not quit_requested:
                # Handle pause
                if paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord("p"):
                        paused = False
                    elif key == ord("q"):
                        quit_requested = True
                    continue

                # Get action
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
                total_steps += 1

                # Get state for visualization
                state = env.get_state()
                euler = Rotations.quaternion_to_euler(state.orientation)
                roll, pitch, yaw = euler

                # Estimate motor values from action (normalized)
                motor_values = (
                    np.clip((action + 1) / 2, 0, 1) if len(action.shape) == 0 else np.ones(4) * 0.6
                )

                # Get yaw error
                yaw_error = info.get("yaw_error", 0.0)

                # Get target position for visualization (used by scene objects)
                _target_pos = np.array(
                    [
                        state.position[0] + np.cos(yaw + yaw_error) * 2,
                        state.position[1] + np.sin(yaw + yaw_error) * 2,
                        state.position[2],
                    ]
                )  # Will be used when scene objects are integrated with mjvScene

                # Render base frame
                frame = env.render()

                if frame is not None:
                    # Update scene objects (would need access to mjvScene)
                    # For now, just render 2D overlay

                    # Update telemetry
                    visualizer.update_telemetry(
                        roll=roll,
                        pitch=pitch,
                        yaw=yaw,
                        yaw_rate=state.angular_velocity[2],
                        altitude=state.position[2],
                        motor_values=motor_values if len(motor_values) == 4 else np.ones(4) * 0.6,
                    )

                    # Update NN visualization
                    visualizer.update_nn(obs, action)

                    # Step visualization
                    visualizer.step(env.dt, reward)

                    # Render overlay
                    frame_with_overlay = visualizer.render_overlay(frame)

                    # Convert for display
                    if CV2_AVAILABLE:
                        display_frame = cv2.cvtColor(frame_with_overlay, cv2.COLOR_RGB2BGR)
                        cv2.imshow(window_name, display_frame)

                        # Write to video
                        if writer is not None:
                            writer.append_data(frame_with_overlay)

                        # Handle keyboard
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            quit_requested = True
                        elif key == ord("r"):
                            done = True  # Force reset
                        elif key == ord("p"):
                            paused = True
                        elif key == ord("s"):
                            # Save screenshot
                            screenshot_path = f"screenshot_{total_steps:06d}.png"
                            cv2.imwrite(screenshot_path, display_frame)
                            print(f"Saved screenshot: {screenshot_path}")

                # Control frame rate
                if not args.fast:
                    time.sleep(max(0, 1.0 / args.fps - 0.001))

            print(f"  Steps: {step}, Reward: {episode_reward:.2f}")
            episode += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        env.close()

        if writer is not None:
            writer.close()
            print(f"Video saved to {args.record}")

        if CV2_AVAILABLE:
            cv2.destroyAllWindows()

    print(f"\nCompleted {episode} episodes, {total_steps} total steps")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run mega visualization with trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model runs/best_model.zip
  %(prog)s --model runs/best_model.zip --perturbations config/perturbations.yaml
  %(prog)s --model runs/best_model.zip --record output.mp4 --episodes 3
  %(prog)s --no-model --viz-mode minimal
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        help="Path to trained model (omit for random policy)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Run with random policy",
    )
    parser.add_argument(
        "--perturbations",
        "-p",
        help="Path to perturbations config YAML",
    )
    parser.add_argument(
        "--viz-mode",
        choices=["full", "minimal", "hud", "nn", "record"],
        default="full",
        help="Visualization mode (default: full)",
    )
    parser.add_argument(
        "--record",
        "-r",
        help="Record video to file (e.g., output.mp4)",
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Max steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for display/recording (default: 30)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Window height (default: 720)",
    )
    parser.add_argument(
        "--hover-height",
        type=float,
        default=1.0,
        help="Hover height in meters (default: 1.0)",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=0.5,
        help="Target angular speed (default: 0.5)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run as fast as possible (no frame rate limit)",
    )

    args = parser.parse_args()

    # Validate
    if args.no_model:
        args.model = None

    if args.record and not IMAGEIO_AVAILABLE:
        print("Error: imageio required for recording")
        sys.exit(1)

    # Run visualization
    run_visualization(args)


if __name__ == "__main__":
    main()
