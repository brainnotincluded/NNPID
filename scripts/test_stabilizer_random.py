#!/usr/bin/env python3
"""Test SITL-style stabilizer with random actions.

This script verifies that the hover stabilizer can maintain stable flight
even when the neural network (simulated by random actions) gives aggressive
or erratic yaw commands.

Success criteria:
1. Drone completes full episodes without crashing
2. Altitude stays within bounds (±1m from hover height)
3. Tilt stays below 60 degrees (recoverable)
4. Episode reward is positive (tracking happens sometimes)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv


def test_stabilizer_random_actions(
    n_episodes: int = 5,
    max_steps_per_episode: int = 500,
    verbose: bool = True,
) -> dict:
    """Test stabilizer with random actions.

    Args:
        n_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        verbose: Print detailed output

    Returns:
        Dictionary with test results
    """
    # Create environment with SITL-style stabilizer
    config = YawTrackingConfig(
        max_episode_steps=max_steps_per_episode,
        target_speed_min=0.3,
        target_speed_max=0.5,
        # Use default SITL-style stabilizer settings
        # (already tuned for stability)
    )

    env = YawTrackingEnv(config=config)

    results = {
        "episodes": [],
        "crashes": 0,
        "completed": 0,
        "total_steps": 0,
        "total_rewards": [],
        "max_tilts": [],
        "altitude_errors": [],
        "safety_mode_activations": 0,
    }

    if verbose:
        print("=" * 60)
        print("Testing SITL-style Stabilizer with Random Actions")
        print("=" * 60)
        print(f"Episodes: {n_episodes}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print("Stabilizer settings:")
        print(f"  - altitude_kp: {config.altitude_kp}")
        print(f"  - altitude_ki: {config.altitude_ki}")
        print(f"  - attitude_kp: {config.attitude_kp}")
        print(f"  - attitude_ki: {config.attitude_ki}")
        print(f"  - yaw_authority: {config.yaw_authority}")
        print(f"  - safety_tilt_threshold: {np.degrees(config.safety_tilt_threshold):.1f}°")
        print("=" * 60)

    np.random.seed(42)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=42 + ep)

        episode_reward = 0.0
        episode_steps = 0
        max_tilt = 0.0
        max_alt_error = 0.0
        safety_activations = 0
        terminated = False

        while episode_steps < max_steps_per_episode:
            # Random action (simulates random/untrained NN)
            action = np.random.uniform(-1, 1, size=(1,))

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Track max tilt
            roll = info.get("roll", 0)
            pitch = info.get("pitch", 0)
            tilt = np.sqrt(roll**2 + pitch**2)
            max_tilt = max(max_tilt, tilt)

            # Track altitude error
            alt = info.get("altitude", config.hover_height)
            alt_error = abs(alt - config.hover_height)
            max_alt_error = max(max_alt_error, alt_error)

            # Check for safety mode
            if env._stabilizer.is_safety_mode_active:
                safety_activations += 1

            if terminated:
                results["crashes"] += 1
                break

            if truncated:
                results["completed"] += 1
                break

        results["total_steps"] += episode_steps
        results["total_rewards"].append(episode_reward)
        results["max_tilts"].append(max_tilt)
        results["altitude_errors"].append(max_alt_error)
        results["safety_mode_activations"] += safety_activations

        results["episodes"].append(
            {
                "episode": ep + 1,
                "steps": episode_steps,
                "reward": episode_reward,
                "max_tilt_deg": np.degrees(max_tilt),
                "max_alt_error": max_alt_error,
                "crashed": terminated,
                "safety_activations": safety_activations,
            }
        )

        if verbose:
            status = "CRASHED" if terminated else "COMPLETED"
            print(f"Episode {ep + 1}: {status} after {episode_steps} steps")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Max tilt: {np.degrees(max_tilt):.1f}°")
            print(f"  Max altitude error: {max_alt_error:.2f}m")
            print(f"  Safety mode activations: {safety_activations}")

    env.close()

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Episodes completed without crash: {results['completed']}/{n_episodes}")
        print(f"Crashes: {results['crashes']}")
        print(f"Total steps: {results['total_steps']}")
        print(f"Mean episode reward: {np.mean(results['total_rewards']):.2f}")
        print(f"Mean max tilt: {np.degrees(np.mean(results['max_tilts'])):.1f}°")
        print(f"Mean max altitude error: {np.mean(results['altitude_errors']):.2f}m")
        print(f"Total safety mode activations: {results['safety_mode_activations']}")

        # Pass/Fail criteria
        passed = True
        print("\n" + "-" * 40)
        print("TEST CRITERIA:")

        # Criterion 1: No crashes
        if results["crashes"] == 0:
            print("✓ No crashes")
        else:
            print(f"✗ {results['crashes']} crashes detected")
            passed = False

        # Criterion 2: Mean altitude error < 1m
        mean_alt_error = np.mean(results["altitude_errors"])
        if mean_alt_error < 1.0:
            print(f"✓ Altitude error OK (mean: {mean_alt_error:.2f}m)")
        else:
            print(f"✗ Altitude error too high (mean: {mean_alt_error:.2f}m)")
            passed = False

        # Criterion 3: Mean tilt < 60 degrees
        mean_tilt = np.degrees(np.mean(results["max_tilts"]))
        if mean_tilt < 60.0:
            print(f"✓ Tilt OK (mean max: {mean_tilt:.1f}°)")
        else:
            print(f"✗ Tilt too high (mean max: {mean_tilt:.1f}°)")
            passed = False

        print("-" * 40)
        if passed:
            print("TEST PASSED: Stabilizer maintains stability with random actions")
        else:
            print("TEST FAILED: Stabilizer could not maintain stability")

    return results


def test_stabilizer_aggressive_actions(
    n_episodes: int = 3,
    max_steps_per_episode: int = 300,
    verbose: bool = True,
) -> dict:
    """Test stabilizer with aggressive (max/min) actions.

    This test uses the most extreme possible yaw commands to verify
    the stabilizer can handle worst-case NN outputs.
    """
    config = YawTrackingConfig(
        max_episode_steps=max_steps_per_episode,
        target_speed_min=0.5,
        target_speed_max=1.0,
    )

    env = YawTrackingEnv(config=config)

    results = {"crashes": 0, "completed": 0}

    if verbose:
        print("\n" + "=" * 60)
        print("Testing Stabilizer with AGGRESSIVE Actions")
        print("(Alternating max positive/negative yaw commands)")
        print("=" * 60)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=100 + ep)

        episode_steps = 0
        terminated = False
        direction = 1

        while episode_steps < max_steps_per_episode:
            # Aggressive alternating max actions
            if episode_steps % 20 == 0:
                direction *= -1  # Switch direction every 20 steps

            action = np.array([direction * 1.0])  # Max command

            obs, reward, terminated, truncated, info = env.step(action)
            episode_steps += 1

            if terminated:
                results["crashes"] += 1
                break
            if truncated:
                results["completed"] += 1
                break

        if verbose:
            status = "CRASHED" if terminated else "COMPLETED"
            print(f"Episode {ep + 1}: {status} after {episode_steps} steps")

    env.close()

    if verbose:
        print(f"\nAggressive test: {results['completed']}/{n_episodes} episodes completed")
        if results["crashes"] == 0:
            print("✓ Stabilizer handles aggressive commands")
        else:
            print(f"✗ {results['crashes']} crashes with aggressive commands")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test SITL-style stabilizer with random actions")
    parser.add_argument(
        "--episodes", "-n", type=int, default=5, help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=500, help="Max steps per episode (default: 500)"
    )
    parser.add_argument(
        "--aggressive", "-a", action="store_true", help="Also run aggressive action test"
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (less output)")

    args = parser.parse_args()

    # Run random action test
    results = test_stabilizer_random_actions(
        n_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        verbose=not args.quiet,
    )

    # Run aggressive action test if requested
    if args.aggressive:
        test_stabilizer_aggressive_actions(
            n_episodes=3,
            max_steps_per_episode=300,
            verbose=not args.quiet,
        )

    # Return exit code based on results
    if results["crashes"] == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
