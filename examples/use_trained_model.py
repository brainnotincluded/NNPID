#!/usr/bin/env python3
"""Example: Using TrainedYawTracker in your control loop.

This example shows how to integrate a trained yaw tracking model
into your own control system.
"""

from pathlib import Path

import numpy as np

# Import the trained tracker
from src.deployment.trained_yaw_tracker import TrainedYawTracker
from src.environments.yaw_tracking_env import YawTrackingEnv, YawTrackingConfig

# Example 1: Basic usage
def example_basic():
    """Basic usage example."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Load trained model
    model_path = Path("runs/analysis_20260126_150455/best_model")
    tracker = TrainedYawTracker.from_path(model_path)

    print(f"Loaded: {tracker}")
    print(f"Info: {tracker.get_info()}")

    # Create environment
    config = YawTrackingConfig(
        target_patterns=["circular"],
        target_speed_min=0.05,
        target_speed_max=0.1,
    )
    env = YawTrackingEnv(config=config)

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")

    # Control loop
    print("\nRunning control loop...")
    for step in range(100):
        # Get yaw rate command from trained model
        yaw_rate_cmd = tracker.predict(obs, deterministic=True)

        # Step environment (this uses the command internally)
        # In your code, you would use yaw_rate_cmd with your stabilizer
        obs, reward, terminated, truncated, info = env.step(np.array([yaw_rate_cmd]))

        if step % 20 == 0:
            yaw_error_deg = np.degrees(info["yaw_error"])
            print(f"  Step {step:3d}: yaw_cmd={yaw_rate_cmd:+.2f}, "
                  f"yaw_error={yaw_error_deg:6.1f}°, reward={reward:+.2f}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break

    env.close()
    print("\nDone!")


# Example 2: Integration with custom control loop
def example_custom_loop():
    """Example showing integration with custom control system."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Control Loop Integration")
    print("=" * 60)

    # Load model
    model_path = Path("runs/analysis_20260126_150455/best_model")
    tracker = TrainedYawTracker.from_path(model_path)

    # Your custom control loop
    print("\nYour control loop structure:")
    print("""
    # Load tracker once at startup
    tracker = TrainedYawTracker.from_path("path/to/model")

    # In your main control loop:
    while running:
        # 1. Get current state from sensors/simulator
        state = get_current_state()  # Your function
        
        # 2. Build observation vector (11 elements for YawTrackingEnv)
        obs = build_observation(state, target)  # Your function
        
        # 3. Get yaw rate command from trained model
        yaw_rate_cmd = tracker.predict(obs, deterministic=True)
        
        # 4. Use command with your stabilizer/controller
        motors = stabilizer.compute_motors(
            state, 
            yaw_rate_cmd=yaw_rate_cmd * max_yaw_rate,  # Scale to rad/s
            dt=dt
        )
        
        # 5. Apply motor commands
        apply_motors(motors)
    """)

    print("\nKey points:")
    print("  - tracker.predict() returns value in [-1, 1]")
    print("  - Scale by max_yaw_rate (e.g., 2.0 rad/s) for actual command")
    print("  - Observation must match training format (11 elements)")
    print("  - Use deterministic=True for deployment, False for exploration")


# Example 3: Observation format
def example_observation_format():
    """Show observation format required by the model."""
    print("\n" + "=" * 60)
    print("Example 3: Observation Format")
    print("=" * 60)

    config = YawTrackingConfig()
    env = YawTrackingEnv(config=config)
    obs, info = env.reset(seed=42)

    print("\nObservation vector (11 elements):")
    print("  [0] target_dir_x: X component of target direction")
    print("  [1] target_dir_y: Y component of target direction")
    print("  [2] target_angular_vel: Target angular velocity (rad/s)")
    print("  [3] current_yaw_rate: Current yaw rate (rad/s)")
    print("  [4] yaw_error: Yaw error angle (rad)")
    print("  [5] roll: Current roll angle (rad)")
    print("  [6] pitch: Current pitch angle (rad)")
    print("  [7] altitude_error: Altitude error (m)")
    print("  [8] velocity_x: X velocity (m/s)")
    print("  [9] velocity_y: Y velocity (m/s)")
    print("  [10] previous_action: Previous action value")

    print(f"\nExample observation: {obs}")
    print(f"Shape: {obs.shape}")

    env.close()


if __name__ == "__main__":
    # Run examples
    try:
        example_basic()
        example_custom_loop()
        example_observation_format()
    except FileNotFoundError as e:
        print(f"\nError: Model not found. {e}")
        print("Please update the model path in the script.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
