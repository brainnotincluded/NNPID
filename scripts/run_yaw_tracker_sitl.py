#!/usr/bin/env python3
"""Run trained yaw tracker on ArduPilot SITL.

This script loads a trained yaw tracking neural network and deploys it
to ArduPilot SITL, controlling the drone's yaw rate to track a moving target.

Prerequisites:
    1. ArduPilot SITL running (e.g., ./build/sitl/bin/arducopter --model JSON)
    2. MuJoCo simulator bridge (scripts/ardupilot_mujoco_final.py) or other sim
    3. Trained model from scripts/train_yaw_tracker.py

Usage:
    # With trained model
    python scripts/run_yaw_tracker_sitl.py --model runs/yaw_tracking/best_model
    
    # Test without model (uses simple P controller)
    python scripts/run_yaw_tracker_sitl.py --no-model --duration 30
    
    # Custom settings
    python scripts/run_yaw_tracker_sitl.py \\
        --model runs/yaw_tracking/best_model \\
        --connection "tcp:127.0.0.1:5760" \\
        --altitude 3.0 \\
        --target-speed 1.0 \\
        --duration 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.yaw_tracker_sitl import (
    YawTrackerSITL,
    YawTrackerSITLConfig,
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run yaw tracker on ArduPilot SITL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to trained model",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Run without NN model (uses P controller)",
    )
    
    # Connection
    parser.add_argument(
        "--connection",
        type=str,
        default="tcp:127.0.0.1:5760",
        help="MAVLink connection string",
    )
    
    # Flight settings
    parser.add_argument(
        "--altitude",
        type=float,
        default=2.0,
        help="Takeoff altitude (meters)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration (seconds)",
    )
    
    # Target settings
    parser.add_argument(
        "--target-speed",
        type=float,
        default=0.5,
        help="Target angular velocity (rad/s)",
    )
    parser.add_argument(
        "--target-radius",
        type=float,
        default=5.0,
        help="Target orbit radius (meters)",
    )
    parser.add_argument(
        "--target-pattern",
        type=str,
        default="circular",
        choices=["circular", "sinusoidal"],
        help="Target motion pattern",
    )
    
    # Control settings
    parser.add_argument(
        "--control-rate",
        type=float,
        default=50.0,
        help="Control loop rate (Hz)",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=2.0,
        help="Maximum yaw rate (rad/s)",
    )
    
    # Output
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        default=None,
        help="Save metrics to file",
    )
    
    args = parser.parse_args()
    
    # Validate model
    model_path = None
    if not args.no_model:
        if args.model is None:
            print("Error: --model required unless --no-model is specified")
            print("Train a model first with: python scripts/train_yaw_tracker.py")
            sys.exit(1)
        model_path = args.model
    
    # Create config
    config = YawTrackerSITLConfig(
        connection_string=args.connection,
        control_rate=args.control_rate,
        max_yaw_rate=args.max_yaw_rate,
        target_radius=args.target_radius,
        target_angular_velocity=args.target_speed,
        target_pattern=args.target_pattern,
    )
    
    # Print settings
    print("=" * 60)
    print("  Yaw Tracker SITL Deployment")
    print("=" * 60)
    print(f"Model: {model_path or 'None (P controller)'}")
    print(f"Connection: {args.connection}")
    print(f"Target pattern: {args.target_pattern}")
    print(f"Target speed: {args.target_speed} rad/s")
    print(f"Duration: {args.duration}s")
    print()
    
    # Create tracker
    tracker = YawTrackerSITL(model_path=model_path, config=config)
    
    try:
        # Connect
        print("Connecting to SITL...")
        if not tracker.connect():
            print("Failed to connect")
            sys.exit(1)
        
        # Wait for state
        time.sleep(1)
        state = tracker.get_state()
        print(f"Current mode: {state['mode']}")
        print(f"Armed: {state['armed']}")
        
        # Arm and takeoff
        if not tracker.arm_and_takeoff(altitude=args.altitude):
            print("Failed to arm and takeoff")
            tracker.shutdown()
            sys.exit(1)
        
        # Wait for stabilization
        print("Stabilizing...")
        time.sleep(3)
        
        # Run tracking
        metrics = tracker.run_loop(
            duration=args.duration,
            verbose=not args.quiet,
        )
        
        # Save metrics
        if args.save_metrics:
            import json
            
            # Convert numpy arrays to lists for JSON
            metrics_json = {
                k: (v.tolist() if hasattr(v, 'tolist') else v)
                for k, v in metrics.items()
            }
            
            with open(args.save_metrics, 'w') as f:
                json.dump(metrics_json, f, indent=2)
            print(f"\nSaved metrics to {args.save_metrics}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        tracker.shutdown()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
