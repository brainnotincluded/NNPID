"""Webots Human Tracker - Track pedestrian with trained neural network.

This script integrates with Webots + ArduPilot SITL to track a moving pedestrian
using the trained yaw tracking neural network.

Usage:
    1. Start Webots with iris_camera_human.wbt
    2. Start ArduPilot SITL: sim_vehicle.py -v ArduCopter -f webots-quad
    3. Run this script: python webots_human_tracker.py --model runs/best_model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from controller import Supervisor

    WEBOTS_AVAILABLE = True
except ImportError:
    WEBOTS_AVAILABLE = False
    print("Warning: Webots Python API not available. Using socket-based tracking.")

from src.deployment.yaw_tracker_sitl import YawTrackerSITL, YawTrackerSITLConfig


class WebotsHumanTracker:
    """Track human in Webots using trained neural network.

    This class integrates Webots supervision API with our trained yaw tracker.
    It gets the human position from Webots and passes it to the neural network
    via the SITL bridge.
    """

    def __init__(
        self,
        model_path: str | Path,
        sitl_connection: str = "tcp:127.0.0.1:5760",
        use_supervisor: bool = True,
    ):
        """Initialize human tracker.

        Args:
            model_path: Path to trained model
            sitl_connection: MAVLink connection string
            use_supervisor: Use Webots Supervisor API (requires supervisor mode)
        """
        self.model_path = model_path
        self.use_supervisor = use_supervisor and WEBOTS_AVAILABLE

        # Initialize Webots supervisor if available
        self.robot = None
        self.pedestrian_node = None
        if self.use_supervisor:
            self.robot = Supervisor()
            self.timestep = int(self.robot.getBasicTimeStep())

            # Get pedestrian node (first Pedestrian in scene)
            self.pedestrian_node = self.robot.getFromDef("PEDESTRIAN")
            if self.pedestrian_node is None:
                # Try to find by type
                root = self.robot.getRoot()
                children_field = root.getField("children")
                for i in range(children_field.getCount()):
                    node = children_field.getMFNode(i)
                    if node.getTypeName() == "Pedestrian":
                        self.pedestrian_node = node
                        break

            if self.pedestrian_node:
                print(f"✅ Found pedestrian node: {self.pedestrian_node.getTypeName()}")
            else:
                print("⚠️  Warning: Could not find pedestrian node")

        # Initialize SITL tracker
        config = YawTrackerSITLConfig(
            connection_string=sitl_connection,
            target_angular_velocity=0.3,  # Human walks ~0.3 rad/s
            target_radius=10.0,  # Track up to 10m away
            max_yaw_rate=1.5,  # Smooth tracking
        )

        self.tracker = YawTrackerSITL(model_path=model_path, config=config)

        print("🚁 Initialized human tracker")
        print(f"   Model: {model_path}")
        print(f"   SITL: {sitl_connection}")
        print(f"   Webots Supervisor: {'Yes' if self.use_supervisor else 'No'}")

    def get_human_position(self) -> np.ndarray | None:
        """Get human position from Webots.

        Returns:
            [x, y, z] position in world frame, or None if not available
        """
        if self.use_supervisor and self.pedestrian_node:
            try:
                translation_field = self.pedestrian_node.getField("translation")
                if translation_field:
                    pos = translation_field.getSFVec3f()
                    return np.array(pos, dtype=np.float32)
            except Exception as e:
                print(f"⚠️  Error getting pedestrian position: {e}")

        return None

    def update_target_from_human(self):
        """Update tracker target position from human position."""
        human_pos = self.get_human_position()

        if human_pos is not None:
            # Update tracker's target position
            # Convert Webots (ENU) to ArduPilot (NED) if needed
            # For now, assume compatible coordinate systems
            self.tracker._target.position = human_pos

            # Estimate human velocity (simple finite difference)
            # In production, use Webots velocity field or better estimation
            dt = 0.02  # 50Hz
            if hasattr(self, '_prev_human_pos') and hasattr(self, '_prev_time'):
                dt = time.time() - self._prev_time
                velocity = (human_pos - self._prev_human_pos) / dt
                self.tracker._target.velocity = velocity

            self._prev_human_pos = human_pos.copy()
            self._prev_time = time.time()

            return True

        return False

    def run(self, duration: float = 300.0):
        """Run human tracking.

        Args:
            duration: Duration in seconds (default: 5 minutes)
        """
        print("\n" + "="*60)
        print("  WEBOTS HUMAN TRACKING")
        print("="*60)

        # Connect to SITL
        print("\n📡 Connecting to ArduPilot SITL...")
        if not self.tracker.connect():
            print("❌ Failed to connect to SITL")
            return

        print("✅ Connected to SITL")

        # Arm and takeoff
        print("\n🚁 Arming and taking off...")
        if not self.tracker.arm_and_takeoff(altitude=3.0):
            print("❌ Failed to arm/takeoff")
            self.tracker.disconnect()
            return

        print("✅ Drone is airborne")

        # Wait for stabilization
        print("\n⏳ Stabilizing...")
        time.sleep(3)

        # Main tracking loop
        print("\n🎯 Starting human tracking...")
        print("   Press Ctrl+C to stop\n")

        start_time = time.time()
        loop_count = 0

        try:
            while time.time() - start_time < duration:
                loop_start = time.time()

                # Step Webots simulation if using supervisor
                if self.use_supervisor and self.robot and self.robot.step(self.timestep) == -1:
                    print("\n⚠️  Webots simulation ended")
                    break

                # Update target from human position
                if not self.update_target_from_human():
                    # Fallback: use circular pattern if can't get human
                    dt = 1.0 / self.tracker.config.control_rate
                    self.tracker.update_target(dt)

                # Run tracker step
                yaw_error, action = self.tracker.step(1.0 / 50.0)

                # Print status every 2 seconds
                if loop_count % 100 == 0:
                    state = self.tracker.get_state()
                    human_pos = self.get_human_position()

                    print(
                        f"[{time.time() - start_time:6.1f}s] "
                        f"Human: {human_pos if human_pos is not None else 'N/A':} | "
                        f"Yaw error: {np.degrees(yaw_error):+6.1f}° | "
                        f"Action: {action:+.2f} | "
                        f"Mode: {state['mode']}"
                    )

                loop_count += 1

                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / 50.0) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user")

        # Shutdown
        print("\n🛬 Landing and shutting down...")
        self.tracker.shutdown()

        # Print final metrics
        print("\n" + "="*60)
        print("  TRACKING RESULTS")
        print("="*60)

        errors = self.tracker._metrics.get("yaw_errors", [])
        if errors:
            print(f"  Mean yaw error:  {np.degrees(np.mean(np.abs(errors))):6.1f}°")
            print(f"  Max yaw error:   {np.degrees(np.max(np.abs(errors))):6.1f}°")
            print(f"  Tracking %:      {100 * np.mean([1 if abs(e) < 0.1 else 0 for e in errors]):6.1f}%")

        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Track human in Webots using trained neural network"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (e.g., runs/best_model)"
    )
    parser.add_argument(
        "--connection",
        type=str,
        default="tcp:127.0.0.1:5760",
        help="MAVLink connection string (default: tcp:127.0.0.1:5760)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Tracking duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Don't use Webots Supervisor API (use fallback tracking)"
    )

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists() and not (model_path.parent / f"{model_path.name}.zip").exists():
        print(f"❌ Error: Model not found at {model_path}")
        print(f"   Searched: {model_path}")
        print(f"   Searched: {model_path.parent / f'{model_path.name}.zip'}")
        sys.exit(1)

    # Create and run tracker
    tracker = WebotsHumanTracker(
        model_path=args.model,
        sitl_connection=args.connection,
        use_supervisor=not args.no_supervisor,
    )

    tracker.run(duration=args.duration)


if __name__ == "__main__":
    main()
