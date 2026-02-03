"""Webots Human Tracker - Track pedestrian with trained neural network.

This script connects to ArduPilot SITL and receives human position from
the Webots controller to track a moving pedestrian using the trained
yaw tracking neural network.

Architecture:
    Webots (drone controller) --UDP:9100--> This script --MAVLink--> SITL
                                 (human pos)              (yaw cmd)

Usage:
    1. Start Webots with iris_camera_human.wbt
    2. Start ArduPilot SITL: scripts/shell/run_sitl.sh (or sim_vehicle.py -v ArduCopter -f JSON)
    3. Run this script: python src/webots/webots_human_tracker.py --model runs/<run_name>/best_model
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.deployment.yaw_tracker_sitl import YawTrackerSITL, YawTrackerSITLConfig


class TrackingDataReceiver:
    """Receive tracking data (drone + human position) from Webots controller via UDP."""

    def __init__(self, port: int = 9100, host: str = "127.0.0.1"):
        """Initialize tracking data receiver.

        Args:
            port: UDP port to listen on (must match Webots controller)
            host: Host to bind to (default: 127.0.0.1 for local only, use 0.0.0.0 for remote)
        """
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.setblocking(False)

        self._latest_data = None
        self._data_lock = threading.Lock()
        self._running = True

        # Start receiver thread
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

        print(f"📡 Listening for tracking data on UDP port {port}")

    def _receive_loop(self):
        """Background loop to receive tracking data."""
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
                parsed = json.loads(data.decode("utf-8"))
                with self._data_lock:
                    self._latest_data = parsed
            except OSError:
                time.sleep(0.001)  # No data available
            except json.JSONDecodeError:
                pass

    def get_latest(self) -> dict | None:
        """Get latest tracking data.

        Returns:
            Dictionary with drone and human data, or None if no data received yet
        """
        with self._data_lock:
            return self._latest_data.copy() if self._latest_data else None

    def get_human_position(self) -> np.ndarray | None:
        """Get latest human position in NED frame.

        Returns:
            [north, east, down] position or None if not available
        """
        data = self.get_latest()
        if data and data.get("human") and data["human"].get("position"):
            return np.array(data["human"]["position"], dtype=np.float32)
        return None

    def close(self):
        """Close receiver."""
        self._running = False
        self._sock.close()


class WebotsHumanTracker:
    """Track human in Webots using trained neural network.

    This class receives human position from the Webots controller via UDP
    and controls the drone's yaw via MAVLink to ArduPilot SITL.
    """

    def __init__(
        self,
        model_path: str | Path,
        sitl_connection: str = "tcp:127.0.0.1:5760",
        tracking_port: int = 9100,
        max_yaw_rate: float = 1.5,
    ):
        """Initialize human tracker.

        Args:
            model_path: Path to trained model
            sitl_connection: MAVLink connection string
            tracking_port: UDP port to receive tracking data from Webots
            max_yaw_rate: Maximum yaw rate in rad/s
        """
        self.model_path = model_path
        self.tracking_port = tracking_port

        # Initialize tracking data receiver
        self.receiver = TrackingDataReceiver(port=tracking_port)

        # Initialize SITL tracker
        config = YawTrackerSITLConfig(
            connection_string=sitl_connection,
            target_angular_velocity=0.5,  # Will be updated from human velocity
            target_radius=15.0,  # Track up to 15m away
            max_yaw_rate=max_yaw_rate,
        )

        self.tracker = YawTrackerSITL(model_path=model_path, config=config)

        # State tracking
        self._prev_human_pos = None
        self._prev_time = None

        print("🚁 Initialized human tracker")
        print(f"   Model: {model_path}")
        print(f"   SITL: {sitl_connection}")
        print(f"   Tracking port: {tracking_port}")

    def update_target_from_human(self) -> bool:
        """Update tracker target position from human position.

        Returns:
            True if human position was successfully updated
        """
        human_pos = self.receiver.get_human_position()

        if human_pos is not None:
            # Update tracker's target position
            self.tracker._target.position = human_pos

            # Estimate human angular velocity relative to drone
            now = time.time()
            if self._prev_human_pos is not None and self._prev_time is not None:
                dt = now - self._prev_time
                if dt > 0.001:
                    velocity = (human_pos - self._prev_human_pos) / dt
                    self.tracker._target.velocity = velocity

                    # Estimate angular velocity based on velocity magnitude
                    # and distance from drone
                    drone_state = self.tracker.get_state()
                    to_human = human_pos - drone_state["position"]
                    distance = np.linalg.norm(to_human[:2])
                    if distance > 0.1:
                        # Angular velocity = tangential velocity / distance
                        tangent_vel = np.linalg.norm(velocity[:2])
                        self.tracker._target.angular_velocity = tangent_vel / distance

            self._prev_human_pos = human_pos.copy()
            self._prev_time = now
            return True

        return False

    def run(self, duration: float = 300.0, altitude: float = 3.0):
        """Run human tracking.

        Args:
            duration: Duration in seconds (default: 5 minutes)
            altitude: Takeoff altitude in meters
        """
        print("\n" + "=" * 60)
        print("  WEBOTS HUMAN TRACKING")
        print("=" * 60)

        # Wait for tracking data
        print("\n⏳ Waiting for tracking data from Webots...")
        timeout = 30
        start_wait = time.time()
        while time.time() - start_wait < timeout:
            data = self.receiver.get_latest()
            if data:
                print("✅ Receiving tracking data from Webots")
                break
            time.sleep(0.5)
        else:
            print("⚠️  No tracking data received - using fallback circular target")

        # Connect to SITL
        print("\n📡 Connecting to ArduPilot SITL...")
        if not self.tracker.connect():
            print("❌ Failed to connect to SITL")
            print("   Make sure SITL is running: scripts/shell/run_sitl.sh")
            return

        print("✅ Connected to SITL")

        # Wait for state
        time.sleep(1)
        state = self.tracker.get_state()
        print(f"   Mode: {state['mode']}, Armed: {state['armed']}")

        # Arm and takeoff
        print(f"\n🚁 Arming and taking off to {altitude}m...")
        if not self.tracker.arm_and_takeoff(altitude=altitude):
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
        errors_sum = 0.0
        on_target_count = 0

        try:
            while time.time() - start_time < duration:
                loop_start = time.time()

                # Update target from human position
                has_human = self.update_target_from_human()
                if not has_human:
                    # Fallback: use circular pattern
                    dt = 1.0 / self.tracker.config.control_rate
                    self.tracker.update_target(dt)

                # Run tracker step
                yaw_error, action = self.tracker.step(1.0 / 50.0)
                errors_sum += abs(yaw_error)
                if abs(yaw_error) < 0.1:  # ~6 degrees
                    on_target_count += 1

                # Print status every 2 seconds
                if loop_count % 100 == 0:
                    state = self.tracker.get_state()
                    human_pos = self.receiver.get_human_position()

                    human_str = (
                        f"[{human_pos[0]:5.1f}, {human_pos[1]:5.1f}]"
                        if human_pos is not None
                        else "N/A"
                    )
                    tracking_pct = 100 * on_target_count / max(loop_count, 1)

                    print(
                        f"[{time.time() - start_time:6.1f}s] "
                        f"Human: {human_str} | "
                        f"Error: {np.degrees(yaw_error):+6.1f}° | "
                        f"Action: {action:+.2f} | "
                        f"Track: {tracking_pct:4.1f}%"
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
        self.receiver.close()

        # Print final metrics
        print("\n" + "=" * 60)
        print("  TRACKING RESULTS")
        print("=" * 60)

        if loop_count > 0:
            mean_error = errors_sum / loop_count
            tracking_pct = 100 * on_target_count / loop_count
            print(f"  Total steps:     {loop_count}")
            print(f"  Mean yaw error:  {np.degrees(mean_error):6.1f}°")
            print(f"  Tracking %:      {tracking_pct:6.1f}%")

        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Track human in Webots using trained neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (e.g., runs/analysis_20260126_150455/best_model)",
    )
    parser.add_argument(
        "--connection", type=str, default="tcp:127.0.0.1:5760", help="MAVLink connection string"
    )
    parser.add_argument(
        "--tracking-port",
        type=int,
        default=9100,
        help="UDP port to receive tracking data from Webots",
    )
    parser.add_argument("--altitude", type=float, default=3.0, help="Takeoff altitude in meters")
    parser.add_argument(
        "--duration", type=float, default=300.0, help="Tracking duration in seconds"
    )
    parser.add_argument("--max-yaw-rate", type=float, default=1.5, help="Maximum yaw rate in rad/s")

    args = parser.parse_args()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try with best_model subdirectory
        if (model_path / "best_model").exists():
            pass  # Will be resolved in tracker
        elif not model_path.with_suffix(".zip").exists():
            print(f"❌ Error: Model not found at {model_path}")
            print("\nAvailable models:")
            runs_dir = Path("runs")
            if runs_dir.exists():
                for d in sorted(runs_dir.iterdir()):
                    if (d / "best_model").exists() or (d / "vec_normalize.pkl").exists():
                        print(f"   {d}")
            sys.exit(1)

    # Create and run tracker
    tracker = WebotsHumanTracker(
        model_path=args.model,
        sitl_connection=args.connection,
        tracking_port=args.tracking_port,
        max_yaw_rate=args.max_yaw_rate,
    )

    tracker.run(duration=args.duration, altitude=args.altitude)


if __name__ == "__main__":
    main()
