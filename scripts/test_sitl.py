#!/usr/bin/env python3
"""Interactive SITL flight test script.

This script provides manual tests for verifying SITL integration:
- Takeoff and hover
- Landing
- Waypoint navigation
- Offboard velocity control
- Emergency stop

Usage:
    # Start PX4 SITL first:
    # Terminal 1: make px4_sitl none_iris
    
    # Then run this script:
    python scripts/test_sitl.py [test_name]
    
    # Available tests:
    python scripts/test_sitl.py takeoff      # Test takeoff and hover
    python scripts/test_sitl.py land         # Test landing sequence
    python scripts/test_sitl.py waypoints    # Test waypoint navigation
    python scripts/test_sitl.py offboard     # Test offboard velocity control
    python scripts/test_sitl.py square       # Fly a square pattern
    python scripts/test_sitl.py emergency    # Test emergency stop
    python scripts/test_sitl.py all          # Run all tests sequentially
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mujoco_sim import create_simulator
from src.core.sensors import SensorSimulator, SensorConfig
from src.communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig
from src.communication.messages import HILSensorMessage, HILGPSMessage, SetpointCommand
from src.utils.transforms import CoordinateTransforms
from src.utils.rotations import Rotations


# Global state for signal handling
running = True
test_passed = False


def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    global running
    print("\n[CTRL+C] Stopping test...")
    running = False


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str


class SITLTester:
    """Interactive SITL test runner."""
    
    def __init__(
        self,
        model: str = "generic",
        host: str = "127.0.0.1",
        port: int = 4560,
    ):
        """Initialize tester.
        
        Args:
            model: MuJoCo model name
            host: MAVLink server host
            port: MAVLink server port
        """
        self.model = model
        self.host = host
        self.port = port
        
        # Components
        self.sim = None
        self.sensors = None
        self.bridge = None
        
        # State
        self.armed = False
        self.in_offboard = False
        
        # Test results
        self.results: List[TestResult] = []
    
    def setup(self) -> bool:
        """Set up simulation and connection.
        
        Returns:
            True if setup successful
        """
        print("\n" + "="*60)
        print("SITL Test Setup")
        print("="*60)
        
        # Create simulator
        print(f"\n[1/3] Loading MuJoCo model: {self.model}")
        self.sim = create_simulator(model=self.model)
        self.sensors = SensorSimulator(SensorConfig())
        print(f"      Timestep: {self.sim.timestep*1000:.1f}ms")
        
        # Create MAVLink bridge
        print(f"\n[2/3] Starting MAVLink server on {self.host}:{self.port}")
        config = MAVLinkConfig(host=self.host, port=self.port, lockstep=True)
        self.bridge = MAVLinkBridge(config)
        
        print("\n[3/3] Waiting for PX4 SITL connection...")
        print("      Start PX4 with: make px4_sitl none_iris")
        
        if not self.bridge.start_server():
            print("\n[ERROR] Failed to connect to PX4 SITL")
            return False
        
        print("\n[OK] Connected to PX4 SITL!")
        
        # Reset simulation
        self.sim.reset(position=np.array([0.0, 0.0, 0.1]))
        self.sensors.reset()
        
        # Send initial sensor data
        for _ in range(50):
            self._send_sensors()
            time.sleep(0.02)
        
        return True
    
    def teardown(self) -> None:
        """Clean up resources."""
        print("\n[CLEANUP] Shutting down...")
        
        if self.armed:
            self._disarm(force=True)
        
        if self.bridge is not None:
            self.bridge.stop()
        
        print("[OK] Cleanup complete")
    
    def _send_sensors(self) -> None:
        """Send sensor data to PX4."""
        if self.bridge is None or self.sim is None:
            return
        
        state = self.sim.get_state()
        timestamp = self.sim.get_time()
        
        # Get IMU data
        gyro, accel = self.sim.get_imu_data()
        
        # Convert to FRD/NED frames
        gyro_frd = CoordinateTransforms.angular_velocity_mujoco_to_frd(gyro)
        accel_frd = CoordinateTransforms.acceleration_mujoco_to_frd(accel)
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(state.velocity)
        
        # Add sensor noise
        imu_data = self.sensors.get_imu(gyro_frd, accel_frd, timestamp)
        
        # Send HIL_SENSOR
        sensor_msg = HILSensorMessage.from_sensor_data(
            time_sec=timestamp,
            gyro=imu_data.gyro,
            accel=imu_data.accel,
            mag=np.array([0.21, 0.0, 0.42]),
            pressure=101325.0,
            temperature=20.0,
            altitude=-pos_ned[2],
        )
        self.bridge.send_hil_sensor(sensor_msg)
        
        # Send GPS at lower rate
        if int(timestamp * 10) % 1 == 0:  # 10 Hz
            gps_data = self.sensors.get_gps(pos_ned, vel_ned, timestamp)
            gps_msg = HILGPSMessage.from_gps_data(
                time_sec=timestamp,
                lat=gps_data.latitude,
                lon=gps_data.longitude,
                alt=gps_data.altitude,
                vel_ned=np.array([
                    gps_data.velocity_north,
                    gps_data.velocity_east,
                    gps_data.velocity_down,
                ]),
            )
            self.bridge.send_hil_gps(gps_msg)
        
        # Heartbeat
        self.bridge.send_heartbeat()
    
    def _step_sim(self, duration: float = 0.02) -> None:
        """Step simulation for given duration.
        
        Args:
            duration: Time to step (seconds)
        """
        if self.sim is None or self.bridge is None:
            return
        
        steps = int(duration / self.sim.timestep)
        
        for _ in range(steps):
            # Get motor commands from PX4
            motors = self.bridge.get_motor_commands(timeout=0.001)
            if motors is None:
                motors = np.zeros(4)
            
            # Step physics
            self.sim.step(motors)
            
            # Send sensors
            self._send_sensors()
    
    def _arm(self) -> bool:
        """Arm the vehicle.
        
        Returns:
            True if armed successfully
        """
        print("  Arming vehicle...")
        
        if self.bridge is None:
            return False
        
        if self.bridge.arm():
            time.sleep(0.5)
            self.armed = True
            print("  [OK] Armed")
            return True
        else:
            print("  [FAIL] Arming failed")
            return False
    
    def _disarm(self, force: bool = False) -> bool:
        """Disarm the vehicle.
        
        Args:
            force: Force disarm
            
        Returns:
            True if disarmed
        """
        print("  Disarming...")
        
        if self.bridge is None:
            return False
        
        if self.bridge.disarm(force=force):
            self.armed = False
            print("  [OK] Disarmed")
            return True
        return False
    
    def _enter_offboard(self) -> bool:
        """Enter offboard mode.
        
        Returns:
            True if successful
        """
        print("  Entering offboard mode...")
        
        if self.bridge is None:
            return False
        
        # Send setpoints before switching mode
        pos = np.array([0.0, 0.0, -1.0])  # 1m up in NED
        for _ in range(50):
            self.bridge.send_position_setpoint(pos)
            self._step_sim(0.02)
        
        if self.bridge.set_offboard_mode():
            self.in_offboard = True
            print("  [OK] Offboard mode active")
            return True
        else:
            print("  [FAIL] Failed to enter offboard mode")
            return False
    
    def _wait_for_altitude(
        self,
        target_alt: float,
        tolerance: float = 0.2,
        timeout: float = 10.0,
    ) -> bool:
        """Wait for drone to reach altitude.
        
        Args:
            target_alt: Target altitude (meters, positive up)
            tolerance: Acceptable error (meters)
            timeout: Timeout (seconds)
            
        Returns:
            True if altitude reached
        """
        start = time.time()
        
        while running and (time.time() - start) < timeout:
            self._step_sim(0.02)
            
            state = self.sim.get_state()
            alt = state.position[2]
            error = abs(alt - target_alt)
            
            print(f"\r  Altitude: {alt:.2f}m (target: {target_alt:.2f}m, error: {error:.2f}m)   ", end="")
            
            if error < tolerance:
                print(f"\n  [OK] Reached altitude {alt:.2f}m")
                return True
        
        print(f"\n  [TIMEOUT] Failed to reach altitude")
        return False
    
    def _fly_to_position(
        self,
        position: np.ndarray,
        tolerance: float = 0.3,
        timeout: float = 15.0,
    ) -> bool:
        """Fly to position using offboard mode.
        
        Args:
            position: Target position [x, y, z] in MuJoCo frame
            tolerance: Acceptable error (meters)
            timeout: Timeout (seconds)
            
        Returns:
            True if position reached
        """
        if self.bridge is None:
            return False
        
        # Convert to NED
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(position)
        
        start = time.time()
        
        while running and (time.time() - start) < timeout:
            # Send setpoint
            self.bridge.send_position_setpoint(pos_ned)
            
            self._step_sim(0.02)
            
            state = self.sim.get_state()
            error = np.linalg.norm(state.position - position)
            
            print(f"\r  Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}] "
                  f"(error: {error:.2f}m)   ", end="")
            
            if error < tolerance:
                print(f"\n  [OK] Reached position")
                return True
        
        print(f"\n  [TIMEOUT] Failed to reach position")
        return False
    
    def _fly_velocity(
        self,
        velocity: np.ndarray,
        duration: float,
    ) -> None:
        """Fly with velocity setpoint.
        
        Args:
            velocity: Velocity [vx, vy, vz] in MuJoCo frame (m/s)
            duration: How long to fly (seconds)
        """
        if self.bridge is None:
            return
        
        # Convert to NED
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(velocity)
        
        start = time.time()
        
        while running and (time.time() - start) < duration:
            self.bridge.send_velocity_setpoint(vel_ned)
            self._step_sim(0.02)
            
            state = self.sim.get_state()
            print(f"\r  Velocity cmd: [{velocity[0]:.1f}, {velocity[1]:.1f}, {velocity[2]:.1f}] m/s | "
                  f"Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]   ", end="")
        
        print()
    
    # =========================================================================
    # Test Cases
    # =========================================================================
    
    def test_takeoff(self) -> TestResult:
        """Test takeoff and hover."""
        print("\n" + "-"*60)
        print("TEST: Takeoff and Hover")
        print("-"*60)
        
        start = time.time()
        
        try:
            # Enter offboard mode
            if not self._enter_offboard():
                return TestResult("takeoff", False, time.time()-start, "Failed to enter offboard mode")
            
            # Arm
            if not self._arm():
                return TestResult("takeoff", False, time.time()-start, "Failed to arm")
            
            # Takeoff to 2m
            print("\n  Taking off to 2m...")
            target_pos = np.array([0.0, 0.0, 2.0])
            pos_ned = CoordinateTransforms.position_mujoco_to_ned(target_pos)
            
            for _ in range(100):
                self.bridge.send_position_setpoint(pos_ned)
                self._step_sim(0.02)
            
            # Wait for altitude
            if not self._wait_for_altitude(2.0, tolerance=0.3, timeout=10.0):
                return TestResult("takeoff", False, time.time()-start, "Failed to reach altitude")
            
            # Hover for 3 seconds
            print("\n  Hovering for 3 seconds...")
            hover_start = time.time()
            while running and (time.time() - hover_start) < 3.0:
                self.bridge.send_position_setpoint(pos_ned)
                self._step_sim(0.02)
                
                state = self.sim.get_state()
                print(f"\r  Hover position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]   ", end="")
            
            print("\n  [OK] Hover complete")
            
            return TestResult("takeoff", True, time.time()-start, "Takeoff and hover successful")
            
        except Exception as e:
            return TestResult("takeoff", False, time.time()-start, f"Error: {e}")
    
    def test_land(self) -> TestResult:
        """Test landing sequence."""
        print("\n" + "-"*60)
        print("TEST: Landing")
        print("-"*60)
        
        start = time.time()
        
        try:
            # First takeoff if not already airborne
            state = self.sim.get_state()
            if state.position[2] < 1.0:
                print("  Taking off first...")
                if not self._enter_offboard():
                    return TestResult("land", False, time.time()-start, "Failed to enter offboard")
                if not self._arm():
                    return TestResult("land", False, time.time()-start, "Failed to arm")
                
                self._fly_to_position(np.array([0.0, 0.0, 2.0]))
            
            # Land command
            print("\n  Sending land command...")
            if self.bridge.land():
                print("  [OK] Land command sent")
            
            # Wait for landing
            print("  Descending...")
            if self._wait_for_altitude(0.1, tolerance=0.1, timeout=15.0):
                # Disarm
                time.sleep(1.0)
                self._disarm()
                return TestResult("land", True, time.time()-start, "Landing successful")
            else:
                return TestResult("land", False, time.time()-start, "Landing timeout")
            
        except Exception as e:
            return TestResult("land", False, time.time()-start, f"Error: {e}")
    
    def test_waypoints(self) -> TestResult:
        """Test waypoint navigation."""
        print("\n" + "-"*60)
        print("TEST: Waypoint Navigation")
        print("-"*60)
        
        start = time.time()
        waypoints = [
            np.array([0.0, 0.0, 2.0]),   # Start position
            np.array([2.0, 0.0, 2.0]),   # Forward
            np.array([2.0, 2.0, 2.0]),   # Right
            np.array([0.0, 2.0, 2.0]),   # Back
            np.array([0.0, 0.0, 2.0]),   # Return
        ]
        
        try:
            # Setup
            if not self.in_offboard:
                if not self._enter_offboard():
                    return TestResult("waypoints", False, time.time()-start, "Failed to enter offboard")
            if not self.armed:
                if not self._arm():
                    return TestResult("waypoints", False, time.time()-start, "Failed to arm")
            
            # Fly waypoints
            for i, wp in enumerate(waypoints):
                print(f"\n  Waypoint {i+1}/{len(waypoints)}: [{wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}]")
                
                if not self._fly_to_position(wp, tolerance=0.4, timeout=10.0):
                    return TestResult("waypoints", False, time.time()-start, f"Failed to reach waypoint {i+1}")
                
                # Hold briefly
                time.sleep(0.5)
            
            print("\n  [OK] All waypoints reached!")
            return TestResult("waypoints", True, time.time()-start, "Waypoint navigation successful")
            
        except Exception as e:
            return TestResult("waypoints", False, time.time()-start, f"Error: {e}")
    
    def test_offboard_velocity(self) -> TestResult:
        """Test offboard velocity control."""
        print("\n" + "-"*60)
        print("TEST: Offboard Velocity Control")
        print("-"*60)
        
        start = time.time()
        
        try:
            # Setup
            if not self.in_offboard:
                if not self._enter_offboard():
                    return TestResult("offboard", False, time.time()-start, "Failed to enter offboard")
            if not self.armed:
                if not self._arm():
                    return TestResult("offboard", False, time.time()-start, "Failed to arm")
            
            # First hover
            print("\n  Hovering at start position...")
            self._fly_to_position(np.array([0.0, 0.0, 2.0]))
            
            # Test velocity commands
            velocities = [
                (np.array([1.0, 0.0, 0.0]), 2.0, "Forward"),
                (np.array([0.0, 1.0, 0.0]), 2.0, "Right"),
                (np.array([-1.0, 0.0, 0.0]), 2.0, "Backward"),
                (np.array([0.0, -1.0, 0.0]), 2.0, "Left"),
                (np.array([0.0, 0.0, 0.5]), 2.0, "Up"),
                (np.array([0.0, 0.0, -0.5]), 2.0, "Down"),
            ]
            
            for vel, duration, name in velocities:
                print(f"\n  Velocity test: {name} ({vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f}) m/s for {duration}s")
                self._fly_velocity(vel, duration)
            
            # Stop
            print("\n  Stopping...")
            self._fly_velocity(np.zeros(3), 1.0)
            
            return TestResult("offboard", True, time.time()-start, "Velocity control successful")
            
        except Exception as e:
            return TestResult("offboard", False, time.time()-start, f"Error: {e}")
    
    def test_square_pattern(self) -> TestResult:
        """Test flying a square pattern."""
        print("\n" + "-"*60)
        print("TEST: Square Pattern")
        print("-"*60)
        
        start = time.time()
        size = 3.0  # 3m square
        altitude = 2.0
        
        corners = [
            np.array([0.0, 0.0, altitude]),
            np.array([size, 0.0, altitude]),
            np.array([size, size, altitude]),
            np.array([0.0, size, altitude]),
            np.array([0.0, 0.0, altitude]),
        ]
        
        try:
            # Setup
            if not self.in_offboard:
                if not self._enter_offboard():
                    return TestResult("square", False, time.time()-start, "Failed to enter offboard")
            if not self.armed:
                if not self._arm():
                    return TestResult("square", False, time.time()-start, "Failed to arm")
            
            print(f"\n  Flying {size}m square at {altitude}m altitude")
            
            for i, corner in enumerate(corners):
                print(f"\n  Corner {i+1}/{len(corners)}")
                if not self._fly_to_position(corner, tolerance=0.3):
                    return TestResult("square", False, time.time()-start, f"Failed at corner {i+1}")
            
            return TestResult("square", True, time.time()-start, "Square pattern complete")
            
        except Exception as e:
            return TestResult("square", False, time.time()-start, f"Error: {e}")
    
    def test_emergency_stop(self) -> TestResult:
        """Test emergency stop (immediate disarm)."""
        print("\n" + "-"*60)
        print("TEST: Emergency Stop")
        print("-"*60)
        
        start = time.time()
        
        try:
            # Setup and takeoff
            if not self.in_offboard:
                if not self._enter_offboard():
                    return TestResult("emergency", False, time.time()-start, "Failed to enter offboard")
            if not self.armed:
                if not self._arm():
                    return TestResult("emergency", False, time.time()-start, "Failed to arm")
            
            self._fly_to_position(np.array([0.0, 0.0, 2.0]))
            
            print("\n  Triggering emergency stop (force disarm)...")
            
            # Emergency disarm
            if self._disarm(force=True):
                print("  [OK] Emergency stop executed")
                
                # Check that motors stopped
                time.sleep(0.5)
                motors = self.bridge.get_motor_commands(timeout=0.1)
                if motors is not None and np.max(motors) < 0.1:
                    print("  [OK] Motors stopped")
                    self.armed = False
                    return TestResult("emergency", True, time.time()-start, "Emergency stop successful")
                else:
                    return TestResult("emergency", False, time.time()-start, "Motors still running")
            else:
                return TestResult("emergency", False, time.time()-start, "Disarm command failed")
            
        except Exception as e:
            return TestResult("emergency", False, time.time()-start, f"Error: {e}")
    
    def run_test(self, test_name: str) -> Optional[TestResult]:
        """Run a single test.
        
        Args:
            test_name: Name of test to run
            
        Returns:
            Test result or None if test not found
        """
        tests = {
            "takeoff": self.test_takeoff,
            "land": self.test_land,
            "waypoints": self.test_waypoints,
            "offboard": self.test_offboard_velocity,
            "square": self.test_square_pattern,
            "emergency": self.test_emergency_stop,
        }
        
        if test_name not in tests:
            print(f"Unknown test: {test_name}")
            print(f"Available tests: {', '.join(tests.keys())}")
            return None
        
        return tests[test_name]()
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all tests sequentially.
        
        Returns:
            List of test results
        """
        results = []
        
        test_order = ["takeoff", "offboard", "waypoints", "square", "land", "emergency"]
        
        for test_name in test_order:
            if not running:
                break
            
            result = self.run_test(test_name)
            if result:
                results.append(result)
            
            # Reset between tests
            if self.armed:
                self._disarm(force=True)
            time.sleep(1.0)
            self.sim.reset(position=np.array([0.0, 0.0, 0.1]))
        
        return results
    
    def print_results(self, results: List[TestResult]) -> None:
        """Print test results summary."""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        for result in results:
            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"  {status} {result.name}: {result.message} ({result.duration:.1f}s)")
        
        print("-"*60)
        print(f"  Total: {passed}/{total} tests passed")
        print("="*60)


def main():
    global running
    
    parser = argparse.ArgumentParser(description="SITL Integration Tests")
    parser.add_argument(
        "test",
        nargs="?",
        default="all",
        choices=["takeoff", "land", "waypoints", "offboard", "square", "emergency", "all"],
        help="Test to run (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="generic",
        help="MuJoCo model name",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4560,
        help="MAVLink port",
    )
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create tester
    tester = SITLTester(model=args.model, port=args.port)
    
    try:
        # Setup
        if not tester.setup():
            sys.exit(1)
        
        # Run tests
        if args.test == "all":
            results = tester.run_all_tests()
            tester.print_results(results)
        else:
            result = tester.run_test(args.test)
            if result:
                tester.print_results([result])
        
    finally:
        tester.teardown()


if __name__ == "__main__":
    main()
