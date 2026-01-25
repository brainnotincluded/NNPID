#!/usr/bin/env python3
"""Pre-built flight scenarios for SITL testing.

This script provides ready-to-run flight scenarios:
- Hover test: Takeoff, hover, land
- Square pattern: Fly a square at fixed altitude
- Figure-8: Fly a figure-8 pattern
- Waypoint mission: Navigate through random waypoints
- Stress test: Rapid maneuvers

Usage:
    python scripts/sitl_scenarios.py hover
    python scripts/sitl_scenarios.py square --size 5
    python scripts/sitl_scenarios.py figure8 --radius 3
    python scripts/sitl_scenarios.py waypoints --count 10
    python scripts/sitl_scenarios.py stress
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
from abc import ABC, abstractmethod

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mujoco_sim import create_simulator, MuJoCoSimulator
from src.core.sensors import SensorSimulator, SensorConfig
from src.communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig
from src.communication.messages import HILSensorMessage, HILGPSMessage
from src.utils.transforms import CoordinateTransforms
from src.utils.rotations import Rotations


# Global running flag
running = True


def signal_handler(sig, frame):
    global running
    print("\n[INTERRUPT] Stopping scenario...")
    running = False


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    name: str
    success: bool
    duration: float
    waypoints_reached: int
    total_distance: float
    max_error: float
    message: str


class FlightScenario(ABC):
    """Base class for flight scenarios."""
    
    def __init__(
        self,
        sim: MuJoCoSimulator,
        bridge: MAVLinkBridge,
        sensors: SensorSimulator,
    ):
        self.sim = sim
        self.bridge = bridge
        self.sensors = sensors
        
        self.armed = False
        self.in_offboard = False
        self.start_time = 0.0
        
        # Tracking
        self.path: List[np.ndarray] = []
        self.waypoints_reached = 0
        self.max_error = 0.0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Scenario name."""
        pass
    
    @abstractmethod
    def run(self) -> ScenarioResult:
        """Run the scenario."""
        pass
    
    def _send_sensors(self) -> None:
        """Send sensor data to PX4."""
        state = self.sim.get_state()
        timestamp = self.sim.get_time()
        
        gyro, accel = self.sim.get_imu_data()
        gyro_frd = CoordinateTransforms.angular_velocity_mujoco_to_frd(gyro)
        accel_frd = CoordinateTransforms.acceleration_mujoco_to_frd(accel)
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(state.velocity)
        
        imu_data = self.sensors.get_imu(gyro_frd, accel_frd, timestamp)
        
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
        
        if int(timestamp * 10) % 1 == 0:
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
        
        self.bridge.send_heartbeat()
    
    def _step(self, duration: float = 0.02) -> None:
        """Step simulation."""
        steps = int(duration / self.sim.timestep)
        
        for _ in range(steps):
            motors = self.bridge.get_motor_commands(timeout=0.001)
            if motors is None:
                motors = np.zeros(4)
            self.sim.step(motors)
            self._send_sensors()
        
        # Track path
        self.path.append(self.sim.get_state().position.copy())
    
    def _enter_offboard(self) -> bool:
        """Enter offboard mode."""
        pos_ned = np.array([0.0, 0.0, -1.0])
        for _ in range(50):
            self.bridge.send_position_setpoint(pos_ned)
            self._step(0.02)
        
        if self.bridge.set_offboard_mode():
            self.in_offboard = True
            return True
        return False
    
    def _arm(self) -> bool:
        """Arm vehicle."""
        if self.bridge.arm():
            self.armed = True
            time.sleep(0.5)
            return True
        return False
    
    def _fly_to(
        self,
        position: np.ndarray,
        tolerance: float = 0.3,
        timeout: float = 15.0,
    ) -> bool:
        """Fly to position."""
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(position)
        start = time.time()
        
        while running and (time.time() - start) < timeout:
            self.bridge.send_position_setpoint(pos_ned)
            self._step(0.02)
            
            state = self.sim.get_state()
            error = np.linalg.norm(state.position - position)
            self.max_error = max(self.max_error, error)
            
            if error < tolerance:
                self.waypoints_reached += 1
                return True
        
        return False
    
    def _fly_velocity(self, velocity: np.ndarray, duration: float) -> None:
        """Fly with velocity."""
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(velocity)
        start = time.time()
        
        while running and (time.time() - start) < duration:
            self.bridge.send_velocity_setpoint(vel_ned)
            self._step(0.02)
    
    def _compute_total_distance(self) -> float:
        """Compute total path distance."""
        if len(self.path) < 2:
            return 0.0
        
        total = 0.0
        for i in range(1, len(self.path)):
            total += np.linalg.norm(self.path[i] - self.path[i-1])
        return total


class HoverScenario(FlightScenario):
    """Simple hover test."""
    
    def __init__(
        self,
        sim: MuJoCoSimulator,
        bridge: MAVLinkBridge,
        sensors: SensorSimulator,
        altitude: float = 2.0,
        duration: float = 10.0,
    ):
        super().__init__(sim, bridge, sensors)
        self.altitude = altitude
        self.duration = duration
    
    @property
    def name(self) -> str:
        return f"Hover at {self.altitude}m for {self.duration}s"
    
    def run(self) -> ScenarioResult:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.name}")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        
        try:
            # Setup
            if not self._enter_offboard():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to enter offboard")
            if not self._arm():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to arm")
            
            # Takeoff
            print(f"\n  Takeoff to {self.altitude}m...")
            if not self._fly_to(np.array([0.0, 0.0, self.altitude])):
                return ScenarioResult(self.name, False, time.time()-self.start_time, 
                                     0, self._compute_total_distance(), self.max_error,
                                     "Takeoff failed")
            
            # Hover
            print(f"  Hovering for {self.duration}s...")
            hover_start = time.time()
            errors = []
            
            target = np.array([0.0, 0.0, self.altitude])
            pos_ned = CoordinateTransforms.position_mujoco_to_ned(target)
            
            while running and (time.time() - hover_start) < self.duration:
                self.bridge.send_position_setpoint(pos_ned)
                self._step(0.02)
                
                state = self.sim.get_state()
                error = np.linalg.norm(state.position - target)
                errors.append(error)
                
                print(f"\r  Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}] "
                      f"error: {error:.3f}m   ", end="")
            
            print()
            
            # Land
            print("  Landing...")
            self.bridge.land()
            if self._fly_to(np.array([0.0, 0.0, 0.1]), tolerance=0.2, timeout=20.0):
                self.bridge.disarm(force=True)
            
            mean_error = np.mean(errors) if errors else 0
            return ScenarioResult(
                self.name, True,
                time.time() - self.start_time,
                self.waypoints_reached,
                self._compute_total_distance(),
                self.max_error,
                f"Hover complete, mean error: {mean_error:.3f}m"
            )
            
        except Exception as e:
            return ScenarioResult(self.name, False, time.time()-self.start_time,
                                 0, 0, 0, f"Error: {e}")


class SquareScenario(FlightScenario):
    """Fly a square pattern."""
    
    def __init__(
        self,
        sim: MuJoCoSimulator,
        bridge: MAVLinkBridge,
        sensors: SensorSimulator,
        size: float = 3.0,
        altitude: float = 2.0,
        laps: int = 1,
    ):
        super().__init__(sim, bridge, sensors)
        self.size = size
        self.altitude = altitude
        self.laps = laps
    
    @property
    def name(self) -> str:
        return f"Square {self.size}m x {self.laps} laps at {self.altitude}m"
    
    def run(self) -> ScenarioResult:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.name}")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        
        try:
            # Setup
            if not self._enter_offboard():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to enter offboard")
            if not self._arm():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to arm")
            
            # Generate waypoints
            corners = [
                np.array([0.0, 0.0, self.altitude]),
                np.array([self.size, 0.0, self.altitude]),
                np.array([self.size, self.size, self.altitude]),
                np.array([0.0, self.size, self.altitude]),
            ]
            
            # Fly laps
            for lap in range(self.laps):
                print(f"\n  Lap {lap+1}/{self.laps}")
                
                for i, corner in enumerate(corners + [corners[0]]):
                    print(f"    Corner {i+1}: [{corner[0]:.1f}, {corner[1]:.1f}, {corner[2]:.1f}]")
                    
                    if not self._fly_to(corner):
                        return ScenarioResult(
                            self.name, False,
                            time.time() - self.start_time,
                            self.waypoints_reached,
                            self._compute_total_distance(),
                            self.max_error,
                            f"Failed at corner {i+1}"
                        )
            
            # Land
            print("\n  Landing...")
            self.bridge.land()
            self._fly_to(np.array([0.0, 0.0, 0.1]), tolerance=0.2, timeout=20.0)
            self.bridge.disarm(force=True)
            
            return ScenarioResult(
                self.name, True,
                time.time() - self.start_time,
                self.waypoints_reached,
                self._compute_total_distance(),
                self.max_error,
                f"Completed {self.laps} laps"
            )
            
        except Exception as e:
            return ScenarioResult(self.name, False, time.time()-self.start_time,
                                 0, 0, 0, f"Error: {e}")


class Figure8Scenario(FlightScenario):
    """Fly a figure-8 pattern."""
    
    def __init__(
        self,
        sim: MuJoCoSimulator,
        bridge: MAVLinkBridge,
        sensors: SensorSimulator,
        radius: float = 2.0,
        altitude: float = 2.0,
        speed: float = 1.0,
        laps: int = 2,
    ):
        super().__init__(sim, bridge, sensors)
        self.radius = radius
        self.altitude = altitude
        self.speed = speed
        self.laps = laps
    
    @property
    def name(self) -> str:
        return f"Figure-8, radius {self.radius}m, {self.laps} laps"
    
    def _figure8_point(self, t: float) -> np.ndarray:
        """Get point on figure-8 at parameter t."""
        x = self.radius * np.sin(t)
        y = self.radius * np.sin(2*t) / 2
        return np.array([x, y, self.altitude])
    
    def run(self) -> ScenarioResult:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.name}")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        
        try:
            # Setup
            if not self._enter_offboard():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to enter offboard")
            if not self._arm():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to arm")
            
            # Takeoff to start
            start_point = self._figure8_point(0)
            print(f"\n  Flying to start position...")
            if not self._fly_to(start_point):
                return ScenarioResult(self.name, False, time.time()-self.start_time,
                                     0, 0, 0, "Failed to reach start")
            
            # Fly figure-8
            print(f"  Flying figure-8...")
            
            num_points = 50 * self.laps
            for i in range(num_points):
                if not running:
                    break
                
                t = 2 * np.pi * i / (num_points / self.laps)
                target = self._figure8_point(t)
                pos_ned = CoordinateTransforms.position_mujoco_to_ned(target)
                
                # Velocity towards target
                state = self.sim.get_state()
                direction = target - state.position
                dist = np.linalg.norm(direction)
                
                if dist > 0.1:
                    velocity = direction / dist * self.speed
                    vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(velocity)
                    self.bridge.send_velocity_setpoint(vel_ned)
                else:
                    self.bridge.send_position_setpoint(pos_ned)
                
                self._step(0.1)
                
                error = np.linalg.norm(state.position - target)
                self.max_error = max(self.max_error, error)
                
                print(f"\r  Progress: {100*i/num_points:.0f}% | "
                      f"Position: [{state.position[0]:.2f}, {state.position[1]:.2f}] | "
                      f"Error: {error:.2f}m   ", end="")
            
            print()
            self.waypoints_reached = num_points
            
            # Land
            print("\n  Landing...")
            self.bridge.land()
            self._fly_to(np.array([0.0, 0.0, 0.1]), tolerance=0.2, timeout=20.0)
            self.bridge.disarm(force=True)
            
            return ScenarioResult(
                self.name, True,
                time.time() - self.start_time,
                self.waypoints_reached,
                self._compute_total_distance(),
                self.max_error,
                "Figure-8 complete"
            )
            
        except Exception as e:
            return ScenarioResult(self.name, False, time.time()-self.start_time,
                                 0, 0, 0, f"Error: {e}")


class WaypointScenario(FlightScenario):
    """Navigate through random waypoints."""
    
    def __init__(
        self,
        sim: MuJoCoSimulator,
        bridge: MAVLinkBridge,
        sensors: SensorSimulator,
        count: int = 5,
        radius: float = 5.0,
        altitude_range: Tuple[float, float] = (1.0, 3.0),
        seed: int = 42,
    ):
        super().__init__(sim, bridge, sensors)
        self.count = count
        self.radius = radius
        self.altitude_range = altitude_range
        self.seed = seed
    
    @property
    def name(self) -> str:
        return f"Waypoints: {self.count} points in {self.radius}m radius"
    
    def run(self) -> ScenarioResult:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.name}")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        np.random.seed(self.seed)
        
        # Generate waypoints
        waypoints = []
        for _ in range(self.count):
            angle = np.random.uniform(0, 2*np.pi)
            r = np.random.uniform(0, self.radius)
            alt = np.random.uniform(*self.altitude_range)
            waypoints.append(np.array([
                r * np.cos(angle),
                r * np.sin(angle),
                alt,
            ]))
        
        try:
            # Setup
            if not self._enter_offboard():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to enter offboard")
            if not self._arm():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to arm")
            
            # Navigate waypoints
            for i, wp in enumerate(waypoints):
                print(f"\n  Waypoint {i+1}/{len(waypoints)}: "
                      f"[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}]")
                
                if not self._fly_to(wp, tolerance=0.4):
                    return ScenarioResult(
                        self.name, False,
                        time.time() - self.start_time,
                        self.waypoints_reached,
                        self._compute_total_distance(),
                        self.max_error,
                        f"Failed at waypoint {i+1}"
                    )
                
                print(f"    Reached!")
            
            # Return home and land
            print("\n  Returning home...")
            self._fly_to(np.array([0.0, 0.0, 2.0]))
            
            print("  Landing...")
            self.bridge.land()
            self._fly_to(np.array([0.0, 0.0, 0.1]), tolerance=0.2, timeout=20.0)
            self.bridge.disarm(force=True)
            
            return ScenarioResult(
                self.name, True,
                time.time() - self.start_time,
                self.waypoints_reached,
                self._compute_total_distance(),
                self.max_error,
                f"All {self.count} waypoints reached"
            )
            
        except Exception as e:
            return ScenarioResult(self.name, False, time.time()-self.start_time,
                                 0, 0, 0, f"Error: {e}")


class StressScenario(FlightScenario):
    """Stress test with rapid maneuvers."""
    
    def __init__(
        self,
        sim: MuJoCoSimulator,
        bridge: MAVLinkBridge,
        sensors: SensorSimulator,
        duration: float = 30.0,
    ):
        super().__init__(sim, bridge, sensors)
        self.duration = duration
    
    @property
    def name(self) -> str:
        return f"Stress test for {self.duration}s"
    
    def run(self) -> ScenarioResult:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.name}")
        print(f"{'='*60}")
        
        self.start_time = time.time()
        
        try:
            # Setup
            if not self._enter_offboard():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to enter offboard")
            if not self._arm():
                return ScenarioResult(self.name, False, 0, 0, 0, 0, "Failed to arm")
            
            # Takeoff
            print("\n  Takeoff...")
            self._fly_to(np.array([0.0, 0.0, 2.0]))
            
            # Rapid maneuvers
            print("  Running stress maneuvers...")
            maneuvers = [
                ("Forward burst", np.array([3.0, 0.0, 0.0]), 0.5),
                ("Stop", np.array([0.0, 0.0, 0.0]), 0.5),
                ("Backward burst", np.array([-3.0, 0.0, 0.0]), 0.5),
                ("Stop", np.array([0.0, 0.0, 0.0]), 0.5),
                ("Right burst", np.array([0.0, 3.0, 0.0]), 0.5),
                ("Stop", np.array([0.0, 0.0, 0.0]), 0.5),
                ("Left burst", np.array([0.0, -3.0, 0.0]), 0.5),
                ("Stop", np.array([0.0, 0.0, 0.0]), 0.5),
                ("Up burst", np.array([0.0, 0.0, 2.0]), 0.5),
                ("Down burst", np.array([0.0, 0.0, -2.0]), 0.5),
            ]
            
            start = time.time()
            maneuver_idx = 0
            
            while running and (time.time() - start) < self.duration:
                maneuver = maneuvers[maneuver_idx % len(maneuvers)]
                print(f"\r  {maneuver[0]:20s}", end="")
                
                self._fly_velocity(maneuver[1], maneuver[2])
                self.waypoints_reached += 1
                maneuver_idx += 1
            
            print()
            
            # Recover and land
            print("\n  Recovering...")
            self._fly_to(np.array([0.0, 0.0, 2.0]))
            
            print("  Landing...")
            self.bridge.land()
            self._fly_to(np.array([0.0, 0.0, 0.1]), tolerance=0.2, timeout=20.0)
            self.bridge.disarm(force=True)
            
            return ScenarioResult(
                self.name, True,
                time.time() - self.start_time,
                self.waypoints_reached,
                self._compute_total_distance(),
                self.max_error,
                f"Completed {maneuver_idx} maneuvers"
            )
            
        except Exception as e:
            return ScenarioResult(self.name, False, time.time()-self.start_time,
                                 0, 0, 0, f"Error: {e}")


def run_scenario(
    scenario_name: str,
    **kwargs,
) -> Optional[ScenarioResult]:
    """Run a flight scenario.
    
    Args:
        scenario_name: Name of scenario to run
        **kwargs: Scenario-specific parameters
        
    Returns:
        Scenario result or None if setup failed
    """
    global running
    running = True
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup simulation
    print("\n" + "="*60)
    print("SITL Flight Scenario Runner")
    print("="*60)
    
    print("\n[1/3] Loading MuJoCo model...")
    sim = create_simulator(model="generic")
    sim.reset(position=np.array([0.0, 0.0, 0.1]))
    sensors = SensorSimulator(SensorConfig())
    
    print("[2/3] Starting MAVLink server...")
    config = MAVLinkConfig(port=kwargs.get("port", 4560))
    bridge = MAVLinkBridge(config)
    
    print("[3/3] Waiting for PX4 connection...")
    if not bridge.start_server():
        print("[ERROR] Failed to connect to PX4")
        return None
    
    print("[OK] Connected to PX4!")
    
    # Send initial sensors
    for _ in range(50):
        state = sim.get_state()
        timestamp = sim.get_time()
        gyro, accel = sim.get_imu_data()
        
        sensor_msg = HILSensorMessage.from_sensor_data(
            time_sec=timestamp,
            gyro=gyro, accel=accel,
            mag=np.array([0.21, 0.0, 0.42]),
            pressure=101325.0, temperature=20.0, altitude=0.1,
        )
        bridge.send_hil_sensor(sensor_msg)
        bridge.send_heartbeat()
        time.sleep(0.02)
    
    # Create scenario
    scenarios = {
        "hover": lambda: HoverScenario(
            sim, bridge, sensors,
            altitude=kwargs.get("altitude", 2.0),
            duration=kwargs.get("duration", 10.0),
        ),
        "square": lambda: SquareScenario(
            sim, bridge, sensors,
            size=kwargs.get("size", 3.0),
            altitude=kwargs.get("altitude", 2.0),
            laps=kwargs.get("laps", 1),
        ),
        "figure8": lambda: Figure8Scenario(
            sim, bridge, sensors,
            radius=kwargs.get("radius", 2.0),
            altitude=kwargs.get("altitude", 2.0),
            speed=kwargs.get("speed", 1.0),
            laps=kwargs.get("laps", 2),
        ),
        "waypoints": lambda: WaypointScenario(
            sim, bridge, sensors,
            count=kwargs.get("count", 5),
            radius=kwargs.get("radius", 5.0),
            seed=kwargs.get("seed", 42),
        ),
        "stress": lambda: StressScenario(
            sim, bridge, sensors,
            duration=kwargs.get("duration", 30.0),
        ),
    }
    
    if scenario_name not in scenarios:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(scenarios.keys())}")
        bridge.stop()
        return None
    
    scenario = scenarios[scenario_name]()
    
    try:
        result = scenario.run()
        
        # Print result
        print("\n" + "="*60)
        print("SCENARIO RESULT")
        print("="*60)
        print(f"  Name: {result.name}")
        print(f"  Success: {'YES' if result.success else 'NO'}")
        print(f"  Duration: {result.duration:.1f}s")
        print(f"  Waypoints: {result.waypoints_reached}")
        print(f"  Distance: {result.total_distance:.1f}m")
        print(f"  Max Error: {result.max_error:.2f}m")
        print(f"  Message: {result.message}")
        print("="*60)
        
        return result
        
    finally:
        bridge.stop()


def main():
    parser = argparse.ArgumentParser(description="SITL Flight Scenarios")
    
    subparsers = parser.add_subparsers(dest="scenario", help="Scenario to run")
    
    # Hover
    hover = subparsers.add_parser("hover", help="Hover test")
    hover.add_argument("--altitude", type=float, default=2.0)
    hover.add_argument("--duration", type=float, default=10.0)
    
    # Square
    square = subparsers.add_parser("square", help="Square pattern")
    square.add_argument("--size", type=float, default=3.0)
    square.add_argument("--altitude", type=float, default=2.0)
    square.add_argument("--laps", type=int, default=1)
    
    # Figure-8
    fig8 = subparsers.add_parser("figure8", help="Figure-8 pattern")
    fig8.add_argument("--radius", type=float, default=2.0)
    fig8.add_argument("--altitude", type=float, default=2.0)
    fig8.add_argument("--speed", type=float, default=1.0)
    fig8.add_argument("--laps", type=int, default=2)
    
    # Waypoints
    waypoints = subparsers.add_parser("waypoints", help="Waypoint navigation")
    waypoints.add_argument("--count", type=int, default=5)
    waypoints.add_argument("--radius", type=float, default=5.0)
    waypoints.add_argument("--seed", type=int, default=42)
    
    # Stress
    stress = subparsers.add_parser("stress", help="Stress test")
    stress.add_argument("--duration", type=float, default=30.0)
    
    # Common args
    for p in [hover, square, fig8, waypoints, stress]:
        p.add_argument("--port", type=int, default=4560)
    
    args = parser.parse_args()
    
    if args.scenario is None:
        parser.print_help()
        sys.exit(1)
    
    run_scenario(args.scenario, **vars(args))


if __name__ == "__main__":
    main()
