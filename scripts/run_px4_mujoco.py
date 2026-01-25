#!/usr/bin/env python3
"""Run MuJoCo + PX4 SITL integration.

This script:
1. Starts PX4 SITL in background
2. Connects to PX4 via MAVLink UDP  
3. Sends sensor data from MuJoCo to PX4
4. Receives motor commands from PX4
5. Applies commands to MuJoCo physics

Usage:
    python scripts/run_px4_mujoco.py
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import threading
from pathlib import Path
from queue import Queue, Empty

import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

from src.core.mujoco_sim import create_simulator
from src.core.sensors import SensorSimulator, SensorConfig
from src.utils.transforms import CoordinateTransforms


running = True


def signal_handler(sig, frame):
    global running
    print("\n[CTRL+C] Stopping...")
    running = False


class MuJoCoPX4Bridge:
    """Bridge between MuJoCo physics and PX4 SITL."""
    
    def __init__(self, px4_home: str = None):
        self.px4_home = px4_home or str(Path.home() / "PX4-Autopilot")
        self.px4_process = None
        
        # MuJoCo
        self.sim = create_simulator(model="generic")
        self.sensors = SensorSimulator(SensorConfig())
        
        # MAVLink
        self.mav = None
        
        # State
        self.armed = False
        self.motor_commands = np.zeros(4)
        
    def start_px4(self) -> bool:
        """Start PX4 SITL in background."""
        print("[PX4] Starting PX4 SITL...")
        
        # Start PX4 with HIL enabled
        cmd = f"cd {self.px4_home} && make px4_sitl_default none"
        
        self.px4_process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        )
        
        # Wait for startup
        print("[PX4] Waiting for startup...")
        time.sleep(5)
        
        return self.px4_process.poll() is None
    
    def stop_px4(self):
        """Stop PX4 SITL."""
        if self.px4_process:
            print("[PX4] Stopping...")
            self.px4_process.terminate()
            try:
                self.px4_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.px4_process.kill()
    
    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to PX4 via MAVLink UDP."""
        print("[MAVLink] Connecting to PX4...")
        
        try:
            self.mav = mavutil.mavlink_connection(
                'udpin:0.0.0.0:14540',
                source_system=255
            )
            
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
            if msg:
                print(f"[MAVLink] Connected to system {self.mav.target_system}")
                return True
            else:
                print("[MAVLink] No heartbeat received")
                return False
                
        except Exception as e:
            print(f"[MAVLink] Connection error: {e}")
            return False
    
    def send_sensors(self, timestamp: float):
        """Send sensor data to PX4."""
        if not self.mav:
            return
        
        state = self.sim.get_state()
        gyro, accel = self.sim.get_imu_data()
        
        # Convert to FRD/NED frames
        gyro_frd = CoordinateTransforms.angular_velocity_mujoco_to_frd(gyro)
        accel_frd = CoordinateTransforms.acceleration_mujoco_to_frd(accel)
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
        
        # Add sensor noise
        imu = self.sensors.get_imu(gyro_frd, accel_frd, timestamp)
        
        # Send HIL_SENSOR
        time_usec = int(timestamp * 1e6)
        self.mav.mav.hil_sensor_send(
            time_usec,
            imu.accel[0], imu.accel[1], imu.accel[2],
            imu.gyro[0], imu.gyro[1], imu.gyro[2],
            0.21, 0.0, 0.42,  # mag
            101325.0 - pos_ned[2] * 12.0,  # pressure (approximate)
            0.0,  # diff pressure
            -pos_ned[2] / 0.0289644,  # pressure altitude
            20.0,  # temperature
            0x1FF  # fields updated
        )
    
    def send_gps(self, timestamp: float):
        """Send GPS data to PX4."""
        if not self.mav:
            return
        
        state = self.sim.get_state()
        pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
        vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(state.velocity)
        
        gps = self.sensors.get_gps(pos_ned, vel_ned, timestamp)
        
        time_usec = int(timestamp * 1e6)
        
        # Compute ground speed and course
        vel_horiz = np.sqrt(vel_ned[0]**2 + vel_ned[1]**2)
        course = np.arctan2(vel_ned[1], vel_ned[0])
        if course < 0:
            course += 2 * np.pi
        cog = int(course * 180 / np.pi * 100) % 36000  # Ensure [0, 36000)
        
        self.mav.mav.hil_gps_send(
            time_usec,
            3,  # fix_type (3D)
            int(gps.latitude * 1e7),
            int(gps.longitude * 1e7),
            int(gps.altitude * 1000),
            10, 10,  # eph, epv (in cm)
            int(vel_horiz * 100),  # vel (cm/s)
            int(vel_ned[0] * 100),  # vn
            int(vel_ned[1] * 100),  # ve
            int(vel_ned[2] * 100),  # vd
            cog,  # cog (cdeg)
            12  # satellites
        )
    
    def receive_actuators(self, timeout: float = 0.001) -> bool:
        """Receive actuator commands from PX4."""
        if not self.mav:
            return False
        
        msg = self.mav.recv_match(
            type='HIL_ACTUATOR_CONTROLS',
            blocking=True,
            timeout=timeout
        )
        
        if msg:
            # Controls are in [-1, 1], convert to [0, 1]
            self.motor_commands = np.array([
                (msg.controls[0] + 1) / 2,
                (msg.controls[1] + 1) / 2,
                (msg.controls[2] + 1) / 2,
                (msg.controls[3] + 1) / 2,
            ])
            return True
        
        return False
    
    def arm(self) -> bool:
        """Arm the vehicle."""
        if not self.mav:
            return False
        
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
        
        ack = self.mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        if ack and ack.result == 0:
            self.armed = True
            return True
        return False
    
    def set_offboard_mode(self) -> bool:
        """Set offboard mode."""
        if not self.mav:
            return False
        
        # Send position setpoints first
        for _ in range(20):
            self.mav.mav.set_position_target_local_ned_send(
                0, self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0
            )
            time.sleep(0.05)
        
        self.mav.mav.command_long_send(
            self.mav.target_system,
            self.mav.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            6, 0, 0, 0, 0, 0  # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        )
        
        ack = self.mav.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
        return ack and ack.result == 0
    
    def send_position_setpoint(self, position: np.ndarray, yaw: float = 0):
        """Send position setpoint."""
        if not self.mav:
            return
        
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            position[0], position[1], position[2],
            0, 0, 0, 0, 0, 0, yaw, 0
        )
    
    def run_simulation_step(self, dt: float = 0.002):
        """Run one simulation step."""
        # Apply motor commands to MuJoCo
        self.sim.step(self.motor_commands)
        
        # Send sensor data
        timestamp = self.sim.get_time()
        self.send_sensors(timestamp)
        
        # Send GPS at lower rate
        if int(timestamp * 100) % 10 == 0:
            self.send_gps(timestamp)
        
        # Receive actuator commands
        self.receive_actuators()
    
    def run_test(self):
        """Run the full integration test."""
        global running
        
        signal.signal(signal.SIGINT, signal_handler)
        
        print("="*60)
        print("MuJoCo + PX4 SITL Integration Test")
        print("="*60)
        
        # Start PX4
        if not self.start_px4():
            print("[ERROR] Failed to start PX4")
            return
        
        try:
            # Connect
            if not self.connect():
                print("[ERROR] Failed to connect to PX4")
                return
            
            # Reset MuJoCo
            self.sim.reset(position=np.array([0.0, 0.0, 0.1]))
            self.sensors.reset()
            
            # Warm up - send sensor data
            print("\n[1] Warming up...")
            for _ in range(100):
                self.run_simulation_step()
                time.sleep(0.01)
            
            # Set offboard mode
            print("\n[2] Setting offboard mode...")
            if self.set_offboard_mode():
                print("    Offboard mode set!")
            else:
                print("    Warning: Failed to set offboard mode")
            
            # Arm
            print("\n[3] Arming...")
            if self.arm():
                print("    Armed!")
            else:
                print("    Warning: Failed to arm")
            
            # Takeoff
            print("\n[4] Taking off to 2m...")
            target = np.array([0.0, 0.0, -2.0])  # NED
            takeoff_start = time.time()
            
            while running and (time.time() - takeoff_start) < 15:
                self.send_position_setpoint(target)
                self.run_simulation_step()
                
                state = self.sim.get_state()
                alt = state.position[2]
                
                print(f"\r    Altitude: {alt:.2f}m | "
                      f"Motors: [{self.motor_commands[0]:.2f}, {self.motor_commands[1]:.2f}, "
                      f"{self.motor_commands[2]:.2f}, {self.motor_commands[3]:.2f}]   ", end="")
                
                if alt > 1.8:
                    print(f"\n    Reached altitude!")
                    break
                
                time.sleep(0.002)
            
            if not running:
                return
            
            # Hover
            print("\n\n[5] Hovering for 5 seconds...")
            hover_start = time.time()
            
            while running and (time.time() - hover_start) < 5:
                self.send_position_setpoint(target)
                self.run_simulation_step()
                
                state = self.sim.get_state()
                print(f"\r    Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]   ", end="")
                
                time.sleep(0.002)
            
            # Done
            print("\n\n[6] Test complete!")
            
            # Print final state
            state = self.sim.get_state()
            print(f"\n    Final position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
            print(f"    Final velocity: [{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}]")
            
        finally:
            self.stop_px4()


def main():
    bridge = MuJoCoPX4Bridge()
    bridge.run_test()


if __name__ == "__main__":
    main()
