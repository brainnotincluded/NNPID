#!/usr/bin/env python3
"""MuJoCo + PX4 SITL with Visualization.

This script:
1. Opens MuJoCo viewer showing the drone
2. Connects to PX4 SITL
3. Sends sensor data to PX4
4. Receives motor commands and applies to physics
5. Shows drone taking off visually

Usage:
    # Terminal 1: Start this script (it will wait for PX4)
    python scripts/mujoco_sitl_viz.py
    
    # Terminal 2: Start PX4 SITL
    cd ~/PX4-Autopilot && make px4_sitl none_iris
"""

import sys
import time
import socket
import threading
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
import mujoco.viewer

from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink


class MuJoCoSITLViz:
    """MuJoCo simulation with PX4 SITL and visualization."""
    
    def __init__(self):
        # Create simple quadrotor model
        self.xml = """
        <mujoco model="quadrotor">
            <option gravity="0 0 -9.81" timestep="0.002"/>
            
            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
                <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5" width="512" height="512"/>
                <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
            </asset>
            
            <worldbody>
                <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
                <geom type="plane" size="50 50 0.1" material="grid"/>
                
                <body name="drone" pos="0 0 0.5">
                    <freejoint name="root"/>
                    <inertial pos="0 0 0" mass="1.5" diaginertia="0.01 0.01 0.02"/>
                    
                    <!-- Main body -->
                    <geom type="box" size="0.1 0.1 0.03" rgba="0.2 0.2 0.2 1"/>
                    
                    <!-- Arms -->
                    <geom type="capsule" fromto="0.15 0.15 0 -0.15 -0.15 0" size="0.01" rgba="0.3 0.3 0.3 1"/>
                    <geom type="capsule" fromto="-0.15 0.15 0 0.15 -0.15 0" size="0.01" rgba="0.3 0.3 0.3 1"/>
                    
                    <!-- Motors (visual) -->
                    <geom type="cylinder" pos="0.15 0.15 0.02" size="0.04 0.01" rgba="1 0 0 1"/>
                    <geom type="cylinder" pos="-0.15 0.15 0.02" size="0.04 0.01" rgba="0 1 0 1"/>
                    <geom type="cylinder" pos="-0.15 -0.15 0.02" size="0.04 0.01" rgba="0 0 1 1"/>
                    <geom type="cylinder" pos="0.15 -0.15 0.02" size="0.04 0.01" rgba="1 1 0 1"/>
                    
                    <!-- Propellers (visual, will spin) -->
                    <site name="prop1" pos="0.15 0.15 0.04" size="0.05"/>
                    <site name="prop2" pos="-0.15 0.15 0.04" size="0.05"/>
                    <site name="prop3" pos="-0.15 -0.15 0.04" size="0.05"/>
                    <site name="prop4" pos="0.15 -0.15 0.04" size="0.05"/>
                </body>
            </worldbody>
            
            <actuator>
                <general name="thrust1" site="prop1" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
                <general name="thrust2" site="prop2" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
                <general name="thrust3" site="prop3" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
                <general name="thrust4" site="prop4" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
            </actuator>
        </mujoco>
        """
        
        # Load model
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        
        # Motor commands
        self.motors = np.zeros(4)
        
        # Network
        self.server = None
        self.conn = None
        self.mav = None
        self.mav_udp = None
        
        # State
        self.connected = False
        self.armed = False
        self.running = True
        self.timestamp = 0
        
    def start_server(self):
        """Start TCP server for PX4 connection."""
        print("\n[1] Starting HIL server on port 4560...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('127.0.0.1', 4560))
        self.server.listen(1)
        self.server.settimeout(1.0)  # Non-blocking for viewer
        print("    Waiting for PX4...")
        print("    Run: cd ~/PX4-Autopilot && make px4_sitl none_iris")
        
    def check_connection(self):
        """Check for PX4 connection."""
        if self.connected:
            return True
            
        try:
            self.conn, addr = self.server.accept()
            self.conn.setblocking(False)
            self.mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
            self.connected = True
            print(f"\n[2] PX4 connected from {addr}!")
            
            # Also setup UDP for commands
            print("    Setting up UDP command channel...")
            time.sleep(2)  # Wait for PX4 to start UDP
            try:
                self.mav_udp = mavutil.mavlink_connection('udpin:0.0.0.0:14540', source_system=255)
                msg = self.mav_udp.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
                if msg:
                    print(f"    UDP connected to system {self.mav_udp.target_system}")
            except:
                print("    UDP not available yet")
            
            return True
        except socket.timeout:
            return False
        except Exception as e:
            return False
    
    def send_sensors(self):
        """Send sensor data to PX4."""
        if not self.connected or self.conn is None:
            return
            
        self.timestamp += 4000  # 4ms
        
        # Get state
        pos = self.data.qpos[:3].copy()
        vel = self.data.qvel[:3].copy()
        
        # Accelerometer (gravity + motion)
        accel = np.array([0.0, 0.0, -9.81])
        
        try:
            # HIL_SENSOR
            msg = self.mav.hil_sensor_encode(
                self.timestamp,
                accel[0], accel[1], accel[2],  # accel
                0.0, 0.0, 0.0,  # gyro
                0.21, 0.0, 0.42,  # mag
                101325.0 - pos[2] * 12.0,  # pressure
                0.0,  # diff_pressure
                pos[2],  # pressure_alt
                20.0,  # temp
                0x1FF  # fields
            )
            self.conn.send(msg.get_msgbuf())
            
            # HIL_GPS at 10Hz
            if self.timestamp % 100000 == 0:
                msg = self.mav.hil_gps_encode(
                    self.timestamp, 3,
                    int((47.397742 + pos[0] * 0.00001) * 1e7),
                    int((8.545594 + pos[1] * 0.00001) * 1e7),
                    int((488.0 + pos[2]) * 1000),
                    10, 10,
                    int(np.sqrt(vel[0]**2 + vel[1]**2) * 100),
                    int(vel[0] * 100), int(vel[1] * 100), int(-vel[2] * 100),
                    0, 12
                )
                self.conn.send(msg.get_msgbuf())
            
            # Heartbeat at 1Hz
            if self.timestamp % 1000000 == 0:
                msg = self.mav.heartbeat_encode(6, 8, 0, 0, 0)
                self.conn.send(msg.get_msgbuf())
                
        except Exception as e:
            pass
    
    def receive_motors(self):
        """Receive motor commands from PX4."""
        if not self.connected or self.conn is None:
            return
            
        try:
            data = self.conn.recv(4096)
            for byte in data:
                m = self.mav.parse_char(bytes([byte]))
                if m and m.get_type() == 'HIL_ACTUATOR_CONTROLS':
                    self.motors = np.clip([
                        (m.controls[0] + 1) / 2,
                        (m.controls[1] + 1) / 2,
                        (m.controls[2] + 1) / 2,
                        (m.controls[3] + 1) / 2,
                    ], 0, 1)
        except BlockingIOError:
            pass
        except Exception as e:
            pass
    
    def send_commands(self):
        """Send arm and takeoff commands via UDP."""
        if self.mav_udp is None or self.armed:
            return
            
        try:
            # Send position setpoints
            self.mav_udp.mav.set_position_target_local_ned_send(
                0, self.mav_udp.target_system, self.mav_udp.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                0b0000111111111000,
                0, 0, -5,  # 5m up
                0, 0, 0, 0, 0, 0, 0, 0
            )
            
            # Set offboard mode
            self.mav_udp.mav.command_long_send(
                self.mav_udp.target_system, self.mav_udp.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                6, 0, 0, 0, 0, 0
            )
            
            # Arm
            self.mav_udp.mav.command_long_send(
                self.mav_udp.target_system, self.mav_udp.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 0, 0, 0, 0, 0, 0
            )
            
            self.armed = True
            print("\n[3] Sent arm and takeoff commands!")
            
        except Exception as e:
            pass
    
    def step(self):
        """Step the simulation."""
        # Apply motor commands
        self.data.ctrl[:] = self.motors
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Exchange data with PX4
        if self.connected:
            self.send_sensors()
            self.receive_motors()
            
            # Send commands once
            if not self.armed and self.timestamp > 2000000:  # After 2 seconds
                self.send_commands()
    
    def run(self):
        """Run simulation with viewer."""
        self.start_server()
        
        print("\n[*] Opening MuJoCo viewer...")
        print("    Press ESC to quit")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -30
            viewer.cam.distance = 8
            viewer.cam.lookat[:] = [0, 0, 2]
            
            start_time = time.time()
            last_print = 0
            
            while viewer.is_running() and self.running:
                step_start = time.time()
                
                # Check for connection
                if not self.connected:
                    self.check_connection()
                
                # Step simulation
                self.step()
                
                # Sync viewer
                viewer.sync()
                
                # Print status
                if time.time() - last_print > 0.5:
                    pos = self.data.qpos[:3]
                    status = "CONNECTED" if self.connected else "Waiting for PX4..."
                    print(f"\r[{status}] Alt: {pos[2]:.2f}m | Motors: {self.motors.round(2)} ", end="", flush=True)
                    last_print = time.time()
                
                # Maintain ~250Hz
                elapsed = time.time() - step_start
                if elapsed < 0.004:
                    time.sleep(0.004 - elapsed)
        
        print("\n\n[*] Viewer closed")
        if self.conn:
            self.conn.close()
        if self.server:
            self.server.close()


def main():
    print("="*50)
    print("MuJoCo + PX4 SITL Visualization")
    print("="*50)
    
    sim = MuJoCoSITLViz()
    sim.run()


if __name__ == "__main__":
    main()
