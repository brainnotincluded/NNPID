#!/usr/bin/env python3
"""
TRUE SITL CONTROL: ArduPilot SITL + MuJoCo with MAVLink

ArduPilot needs a MAVLink connection (GCS) to start.
We connect via pymavlink and also handle the JSON sensor interface.
"""

import socket
import struct
import json
import time
import threading
import numpy as np
import mujoco
import imageio
from pymavlink import mavutil

print("=" * 60)
print("  ARDUPILOT SITL + MUJOCO - TRUE SITL CONTROL")
print("=" * 60)

# MuJoCo drone model
XML = """
<mujoco model="ardupilot_quad">
    <visual><global offwidth="1280" offheight="720"/></visual>
    <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.1 0.2" width="512" height="512"/>
        <texture name="ground" type="2d" builtin="checker" rgb1="0.2 0.3 0.2" rgb2="0.15 0.25 0.15" width="256" height="256"/>
        <material name="ground" texture="ground" texrepeat="20 20"/>
    </asset>
    
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="1 1 1"/>
        <geom type="plane" size="50 50 0.1" material="ground"/>
        
        <body name="drone" pos="0 0 0.15">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="1.5" diaginertia="0.02 0.02 0.04"/>
            <geom type="box" size="0.1 0.1 0.03" rgba="0.2 0.2 0.2 1"/>
            <geom type="box" pos="0.15 0 0" size="0.08 0.015 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="box" pos="-0.15 0 0" size="0.08 0.015 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="box" pos="0 0.15 0" size="0.015 0.08 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="box" pos="0 -0.15 0" size="0.015 0.08 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="cylinder" pos="0.18 0.18 0.02" size="0.05 0.015" rgba="1 0.2 0.2 1"/>
            <geom type="cylinder" pos="-0.18 0.18 0.02" size="0.05 0.015" rgba="0.2 1 0.2 1"/>
            <geom type="cylinder" pos="-0.18 -0.18 0.02" size="0.05 0.015" rgba="0.2 0.2 1 1"/>
            <geom type="cylinder" pos="0.18 -0.18 0.02" size="0.05 0.015" rgba="1 1 0.2 1"/>
            <site name="motor1" pos="0.18 0.18 0.04"/>
            <site name="motor2" pos="-0.18 0.18 0.04"/>
            <site name="motor3" pos="-0.18 -0.18 0.04"/>
            <site name="motor4" pos="0.18 -0.18 0.04"/>
        </body>
    </worldbody>
    
    <actuator>
        <general site="motor1" gear="0 0 7 0 0 0.1" ctrllimited="true" ctrlrange="0 1"/>
        <general site="motor2" gear="0 0 7 0 0 -0.1" ctrllimited="true" ctrlrange="0 1"/>
        <general site="motor3" gear="0 0 7 0 0 0.1" ctrllimited="true" ctrlrange="0 1"/>
        <general site="motor4" gear="0 0 7 0 0 -0.1" ctrllimited="true" ctrlrange="0 1"/>
    </actuator>
</mujoco>
"""

# Initialize MuJoCo
print("[1] Initializing MuJoCo...")
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=1280)
print("    Done!")

# JSON UDP Socket for servo data
print("[2] Setting up UDP socket on port 9002...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 9002))
sock.setblocking(False)
print("    Done!")

# Connect MAVLink
print("[3] Connecting MAVLink to ArduPilot on port 5760...")
try:
    mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760', autoreconnect=True)
    print("    Waiting for heartbeat...")
    hb = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=30)
    if hb:
        print(f"    Connected! System {mav.target_system}, Component {mav.target_component}")
    else:
        print("    No heartbeat received, continuing anyway...")
        mav = None
except Exception as e:
    print(f"    MAVLink connection failed: {e}")
    mav = None

# Video
video_path = "/Users/mac/projects/NNPID/ardupilot_sitl_flight.mp4"
writer = imageio.get_writer(video_path, fps=30, quality=8)

# State
pwm_values = [1000] * 16
ardupilot_addr = None
packets_received = 0
frame_count = 0
sim_time = 0.0
max_alt = 0.0
armed = False

print("[4] Running simulation loop...")
start_time = time.time()

try:
    for step in range(30000):  # ~60 seconds at 500Hz
        # === RECEIVE PWM FROM ARDUPILOT ===
        try:
            raw_data, addr = sock.recvfrom(1024)
            
            if packets_received == 0:
                print(f"    First PWM packet from {addr}, len={len(raw_data)}")
            
            if len(raw_data) >= 40:
                magic, frame_rate, frame_count_ap = struct.unpack('<HHI', raw_data[0:8])
                if magic == 18458:
                    pwm_values = list(struct.unpack('<16H', raw_data[8:40]))
                    ardupilot_addr = addr
                    packets_received += 1
        except BlockingIOError:
            pass

        # === CONVERT PWM TO MOTOR THRUST ===
        motors = np.zeros(4)
        for i in range(4):
            motors[i] = np.clip((pwm_values[i] - 1000) / 1000.0, 0.0, 1.0)
        
        data.ctrl[:] = motors
        
        # === STEP MUJOCO PHYSICS ===
        mujoco.mj_step(model, data)
        sim_time = data.time

        # === GET STATE ===
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()
        vel = data.qvel[:3].copy()
        omega = data.qvel[3:6].copy()
        
        max_alt = max(max_alt, pos[2])

        # === SEND SENSOR DATA BACK TO ARDUPILOT ===
        if ardupilot_addr:
            pos_ned = [float(pos[0]), float(pos[1]), float(-pos[2])]
            vel_ned = [float(vel[0]), float(vel[1]), float(-vel[2])]
            
            w, x, y, z = quat
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            
            sensor_data = {
                "timestamp": sim_time,
                "imu": {
                    "gyro": [float(omega[0]), float(omega[1]), float(-omega[2])],
                    "accel_body": [0.0, 0.0, -9.81]
                },
                "position": pos_ned,
                "velocity": vel_ned,
                "attitude": [float(roll), float(pitch), float(yaw)]
            }
            
            try:
                sock.sendto(json.dumps(sensor_data).encode(), ardupilot_addr)
            except:
                pass

        # === MAVLINK COMMANDS ===
        if mav and step == 5000 and not armed:
            print("    Arming drone via MAVLink...")
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            armed = True
        
        if mav and step == 6000:
            print("    Sending TAKEOFF command...")
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, 3.0  # 3m altitude
            )

        # === RENDER VIDEO ===
        if step % 16 == 0:  # ~30fps
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [pos[0], pos[1], max(pos[2], 0.5)]
            cam.distance = 5.0
            cam.azimuth = 45 + step * 0.02
            cam.elevation = -20
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            frame_count += 1

        # === STATUS ===
        if step % 500 == 0:
            elapsed = time.time() - start_time
            print(f"  t={sim_time:.1f}s | Alt={pos[2]:.2f}m | Motors={motors.round(2)} | PWM={packets_received}")

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n  Interrupted!")

writer.close()
sock.close()

print(f"\n{'=' * 60}")
print(f"  RESULTS:")
print(f"    Simulation time: {sim_time:.1f}s")
print(f"    Max altitude: {max_alt:.2f}m")
print(f"    PWM packets received: {packets_received}")
print(f"    Video: {video_path}")

if packets_received > 100:
    print("\n  *** TRUE ARDUPILOT SITL CONTROL ACHIEVED! ***")
print("=" * 60)
