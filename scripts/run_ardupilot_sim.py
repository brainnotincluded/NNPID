#!/usr/bin/env python3
"""
WORKING ArduPilot SITL + MuJoCo
Based on ArduPilot's pybullet example - continuous tight loop.
"""

import socket
import struct
import json
import time
import numpy as np
import mujoco
import imageio
from pymavlink import mavutil

print("=" * 60)
print("  ArduPilot SITL + MuJoCo - Working Version")
print("=" * 60)

# MuJoCo XML
XML = """
<mujoco>
    <visual><global offwidth="1280" offheight="720"/></visual>
    <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
    <worldbody>
        <light pos="0 0 10"/>
        <geom type="plane" size="20 20 0.1" rgba="0.3 0.4 0.3 1"/>
        <body name="drone" pos="0 0 0.1">
            <freejoint/>
            <inertial pos="0 0 0" mass="1.5" diaginertia="0.02 0.02 0.04"/>
            <geom type="box" size="0.15 0.15 0.04" rgba="0.3 0.3 0.3 1"/>
            <site name="m1" pos="0.15 0.15 0.03"/>
            <site name="m2" pos="-0.15 0.15 0.03"/>
            <site name="m3" pos="-0.15 -0.15 0.03"/>
            <site name="m4" pos="0.15 -0.15 0.03"/>
        </body>
    </worldbody>
    <actuator>
        <general site="m1" gear="0 0 6 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general site="m2" gear="0 0 6 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general site="m3" gear="0 0 6 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general site="m4" gear="0 0 6 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=1280)
print("[1] MuJoCo ready")

# UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', 9002))
sock.settimeout(0.1)
print("[2] UDP ready on port 9002")

# Video
video_path = "/Users/mac/projects/NNPID/ardupilot_working.mp4"
writer = imageio.get_writer(video_path, fps=30, quality=8)
print(f"[3] Video: {video_path}")

# State
sim_time = 0.0
frame_count = 0
pwm_packets = 0
last_frame = -1
max_alt = 0.0
motors_nonzero = 0

# Connect MAVLink to wake up ArduPilot
print("[4] Connecting MAVLink on port 5760...")
try:
    mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760', autoreconnect=True)
    hb = mav.wait_heartbeat(timeout=10)
    if hb:
        print(f"    Connected! System {mav.target_system}")
    else:
        print("    No heartbeat, but continuing...")
except Exception as e:
    print(f"    MAVLink error: {e}")
    mav = None

print("\n[5] Running - waiting for JSON packets...")

start_time = time.time()
connected = False

try:
    while time.time() - start_time < 40:
        try:
            raw, addr = sock.recvfrom(100)
        except socket.timeout:
            time.sleep(0.01)
            continue

        if len(raw) < 40:
            continue

        # Parse PWM packet
        magic, frame_rate, frame_num = struct.unpack('<HHI', raw[:8])
        if magic != 18458:
            continue

        pwm = struct.unpack('<16H', raw[8:40])
        pwm_packets += 1

        if not connected:
            print(f"    Connected to {addr}!")
            connected = True

        # Reset on new session
        if frame_num < last_frame:
            mujoco.mj_resetData(model, data)
            sim_time = 0.0
            print("    Reset simulation")
        last_frame = frame_num

        # Convert PWM to motor thrust
        motors = np.array([max(0, (p - 1000) / 1000.0) for p in pwm[:4]])
        if np.sum(motors) > 0.1:
            motors_nonzero += 1
        
        data.ctrl[:] = motors

        # Step physics (multiple sub-steps for stability)
        for _ in range(2):
            mujoco.mj_step(model, data)
        sim_time = data.time

        # Get state
        pos = data.qpos[:3]
        vel = data.qvel[:3]
        omega = data.qvel[3:6]
        quat = data.qpos[3:7]  # w,x,y,z

        max_alt = max(max_alt, pos[2])

        # Quaternion to euler
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        # Convert to NED
        pos_ned = (float(pos[0]), float(-pos[1]), float(-pos[2]))
        vel_ned = (float(vel[0]), float(-vel[1]), float(-vel[2]))
        gyro = (float(omega[0]), float(-omega[1]), float(-omega[2]))
        attitude = (float(roll), float(-pitch), float(-yaw))

        # Sensor packet
        sensor = {
            "timestamp": sim_time,
            "imu": {
                "gyro": gyro,
                "accel_body": (0.0, 0.0, -9.8)
            },
            "position": pos_ned,
            "velocity": vel_ned,
            "attitude": attitude
        }

        # Send immediately
        sock.sendto((json.dumps(sensor, separators=(',', ':')) + "\n").encode(), addr)

        # Render video
        if pwm_packets % 16 == 0:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [pos[0], pos[1], max(pos[2], 0.5)]
            cam.distance = 4
            cam.azimuth = 30 + pwm_packets * 0.02
            cam.elevation = -15
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            frame_count += 1

        # Status
        if pwm_packets % 500 == 0:
            print(f"  t={sim_time:.1f}s Alt={pos[2]:.2f}m PWM={pwm[:4]} Motors={motors.round(2)}")

except KeyboardInterrupt:
    print("\n  Stopped")

writer.close()
sock.close()

print(f"\n{'=' * 60}")
print(f"  Results:")
print(f"    PWM packets: {pwm_packets}")
print(f"    Motors non-zero: {motors_nonzero} times")
print(f"    Max altitude: {max_alt:.2f}m")
print(f"    Video frames: {frame_count}")
print(f"    Video: {video_path}")
print("=" * 60)
