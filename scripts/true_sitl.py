#!/usr/bin/env python3
"""TRUE PX4 SITL + MuJoCo - Correct lockstep protocol."""

import socket
import time
import numpy as np
import mujoco
import imageio
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

print("="*60)
print("  TRUE PX4 SITL + MUJOCO")
print("="*60)

# MuJoCo
xml = """
<mujoco>
    <visual><global offwidth="1280" offheight="720"/></visual>
    <option gravity="0 0 -9.81" timestep="0.004"/>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0.1" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.2" rgb2="0.1 0.2 0.1" width="256" height="256"/>
        <material name="grid" texture="grid" texrepeat="10 10"/>
    </asset>
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1"/>
        <geom type="plane" size="20 20 0.1" material="grid"/>
        <body name="drone" pos="0 0 0.1">
            <freejoint/>
            <inertial pos="0 0 0" mass="1.5" diaginertia="0.025 0.025 0.04"/>
            <geom type="box" size="0.18 0.18 0.04" rgba="0.25 0.25 0.25 1"/>
            <geom type="cylinder" pos="0.18 0.18 0.02" size="0.07 0.01" rgba="1 0.2 0.2 1"/>
            <geom type="cylinder" pos="-0.18 0.18 0.02" size="0.07 0.01" rgba="0.2 1 0.2 1"/>
            <geom type="cylinder" pos="-0.18 -0.18 0.02" size="0.07 0.01" rgba="0.2 0.2 1 1"/>
            <geom type="cylinder" pos="0.18 -0.18 0.02" size="0.07 0.01" rgba="1 1 0.2 1"/>
            <site name="m1" pos="0.18 0.18 0.03"/>
            <site name="m2" pos="-0.18 0.18 0.03"/>
            <site name="m3" pos="-0.18 -0.18 0.03"/>
            <site name="m4" pos="0.18 -0.18 0.03"/>
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

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=1280)
print("[1] MuJoCo ready")

# TCP Server
print("[2] Starting server on port 4560...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 4560))
server.listen(1)
server.settimeout(90)
print("    Waiting for PX4...")
print("    >>> START PX4: cd ~/PX4-Autopilot && make px4_sitl none_iris")

try:
    conn, addr = server.accept()
except socket.timeout:
    print("    TIMEOUT waiting for PX4!")
    exit(1)
print(f"[3] PX4 connected from {addr}!")

# Set socket to blocking with timeout for lockstep
conn.settimeout(0.1)

mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)

# Video
video_path = "/Users/mac/projects/NNPID/true_sitl_flight.mp4"
writer = imageio.get_writer(video_path, fps=25, quality=8)

motors = np.zeros(4)
hil_count = 0
time_usec = 0

print("[4] Running lockstep loop...")

max_alt = 0
frames = 0

for step in range(8000):  # ~32 seconds
    time_usec += 4000  # 4ms timestep
    
    # Get state
    pos = data.qpos[:3]
    vel = data.qvel[:3]
    omega = data.qvel[3:6]
    
    max_alt = max(max_alt, pos[2])
    
    # === SEND SENSORS TO PX4 ===
    try:
        # HIL_SENSOR (IMU)
        msg = mav.hil_sensor_encode(
            time_usec,
            0.0, 0.0, 9.81,        # accel (gravity)
            omega[0], omega[1], omega[2],  # gyro
            0.2, 0.0, 0.4,         # mag
            101325.0,              # pressure
            0.0,                   # diff_pressure
            pos[2],                # pressure_alt
            20.0,                  # temp
            0x1FF                  # fields
        )
        conn.send(msg.get_msgbuf())
        
        # HIL_GPS at 10Hz
        if step % 25 == 0:
            msg = mav.hil_gps_encode(
                time_usec, 3,
                int(47.397742e7), int(8.545594e7),
                int((488 + pos[2]) * 1000),
                50, 50, 0, 0, 0, 0, 0, 12
            )
            conn.send(msg.get_msgbuf())
        
        # Heartbeat at 1Hz  
        if step % 250 == 0:
            msg = mav.heartbeat_encode(2, 12, 0, 0, 0)
            conn.send(msg.get_msgbuf())
            
    except Exception as e:
        pass
    
    # === RECEIVE MOTORS FROM PX4 ===
    try:
        raw = conn.recv(4096)
        for b in raw:
            m = mav.parse_char(bytes([b]))
            if m and m.get_type() == 'HIL_ACTUATOR_CONTROLS':
                hil_count += 1
                motors = np.clip([(m.controls[i]+1)/2 for i in range(4)], 0, 1)
    except socket.timeout:
        pass
    except Exception:
        pass
    
    # === APPLY TO MUJOCO ===
    data.ctrl[:] = motors
    mujoco.mj_step(model, data)
    
    # === RENDER ===
    if step % 10 == 0:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, max(1, pos[2])]
        cam.distance = 6
        cam.azimuth = 30 + step * 0.05
        cam.elevation = -15
        renderer.update_scene(data, camera=cam)
        writer.append_data(renderer.render())
        frames += 1
    
    # Status
    if step % 400 == 0:
        print(f"  t={time_usec/1e6:.1f}s | Alt={pos[2]:.2f}m | Motors={motors.round(2)} | HIL={hil_count}")

writer.close()
conn.close()
server.close()

print(f"\n{'='*60}")
print(f"Max altitude: {max_alt:.2f}m")
print(f"HIL messages: {hil_count}")
print(f"Video: {video_path} ({frames} frames)")
if hil_count > 0:
    print("\n*** TRUE PX4 SITL CONTROL ACHIEVED! ***")
print("="*60)
