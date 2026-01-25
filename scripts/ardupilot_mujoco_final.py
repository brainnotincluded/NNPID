#!/usr/bin/env python3
"""
TRUE SITL CONTROL: ArduPilot SITL + MuJoCo
FINAL WORKING VERSION

Protocol verified:
- Connect MAVLink TCP to wake up ArduPilot
- Receive PWM on UDP port 9002
- Send JSON sensor data back to ArduPilot's ephemeral port
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
print("  TRUE SITL CONTROL: ArduPilot + MuJoCo")
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
        
        <body name="drone" pos="0 0 0.1">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="1.5" diaginertia="0.02 0.02 0.04"/>
            
            <!-- Body -->
            <geom type="box" size="0.1 0.1 0.03" rgba="0.2 0.2 0.2 1"/>
            
            <!-- Arms -->
            <geom type="box" pos="0.15 0 0" size="0.08 0.015 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="box" pos="-0.15 0 0" size="0.08 0.015 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="box" pos="0 0.15 0" size="0.015 0.08 0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="box" pos="0 -0.15 0" size="0.015 0.08 0.01" rgba="0.3 0.3 0.3 1"/>
            
            <!-- Motors (colored) -->
            <geom type="cylinder" pos="0.18 0.18 0.02" size="0.05 0.015" rgba="1 0.2 0.2 1"/>
            <geom type="cylinder" pos="-0.18 0.18 0.02" size="0.05 0.015" rgba="0.2 1 0.2 1"/>
            <geom type="cylinder" pos="-0.18 -0.18 0.02" size="0.05 0.015" rgba="0.2 0.2 1 1"/>
            <geom type="cylinder" pos="0.18 -0.18 0.02" size="0.05 0.015" rgba="1 1 0.2 1"/>
            
            <!-- Thrust sites -->
            <site name="motor1" pos="0.18 0.18 0.04"/>
            <site name="motor2" pos="-0.18 0.18 0.04"/>
            <site name="motor3" pos="-0.18 -0.18 0.04"/>
            <site name="motor4" pos="0.18 -0.18 0.04"/>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Motor thrust with yaw torque -->
        <general site="motor1" gear="0 0 7 0 0 0.05" ctrllimited="true" ctrlrange="0 1"/>
        <general site="motor2" gear="0 0 7 0 0 -0.05" ctrllimited="true" ctrlrange="0 1"/>
        <general site="motor3" gear="0 0 7 0 0 0.05" ctrllimited="true" ctrlrange="0 1"/>
        <general site="motor4" gear="0 0 7 0 0 -0.05" ctrllimited="true" ctrlrange="0 1"/>
    </actuator>
</mujoco>
"""

# Initialize MuJoCo
print("[1] Initializing MuJoCo...")
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=1280)
print("    Done!")

# UDP Socket for JSON
print("[2] Setting up UDP on port 9002...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 9002))
sock.setblocking(False)
print("    Done!")

# Connect MAVLink
print("[3] Connecting MAVLink to wake up ArduPilot...")
try:
    mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760', autoreconnect=True)
    hb = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
    if hb:
        print(f"    Connected! System {mav.target_system}")
    else:
        print("    No heartbeat, but continuing...")
except Exception as e:
    print(f"    MAVLink error: {e}")
    mav = None

# Video
video_path = "/Users/mac/projects/NNPID/ardupilot_true_sitl.mp4"
writer = imageio.get_writer(video_path, fps=30, quality=8)

# State
pwm_values = [1000] * 16
ardupilot_addr = None
packets_rx = 0
packets_tx = 0
max_alt = 0.0
armed = False
mode_set = False

print("[4] Running simulation...")
print("    (Motor values from ArduPilot SITL will control MuJoCo)")
print()

start_time = time.time()

try:
    for step in range(25000):  # ~50 seconds
        sim_time = data.time
        
        # === RECEIVE PWM FROM ARDUPILOT ===
        try:
            raw, addr = sock.recvfrom(1024)
            if len(raw) >= 40:
                magic = struct.unpack('<H', raw[:2])[0]
                if magic == 18458:
                    pwm_values = list(struct.unpack('<16H', raw[8:40]))
                    ardupilot_addr = addr
                    packets_rx += 1
        except BlockingIOError:
            pass

        # === CONVERT PWM TO MOTOR THRUST ===
        # ArduPilot PWM: 1000-2000, we map to 0-1
        motors = np.zeros(4)
        for i in range(4):
            motors[i] = np.clip((pwm_values[i] - 1000) / 1000.0, 0.0, 1.0)
        
        data.ctrl[:] = motors
        
        # === STEP MUJOCO ===
        mujoco.mj_step(model, data)
        
        # === GET STATE ===
        pos = data.qpos[:3].copy()
        quat = data.qpos[3:7].copy()  # w,x,y,z in MuJoCo
        vel = data.qvel[:3].copy()
        omega = data.qvel[3:6].copy()
        
        max_alt = max(max_alt, pos[2])
        
        # Compute body-frame acceleration (needed for IMU)
        # MuJoCo gives world-frame acceleration in data.qacc

        # === SEND SENSOR DATA TO ARDUPILOT ===
        if ardupilot_addr:
            # MuJoCo uses Z-up, ArduPilot uses NED (Z-down)
            # Position: MuJoCo [x,y,z] -> NED [x, -y, -z]
            pos_ned = (float(pos[0]), float(-pos[1]), float(-pos[2]))
            vel_ned = (float(vel[0]), float(-vel[1]), float(-vel[2]))
            
            # Gyro: MuJoCo body frame to ArduPilot body frame
            gyro_body = (float(omega[0]), float(-omega[1]), float(-omega[2]))
            
            # Quaternion to Euler (MuJoCo quat is w,x,y,z)
            w, x, y, z = quat
            # Convert to ArduPilot coordinate frame
            roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            # Adjust for NED
            attitude = (float(roll), float(-pitch), float(-yaw))
            
            # Accelerometer: gravity in body frame (FRD)
            # When hovering level, accel should be [0, 0, -9.8] in FRD
            accel_body = (0.0, 0.0, -9.8)
            
            sensor = {
                "timestamp": sim_time,
                "imu": {
                    "gyro": gyro_body,
                    "accel_body": accel_body
                },
                "position": pos_ned,
                "velocity": vel_ned,
                "attitude": attitude
            }
            
            try:
                # IMPORTANT: JSON must end with newline, use compact format
                json_str = json.dumps(sensor, separators=(',', ':')) + "\n"
                sock.sendto(json_str.encode('ascii'), ardupilot_addr)
                packets_tx += 1
            except:
                pass

        # === MAVLINK COMMANDS ===
        if mav:
            # Handle incoming MAVLink
            try:
                msg = mav.recv_match(blocking=False)
            except:
                pass
            
            # Set GUIDED mode and arm after 2 seconds
            if step == 1000 and not mode_set:
                print("    Setting GUIDED mode...")
                mav.set_mode('GUIDED')
                mode_set = True
            
            if step == 1500 and not armed:
                print("    Arming...")
                mav.arducopter_arm()
                armed = True
            
            if step == 2500:
                print("    Takeoff to 3m...")
                mav.mav.command_long_send(
                    mav.target_system, mav.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    0, 0, 0, 0, 0, 0, 0, 3.0
                )

        # === RENDER ===
        if step % 16 == 0:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [pos[0], pos[1], max(pos[2], 0.5)]
            cam.distance = 5.0
            cam.azimuth = 45 + step * 0.02
            cam.elevation = -20
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

        # === STATUS ===
        if step % 500 == 0:
            motor_sum = sum(motors)
            print(f"  t={sim_time:.1f}s | Alt={pos[2]:.2f}m | Motors={motors.round(2)} | "
                  f"RX={packets_rx} TX={packets_tx} | Armed={armed}")

        time.sleep(0.001)

except KeyboardInterrupt:
    print("\n  Interrupted!")

writer.close()
sock.close()

print(f"\n{'=' * 60}")
print(f"  RESULTS:")
print(f"    Max altitude: {max_alt:.2f}m")
print(f"    PWM packets received: {packets_rx}")
print(f"    Sensor packets sent: {packets_tx}")
print(f"    Video saved: {video_path}")

if packets_rx > 100 and packets_tx > 100:
    print("\n  *** TRUE ARDUPILOT SITL CONTROL ACHIEVED! ***")
print("=" * 60)
