#!/usr/bin/env python3
"""MuJoCo + PX4 SITL Takeoff Demo."""

import socket
import numpy as np
import time
import mujoco
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

print("="*60)
print("    MUJOCO + PX4 SITL TAKEOFF")
print("="*60)

# MuJoCo model
xml = """
<mujoco model="quadrotor">
    <option gravity="0 0 -9.81" timestep="0.002"/>
    <worldbody>
        <body name="drone" pos="0 0 0.5">
            <freejoint/>
            <inertial pos="0 0 0" mass="1.5" diaginertia="0.01 0.01 0.02"/>
            <geom type="box" size="0.15 0.15 0.03"/>
            <site name="m1" pos="0.15 0.15 0"/>
            <site name="m2" pos="-0.15 0.15 0"/>
            <site name="m3" pos="-0.15 -0.15 0"/>
            <site name="m4" pos="0.15 -0.15 0"/>
        </body>
    </worldbody>
    <actuator>
        <general name="t1" site="m1" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general name="t2" site="m2" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general name="t3" site="m3" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general name="t4" site="m4" gear="0 0 15 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
print("[1] MuJoCo ready")

# TCP SERVER - PX4 connects to us
print("[2] Starting server on TCP:4560...")
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 4560))
server.listen(1)
server.settimeout(60)
print("    Waiting for PX4 to connect...")
print("    Run: cd ~/PX4-Autopilot && make px4_sitl none_iris")
sock, addr = server.accept()
print(f"    PX4 connected from {addr}!")
sock.setblocking(False)

mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)

def send(msg):
    try: sock.send(msg.get_msgbuf())
    except: pass

def recv():
    msgs = []
    try:
        for b in sock.recv(4096):
            m = mav.parse_char(bytes([b]))
            if m: msgs.append(m)
    except: pass
    return msgs

motors = np.zeros(4)
ts = 0

# Warm up
print("[3] Warming up...")
for i in range(300):
    ts += 4000
    pos = data.qpos[:3]
    send(mav.hil_sensor_encode(ts, 0,0,-9.81, 0,0,0, 0.21,0,0.42, 101325,0,pos[2],20, 0x1FF))
    if i%25==0: send(mav.hil_gps_encode(ts,3, 473977420,85455940,488000, 10,10,0,0,0,0,0,12))
    if i%100==0: send(mav.heartbeat_encode(6,8,0,0,0))
    for m in recv():
        if m.get_type()=='HIL_ACTUATOR_CONTROLS':
            motors = np.clip([(m.controls[j]+1)/2 for j in range(4)], 0, 1)
    data.ctrl[:] = motors
    mujoco.mj_step(model, data)
    time.sleep(0.002)

# UDP
print("[4] UDP connection...")
mav_udp = mavutil.mavlink_connection('udpin:0.0.0.0:14540', source_system=255)
msg = mav_udp.recv_match(type='HEARTBEAT', blocking=True, timeout=5)
print(f"    System {mav_udp.target_system}" if msg else "    No UDP")

# Arm
print("[5] Arming...")
for _ in range(50):
    ts += 4000
    pos = data.qpos[:3]
    send(mav.hil_sensor_encode(ts, 0,0,-9.81, 0,0,0, 0.21,0,0.42, 101325,0,pos[2],20, 0x1FF))
    mav_udp.mav.set_position_target_local_ned_send(0, mav_udp.target_system, mav_udp.target_component, 1, 0b0000111111111000, 0,0,-5, 0,0,0,0,0,0,0,0)
    for m in recv(): 
        if m.get_type()=='HIL_ACTUATOR_CONTROLS': motors = np.clip([(m.controls[j]+1)/2 for j in range(4)], 0, 1)
    data.ctrl[:] = motors
    mujoco.mj_step(model, data)
    time.sleep(0.01)

mav_udp.mav.command_long_send(mav_udp.target_system, mav_udp.target_component, 176, 0, 1, 6, 0,0,0,0,0)
time.sleep(0.3)
mav_udp.mav.command_long_send(mav_udp.target_system, mav_udp.target_component, 400, 0, 1, 0,0,0,0,0,0)
print("    Armed!")

# FLY
print("\n" + "─"*60)
print("FLYING (PX4 -> MuJoCo)")
print("─"*60)

max_alt = 0
hil = 0

for step in range(1500):
    ts += 4000
    pos = data.qpos[:3]
    vel = data.qvel[:3]
    max_alt = max(max_alt, pos[2])
    
    # Sensors
    send(mav.hil_sensor_encode(ts, 0,0,-9.81, 0,0,0, 0.21,0,0.42, 101325-pos[2]*12,0,pos[2],20, 0x1FF))
    if step%25==0: send(mav.hil_gps_encode(ts,3, 473977420,85455940, int((488+pos[2])*1000), 10,10,0,0,0,0,0,12))
    
    # Setpoint
    mav_udp.mav.set_position_target_local_ned_send(0, mav_udp.target_system, mav_udp.target_component, 1, 0b0000111111111000, 0,0,-5, 0,0,0,0,0,0,0,0)
    
    # Motors from PX4
    for m in recv():
        if m.get_type()=='HIL_ACTUATOR_CONTROLS':
            hil += 1
            motors = np.clip([(m.controls[j]+1)/2 for j in range(4)], 0, 1)
    
    # MuJoCo physics
    data.ctrl[:] = motors
    mujoco.mj_step(model, data)
    
    if step % 150 == 0:
        bar = "█" * min(int(pos[2]*5), 25) + "░" * max(0, 25 - int(pos[2]*5))
        print(f"[{bar}] {pos[2]:5.2f}m | motors={motors.round(2)} | hil={hil}")
    
    if pos[2] > 4.5:
        print(f"\n*** REACHED {pos[2]:.2f}m ***")
        break
    
    time.sleep(0.002)

print(f"\n{'='*60}")
print(f"Max altitude: {max_alt:.2f}m")
print(f"HIL messages: {hil}")
if hil > 0 and max_alt > 1: print("✅ PX4 SITL controlled MuJoCo drone!")
sock.close()
