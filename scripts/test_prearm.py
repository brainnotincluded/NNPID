#!/usr/bin/env python3
"""Test ArduPilot pre-arm and arming"""
import socket
import struct
import json
import time
from pymavlink import mavutil

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 9002))
sock.settimeout(0.1)

mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
mav.wait_heartbeat(timeout=5)
print(f"MAVLink: system {mav.target_system}")

t = 0.0
ap_addr = None

def pump():
    global t, ap_addr
    try:
        data, addr = sock.recvfrom(100)
        if len(data) >= 40:
            magic = struct.unpack('<H', data[:2])[0]
            if magic == 18458:
                ap_addr = addr
                sensor = {"timestamp": t, "imu": {"gyro": [0,0,0], "accel_body": [0,0,-9.8]},
                         "position": [0,0,0], "velocity": [0,0,0], "attitude": [0,0,0]}
                sock.sendto((json.dumps(sensor, separators=(',', ':')) + "\n").encode(), addr)
                t += 0.002
                return struct.unpack('<16H', data[8:40])[:4]
    except:
        pass
    return None

# Initial pump
print("Establishing communication...")
for _ in range(300):
    pump()
print(f"Sent {int(t/0.002)} packets")

# Check status
print("\nChecking status messages...")
for _ in range(200):
    pump()
    msg = mav.recv_match(blocking=False)
    if msg and msg.get_type() == 'STATUSTEXT':
        print(f"  STATUS: {msg.text}")

# Disable pre-arm checks
print("\nDisabling ARMING_CHECK...")
mav.mav.param_set_send(mav.target_system, mav.target_component,
    b'ARMING_CHECK', 0, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
time.sleep(0.5)

for _ in range(100):
    pump()

# Set GUIDED
print("\nSetting GUIDED mode...")
mav.set_mode('GUIDED')
time.sleep(0.5)

for _ in range(100):
    pump()

# Check mode
msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
if msg:
    from pymavlink.mavutil import mode_string_v10
    print(f"Current mode: {mode_string_v10(msg)}")

# ARM
print("\nArming...")
mav.arducopter_arm()
time.sleep(1)

# Check messages and PWM
print("\nChecking response...")
for i in range(200):
    pwm = pump()
    if pwm and i % 40 == 0:
        print(f"PWM: {pwm}")
    msg = mav.recv_match(blocking=False)
    if msg:
        if msg.get_type() == 'STATUSTEXT':
            print(f"STATUS: {msg.text}")
        elif msg.get_type() == 'COMMAND_ACK':
            print(f"ACK: cmd={msg.command} result={msg.result}")

sock.close()
print("\nDone")
