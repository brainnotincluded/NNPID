#!/usr/bin/env python3
"""Test ArduPilot arming with MuJoCo sensor data"""
import socket
import struct
import json
import time
from pymavlink import mavutil

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 9002))
sock.settimeout(0.1)

# First establish sensor communication
print("Establishing sensor communication...")
ap_addr = None
t = 0.0

for i in range(100):
    try:
        data, addr = sock.recvfrom(100)
        if len(data) >= 8:
            magic = struct.unpack('<H', data[:2])[0]
            if magic == 18458:
                ap_addr = addr
                sensor = {"timestamp": t, "imu": {"gyro": [0,0,0], "accel_body": [0,0,-9.8]},
                         "position": [0,0,0], "velocity": [0,0,0], "attitude": [0,0,0]}
                sock.sendto((json.dumps(sensor, separators=(',', ':')) + "\n").encode(), addr)
                t += 0.002
                if i == 0:
                    print(f"Sensor comm established with {addr}")
    except socket.timeout:
        pass
    time.sleep(0.01)

print(f"Sent {int(t/0.002)} sensor packets")

# Now connect MAVLink
print("\nConnecting MAVLink...")
mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
print("Waiting for heartbeat...")
msg = mav.wait_heartbeat(blocking=True, timeout=10)
if msg:
    print(f"Got heartbeat from system {mav.target_system}")
else:
    print("No heartbeat - continuing anyway")
    mav.target_system = 1
    mav.target_component = 1

t = 0.0
ap_addr = None

def send_sensor(t, addr):
    sensor = {
        "timestamp": t,
        "imu": {"gyro": [0, 0, 0], "accel_body": [0, 0, -9.8]},
        "position": [0, 0, 0],
        "velocity": [0, 0, 0],
        "attitude": [0, 0, 0]
    }
    sock.sendto((json.dumps(sensor, separators=(',', ':')) + "\n").encode(), addr)

def recv_pwm():
    global ap_addr, t
    try:
        data, addr = sock.recvfrom(100)
        if len(data) >= 40:
            magic = struct.unpack('<H', data[:2])[0]
            if magic == 18458:
                pwm = struct.unpack('<16H', data[8:40])
                ap_addr = addr
                send_sensor(t, addr)
                t += 0.002
                return pwm[:4]
    except socket.timeout:
        pass
    return None

# Initial sensor data
print("\nSending initial sensor data...")
for i in range(300):
    pwm = recv_pwm()
    if pwm and i % 100 == 0:
        print(f"[{i}] PWM: {pwm}")

# Set GUIDED mode
print("\nSetting GUIDED mode...")
mav.set_mode('GUIDED')
time.sleep(0.5)

# Wait and check
for i in range(100):
    pwm = recv_pwm()

# Check messages
while True:
    msg = mav.recv_match(blocking=False)
    if not msg:
        break
    if msg.get_type() == 'STATUSTEXT':
        print(f"STATUS: {msg.text}")

# Force ARM
print("\nForce arming...")
mav.mav.command_long_send(
    mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0  # force arm
)

# Check ACK
for i in range(50):
    msg = mav.recv_match(blocking=False)
    if msg and msg.get_type() == 'COMMAND_ACK':
        print(f"ARM ACK: result={msg.result}")
        break
    pwm = recv_pwm()
    time.sleep(0.02)

time.sleep(1)

# Check PWM after arming
print("\nPWM after arming:")
for i in range(200):
    pwm = recv_pwm()
    if pwm and i % 50 == 0:
        print(f"PWM: {pwm}")

# Takeoff
print("\nSending takeoff...")
mav.mav.command_long_send(
    mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, 3  # 3m
)

# Final check
print("\nFinal PWM check:")
for i in range(300):
    pwm = recv_pwm()
    if pwm and i % 50 == 0:
        print(f"PWM: {pwm}")

sock.close()
print("\nDone")
