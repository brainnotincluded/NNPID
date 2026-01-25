#!/usr/bin/env python3
"""Quick test of ArduPilot JSON interface"""
import socket
import struct
import time
from pymavlink import mavutil

print("=== Testing ArduPilot JSON Interface ===")

# 1. Connect MAVLink to wake up ArduPilot
print("\n[1] Connecting MAVLink to port 5760...")
try:
    mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760', autoreconnect=True)
    hb = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=15)
    if hb:
        print(f"    Connected! System {mav.target_system}")
    else:
        print("    No heartbeat!")
except Exception as e:
    print(f"    Error: {e}")

# 2. Set up UDP socket for JSON
print("\n[2] Setting up UDP socket on port 9002...")
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 9002))
sock.setblocking(False)
print("    Done!")

# 3. Listen for PWM packets
print("\n[3] Listening for PWM packets (5 seconds)...")
packets = 0
pwm_addr = None
start = time.time()

while time.time() - start < 5:
    try:
        data, addr = sock.recvfrom(1024)
        if len(data) >= 8:
            magic = struct.unpack('<H', data[:2])[0]
            if magic == 18458:
                packets += 1
                pwm_addr = addr
                if packets == 1:
                    print(f"    First PWM packet from {addr}!")
                    pwm_values = struct.unpack('<16H', data[8:40])
                    print(f"    PWM values: {pwm_values[:4]}")
    except BlockingIOError:
        pass
    time.sleep(0.001)

print(f"\n    Received {packets} PWM packets")

if pwm_addr:
    print("\n[4] *** SUCCESS! ArduPilot JSON interface is working! ***")
else:
    print("\n[4] No PWM packets received")

sock.close()
