#!/usr/bin/env python3
"""Debug ArduPilot JSON interface"""
import socket
import struct
import json
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('0.0.0.0', 9002))
sock.settimeout(0.1)

print("Waiting for ArduPilot PWM packets...")
ap_addr = None
rx_count = 0
tx_count = 0
start = time.time()

while time.time() - start < 10:
    try:
        data, addr = sock.recvfrom(1024)
        if len(data) >= 8:
            magic = struct.unpack('<H', data[:2])[0]
            if magic == 18458:
                ap_addr = addr
                rx_count += 1
                
                # Send sensor JSON back immediately
                sensor = {
                    "timestamp": time.time() - start,
                    "imu": {
                        "gyro": [0.0, 0.0, 0.0],
                        "accel_body": [0.0, 0.0, -9.8]
                    },
                    "position": [0.0, 0.0, 0.0],
                    "velocity": [0.0, 0.0, 0.0],
                    "attitude": [0.0, 0.0, 0.0]
                }
                
                json_bytes = json.dumps(sensor).encode('ascii')
                sock.sendto(json_bytes, addr)
                tx_count += 1
                
                if rx_count == 1:
                    print(f"First PWM from {addr}")
                    print(f"Sending JSON ({len(json_bytes)} bytes): {json_bytes[:80]}...")
                
    except socket.timeout:
        pass

print(f"\nRX: {rx_count}, TX: {tx_count}")
sock.close()
