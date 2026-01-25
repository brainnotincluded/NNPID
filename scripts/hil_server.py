#!/usr/bin/env python3
"""HIL Server - Simulates drone physics and connects to PX4 SITL."""

import socket
import time
import numpy as np
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mavlink

def main():
    print("="*50)
    print("HIL SIMULATOR SERVER")
    print("="*50)
    
    # Start TCP server
    print("\n[1] Starting server on port 4560...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 4560))
    server.listen(1)
    server.settimeout(120)
    
    print("    Waiting for PX4 to connect...")
    print("    Run: cd ~/PX4-Autopilot && make px4_sitl none_iris")
    
    try:
        conn, addr = server.accept()
    except socket.timeout:
        print("    Timeout waiting for PX4!")
        return
    
    print(f"    PX4 connected from {addr}!")
    conn.setblocking(False)
    
    # MAVLink encoder
    mav = mavutil.mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
    
    def send(msg):
        try:
            conn.send(msg.get_msgbuf())
        except:
            pass
    
    def recv():
        msgs = []
        try:
            for b in conn.recv(4096):
                m = mav.parse_char(bytes([b]))
                if m:
                    msgs.append(m)
        except:
            pass
        return msgs
    
    # Simulation state
    alt = 0.0
    vz = 0.0
    motors = np.zeros(4)
    ts = 0
    max_alt = 0
    
    print("\n[2] Running simulation loop...")
    print("    (PX4 will arm and takeoff automatically in offboard mode)\n")
    
    try:
        for i in range(5000):  # ~20 seconds
            ts += 4000  # 4ms timestep (250Hz)
            
            # Simple physics
            thrust = np.mean(motors)
            accel = (thrust - 0.5) * 20.0  # Thrust produces acceleration
            vz += accel * 0.004  # Integrate velocity
            vz -= 9.81 * 0.004  # Gravity
            alt += vz * 0.004   # Integrate position
            
            # Ground contact
            if alt < 0:
                alt = 0
                vz = 0
            
            max_alt = max(max_alt, alt)
            
            # Send HIL_SENSOR (250Hz)
            send(mav.hil_sensor_encode(
                ts,
                0.0, 0.0, -9.81 + accel,  # accel (with thrust)
                0.0, 0.0, 0.0,  # gyro
                0.21, 0.0, 0.42,  # mag
                101325.0 - alt * 12.0,  # pressure
                0.0,  # diff_pressure
                alt,  # pressure_alt
                20.0,  # temp
                0x1FF  # fields
            ))
            
            # Send HIL_GPS (10Hz)
            if i % 25 == 0:
                send(mav.hil_gps_encode(
                    ts, 3,  # time, fix
                    473977420, 85455940,  # lat, lon (degE7)
                    int((488.0 + alt) * 1000),  # alt (mm)
                    10, 10,  # eph, epv
                    0,  # vel
                    0, 0, int(-vz * 100),  # vn, ve, vd (cm/s)
                    0, 12  # cog, sats
                ))
            
            # Heartbeat (1Hz)
            if i % 250 == 0:
                send(mav.heartbeat_encode(
                    mavlink.MAV_TYPE_GENERIC,
                    mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0
                ))
            
            # Receive actuator commands
            for m in recv():
                if m.get_type() == 'HIL_ACTUATOR_CONTROLS':
                    motors = np.clip([(m.controls[j] + 1) / 2 for j in range(4)], 0, 1)
            
            # Print status
            if i % 125 == 0:  # 2Hz
                thrust_pct = np.mean(motors) * 100
                print(f"t={ts/1e6:5.1f}s | Alt: {alt:6.2f}m | Vz: {vz:+5.2f}m/s | Thrust: {thrust_pct:5.1f}%")
            
            time.sleep(0.004)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    
    print(f"\n[3] Results:")
    print(f"    Maximum altitude: {max_alt:.2f}m")
    print(f"    Final altitude: {alt:.2f}m")
    print(f"    Final motors: {motors.round(3)}")
    
    conn.close()
    server.close()
    print("\nServer closed.")

if __name__ == "__main__":
    main()
