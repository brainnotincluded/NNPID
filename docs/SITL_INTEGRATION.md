# ArduPilot SITL Integration Guide

This guide explains how to connect the MuJoCo simulation with ArduPilot SITL (Software In The Loop).

## Overview

The integration uses two communication channels:

```
┌─────────────────┐                    ┌─────────────────┐
│  MuJoCo Sim     │◄── JSON (UDP) ───► │  ArduPilot SITL │
│  (Python)       │                    │                 │
│                 │◄── MAVLink (TCP) ──│                 │
└─────────────────┘                    └─────────────────┘

JSON Protocol (UDP):
  - Port 9002: MuJoCo → ArduPilot (sensor data)
  - Port 9003: ArduPilot → MuJoCo (motor commands)

MAVLink Protocol (TCP):
  - Port 5760: High-level commands (arm, takeoff, mode)
```

## Prerequisites

### Install ArduPilot SITL

#### macOS (Homebrew)

```bash
brew tap ArduPilot/homebrew-px4
brew install ardupilot-sitl
```

#### Linux (Ubuntu/Debian)

```bash
# Clone ArduPilot
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot
git submodule update --init --recursive

# Install prerequisites
./Tools/environment_install/install-prereqs-ubuntu.sh -y

# Build SITL
./waf configure --board sitl
./waf copter

# Add to PATH
echo 'export PATH=$PATH:~/ardupilot/Tools/autotest' >> ~/.bashrc
source ~/.bashrc
```

### Install pymavlink

```bash
pip install pymavlink
```

## Quick Start

### 1. Start ArduPilot SITL

```bash
# Terminal 1: Start SITL with JSON backend
sim_vehicle.py -v ArduCopter -f JSON --console --map
```

Options:
- `-v ArduCopter`: Vehicle type (copter)
- `-f JSON`: Use JSON interface for physics
- `--console`: Show MAVLink console
- `--map`: Show map (optional)

### 2. Run MuJoCo Bridge

```bash
# Terminal 2: Start MuJoCo simulation
python scripts/run_ardupilot_sim.py
```

The drone should:
1. Connect to SITL
2. Wait for arming
3. Receive motor commands
4. Send sensor data back

### 3. Control via MAVProxy

In the SITL console:

```
# Arm the drone
arm throttle

# Takeoff to 10m
mode GUIDED
takeoff 10

# Change mode
mode LOITER
mode LAND
```

## Deploy Trained Model

### Run Yaw Tracker on SITL

```bash
# Start SITL (Terminal 1)
sim_vehicle.py -v ArduCopter -f JSON --console

# Run trained model (Terminal 2)
python scripts/run_yaw_tracker_sitl.py \
    --model runs/best_model \
    --duration 60 \
    --target-speed 0.2
```

The trained neural network will control the drone's yaw to track a simulated target.

## Protocol Details

### JSON Protocol (Sensor Data)

MuJoCo sends sensor data to ArduPilot:

```json
{
    "timestamp": 1234567890.123,
    "imu": {
        "gyro": [0.01, -0.02, 0.005],      // rad/s
        "accel_body": [0.1, -0.05, -9.8]   // m/s²
    },
    "position": [0.0, 0.0, 10.0],          // m (NED)
    "velocity": [0.1, -0.05, 0.02],        // m/s (NED)
    "quaternion": [1.0, 0.0, 0.0, 0.0]     // [w, x, y, z]
}
```

### JSON Protocol (Motor Commands)

ArduPilot sends PWM commands:

```json
{
    "pwm": [1500, 1500, 1500, 1500]  // 1000-2000 range
}
```

Conversion: `thrust = (pwm - 1000) / 1000`

### MAVLink Commands

```python
from pymavlink import mavutil

# Connect
mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
mav.wait_heartbeat()

# Arm
mav.arducopter_arm()

# Set mode
mav.set_mode('GUIDED')

# Takeoff
mav.mav.command_long_send(
    mav.target_system,
    mav.target_component,
    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
    0, 0, 0, 0, 0, 0, 0, 10  # 10m altitude
)

# Send yaw rate command
mav.mav.set_attitude_target_send(
    0,  # timestamp
    mav.target_system,
    mav.target_component,
    0b01111011,  # type_mask: only yaw rate
    [1, 0, 0, 0],  # quaternion (ignored)
    0, 0, 0.5,  # roll_rate, pitch_rate, yaw_rate
    0.5  # thrust
)
```

## Script Reference

### `scripts/run_ardupilot_sim.py`

Main MuJoCo-SITL bridge:

```bash
python scripts/run_ardupilot_sim.py \
    --model models/quadrotor_x500.xml \  # MuJoCo model
    --sitl-address 127.0.0.1 \           # SITL IP
    --sensor-port 9002 \                  # Sensor data port
    --motor-port 9003 \                   # Motor command port
    --mavlink-port 5760                   # MAVLink port
```

### `scripts/run_yaw_tracker_sitl.py`

Deploy trained NN to SITL:

```bash
python scripts/run_yaw_tracker_sitl.py \
    --model runs/best_model \            # Trained model path
    --connection tcp:127.0.0.1:5760 \    # MAVLink connection
    --duration 60 \                       # Run time (seconds)
    --target-speed 0.2 \                  # Target angular velocity
    --target-pattern circular             # Target motion pattern
```

## Troubleshooting

### SITL Won't Connect

```bash
# Check if SITL is running
ps aux | grep arducopter

# Check ports
netstat -an | grep 9002
netstat -an | grep 5760
```

### "No heartbeat" Error

```python
# Increase timeout
mav = mavutil.mavlink_connection('tcp:127.0.0.1:5760')
mav.wait_heartbeat(timeout=30)  # Wait longer
```

### Drone Flips on Takeoff

- Check motor order matches ArduPilot expectations
- Verify PWM-to-thrust conversion
- Check coordinate frame (NED vs ENU)

### Physics Mismatch

- Ensure MuJoCo timestep matches SITL (500Hz default)
- Check mass/inertia in MuJoCo model
- Verify gravity direction

## Coordinate Frames

### MuJoCo Frame (ENU)
- X: Forward
- Y: Left
- Z: Up

### ArduPilot Frame (NED)
- X: North (Forward)
- Y: East (Right)
- Z: Down

### Conversion

```python
# MuJoCo to NED
ned_pos = [mujoco_pos[0], -mujoco_pos[1], -mujoco_pos[2]]
ned_vel = [mujoco_vel[0], -mujoco_vel[1], -mujoco_vel[2]]

# Quaternion: MuJoCo [w,x,y,z] to NED
ned_quat = [quat[0], quat[1], -quat[2], -quat[3]]
```

## Advanced Usage

### Custom Flight Modes

```python
# Switch to GUIDED mode for position control
mav.set_mode('GUIDED')

# Send position setpoint
mav.mav.set_position_target_local_ned_send(
    0, mav.target_system, mav.target_component,
    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
    0b0000111111111000,  # position only
    10, 0, -5,  # x, y, z (NED, so -5 = 5m up)
    0, 0, 0,    # velocity
    0, 0, 0,    # acceleration
    0, 0        # yaw, yaw_rate
)
```

### Recording Flights

```python
from src.visualization import FlightRecorder

recorder = FlightRecorder()
recorder.start()

# ... run simulation ...

recorder.stop()
recorder.save("flight_log.csv")
```

## References

- [ArduPilot SITL Documentation](https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html)
- [MAVLink Protocol](https://mavlink.io/en/)
- [pymavlink Documentation](https://mavlink.io/en/mavgen_python/)
