# SITL with MuJoCo - Complete Documentation
Generated: January 23, 2026

## Table of Contents
1. [ArduPilot JSON Backend Protocol](#ardupilot-json-backend-protocol)
2. [ArduPilot SITL Guide](#ardupilot-sitl-guide)
3. [PX4 Simulation Overview](#px4-simulation-overview)
4. [MuJoCo Python Bindings](#mujoco-python-bindings)
5. [MuJoCo Simulation Programming](#mujoco-simulation-programming)

---

## ArduPilot JSON Backend Protocol

**Source:** https://ardupilot.org/dev/docs/sitl-with-JSON.html

### JSON SITL backend

The JSON SITL backend is designed to allow ArduPilot SITL to be used with external simulators. It uses JSON for communication with the simulator, sending IMU data and GPS data from the simulator to ArduPilot, and sending servo output from ArduPilot to the simulator.

### Starting SITL

To start ArduPilot SITL in JSON mode you use the sim_vehicle.py command with the -f parameter set to the name of your vehicle model. For example:

```bash
sim_vehicle.py -v ArduCopter -f quad --model JSON:127.0.0.1 --map --console
```

This starts ArduPilot SITL with the quad frame and connects to a JSON simulator on 127.0.0.1.

### Communication Protocol

The JSON backend uses UDP for communication. ArduPilot binds to port 9002 and sends PWM output data to the simulator. The simulator should send sensor data back to ArduPilot on the same port.

### PWM Output Format

ArduPilot sends PWM output as a binary structure:

```c
struct {
    uint16_t magic = 18458;  // magic value
    uint16_t frame_rate;     // frame rate in Hz
    uint32_t frame_count;    // frame number
    uint16_t pwm[16];        // PWM values for up to 16 channels
};
```

The magic value is used to detect the start of a packet. The frame_rate is the rate at which ArduPilot is running. The frame_count increments for each frame.

### JSON Input Format

The simulator should send sensor data as JSON. The JSON should be sent as a single packet (no line breaks within the JSON).

#### Required Fields

The following fields are required in every JSON packet:

```json
{
  "timestamp": 0.0,
  "imu": {
    "gyro": [0.0, 0.0, 0.0],
    "accel_body": [0.0, 0.0, -9.8]
  },
  "position": [0.0, 0.0, 0.0],
  "velocity": [0.0, 0.0, 0.0],
  "attitude": [0.0, 0.0, 0.0]
}
```

- **timestamp**: simulation time in seconds
- **imu.gyro**: gyroscope data in radians/second (body frame)
- **imu.accel_body**: accelerometer data in m/s² (body frame)
- **position**: position in meters (NED earth frame)
- **velocity**: velocity in m/s (NED earth frame)
- **attitude**: roll, pitch, yaw in radians

Alternatively, you can use quaternion instead of attitude:

```json
{
  "quaternion": [1.0, 0.0, 0.0, 0.0]
}
```

#### Optional Fields

Additional optional fields include:

- **imu.accel_true**: true acceleration including gravity (m/s²)
- **airspeed**: airspeed in m/s
- **battery**: battery data with voltage, current, remaining
- **rangefinder**: rangefinder distance array
- **wind**: wind velocity vector
- **mag**: magnetometer data

### Frame Conventions

- **Position/Velocity**: North-East-Down (NED) earth frame
- **IMU data**: Body frame (forward-right-down)
- **Angles**: Radians
- **Attitude**: Roll-Pitch-Yaw (intrinsic rotation)

### Example Python Implementation

```python
import socket
import struct
import json

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('127.0.0.1', 9002))

while True:
    # Receive PWM data from ArduPilot
    data, addr = sock.recvfrom(1024)

    # Unpack PWM data
    magic, frame_rate, frame_count = struct.unpack('<HHI', data[0:8])
    pwm = struct.unpack('<16H', data[8:40])

    # Run physics simulation with PWM values
    # ... your simulation code here ...

    # Send sensor data back
    sensor_data = {
        "timestamp": time,
        "imu": {
            "gyro": [gx, gy, gz],
            "accel_body": [ax, ay, az]
        },
        "position": [x, y, z],
        "velocity": [vx, vy, vz],
        "attitude": [roll, pitch, yaw]
    }

    sock.sendto(json.dumps(sensor_data).encode(), addr)
```


---

## ArduPilot SITL Guide

**Source:** https://ardupilot.org/dev/docs/using-sitl-for-ardupilot-testing.html

### What is SITL?

SITL (Software In The Loop) is a simulator that allows you to run ArduPilot on your PC without any hardware. It is a build of the autopilot code using an ordinary C++ compiler, giving you a native executable that allows you to test your code without hardware.

### Key Features

- No hardware required
- Complete autopilot functionality
- Multiple vehicle types supported (Copter, Plane, Rover, Sub, etc.)
- Integration with various simulators (Gazebo, RealFlight, AirSim, etc.)
- MAVProxy and Mission Planner compatible
- Supports multiple instances for swarm testing

### Starting SITL

The easiest way to start SITL is with sim_vehicle.py:

```bash
cd ardupilot/ArduCopter
sim_vehicle.py --map --console
```

Common options:
- `--map`: Show map window
- `--console`: Show console window
- `-v VEHICLE`: Specify vehicle type (ArduCopter, ArduPlane, etc.)
- `-f FRAME`: Specify frame type (quad, plane, rover, etc.)
- `--model MODEL`: External simulator model
- `-L LOCATION`: Starting location

### Simulator Backends

SITL supports multiple physics backends:

1. **Internal Physics** (default for copter/rover)
2. **JSBSim** (aircraft simulation)
3. **Gazebo** (robotics simulator)
4. **RealFlight** (commercial RC simulator)
5. **AirSim** (Microsoft simulator)
6. **JSON** (custom external simulators)

### Using Custom Locations

You can start at specific GPS coordinates:

```bash
sim_vehicle.py -L CMAC
sim_vehicle.py --location=51.477,-0.461,584,270  # lat,lon,alt,heading
```

### Multiple Instances

Run multiple vehicles simultaneously:

```bash
sim_vehicle.py -n 4  # Run 4 copters
```

Each instance gets unique ports:
- Instance 0: TCP 5760, 5762, 5763
- Instance 1: TCP 5770, 5772, 5773
- Instance 2: TCP 5780, 5782, 5783
- etc.

### Parameters and EEPROM

SITL maintains parameters in `eeprom.bin` file. To reset:

```bash
sim_vehicle.py --wipe
```

### Speedup and Time Control

Control simulation speed:

```bash
sim_vehicle.py --speedup=10  # Run 10x faster
```

In MAVProxy:
```
set speedup 10
```

### Testing Scenarios

SITL is useful for:
- Algorithm development
- Mission planning testing
- Parameter tuning
- Failsafe testing
- Swarm behavior
- CI/CD automated testing

### Integration with Ground Stations

Connect MAVProxy:
```bash
mavproxy.py --master=tcp:127.0.0.1:5760
```

Connect Mission Planner: TCP connection to 127.0.0.1:5760

Connect QGroundControl: Auto-detects SITL on localhost

### Advanced Usage

**Custom Terrain:**
```bash
sim_vehicle.py --use-terrain
```

**Wind Simulation:**
```
param set SIM_WIND_SPD 10
param set SIM_WIND_DIR 270
```

**Sensor Failures:**
```
param set SIM_GPS_DISABLE 1
param set SIM_BARO_DISABLE 1
```

