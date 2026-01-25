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


---

## PX4 Simulation Overview

**Source:** https://docs.px4.io/main/en/simulation/

### Simulation Architecture

PX4 supports multiple simulation environments for testing autonomous vehicle software:

- **Gazebo Classic** (deprecated)
- **Gazebo** (recommended, formerly Gazebo Ignition)
- **JSBSim**
- **FlightGear**
- **AirSim**
- **ROS 2/Gazebo**

### Simulator MAVLink API

PX4 uses MAVLink Hardware-In-The-Loop (HIL) protocol for external simulators. The simulator acts as a TCP server on port 4560, and PX4 connects as a client.

#### Key MAVLink Messages

**From PX4 to Simulator:**
- `HIL_ACTUATOR_CONTROLS`: Actuator commands (motor PWM, servo positions)

**From Simulator to PX4:**
- `HIL_SENSOR`: IMU data (gyro, accel, mag, baro, temperature)
- `HIL_GPS`: GPS position, velocity, and status
- `HIL_STATE_QUATERNION`: Ground truth vehicle state
- `HIL_OPTICAL_FLOW`: Optical flow sensor data (optional)

#### Message Rates

Recommended update rates:
- `HIL_SENSOR`: 200-400 Hz (IMU rate)
- `HIL_GPS`: 5-10 Hz
- `HIL_STATE_QUATERNION`: 50-100 Hz

### Connection Setup

1. Simulator starts TCP server on port 4560
2. Launch PX4 SITL:
```bash
make px4_sitl
```
3. PX4 connects to simulator
4. Message exchange begins

### Coordinate Systems

**NED (North-East-Down):**
- Position and velocity messages
- Standard aviation convention

**FRD (Forward-Right-Down):**
- Body-frame sensor data (IMU)
- Actuator directions

### HIL_SENSOR Message Format

```c
typedef struct __mavlink_hil_sensor_t {
    uint64_t time_usec;      // Timestamp (microseconds)
    float xacc;              // X acceleration (m/s²)
    float yacc;              // Y acceleration (m/s²)
    float zacc;              // Z acceleration (m/s²)
    float xgyro;             // Angular velocity X (rad/s)
    float ygyro;             // Angular velocity Y (rad/s)
    float zgyro;             // Angular velocity Z (rad/s)
    float xmag;              // Magnetic field X (gauss)
    float ymag;              // Magnetic field Y (gauss)
    float zmag;              // Magnetic field Z (gauss)
    float abs_pressure;      // Absolute pressure (millibar)
    float diff_pressure;     // Differential pressure (millibar)
    float pressure_alt;      // Pressure altitude (m)
    float temperature;       // Temperature (degC)
    uint32_t fields_updated; // Bitmap for updated fields
} mavlink_hil_sensor_t;
```

### HIL_GPS Message Format

```c
typedef struct __mavlink_hil_gps_t {
    uint64_t time_usec;      // Timestamp (microseconds)
    int32_t lat;             // Latitude (degE7)
    int32_t lon;             // Longitude (degE7)
    int32_t alt;             // Altitude (mm, AMSL)
    uint16_t eph;            // GPS HDOP (cm)
    uint16_t epv;            // GPS VDOP (cm)
    uint16_t vel;            // GPS ground speed (cm/s)
    int16_t vn;              // GPS North velocity (cm/s)
    int16_t ve;              // GPS East velocity (cm/s)
    int16_t vd;              // GPS Down velocity (cm/s)
    uint16_t cog;            // Course over ground (cdeg)
    uint8_t fix_type;        // 0-1: no fix, 2: 2D fix, 3: 3D fix
    uint8_t satellites_visible; // Number of visible satellites
} mavlink_hil_gps_t;
```

### HIL_ACTUATOR_CONTROLS Format

```c
typedef struct __mavlink_hil_actuator_controls_t {
    uint64_t time_usec;      // Timestamp (microseconds)
    float controls[16];      // Control outputs [-1..1]
    uint8_t mode;            // System mode
    uint64_t flags;          // Flags
} mavlink_hil_actuator_controls_t;
```

### Motor Mapping

For multirotors, `controls[]` array maps to motors:
- controls[0-7]: Motors 1-8
- Typical quad: controls[0-3] for motors 1-4
- Values normalized: -1.0 to 1.0

### Example Simulator Implementation

**Python with pymavlink:**

```python
from pymavlink import mavutil
import socket

# Create TCP server
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('127.0.0.1', 4560))
sock.listen(1)

conn, addr = sock.accept()
mav = mavutil.mavlink_connection(conn, dialect='common')

while True:
    # Receive actuator controls
    msg = mav.recv_match(type='HIL_ACTUATOR_CONTROLS', blocking=True)
    motor_commands = msg.controls

    # Run physics simulation
    # ... your simulation code ...

    # Send sensor data
    mav.mav.hil_sensor_send(
        time_usec=timestamp_us,
        xacc=ax, yacc=ay, zacc=az,
        xgyro=gx, ygyro=gy, zgyro=gz,
        xmag=mx, ymag=my, zmag=mz,
        abs_pressure=pressure,
        diff_pressure=0,
        pressure_alt=altitude,
        temperature=temp,
        fields_updated=0x1FFF
    )

    # Send GPS data
    mav.mav.hil_gps_send(
        time_usec=timestamp_us,
        lat=int(lat * 1e7),
        lon=int(lon * 1e7),
        alt=int(alt * 1000),
        eph=100, epv=100,
        vel=int(vel * 100),
        vn=int(vn * 100),
        ve=int(ve * 100),
        vd=int(vd * 100),
        cog=int(cog * 100),
        fix_type=3,
        satellites_visible=10
    )
```

### Simulation Timing

Important timing considerations:
1. Use monotonic timestamps
2. Match IMU rate (200-400 Hz typical)
3. Synchronize with real-time or use lockstep mode
4. PX4 can run faster than real-time in lockstep

### Lockstep Mode

Lockstep ensures deterministic simulation:
- Simulator controls time advancement
- PX4 waits for sensor data before stepping
- Enables reproducible tests


---

## MuJoCo Python Bindings

**Source:** https://mujoco.readthedocs.io/en/stable/python.html

### Installation

```bash
pip install mujoco
```

Requirements:
- Python 3.8+
- NumPy
- OpenGL (for rendering)

### Basic Usage

```python
import mujoco

# Load model from XML file
model = mujoco.MjModel.from_xml_path('model.xml')

# Create data structure for simulation state
data = mujoco.MjData(model)

# Step the simulation
mujoco.mj_step(model, data)
```

### Model Loading

**From XML file:**
```python
model = mujoco.MjModel.from_xml_path('model.xml')
```

**From XML string:**
```python
xml = '''
<mujoco>
  <worldbody>
    <body>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
'''
model = mujoco.MjModel.from_xml_string(xml)
```

**From binary MJB file:**
```python
model = mujoco.MjModel.from_binary_path('model.mjb')
```

### Data Structure

**MjModel** - Model specification (read-only after creation):
- `model.nq` - Number of position coordinates
- `model.nv` - Number of velocity coordinates  
- `model.nu` - Number of actuators
- `model.nbody` - Number of bodies
- `model.njnt` - Number of joints

**MjData** - Simulation state (mutable):
- `data.qpos` - Position coordinates
- `data.qvel` - Velocity coordinates
- `data.qacc` - Acceleration coordinates
- `data.ctrl` - Control inputs
- `data.time` - Simulation time

### Named Access

Access elements by name:

```python
# Get body position
pos = data.body('my_body').xpos

# Get sensor data
sensor_val = data.sensor('my_sensor').data

# Get joint angle
angle = data.joint('my_joint').qpos

# Get actuator data
actuator = data.actuator('my_actuator')
```

### Simulation Functions

**Basic stepping:**
```python
mujoco.mj_step(model, data)  # Full simulation step
```

**Advanced control:**
```python
mujoco.mj_forward(model, data)    # Forward dynamics
mujoco.mj_inverse(model, data)    # Inverse dynamics
mujoco.mj_resetData(model, data)  # Reset to initial state
```

**Sub-steps:**
```python
mujoco.mj_step1(model, data)  # Before collision detection
mujoco.mj_step2(model, data)  # After collision detection
```

### Control and Actuation

**Set control inputs:**
```python
data.ctrl[:] = [1.0, 0.5, -0.3]  # Set all actuators
data.ctrl[0] = 2.0                # Set specific actuator
```

**Apply external forces:**
```python
data.xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz]
```

**Set initial state:**
```python
data.qpos[:] = initial_positions
data.qvel[:] = initial_velocities
mujoco.mj_forward(model, data)  # Update dependent variables
```

### Rendering

**Offscreen rendering:**
```python
renderer = mujoco.Renderer(model, height=480, width=640)

# Update scene
renderer.update_scene(data)

# Render and get pixels
pixels = renderer.render()

# Save image
import imageio
imageio.imwrite('frame.png', pixels)
```

**Interactive viewer:**
```python
import mujoco.viewer

# Launch interactive viewer
mujoco.viewer.launch(model, data)
```

**Passive viewer (programmatic control):**
```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

### Sensor Data

**Read sensor values:**
```python
# By index
sensor_data = data.sensordata[sensor_id]

# By name
sensor_data = data.sensor('gyro').data
```

**Common sensor types:**
- `framepos`: Body position (3D)
- `framequat`: Body orientation (4D quaternion)
- `framelinvel`: Body linear velocity (3D)
- `frameangvel`: Body angular velocity (3D)
- `gyro`: Gyroscope (3D angular velocity)
- `accelerometer`: Accelerometer (3D linear acceleration)

### Contact Detection

**Access contact information:**
```python
for i in range(data.ncon):
    contact = data.contact[i]
    print(f"Contact {i}:")
    print(f"  Body 1: {model.body(contact.geom1).name}")
    print(f"  Body 2: {model.body(contact.geom2).name}")
    print(f"  Position: {contact.pos}")
    print(f"  Normal: {contact.frame[:3]}")
    print(f"  Force: {contact.force}")
```

### Threading

MuJoCo Python bindings are thread-safe:
- GIL released during C function calls
- Multiple simulations can run in parallel
- Each thread needs its own MjData instance

**Example parallel simulation:**
```python
import concurrent.futures

def run_simulation(model, steps):
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos.copy()

model = mujoco.MjModel.from_xml_path('model.xml')

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_simulation, model, 1000) 
               for _ in range(4)]
    results = [f.result() for f in futures]
```

### Performance Tips

1. **Avoid Python loops**: Use NumPy operations on arrays
2. **Batch operations**: Process multiple timesteps efficiently
3. **Profile bottlenecks**: Use cProfile to find slow code
4. **Pre-allocate arrays**: Reuse arrays instead of creating new ones
5. **Use compiled rendering**: Offscreen faster than interactive

### Example: Quadrotor Simulation

```python
import mujoco
import numpy as np

# Load quadrotor model
model = mujoco.MjModel.from_xml_path('quadrotor.xml')
data = mujoco.MjData(model)

# Simulation parameters
dt = model.opt.timestep
duration = 10.0
steps = int(duration / dt)

# Control gains (simple PD controller)
kp = 10.0
kd = 5.0

# Simulation loop
for i in range(steps):
    # Get current state
    pos = data.qpos[:3]
    vel = data.qvel[:3]

    # Compute control (hover at z=1.0)
    target = np.array([0, 0, 1.0])
    error = target - pos
    error_dot = -vel

    # PD control for thrust
    thrust = kp * error[2] + kd * error_dot[2] + model.opt.gravity[2]

    # Set motor commands (4 motors)
    data.ctrl[:] = thrust / 4.0

    # Step simulation
    mujoco.mj_step(model, data)

    if i % 100 == 0:
        print(f"Time: {data.time:.2f}, Height: {pos[2]:.3f}")
```

