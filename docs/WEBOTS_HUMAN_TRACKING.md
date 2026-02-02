# Webots Human Tracking Guide

This guide explains how to use the trained yaw tracking neural network to track a moving pedestrian in Webots simulator.

## Overview

The system integrates three components:

```
┌─────────────────────────────────────────────────────────────┐
│                   HUMAN TRACKING SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐                                            │
│  │   WEBOTS    │  1. Simulates drone physics                │
│  │  Simulator  │  2. Simulates pedestrian motion            │
│  │             │  3. Provides pedestrian position           │
│  └──────┬──────┘                                            │
│         │ JSON (sensors) + Supervisor API (human pos)       │
│         │                                                    │
│  ┌──────▼──────┐                                            │
│  │  ArduPilot  │  4. Stabilizes drone (roll/pitch)         │
│  │    SITL     │  5. Handles altitude control               │
│  │             │  6. Executes yaw rate commands             │
│  └──────┬──────┘                                            │
│         │ MAVLink (telemetry + commands)                    │
│         │                                                    │
│  ┌──────▼───────────────────────┐                           │
│  │   webots_human_tracker.py    │  Our tracking system     │
│  │                              │                           │
│  │  ├─ Get human position       │  7. Get human pos        │
│  │  ├─ Get drone state          │  8. Get drone state      │
│  │  ├─ Compute observation      │  9. Calculate direction  │
│  │  ├─ Neural network           │ 10. NN prediction        │
│  │  └─ Send yaw_rate command    │ 11. Send command         │
│  └──────────────────────────────┘                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup

If Webots is not installed, see `INSTALL_WEBOTS.md`.

```bash
# Install dependencies
pip install pymavlink opencv-python

# First run only: install Webots controllers
scripts/shell/setup_controllers.sh

# Ensure you have the trained model
ls runs/<run_name>/best_model/best_model.zip
```

### 2. Start Webots

```bash
# Open Webots and load the scene
webots iris_camera_human.wbt
```

The scene includes:
- **Iris quadcopter** at (0, 1.4, 0.8) with camera
- **Pedestrian** at (22.3, 5.6, 1.3) walking with trajectory
- **Environment** with obstacles, animals, and markers

### 3. Start ArduPilot SITL

```bash
# Terminal 1: Start ArduPilot for Webots
scripts/shell/run_sitl.sh

# Or manually:
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f JSON --console --map
```

**Important**: ArduPilot SITL must be started AFTER Webots is running.

### 4. Run Human Tracker

```bash
# Terminal 2: Run the tracking system
python src/webots/webots_human_tracker.py \
    --model runs/<run_name>/best_model \
    --duration 300
```

The drone will:
1. Connect to SITL ✅
2. Arm and takeoff to 3m ✅
3. Track the pedestrian using the neural network ✅
4. Land when done ✅

## Scene Structure

### iris_camera_human.wbt

The Webots world contains:

```python
WorldInfo {
    basicTimeStep 2  # 2ms = 500Hz (matches our training)
}

Pedestrian {
    translation 22.3 5.6 1.3  # Initial position
    controllerArgs [
        "--trajectory=20 8, 27 1"  # Waypoints to walk
    ]
}

Iris {
    controller "ardupilot_vehicle_controller"
    controllerArgs [
        "--motors" "m1_motor, m2_motor, m3_motor, m4_motor"
        "--camera" "camera"
        "--camera-port" "5599"
    ]
    extensionSlot [
        Camera {
            width 640
            height 640
        }
    ]
}
```

### Key Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| `basicTimeStep` | 2ms | 500Hz simulation (same as training) |
| Drone mass | 1.5 kg | Iris weight |
| Camera resolution | 640x640 | Grayscale |
| Camera port | 5599 | TCP stream |
| Pedestrian speed | ~1 m/s | Walking speed |

## How It Works

### 1. Position Tracking

```python
# Get human position from Webots Supervisor API
pedestrian_node = supervisor.getFromDef("PEDESTRIAN")
translation = pedestrian_node.getField("translation").getSFVec3f()
human_pos = np.array(translation)  # [x, y, z]
```

### 2. Observation Computation

The system builds an 11-element observation vector:

```python
observation = [
    target_dir_x,        # Direction to human (body frame)
    target_dir_y,        # Direction to human (body frame)
    target_angular_vel,  # Estimated human angular velocity
    current_yaw_rate,    # Drone yaw rate from SITL
    yaw_error,           # Angle error to human
    roll,                # Drone roll from SITL
    pitch,               # Drone pitch from SITL
    altitude_error,      # Altitude error from target
    velocity_x,          # Drone velocity from SITL
    velocity_y,          # Drone velocity from SITL
    previous_action,     # Previous yaw rate command
]
```

### 3. Neural Network Prediction

```python
# Trained model predicts yaw rate command
yaw_rate_cmd = tracker.predict(observation, deterministic=True)
# Output: [-1, 1] scaled to actual yaw rate in rad/s
```

### 4. Command Execution

```python
# Send to ArduPilot SITL via MAVLink
tracker.send_guided_velocity(
    vx=0, vy=0, vz=0,  # Hover in place
    yaw_rate=yaw_rate_cmd * max_yaw_rate
)
```

## Camera Stream (Optional)

View the drone's camera feed:

```bash
# Terminal 3: Capture camera
python src/webots/webots_capture.py
```

This displays the 640x640 grayscale image stream from the drone's camera.

## Coordinate Systems

### Webots (ENU)
- X: Forward (East)
- Y: Left (North)
- Z: Up

### ArduPilot (NED)
- X: North (Forward)
- Y: East (Right)
- Z: Down

The tracker handles coordinate conversions automatically via the MAVLink bridge.

## Troubleshooting

### SITL Connection Failed

```
❌ Failed to connect to SITL
```

**Solution**:
1. Check Webots is running first
2. Check SITL is running: `ps aux | grep arducopter`
3. Check MAVLink port: `netstat -an | grep 5760`
4. Try increasing timeout: edit `YawTrackerSITLConfig(connection_timeout=60)`

### Pedestrian Not Found

```
⚠️  Warning: Could not find pedestrian node
```

**Solution**:
1. Ensure pedestrian is in the scene (check Webots scene tree)
2. Add `DEF PEDESTRIAN` to the Pedestrian node in .wbt file:
   ```
   DEF PEDESTRIAN Pedestrian {
       ...
   }
   ```
3. Or run without supervisor: `--no-supervisor` (uses fallback circular pattern)

### Drone Not Tracking

**Possible causes**:
1. **Model not loaded**: Check model path exists
2. **Human too far**: Increase `target_radius` in config
3. **Physics mismatch**: Webots vs MuJoCo parameters differ

**Debug**:
```python
# Add verbose logging
tracker.run(duration=300)  # Watch console output

# Check observation values
obs = tracker.compute_observation()
print(f"Observation: {obs}")
print(f"Target position: {tracker._target.position}")
```

### Poor Tracking Performance

If tracking is unstable:

1. **Adjust control parameters**:
```python
config = YawTrackerSITLConfig(
    max_yaw_rate=1.0,  # Reduce from 1.5 for smoother tracking
    target_angular_velocity=0.2,  # Match actual human speed
)
```

2. **Check physics match**:
   - Webots drone mass should match training (2.0 kg in MuJoCo)
   - Adjust inertia if needed

3. **Retune if necessary**:
   - The model was trained in MuJoCo
   - If Webots physics differs significantly, consider sim-to-sim transfer
   - Or fine-tune the model in Webots environment

## Advanced Usage

### Custom Pedestrian Trajectory

Edit the Webots .wbt file:

```
Pedestrian {
    controllerArgs [
        "--trajectory=10 5, 15 10, 20 5, 10 0"  # Custom waypoints
    ]
}
```

### Multi-Human Tracking

To track multiple humans:

1. Add multiple Pedestrian nodes with DEF names
2. Modify `webots_human_tracker.py` to select target:
```python
# Track closest human
all_pedestrians = [
    supervisor.getFromDef("PEDESTRIAN1"),
    supervisor.getFromDef("PEDESTRIAN2"),
]
closest = min(all_pedestrians, key=lambda p: distance_to_drone(p))
```

### Use Different Model

Train a new model optimized for human tracking:

```bash
# Train with human-like target speeds
python scripts/train_yaw_tracker.py \
    --target-speed-min 0.1 \
    --target-speed-max 0.5 \
    --target-patterns circular,figure8 \
    --timesteps 1000000
```

Then use the new model:
```bash
python webots_human_tracker.py --model runs/new_model
```

## Performance Metrics

Expected performance:
- **Tracking accuracy**: 85-95% (within ±6° of human)
- **Mean yaw error**: 3-8 degrees
- **Control frequency**: 50 Hz
- **Latency**: < 20ms (observation → command)

## Integration with Vision

To add computer vision for human detection:

```python
# In webots_human_tracker.py, add:
import cv2
from human_detector import detect_human  # Your detector

def get_human_from_camera(camera_image):
    """Detect human from camera instead of Supervisor API."""
    detections = detect_human(camera_image)
    if detections:
        # Convert 2D image coords to 3D world position
        human_pos_3d = estimate_3d_position(detections[0])
        return human_pos_3d
    return None
```

This enables:
- Pure vision-based tracking (no Supervisor API)
- More realistic deployment scenario
- Better sim-to-real transfer

## Next Steps

1. **Tune for your scenario**: Adjust parameters for your specific use case
2. **Add vision**: Integrate computer vision for human detection
3. **Test real drone**: Deploy to real hardware (requires real SITL or flight controller)
4. **Extend to 3D**: Track altitude changes in addition to yaw

## References

- [ArduPilot Webots Integration](https://ardupilot.org/dev/docs/sitl-with-webots-python.html)
- [Webots Documentation](https://cyberbotics.com/doc/guide/index)
- [Training Guide](TRAINING.md) - Train custom models
- [Architecture](../ARCHITECTURE.md) - System overview
