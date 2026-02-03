# Quickstart: Human Tracking in Webots

This guide explains how to run the complete system: a drone in Webots tracking
a pedestrian using a trained neural network via ArduPilot SITL.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌─────────────┐      JSON/Binary      ┌─────────────┐                 │
│  │   WEBOTS    │◄──────────────────────►│  ArduPilot  │                 │
│  │  Simulator  │     (sensor/motor)     │    SITL     │                 │
│  │             │                        │             │                 │
│  │  - Drone    │                        │ - Stabilize │                 │
│  │  - Human    │                        │ - Execute   │                 │
│  └──────┬──────┘                        └──────┬──────┘                 │
│         │                                      │                        │
│         │ UDP:9100                             │ MAVLink:5760           │
│         │ (human pos)                          │ (yaw rate cmd)         │
│         │                                      │                        │
│         ▼                                      ▼                        │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              webots_human_tracker.py                         │       │
│  │                                                              │       │
│  │   1. Receive human position from Webots                      │       │
│  │   2. Get drone state from MAVLink                            │       │
│  │   3. Compute observation (11 dims)                           │       │
│  │   4. Neural Network → yaw rate command                       │       │
│  │   5. Send SET_POSITION_TARGET to SITL                        │       │
│  │                                                              │       │
│  │   Model: runs/<run_name>/best_model                          │       │
│  │   VecNormalize: runs/<run_name>/vec_normalize.pkl             │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Webots R2023a+** installed
2. **ArduPilot SITL** built (see below)
3. **Python dependencies**: `pip install -e ".[full]"`
4. **First run only**: install Webots controllers with `scripts/shell/setup_controllers.sh`

### Install ArduPilot SITL

See `docs/SITL_INTEGRATION.md` for full setup instructions.
The quickstart assumes ArduPilot is available at `~/ardupilot`.

## Quick Start (3 Terminals)

### Terminal 1: Start Webots

```bash
# Recommended helper script
scripts/shell/run_webots.sh

# Or run directly
webots iris_camera_human.wbt
```

The scene includes:
- **Iris drone** with camera at position (0, 1.4, 0.8)
- **Pedestrian** walking between waypoints (20,8) and (27,1)

### Terminal 2: Start ArduPilot SITL

```bash
# Use the provided script
scripts/shell/run_sitl.sh

# Or manually:
cd ~/ardupilot/ArduCopter
sim_vehicle.py -v ArduCopter -f JSON --console
```

Wait for SITL to show:
```
Ready to fly.
```

### Terminal 3: Run Human Tracker

```bash
# Use a trained model from runs/<run_name>/best_model
python src/webots/webots_human_tracker.py \
    --model runs/<run_name>/best_model \
    --altitude 3.0 \
    --duration 120
```

You should see:
```
🚁 Initialized human tracker
   Model: runs/<run_name>/best_model
   SITL: tcp:127.0.0.1:5760
   Tracking port: 9100
✅ Loaded VecNormalize from ...
📡 Listening for tracking data on UDP port 9100

📡 Connecting to ArduPilot SITL...
✅ Connected to SITL
🚁 Arming and taking off to 3.0m...
✅ Drone is airborne
🎯 Starting human tracking...

[  2.0s] Human: [ 22.3,   5.6] | Error:  +45.2° | Action: +0.85 | Track:  0.0%
[  4.0s] Human: [ 22.1,   5.8] | Error:  +12.3° | Action: +0.32 | Track: 35.0%
[  6.0s] Human: [ 21.9,   6.0] | Error:   -2.1° | Action: -0.05 | Track: 68.0%
```

## Command Line Options

```bash
python src/webots/webots_human_tracker.py \
    --model PATH            # Path to trained model (required)
    --connection STRING     # MAVLink connection (default: tcp:127.0.0.1:5760)
    --tracking-port PORT    # UDP port for Webots data (default: 9100)
    --altitude METERS       # Takeoff altitude (default: 3.0)
    --duration SECONDS      # Tracking duration (default: 300)
    --max-yaw-rate RAD/S    # Max yaw rate (default: 1.5, set to match training)
```

## Troubleshooting

### "No tracking data received"

The Webots controller isn't sending data. Check:
1. Webots is running with `iris_camera_human.wbt`
2. The drone controller is active (check Webots console)
3. UDP port 9100 is not blocked

### "Failed to connect to SITL"

SITL isn't running or MAVLink port is wrong. Check:
1. SITL shows "Ready to fly"
2. Port 5760 is accessible: `nc -zv 127.0.0.1 5760`

### "VecNormalize not found"

The model will still work but may produce poor results. Ensure
`runs/<run_name>/vec_normalize.pkl` exists next to `runs/<run_name>/best_model/`.

### Drone drifts or doesn't track well

1. Check yaw_rate is being sent: look at SITL console for attitude commands
2. Verify the observation space matches training (11 dimensions)
3. Try reducing `--max-yaw-rate` for smoother tracking

### Webots simulation runs at 0.00x speed

The JSON protocol connection failed. See Issue #005:
1. Restart SITL first, then Webots
2. Check UDP port 9002 is free

## Available Models

```bash
# List available runs with normalization
ls -la runs/*/vec_normalize.pkl

# Example model path
runs/<run_name>/best_model
```

## Testing Without Webots

Run the tracker with a simulated circular target:

```bash
# This uses the fallback when no Webots data is received
python scripts/run_yaw_tracker_sitl.py \
    --model runs/<run_name>/best_model \
    --no-model   # Or omit --model to test P-controller
```

## Key Files

| File | Purpose |
|------|---------|
| `iris_camera_human.wbt` | Webots scene with drone and pedestrian |
| `src/webots/webots_human_tracker.py` | Main tracking script |
| `controllers/ardupilot_vehicle_controller/` | Webots drone controller |
| `src/deployment/yaw_tracker_sitl.py` | SITL interface with model loading |
| `scripts/shell/run_sitl.sh` | Helper script to launch SITL |

## Next Steps

- Modify pedestrian trajectory in `iris_camera_human.wbt`
- Train a new model: `python scripts/train_yaw_tracker.py`
- Add computer vision for real human detection (see `docs/WEBOTS_HUMAN_TRACKING.md`)
