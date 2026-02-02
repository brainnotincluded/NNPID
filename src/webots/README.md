# Webots Integration

This directory contains Webots-specific integration modules.

## Files

- `webots_human_tracker.py` - Main script for tracking pedestrians in Webots using trained neural networks
- `webots_capture.py` - Utility for capturing camera streams from Webots

## Usage

### Human Tracking

Track a moving pedestrian in Webots using a trained yaw tracking model:

```bash
python src/webots/webots_human_tracker.py \
    --model runs/<run_name>/best_model \
    --altitude 3.0 \
    --duration 120
```

See `docs/QUICKSTART_HUMAN_TRACKING.md` for complete setup instructions.

### Camera Capture

Capture and display Webots camera stream:

```bash
python src/webots/webots_capture.py
```

## Requirements

- Webots R2023a or newer
- ArduPilot SITL
- PyMAVLink
- Stable-Baselines3 (for model loading)

## Documentation

- `docs/QUICKSTART_HUMAN_TRACKING.md` - Quick start guide
- `docs/WEBOTS_HUMAN_TRACKING.md` - Detailed concepts and troubleshooting
- `docs/DEPLOYMENT_WEBOTS.md` - Deployment notes
