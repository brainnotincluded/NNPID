# Webots Controllers Setup

## Problem

The Webots scene `iris_camera_human.wbt` requires ArduPilot controllers:

- `ardupilot_vehicle_controller` for the Iris drone
- `pedestrian` for human motion

## Recommended: use the provided setup script

```bash
scripts/shell/setup_controllers.sh
```

The script copies the controllers from `~/ardupilot` if present, otherwise it
clones ArduPilot and installs the controllers into `./controllers/`.

## Manual install (if you prefer)

### Copy from an existing ArduPilot checkout

```bash
cp -r ~/ardupilot/libraries/SITL/examples/Webots_Python/controllers ./
```

### Or download from GitHub

```bash
git clone --depth 1 https://github.com/ArduPilot/ardupilot.git /tmp/ardupilot_temp
cp -r /tmp/ardupilot_temp/libraries/SITL/examples/Webots_Python/controllers ./
rm -rf /tmp/ardupilot_temp
```

## Alternative: run scenes from ArduPilot

You can run Webots scenes directly from the ArduPilot repository:

```bash
cd ~/ardupilot/libraries/SITL/examples/Webots_Python/worlds
webots iris_camera.wbt
```

This is useful for reference, but the NNPID project assumes `iris_camera_human.wbt`.

## Temporary test: disable controllers

To verify the scene loads, you can temporarily disable controllers:

```vrml
Iris {
    controller "<none>"
    ...
}

Pedestrian {
    controller "<none>"
    ...
}
```

## Expected controller layout

```
controllers/
├── ardupilot_vehicle_controller/
│   ├── ardupilot_vehicle_controller.py
│   ├── webots_vehicle.py
│   └── drone_interface.py
├── pedestrian/
│   └── pedestrian.py
└── ...
```

## Verify installation

```bash
ls controllers/ardupilot_vehicle_controller/
ls controllers/pedestrian/
```

## After installation

1. Start Webots: `scripts/shell/run_webots.sh`
2. Start SITL: `scripts/shell/run_sitl.sh`
3. Run the tracker: `python src/webots/webots_human_tracker.py --model runs/<run_name>/best_model`

## Troubleshooting

### "No module named 'webots_vehicle'"

Cause: not all controller files were copied.

Fix: copy the entire `controllers/` folder from ArduPilot.

### "pedestrian.py: command not found"

Cause: Webots cannot execute the controller.

Fix: ensure the file exists and is executable:

```bash
chmod +x controllers/*/*.py
```

### Controller does not start

Check:
- Shebang at top of controller files (`#!/usr/bin/env python3`)
- Python version (controllers require Python 3.7+)

## References

- https://github.com/ArduPilot/ardupilot/tree/master/libraries/SITL/examples/Webots_Python
- https://cyberbotics.com/doc/guide/controller-programming
- https://ardupilot.org/dev/docs/sitl-with-webots-python.html
