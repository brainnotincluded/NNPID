# Shell Scripts

This directory contains shell scripts for common project tasks.

## Scripts

- `run_sitl.sh` - Launch ArduPilot SITL for Webots integration
- `run_webots.sh` - Quick launcher for Webots simulator
- `setup_controllers.sh` - Setup Webots controllers from ArduPilot
- `setup_webots_env.sh` - Setup Webots environment and paths

## Usage

All scripts should be run from the project root directory:

```bash
# From project root
scripts/shell/run_sitl.sh
scripts/shell/run_webots.sh
scripts/shell/setup_controllers.sh
scripts/shell/setup_webots_env.sh
```

## Notes

- Scripts automatically detect and use the project root directory
- Make sure scripts are executable: `chmod +x scripts/shell/*.sh`
- See individual scripts for detailed usage and options
