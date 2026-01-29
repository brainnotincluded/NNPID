# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

NNPID is a drone simulation framework using MuJoCo physics with ArduPilot SITL integration for training neural network controllers via reinforcement learning. The primary task is yaw tracking—training a NN to keep a quadrotor facing a moving target.

## Essential Commands

```bash
# Install dependencies
uv sync                     # recommended
pip install -e ".[dev]"     # alternative

# Run tests
pytest tests/ -v            # all tests
pytest tests/ -v -m "not slow"  # fast tests only
pytest tests/test_mujoco_sim.py  # single file

# Lint and format
make lint                   # ruff check + format check
make format                 # auto-fix formatting
make check                  # lint + security + complexity
ruff check src/ --fix       # fix linting issues

# Type checking
mypy src/ --ignore-missing-imports

# Training
python scripts/train_yaw_tracker.py --timesteps 500000
python scripts/train_yaw_tracker.py --config config/yaw_tracking.yaml

# Evaluation
python scripts/evaluate_yaw_tracker.py --model runs/best_model

# Visualization
python scripts/run_mega_viz.py --model runs/best_model.zip
python scripts/model_inspector.py arch runs/model.zip --diagram

# ArduPilot SITL (requires two terminals)
# Terminal 1: sim_vehicle.py -v ArduCopter -f JSON --console
# Terminal 2: python scripts/run_ardupilot_sim.py
```

## Architecture

```
src/
├── core/           # MuJoCo physics (mujoco_sim.py is main simulator)
├── environments/   # Gymnasium RL environments (YawTrackingEnv is primary)
├── controllers/    # PID (hover_stabilizer.py) and NN controllers
├── communication/  # ArduPilot SITL protocols (MAVLink, JSON)
├── deployment/     # Export and deploy models to SITL
├── perturbations/  # Wind, delays, sensor noise for robust training
├── visualization/  # MuJoCo viewer, HUD, NN visualizer
└── utils/          # Quaternion math (rotations.py), transforms
```

Key files to understand:
- `src/core/mujoco_sim.py` — Main physics simulator class
- `src/environments/yaw_tracking_env.py` — Primary training environment
- `src/controllers/hover_stabilizer.py` — PID stabilizer for hover (requires ≥100Hz)
- `config/yaw_tracking.yaml` — Training configuration

## Critical: Coordinate System & Motor Layout

MuJoCo uses **X-forward, Y-left, Z-up** coordinates:

```
       Front (X+)
         ▲
  M1 ────┼──── M4     M1: Front-Left  CCW (+yaw)
 (FL)    │    (FR)    M2: Back-Left   CW  (-yaw)
         │            M3: Back-Right  CCW (+yaw)
  M2 ────┼──── M3     M4: Front-Right CW  (-yaw)
 (BL)    │    (BR)
         ▼
       Back (X-)
```

Motor mixing (signs are critical for correct behavior):
```python
m1 = thrust + roll - pitch + yaw   # FL CCW
m2 = thrust + roll + pitch - yaw   # BL CW
m3 = thrust - roll + pitch + yaw   # BR CCW
m4 = thrust - roll - pitch - yaw   # FR CW
```

## Common Issues

1. **Drone crashes immediately** — Check `base_thrust` (~0.62 for 2kg), verify motor mixing signs, ensure control_frequency ≥100Hz
2. **Drone spins uncontrollably** — Motor positions wrong, yaw signs inverted, reduce `yaw_authority`
3. **Training doesn't converge** — Reduce target speed, increase episode length, check reward scaling
4. **Hover oscillates** — Reduce attitude PID gains (Kp=15, Kd=5 for 100Hz), check anti-windup

See `docs/issues/` for detailed debugging guides.

## Key Parameters

- Physics timestep: 0.002s (500Hz)
- Drone mass: 2.0 kg
- Motor max thrust: 8N each
- Default hover height: 1.0m
- Max yaw rate: 2.0 rad/s

## Code Conventions

- Python 3.10+ with type hints required
- Google-style docstrings
- Use `from src.utils.logger import get_logger` instead of `print()`
- Run `pre-commit run --all-files` before committing

## When Modifying Code

### Adding New Environment
1. Create `src/environments/new_env.py`
2. Inherit from `BaseDroneEnv`
3. Implement: `_get_observation()`, `_compute_reward()`, `_check_termination()`
4. Register with `gym.register()`
5. Add config in `config/`

### Modifying Physics
1. Edit `models/quadrotor_x500.xml`
2. Update `src/core/mujoco_sim.py` if new sensors/actuators

### Adding New Controller
1. Create `src/controllers/new_controller.py`
2. Implement `compute_action(state) -> np.ndarray`

## Do Not

- Modify MuJoCo model (`models/quadrotor_x500.xml`) without updating physics code
- Change motor order without updating mixing matrix
- Use blocking I/O in `step()` functions
- Forget to call `env.close()` after use

## Deployment to Webots via SITL

**Critical**: Model requires VecNormalize (stored as `vec_normalize.pkl`) for correct inference!

### Local Verification
```bash
# Run best model in MuJoCo (verifies VecNormalize works)
python scripts/run_model_mujoco.py --model runs/analysis_20260126_150455/best_model --episodes 3
```

### Server Deployment (3 terminals)
```bash
# Terminal 1: SITL
sim_vehicle.py -v ArduCopter -f JSON --console

# Terminal 2: Webots
webots --mode=fast --minimize --batch webots_scenario.wbt

# Terminal 3: NN Bridge (controls Webots drone via SITL)
python scripts/run_yaw_tracker_sitl.py \
    --model runs/analysis_20260126_150455/best_model \
    --altitude 2.0 --duration 120
```

**VecNormalize Auto-loads** — No extra steps needed if `vec_normalize.pkl` exists in model directory.

See `docs/DEPLOYMENT_WEBOTS.md` for detailed setup and troubleshooting.

## Reference Documentation

- `ARCHITECTURE.md` — Detailed code structure for developers
- `CONTRIBUTING.md` — Full code standards and PR process
- `docs/TRAINING.md` — How to train models
- `docs/SITL_INTEGRATION.md` — ArduPilot connection guide
- `docs/DEPLOYMENT_WEBOTS.md` — Deploy to Webots via SITL (VecNormalize details, server setup)
