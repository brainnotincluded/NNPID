# NNPID SITL MuJoCo

A production-grade drone simulation framework connecting **PX4 SITL** with **MuJoCo** physics, featuring **Gymnasium-compatible** environments for reinforcement learning.

## Features

- **High-Fidelity Physics**: MuJoCo-based quadrotor simulation with realistic dynamics
- **PX4 Integration**: MAVLink HIL protocol for connecting to PX4 SITL autopilot
- **RL Training**: Gymnasium-compatible environments for training neural network controllers
- **Lockstep Simulation**: Deterministic simulation for reproducible experiments
- **Modular Architecture**: Clean separation between physics, control, and learning components

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NNPID SITL MuJoCo                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   MuJoCo     │    │   MAVLink    │    │   Gymnasium      │  │
│  │   Physics    │◄──►│   Bridge     │◄──►│   Environments   │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                     │             │
│         ▼                   ▼                     ▼             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Sensor     │    │   PX4 SITL   │    │   RL Training    │  │
│  │   Simulation │    │   (external) │    │   (SB3/PyTorch)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- MuJoCo 3.0+
- PX4 SITL (for HIL simulation)

### Install from source

```bash
# Clone repository
git clone https://github.com/your-org/nnpid-sitl-mujoco.git
cd nnpid-sitl-mujoco

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Install with uv (recommended)

```bash
uv pip install -e ".[dev]"
```

## Quick Start

### 1. Run Standalone Simulation

Test the MuJoCo simulation with a PID controller:

```bash
python scripts/visualize.py view --model x500 --target 0 0 1
```

### 2. Connect to PX4 SITL

Start PX4 SITL first, then run the MuJoCo simulation:

```bash
# Terminal 1: Start PX4 SITL (in PX4 directory)
make px4_sitl none_iris

# Terminal 2: Start MuJoCo simulation
python scripts/run_sitl_sim.py --model x500
```

### 3. Train RL Controller

Train a neural network controller using PPO:

```bash
python scripts/train_controller.py --env hover --config config/training.yaml
```

### 4. Evaluate Trained Policy

```bash
python scripts/evaluate.py runs/hover_*/best_model/best_model.zip --env hover --episodes 10
```

## Project Structure

```
nnpid-sitl-mujoco/
├── src/
│   ├── core/                    # Physics simulation
│   │   ├── mujoco_sim.py       # MuJoCo wrapper
│   │   ├── quadrotor.py        # Dynamics model
│   │   └── sensors.py          # Sensor simulation
│   ├── communication/           # PX4 SITL bridge
│   │   ├── mavlink_bridge.py   # MAVLink HIL protocol
│   │   ├── messages.py         # Message definitions
│   │   └── lockstep.py         # Timing synchronization
│   ├── environments/            # Gymnasium environments
│   │   ├── base_drone_env.py   # Base environment
│   │   ├── hover_env.py        # Position hold task
│   │   ├── waypoint_env.py     # Waypoint navigation
│   │   └── trajectory_env.py   # Trajectory tracking
│   ├── controllers/             # Control interfaces
│   │   ├── base_controller.py  # Abstract controller
│   │   ├── sitl_controller.py  # PX4 passthrough
│   │   └── nn_controller.py    # Neural network
│   ├── utils/                   # Utilities
│   │   ├── transforms.py       # Coordinate transforms
│   │   ├── rotations.py        # Quaternion math
│   │   └── logger.py           # Telemetry logging
│   └── visualization/           # Rendering
│       ├── viewer.py           # MuJoCo viewer
│       └── dashboard.py        # Real-time plots
├── models/                      # MuJoCo XML models
│   ├── quadrotor_x500.xml      # X500 frame
│   └── quadrotor_generic.xml   # Simplified model
├── config/                      # Configuration files
│   ├── default.yaml            # Default settings
│   ├── training.yaml           # RL hyperparameters
│   └── px4_sitl.yaml          # PX4 connection
├── scripts/                     # CLI tools
│   ├── run_sitl_sim.py        # SITL simulation
│   ├── train_controller.py    # RL training
│   ├── evaluate.py            # Policy evaluation
│   └── visualize.py           # Visualization
└── tests/                       # Test suite
```

## Configuration

Configuration is loaded from YAML files in `config/`. Key parameters:

### Simulation (`default.yaml`)

```yaml
simulation:
  timestep: 0.002          # 500 Hz physics
  gravity: [0, 0, -9.81]
  
quadrotor:
  mass: 2.0
  max_thrust_per_motor: 8.0
  arm_length: 0.25
```

### Training (`training.yaml`)

```yaml
algorithm:
  name: "PPO"
  learning_rate: 3e-4
  n_steps: 2048
  
training:
  total_timesteps: 1_000_000
  n_envs: 8
```

### PX4 Connection (`px4_sitl.yaml`)

```yaml
px4:
  host: "127.0.0.1"
  port: 4560
  lockstep: true
```

## Environments

### HoverEnv

Maintain position at a target location.

```python
import gymnasium as gym
from src.environments.hover_env import HoverEnv

env = HoverEnv()
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### WaypointEnv

Navigate through a sequence of waypoints.

### TrajectoryEnv

Track a reference trajectory (circle, figure-8, etc.).

## Training

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
from src.environments.hover_env import HoverEnv

env = HoverEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("hover_policy")
```

### Command Line

```bash
# Basic training
python scripts/train_controller.py --env hover

# With custom config
python scripts/train_controller.py --env hover --config my_config.yaml

# Resume training
python scripts/train_controller.py --env hover --resume checkpoints/model_500000.zip
```

## PX4 SITL Integration

### Protocol

The simulation uses MAVLink HIL (Hardware-In-the-Loop) protocol:

- **Outgoing**: `HIL_SENSOR` (250 Hz), `HIL_GPS` (10 Hz)
- **Incoming**: `HIL_ACTUATOR_CONTROLS` (motor commands)

### Lockstep Mode

In lockstep mode, the simulation waits for each actuator command before stepping physics. This ensures:

- Deterministic simulation
- Synchronized timing with PX4
- Reproducible experiments

### Coordinate Frames

- **MuJoCo**: Z-up, X-forward
- **PX4/MAVLink**: NED (North-East-Down)
- **Body frame**: FRD (Forward-Right-Down)

Transforms are handled automatically by `src/utils/transforms.py`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_mujoco_sim.py -v
```

## Development

### Code Style

```bash
# Format code
black src/ tests/ scripts/

# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/
```

### Adding New Environments

1. Create new file in `src/environments/`
2. Inherit from `BaseDroneEnv`
3. Implement `_compute_reward()`, `_get_target()`, `_is_success()`
4. Register in `src/environments/__init__.py`

## License

MIT License

## Citation

```bibtex
@software{nnpid_sitl_mujoco,
  title = {NNPID SITL MuJoCo: Drone Simulation for RL},
  year = {2026},
  url = {https://github.com/your-org/nnpid-sitl-mujoco}
}
```

## Acknowledgments

- [MuJoCo](https://mujoco.org/) - Physics simulation
- [PX4](https://px4.io/) - Autopilot stack
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environment API
