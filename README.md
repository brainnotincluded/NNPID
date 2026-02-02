# NNPID - Neural Network PID Controller for Drone Simulation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/physics-MuJoCo-green.svg)](https://mujoco.org/)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium-orange.svg)](https://gymnasium.farama.org/)

A high-fidelity drone simulation framework integrating **MuJoCo physics** with **ArduPilot SITL** for training neural network controllers using reinforcement learning.

## Overview

This project provides:

- **MuJoCo-based quadrotor simulation** with realistic physics
- **ArduPilot SITL integration** via JSON/MAVLink protocols
- **Gymnasium environments** for reinforcement learning
- **Neural network controllers** trained with PPO/SAC
- **Yaw tracking task** as a demonstration of NN control

## Quick Start

```bash
# Clone repository
git clone https://github.com/brainnotincluded/NNPID.git
cd NNPID

# Install dependencies (using uv - recommended)
uv sync

# Or using pip
pip install -e .

# Train yaw tracking model
python scripts/train_yaw_tracker.py --timesteps 500000

# Evaluate trained model
python scripts/evaluate_yaw_tracker.py --model runs/<run_name>/best_model

# List available runs
ls runs/
```

For full setup details, see `docs/GETTING_STARTED.md`.

## Project Structure

```
NNPID/
├── src/                    # Main source code
│   ├── core/              # MuJoCo simulation core
│   │   ├── mujoco_sim.py  # Main simulator class
│   │   ├── quadrotor.py   # Quadrotor dynamics
│   │   └── sensors.py     # Sensor models
│   ├── environments/      # Gymnasium environments
│   │   ├── base_drone_env.py      # Base environment
│   │   ├── yaw_tracking_env.py    # Yaw tracking task
│   │   ├── hover_env.py           # Hover task
│   │   └── waypoint_env.py        # Waypoint navigation
│   ├── controllers/       # Control algorithms
│   │   ├── base_controller.py     # PID controller
│   │   ├── nn_controller.py       # Neural network controller
│   │   └── yaw_rate_controller.py # Yaw rate controller
│   ├── communication/     # SITL communication
│   │   ├── mavlink_bridge.py      # MAVLink protocol
│   │   └── messages.py            # Message definitions
│   ├── deployment/        # Model deployment
│   │   ├── yaw_tracker_sitl.py    # Deploy to ArduPilot SITL
│   │   └── model_export.py        # Export models
│   ├── utils/             # Utilities
│   │   ├── rotations.py   # Quaternion math
│   │   └── transforms.py  # Coordinate transforms
│   ├── visualization/     # Visualization tools
│   │   ├── viewer.py              # MuJoCo viewer
│   │   ├── scene_objects.py       # 3D scene objects
│   │   ├── nn_visualizer.py       # Neural network visualizer
│   │   ├── telemetry_hud.py       # Real-time HUD
│   │   └── mujoco_overlay.py      # Combined overlay system
│   └── perturbations/     # Realistic perturbations
│       ├── wind.py                # Wind effects
│       ├── delays.py              # Sensor/actuator delays
│       └── ...                    # Other perturbations
├── scripts/               # Executable scripts
│   ├── train_yaw_tracker.py       # Train yaw tracking
│   ├── evaluate_yaw_tracker.py    # Evaluate models
│   ├── run_mega_viz.py            # Full visualization
│   ├── model_inspector.py         # CLI model analysis
│   ├── run_ardupilot_sim.py       # Run with ArduPilot SITL
│   └── run_yaw_tracker_sitl.py    # Deploy NN to SITL
├── models/                # MuJoCo XML models
│   └── quadrotor_x500.xml # X500 quadrotor model
├── config/                # Configuration files
│   ├── yaw_tracking.yaml  # Yaw tracking config
│   └── simulation.yaml    # Simulation settings
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── runs/                  # Training runs & checkpoints
```

## Key Components

### 1. MuJoCo Simulator (`src/core/mujoco_sim.py`)

High-fidelity physics simulation with:
- 500Hz physics timestep
- Accurate motor dynamics
- IMU, gyroscope, accelerometer sensors
- Ground contact detection

```python
from src.core.mujoco_sim import MuJoCoSimulator

sim = MuJoCoSimulator("models/quadrotor_x500.xml")
sim.reset(position=[0, 0, 1])
sim.step(motor_commands=[0.5, 0.5, 0.5, 0.5])
state = sim.get_state()
```

### 2. Gymnasium Environments (`src/environments/`)

Standard RL interface for training:

```python
import gymnasium as gym
from src.environments import YawTrackingEnv

env = YawTrackingEnv(render_mode="rgb_array")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### 3. Controllers (`src/controllers/`)

Both classical and neural network controllers:

```python
from src.controllers import PIDController, NNController

# Classical PID
pid = PIDController(kp=1.0, kd=0.1)

# Neural network (trained)
nn = NNController.load("runs/<run_name>/best_model")
action = nn.predict(observation)
```

### 4. ArduPilot SITL Integration

Connect MuJoCo simulation to ArduPilot:

```bash
# Terminal 1: Start ArduPilot SITL
sim_vehicle.py -v ArduCopter -f JSON --console

# Terminal 2: Run MuJoCo bridge
python scripts/run_ardupilot_sim.py
```

## Training

### Yaw Tracking Task

Train a neural network to keep the drone facing a moving target:

```bash
# Basic training
python scripts/train_yaw_tracker.py

# With custom settings
python scripts/train_yaw_tracker.py \
    --config config/yaw_tracking.yaml \
    --timesteps 1000000 \
    --n-envs 8
```

### Configuration

Edit `config/yaw_tracking.yaml`:

```yaml
environment:
  hover_height: 1.0
  max_episode_steps: 1000
  target_patterns: ["circular", "random"]
  target_speed_min: 0.1
  target_speed_max: 0.3

training:
  algorithm: PPO
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  policy_kwargs:
    net_arch: [64, 64]
```

## Visualization

### Mega Visualization System

Run full visualization with all effects:

```bash
# Run with trained model
python scripts/run_mega_viz.py --model runs/<run_name>/best_model/best_model.zip

# With perturbations and video recording
python scripts/run_mega_viz.py \
    --model runs/<run_name>/best_model/best_model.zip \
    --perturbations config/perturbations.yaml \
    --record output.mp4
```

Features:
- **3D Scene Objects**: Wind arrows, force vectors, trajectory trails, VRS danger zones
- **Neural Network Visualizer**: Real-time activation display
- **Telemetry HUD**: Roll/pitch/yaw graphs, motor indicators, attitude display
- **Perturbation Effects**: Visual wind, ground effects, sensor delays

### Model Inspector CLI

Analyze trained models without running simulation:

```bash
# Show architecture with ASCII diagram
python scripts/model_inspector.py arch runs/model.zip --diagram

# Visualize weights as heatmap
python scripts/model_inspector.py weights runs/model.zip --heatmap

# Analyze activations over episodes
python scripts/model_inspector.py activations runs/model.zip --episodes 5

# Export statistics to JSON
python scripts/model_inspector.py stats runs/model.zip --output stats.json
```

## Perturbations

Add realistic disturbances for robust training:

```bash
# Train with perturbations
python scripts/train_yaw_tracker.py --perturbations config/perturbations.yaml
```

Available perturbations:
- **Wind**: Steady wind, gusts, turbulence (Dryden model)
- **Delays**: Sensor latency, actuator delays, jitter
- **Sensor Noise**: Gaussian noise, drift, outliers, GPS loss
- **Physics**: CoM offset, motor variations, ground effect
- **Aerodynamics**: Air drag, blade flapping, VRS
- **External Forces**: Impulses, vibrations, EMI

## Deployment

Deploy trained model to ArduPilot SITL:

```bash
python scripts/run_yaw_tracker_sitl.py \
    --model runs/<run_name>/best_model \
    --connection udp:127.0.0.1:14550
```

## Requirements

- Python 3.10+
- MuJoCo 3.0+
- PyTorch or JAX (for neural networks)
- ArduPilot SITL (optional, for deployment)

### Install ArduPilot SITL

See `docs/SITL_INTEGRATION.md` for full setup instructions on macOS and Linux.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Setup and first steps
- [Architecture](ARCHITECTURE.md) - Code structure for developers
- [Training Guide](docs/TRAINING.md) - How to train models
- [Using Trained Models](docs/TRAINED_MODEL_USAGE.md) - Load and use trained models
- [Webots Human Tracking](docs/WEBOTS_HUMAN_TRACKING.md) - Track pedestrians in Webots
- [Webots Quickstart](docs/QUICKSTART_HUMAN_TRACKING.md) - Fast setup for Webots tracking
- [SITL Integration](docs/SITL_INTEGRATION.md) - ArduPilot connection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- [MuJoCo](https://mujoco.org/) - Physics simulation
- [Gymnasium](https://gymnasium.farama.org/) - RL interface
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [ArduPilot](https://ardupilot.org/) - Flight controller
