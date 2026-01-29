# Getting Started with NNPID

This guide walks you through setting up the project and running your first simulation.

## Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **MuJoCo 3.0+** (automatically installed with `mujoco` package)
- **Git** for version control
- **8GB+ RAM** recommended for training

## Installation

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager.

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/brainnotincluded/NNPID.git
cd NNPID
uv sync

# Activate environment
source .venv/bin/activate
```

### Option 2: Using pip

```bash
# Clone repository
git clone https://github.com/brainnotincluded/NNPID.git
cd NNPID

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install package
pip install -e .
```

### Option 3: Development Install

```bash
git clone https://github.com/brainnotincluded/NNPID.git
cd NNPID

# Install with dev dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Test MuJoCo works
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__}')"

# Test environment loads
python -c "from src.environments import YawTrackingEnv; env = YawTrackingEnv(); print('OK')"

# Run tests
pytest tests/ -v
```

## Your First Simulation

### 1. Run Basic Simulation

```python
from src.core.mujoco_sim import MuJoCoSimulator
import numpy as np

# Create simulator
sim = MuJoCoSimulator("models/quadrotor_x500.xml")

# Reset at 1m height
sim.reset(position=np.array([0, 0, 1.0]))

# Hover (equal thrust on all motors)
hover_thrust = 0.62  # ~mg/max_thrust for 2kg drone

# Simulate 100 steps
for i in range(100):
    sim.step(np.array([hover_thrust] * 4))
    state = sim.get_state()
    if i % 20 == 0:
        print(f"Step {i}: z={state.position[2]:.2f}m")
```

### 2. Run Gymnasium Environment

```python
from src.environments import YawTrackingEnv

# Create environment
env = YawTrackingEnv(render_mode="human")

# Run episode
obs, info = env.reset(seed=42)
for step in range(500):
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 50 == 0:
        print(f"Step {step}: yaw_error={info['yaw_error']:.2f} rad")
    
    if terminated or truncated:
        break

env.close()
```

### 3. Train Your First Model

```bash
# Quick training (5 minutes)
python scripts/train_yaw_tracker.py --timesteps 50000

# Check results
ls runs/
```

## Project Structure Quick Tour

```
NNPID/
├── src/
│   ├── core/              # MuJoCo physics
│   ├── environments/      # Gym environments
│   ├── controllers/       # PID & NN controllers
│   └── utils/             # Helper functions
├── scripts/               # Run these!
├── models/                # MuJoCo XML models
├── config/                # YAML configurations
└── runs/                  # Training outputs
```

## Next Steps

1. **Train a model**: See [Training Guide](TRAINING.md)
2. **Use trained models**: See [Using Trained Models](TRAINED_MODEL_USAGE.md)
3. **Deploy to SITL**: See [SITL Integration](SITL_INTEGRATION.md)
4. **Understand code**: See [Architecture](../ARCHITECTURE.md)

## Common Issues

### MuJoCo Error: "GLFW not found"

```bash
# macOS
brew install glfw

# Ubuntu
sudo apt-get install libglfw3-dev
```

### Import Error: "No module named 'src'"

Make sure you installed the package:
```bash
pip install -e .
```

### Slow Training

Use parallel environments:
```bash
python scripts/train_yaw_tracker.py --n-envs 8
```

## Getting Help

- Check [Architecture Guide](../ARCHITECTURE.md) for code structure
- Open an issue on GitHub for bugs
- Read MuJoCo docs: https://mujoco.readthedocs.io/
