# NNPID - Neural Network PID Replacement for Drone Tracking

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Neural network-based adaptive controller to replace traditional PID for drone target tracking using **Recurrent Soft Actor-Critic (RSAC)** with GRU.

## Overview

This project implements a deep RL system that learns to track moving targets with a drone, adapting in real-time to:
- Target behavior changes (speed, direction, patterns)
- Drone dynamics (mass, inertia, motor response)
- Environmental conditions (wind, air density)

**Key Innovation**: GRU hidden state provides "memory" for adaptation without requiring backpropagation during flight.

## Quick Start

### Installation (using UV - recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourname/NNPID.git
cd NNPID
uv sync

# Optional: Install with extra dependencies
uv sync --extra noise      # Perlin noise for trajectories
uv sync --extra drone      # PyMAVLink for real drone
uv sync --extra all        # Everything
```

### Running

```bash
# Launch web dashboard (recommended)
uv run python scripts/dashboard.py
# Then open http://localhost:8000 in your browser

# Or use the CLI
uv run python main.py --help
uv run python main.py dashboard           # Web dashboard
uv run python main.py train --steps 50000 # Train model
uv run python main.py demo                 # Quick demo
uv run python main.py info                 # System info
```

### Alternative: pip install

```bash
pip install -e .

# Run commands
python main.py dashboard
python scripts/train.py --steps 50000
```

## Features

### Web Dashboard
- 🎮 **Live Demo** - Real-time simulation visualization with first-person camera view
- 📊 **Training Dashboard** - Live metrics, loss curves, reward plots
- ⚙️ **Training Control** - Start/stop training, configure parameters, view logs
- 🎯 **13 Trajectory Patterns** - From stationary to adversarial "Predator" mode

### Trajectory Types
| Type | Difficulty | Description |
|------|------------|-------------|
| Stationary | ⭐ | Fixed position (warmup) |
| Linear | ⭐⭐ | Constant velocity |
| Circular | ⭐⭐⭐ | Circular motion |
| Lissajous | ⭐⭐⭐⭐ | Figure-8 patterns |
| Spiral Dive | ⭐⭐⭐⭐⭐ | 3D spiral with altitude |
| Chaotic | ⭐⭐⭐⭐⭐⭐⭐⭐ | Multi-frequency overlay |
| Evasive | ⭐⭐⭐⭐⭐⭐⭐⭐⭐ | Fighter jet maneuvers |
| Predator | ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐ | Adversarial AI evasion |

### Core Components
- ✅ **RSAC-Share Architecture** - 2x faster training with shared GRU encoder
- ✅ **Domain Randomization** - Sim-to-real transfer (mass, thrust, wind, latency)
- ✅ **Safety Layer** - Geofence, velocity limits, fallback PID
- ✅ **Curriculum Learning** - Progressive difficulty scaling
- ✅ **Recurrent Replay Buffer** - Episode storage with BPTT chunks

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RSAC-Share Network                   │
├─────────────────────────────────────────────────────────┤
│  Observation (12D)                                      │
│       ↓                                                 │
│  ┌─────────────────┐                                    │
│  │  GRU Encoder    │  ← Hidden state = "Memory"        │
│  │  (2×64 units)   │                                    │
│  └────────┬────────┘                                    │
│           ↓                                             │
│  ┌────────┴────────┐                                    │
│  ↓                 ↓                                    │
│ Actor MLP      Critic MLP (×2)                          │
│ (256×256)      (256×256)                                │
│  ↓                 ↓                                    │
│ Action (3D)    Q-values                                 │
│ [vx,vy,vz]                                              │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
NNPID/
├── main.py                  # CLI entry point
├── pyproject.toml           # UV/pip dependencies
├── config/
│   └── training_config.yaml # Training configuration
├── scripts/
│   ├── train.py             # Training script
│   └── dashboard.py         # Web dashboard launcher
├── web/
│   ├── app.py               # FastAPI backend
│   └── templates/           # HTML templates
├── src/
│   ├── models/              # Neural networks
│   │   ├── gru_networks.py  # RSAC-Share architecture
│   │   └── replay_buffer.py # Recurrent replay
│   ├── training/            # Training loop
│   │   ├── rsac_trainer.py  # Main trainer
│   │   └── reward_shaper.py # Reward functions
│   ├── environment/         # Simulation
│   │   └── simple_drone_sim.py
│   └── utils/               # Utilities
│       ├── trajectory_generator.py
│       ├── domain_randomization.py
│       └── safety.py
├── logs/                    # Training logs
└── checkpoints/             # Model checkpoints
```

## Configuration

Edit `config/training_config.yaml` for:
- Network architecture (hidden_dim, gru_layers)
- RSAC hyperparameters (gamma, tau, alpha, learning rates)
- Domain randomization ranges
- Reward shaping weights
- Safety limits

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Lint code
uv run ruff check .
uv run ruff format .
```

## Roadmap

- [x] RSAC-Share neural network architecture
- [x] Simple Python drone simulator
- [x] 13 trajectory patterns (stationary → predator)
- [x] Domain randomization for sim-to-real
- [x] Web dashboard with live visualization
- [x] Training control center
- [ ] Webots integration
- [ ] ArduPilot SITL connection
- [ ] ONNX export for embedded deployment
- [ ] Real drone flight tests

## License

MIT

---

*"Safety first, performance second" - Never trust neural networks blindly on real hardware.*
