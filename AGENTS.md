# AGENTS.md

**Purpose**: Guidance for WARP (warp.dev) agents in this repo.
For Cursor/LLM guidance, see `.cursorrules`. For human standards, see
`CONTRIBUTING.md` and `CODE_STYLE.md`.

## Quick Links

- `.cursorrules`
- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `CODE_STYLE.md`
- `docs/TRAINING.md`
- `docs/SITL_INTEGRATION.md`
- `docs/DEPLOYMENT_WEBOTS.md`

## Essential Commands

```bash
# Install dependencies
uv sync                     # recommended
pip install -e ".[dev]"     # alternative

# Run tests
pytest tests/ -v               # all tests
pytest tests/ -v -m "not slow" # fast tests only
pytest tests/test_mujoco_sim.py

# Lint and format
make lint
make format
make check
ruff check src/ --fix

# Type checking
mypy src/ --ignore-missing-imports

# Training
python scripts/train_yaw_tracker.py --timesteps 500000
python scripts/train_yaw_tracker.py --config config/yaw_tracking.yaml

# Evaluation
python scripts/evaluate_yaw_tracker.py --model runs/<run_name>/best_model

# Visualization
python scripts/visualize_mujoco.py --mode interactive --model runs/<run_name>/best_model
python scripts/run_mega_viz.py --model runs/<run_name>/best_model
python scripts/model_inspector.py arch runs/<run_name>/best_model --diagram

# ArduPilot SITL (two terminals)
# Terminal 1: sim_vehicle.py -v ArduCopter -f JSON --console
# Terminal 2: python scripts/run_ardupilot_sim.py
```

## Webots Deployment via SITL (Summary)

**VecNormalize is required**: `vec_normalize.pkl` must exist in the run directory.

```bash
# Terminal 1: SITL
scripts/shell/run_sitl.sh

# Terminal 2: Webots
scripts/shell/run_webots.sh

# Terminal 3: NN Bridge
python scripts/run_yaw_tracker_sitl.py \
    --model runs/<run_name>/best_model \
    --altitude 2.0 --duration 120
```

See `docs/DEPLOYMENT_WEBOTS.md` for full setup and troubleshooting.
