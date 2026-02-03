# Training Guide

This guide explains how to train neural network controllers for drone control.

## Quick Start

Note: Training requires the full dependency set:
`pip install -e ".[full]"` or `uv sync --all-extras`.

```bash
# Train yaw tracking (default settings)
python scripts/train_yaw_tracker.py

# Train with detailed analytics + VecNormalize (recommended for deployment)
python scripts/train_with_analysis.py

# Train with custom timesteps
python scripts/train_yaw_tracker.py --timesteps 1000000

# Train with more parallel environments
python scripts/train_yaw_tracker.py --n-envs 8 --timesteps 500000
```

## Training Tasks

### 1. Yaw Tracking (Main Task)

Train a neural network to keep the drone facing a moving target.

**Task Description:**
- Target moves in circular/random patterns around the drone
- NN outputs yaw rate commands [-1, 1]
- Internal PID handles altitude and attitude stabilization
- Reward: minimize yaw error, smooth actions

**Configuration:** `config/yaw_tracking.yaml`

```bash
python scripts/train_yaw_tracker.py \
    --config config/yaw_tracking.yaml \
    --timesteps 1000000 \
    --n-envs 8
```

### 2. Hover (Simple Task)

Train to maintain position at a setpoint.

```bash
python scripts/train_controller.py \
    --env hover \
    --timesteps 200000
```

### 3. Waypoint Navigation

Train to navigate through a series of waypoints.

```bash
python scripts/train_setpoint.py \
    --env waypoint \
    --timesteps 500000
```

## Configuration

### Environment Settings

```yaml
# config/yaw_tracking.yaml
environment:
  hover_height: 1.0           # Target altitude (m)
  max_episode_steps: 1000     # Max steps per episode
  max_yaw_rate: 1.0           # Max yaw rate (rad/s)
  action_dead_zone: 0.08      # Dead zone for small actions
  
  # Target motion
  target_patterns:
    - circular              # Circular motion
    - random                # Random direction changes
    - sinusoidal           # Oscillating
    - step                 # Sudden jumps
  target_speed_min: 0.1     # Min angular velocity (rad/s)
  target_speed_max: 0.3     # Max angular velocity (rad/s)
  target_radius: 3.0        # Distance from drone (m)
  
  # Reward weights (see config for full list)
  rewards:
    facing_reward_weight: 1.5
    action_rate_penalty_weight: 0.03
    sustained_tracking_bonus: 0.3
  
  # Termination conditions
  max_tilt_angle: 0.6       # Max roll/pitch before reset (rad)
  max_altitude_error: 2.0   # Max height deviation (m)
```

### Training Settings

```yaml
training:
  algorithm: PPO              # PPO or SAC
  total_timesteps: 500000     # Total training steps
  n_envs: 4                   # Parallel environments
  
  # PPO hyperparameters
  learning_rate: 0.0003
  n_steps: 2048               # Steps per update
  batch_size: 64
  n_epochs: 10
  gamma: 0.99                 # Discount factor
  gae_lambda: 0.95            # GAE parameter
  clip_range: 0.2             # PPO clip range
  
  # Network architecture
  policy_kwargs:
    net_arch:
      pi: [64, 64]            # Policy network
      vf: [64, 64]            # Value network
    activation_fn: tanh
  
  # Callbacks
  eval_freq: 10000            # Evaluate every N steps
  save_freq: 25000            # Save checkpoint every N steps
  n_eval_episodes: 10         # Episodes per evaluation
```

### Curriculum Learning

Gradually increase difficulty:

```yaml
curriculum:
  enabled: true
  stages:
    - timesteps: 0           # Start
      target_speed_max: 0.1   # Very slow target
      target_patterns: ["circular"]
      
    - timesteps: 100000      # After 100k steps
      target_speed_max: 0.15
      
    - timesteps: 200000      # After 200k steps
      target_speed_max: 0.2
      target_patterns: ["circular", "random"]
      
    - timesteps: 300000      # After 300k steps
      target_speed_max: 0.3
      target_patterns: ["circular", "random", "sinusoidal"]
```

## Training Scripts

### `train_yaw_tracker.py`

Main training script with full options:

```bash
python scripts/train_yaw_tracker.py \
    --config config/yaw_tracking.yaml \  # Config file
    --timesteps 1000000 \                 # Training steps
    --n-envs 8 \                          # Parallel envs
    --seed 42 \                           # Random seed
    --device auto                         # cpu/cuda/auto
```

### `train_with_analysis.py`

Detailed training with analytics, evaluation plots, and `vec_normalize.pkl`:

```bash
python scripts/train_with_analysis.py
```

Default total timesteps in this script are **3,000,000** (adjust in script if needed).

### Output Structure

```
runs/yaw_tracking_20260123_120000/
├── config.yaml           # Saved configuration
├── best_model/
│   └── best_model.zip    # Best model (by eval reward)
├── final_model.zip       # Final model
├── vec_normalize.pkl     # Observation normalization (train_with_analysis.py)
├── checkpoints/
│   ├── yaw_tracker_25000_steps.zip
│   ├── yaw_tracker_50000_steps.zip
│   └── ...
├── eval_logs/
│   └── evaluations.npz   # Evaluation history
├── analytics/            # Detailed per-episode analytics (train_with_analysis.py)
│   └── training_analytics.jsonl
└── tensorboard/
    └── PPO_1/            # TensorBoard logs
```

Note: `train_yaw_tracker.py` does not write `vec_normalize.pkl`. Use
`train_with_analysis.py` for deployment-ready runs (includes normalization stats).

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open http://localhost:6006
```

Key metrics to watch:
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/ep_len_mean`: Episode length (should increase)
- `train/loss`: Training loss (should decrease then stabilize)
- `train/entropy_loss`: Exploration (should slowly decrease)

### Console Output

```
Eval num_timesteps=100000, episode_reward=45.32 +/- 28.17
Episode length: 523.40 +/- 287.65
-----------------------------------------
| eval/                   |             |
|    mean_ep_length       | 523         |
|    mean_reward          | 45.3        |
| train/                  |             |
|    approx_kl            | 0.0025      |  # Should be < 0.02
|    clip_fraction        | 0.018       |  # 0.1-0.2 is normal
|    entropy_loss         | -1.35       |  # Exploration
|    learning_rate        | 0.0003      |
|    loss                 | 12.5        |
|    explained_variance   | 0.85        |  # Should be > 0.5
-----------------------------------------
```

## Evaluation

### Evaluate Trained Model

```bash
# Basic evaluation
python scripts/evaluate_yaw_tracker.py \
    --model runs/<run_name>/best_model

# With video rendering
python scripts/evaluate_yaw_tracker.py \
    --model runs/<run_name>/best_model \
    --render \
    --output evaluation_video.mp4

# Multiple episodes
python scripts/evaluate_yaw_tracker.py \
    --model runs/<run_name>/best_model \
    --episodes 20
```

To locate a run, check the `runs/` directory and use the run folder name in
`runs/<run_name>/best_model`.

### Metrics

- **Mean Reward**: Higher is better (>50 is good)
- **Episode Length**: Longer = more stable (1000 = max)
- **Tracking Percentage**: Time spent on target (>50% is good)
- **Mean Yaw Error**: Lower is better (<15° is good)

## Tips & Tricks

### Speed Up Training

1. **More environments**: `--n-envs 8` or `--n-envs 16`
2. **Bigger batches**: Increase `batch_size` in config
3. **Disable rendering**: Don't pass `render_mode`

### Improve Performance

1. **Curriculum learning**: Start easy, gradually increase difficulty
2. **Reward shaping**: Adjust reward weights in config
3. **Network size**: Try `[128, 128]` for complex tasks
4. **Longer training**: 1M+ timesteps for best results

### Debug Training Issues

1. **Reward not increasing**
   - Check reward function makes sense
   - Try higher learning rate
   - Simplify task (slower targets)

2. **High variance in rewards**
   - Increase `n_envs` for more stable gradients
   - Use larger `batch_size`
   - Try SAC instead of PPO

3. **Episode length stays low**
   - Drone might be crashing
   - Check termination conditions
   - Make stabilization easier

## Example Workflow

```bash
# 1. Start with quick experiment
python scripts/train_yaw_tracker.py --timesteps 100000 --n-envs 4

# 2. Check results
python scripts/evaluate_yaw_tracker.py --model runs/<run_name>/best_model

# 3. If promising, train longer
python scripts/train_yaw_tracker.py --timesteps 1000000 --n-envs 8

# 4. Final evaluation with video
python scripts/evaluate_yaw_tracker.py \
    --model runs/<run_name>/best_model \
    --render \
    --episodes 10
```

## Using Trained Models

After training, you can integrate trained models into your own control systems using the `TrainedYawTracker` wrapper class.

**Quick Example:**

```python
from src.deployment.trained_yaw_tracker import TrainedYawTracker

# Load model
tracker = TrainedYawTracker.from_path("runs/<run_name>/best_model")

# In your control loop
yaw_rate_cmd = tracker.predict(observation, deterministic=True)
```

**Full Guide:** See [Using Trained Models](TRAINED_MODEL_USAGE.md) for complete integration guide, API reference, and examples.
