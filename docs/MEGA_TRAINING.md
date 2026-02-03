# Mega Training Session - 20M Timesteps

> Note: This document is a historical training log. For current defaults and
> recommended scripts, see `docs/TRAINING.md`.

**Started:** 2026-01-26 02:12:46  
**Run Directory:** `runs/yaw_tracking_20260126_021246`  
**Status:** 🟢 RUNNING

## Configuration

### Training Parameters
- **Total Timesteps:** 20,000,000
- **Parallel Environments:** 16
- **Algorithm:** PPO
- **Learning Rate:** 3e-4
- **Batch Size:** 128
- **N Steps:** 4096

### Network Architecture
```
Policy Network:  [512, 512, 256, 128] (tanh activation)
Value Network:   [512, 512, 256, 128]
Total Parameters: ~1.2M
Model Size: ~10MB per checkpoint
```

### Environment Features

#### Target Trajectories (9 patterns)
1. **Circular** - constant angular velocity orbit
2. **Sinusoidal** - smooth oscillating motion
3. **Figure8** - infinity pattern
4. **Lissajous** - parametric curves (2D harmonic)
5. **Spiral** - expanding/contracting radius
6. **Multi-frequency** - combined sine waves
7. **Random** - unpredictable jumps
8. **Step** - sudden direction changes
9. **Evasive** - adversarial maneuvers

#### Perturbations (Realistic Disturbances)
- **Wind** - gusts and steady wind (intensity 0.5, 30% gust probability)
- **Motor Failures** - partial power loss (5% failure rate)
- **Sensor Noise** - bias drift (intensity 0.4)
- **Communication Delays** - up to 50ms latency (2% packet loss)
- **Inertia Variations** - ±15% variation
- **Mass Variations** - ±10% variation (payload changes)
- **Random Enable** - 70% chance per episode

## Curriculum Learning (14 Stages)

### Phase 1: Foundations (0-2M)
| Stage | Timesteps | Speed | Patterns | Perturbations | Description |
|-------|-----------|-------|----------|---------------|-------------|
| 1 | 0 | 0.1 | circular | disabled | Basic tracking |
| 2 | 200k | 0.15 | +sinusoidal | disabled | Smooth patterns |
| 3 | 500k | 0.2 | +figure8 | 0.2 intensity | Light disturbances |
| 4 | 1M | 0.3 | +lissajous | 0.3 intensity | Complex curves |

### Phase 2: Intermediate (2M-6M)
| Stage | Timesteps | Speed | Patterns | Perturbations | Description |
|-------|-----------|-------|----------|---------------|-------------|
| 5 | 2M | 0.4 | +spiral | 0.4 intensity | Variable radius |
| 6 | 3M | 0.5 | +multi_freq | 0.5 intensity | Multi-frequency |
| 7 | 4M | 0.6 | +random | 0.6 intensity | Unpredictable |
| 8 | 5M | 0.7 | +step | 0.65 intensity | Sudden changes |

### Phase 3: Advanced (6M-12M)
| Stage | Timesteps | Speed | Patterns | Perturbations | Description |
|-------|-----------|-------|----------|---------------|-------------|
| 9 | 6M | 0.8 | all (8) | 0.7 intensity | Fast targets |
| 10 | 8M | 0.9 | +evasive | 0.75 intensity | Adversarial |
| 11 | 10M | 1.0 | all (9) | 0.8 intensity | Full speed |

### Phase 4: Expert Refinement (12M-20M)
| Stage | Timesteps | Speed | Patterns | Perturbations | Description |
|-------|-----------|-------|----------|---------------|-------------|
| 12 | 12M | 1.0 | all | 0.85 (80% prob) | Expert mode |
| 13 | 15M | 1.0 | all | 0.9 (90% prob) | Near-max |
| 14 | 18M | 1.0 | all | 1.0 (100% prob) | Maximum difficulty |

## Current Progress

**Current Stage:** 2 (Sinusoidal patterns, no perturbations yet)  
**Timesteps:** 200,000 / 20,000,000 (1.0%)  
**Latest Reward:** 64.70  
**Checkpoints Saved:** 2 (100k, 200k)  

### Expected Timeline
- **Steps per second:** ~1,400 steps/sec (16 envs)
- **Total training time:** ~4-5 hours
- **Checkpoint frequency:** Every 100k steps (200 checkpoints total)
- **Evaluation frequency:** Every 50k steps (400 evaluations total)

## Monitoring

### Real-time Monitor
```bash
python scripts/monitor_training.py runs/yaw_tracking_20260126_021246
```

### TensorBoard
```bash
tensorboard --logdir runs/yaw_tracking_20260126_021246/tensorboard
```

### Quick Status Check
```bash
python -c "
import numpy as np
data = np.load('runs/yaw_tracking_20260126_021246/eval_logs/evaluations.npz')
print(f'Steps: {int(data[\"timesteps\"][-1]):,}')
print(f'Reward: {np.mean(data[\"results\"][-1]):.2f}')
"
```

## Files

### Model Checkpoints
- `checkpoints/yaw_tracker_<steps>_steps.zip` - Periodic checkpoints
- `best_model/` (contains `best_model.zip`) - Best performing model
- `final_model.zip` - Final trained model (after 20M steps)

### Logs
- `eval_logs/evaluations.npz` - Evaluation metrics
- `tensorboard/` - TensorBoard event files
- `config.yaml` - Training configuration snapshot

## Post-Training

After training completes (~4-5 hours), evaluate the model:

```bash
# Evaluate on all patterns
python scripts/evaluate_yaw_tracker.py \
    --model runs/yaw_tracking_20260126_021246/best_model \
    --episodes 100 \
    --render

# Test specific patterns
python scripts/evaluate_yaw_tracker.py \
    --model runs/yaw_tracking_20260126_021246/best_model \
    --patterns evasive spiral multi_frequency \
    --episodes 50
```

## Expected Results

Based on curriculum design, the trained model should:

1. **Track slow targets (0.1-0.3 rad/s):** Near-perfect tracking, <5° error
2. **Track medium targets (0.3-0.6 rad/s):** Good tracking, <10° error
3. **Track fast targets (0.6-1.0 rad/s):** Acceptable tracking, <15° error
4. **Handle perturbations:** Recover from disturbances within 1-2 seconds
5. **Avoid crashes:** <1% crash rate even with max perturbations
6. **Smooth control:** Minimal jerk, efficient yaw commands

## Notes

- Using **HoverStabilizer** with corrected motor mixing for guaranteed hover stability
- Drone will always remain stable and hovering - only yaw control trained by NN
- Large network (512-512-256-128) to handle complex patterns + perturbations
- 16 parallel environments for faster training
- Aggressive curriculum to prevent catastrophic forgetting
