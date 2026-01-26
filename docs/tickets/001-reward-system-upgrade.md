# Ticket #001: Reward System Upgrade for YawTrackingEnv

**Priority:** High  
**Type:** Enhancement  
**Component:** `src/environments/yaw_tracking_env.py`  
**Created:** 2026-01-26  
**Status:** Open

---

## Summary

Полная переработка системы вознаграждений для улучшения обучения yaw tracking. Текущая система показывает только 3.6% tracking после 20M шагов обучения.

---

## Background

После исправления знака yaw (commit `bdb96c1`) P-контроллер показал что окружение работает корректно. Однако обученная модель всё ещё показывает плохие результаты из-за проблем с reward функцией.

### Текущие метрики
- Tracking (<10°): **3.6%**
- Mean error: **92°**
- Model learned wrong behavior due to conflicting rewards

---

## Problems Analysis

### 🔴 Problem 1: Soft facing_reward doesn't create gradient

**Current:**
```python
facing_reward = exp(-5.0 * yaw_error²)
```

**Issue:** At 30° error (0.52 rad), reward = 0.26 — still positive! Agent can "sit" at 30° and receive reward.

| Error | Current Reward |
|-------|----------------|
| 0° | +1.00 |
| 10° | +0.96 |
| 30° | +0.26 ← Still positive! |
| 60° | +0.01 |
| 90° | ~0 |

---

### 🔴 Problem 2: yaw_rate_penalty conflicts with task

**Current:**
```python
reward -= 0.1 * yaw_rate²
```

**Issue:** To track target moving at 1 rad/s, drone MUST have yaw_rate ≈ 1 rad/s. But this gives penalty = -0.1!

---

### 🔴 Problem 3: No reward for progress toward target

**Current:** Agent gets reward only for current position, not for improvement.

**Issue:** No immediate feedback for correct actions when far from target.

---

### 🔴 Problem 4: Sparse sustained_tracking_bonus

**Current:**
```python
if time_on_target >= 0.5:
    reward += 0.5
```

**Issue:** Binary bonus — agent either gets full 0.5 or nothing. Hard to connect actions with outcome.

---

### 🔴 Problem 5: alive_bonus dilutes signal

**Current:**
```python
reward += 0.1  # Every step, regardless of quality
```

**Issue:** Agent gets +0.1 even when facing opposite direction from target.

---

## Solution Design

### 1. Zone-Based Facing Reward

Replace exponential with zones that have negative values for bad tracking:

```python
def _compute_facing_reward(self, yaw_error: float) -> float:
    """Zone-based facing reward with clear gradient."""
    abs_error = abs(yaw_error)
    
    if abs_error < 0.1:  # < 6° — On target zone
        # High reward, peaks at 0
        return 1.0 - 5.0 * abs_error
    elif abs_error < 0.35:  # 6-20° — Close tracking zone
        # Positive but decreasing
        return 0.5 * (1.0 - (abs_error - 0.1) / 0.25)
    elif abs_error < 1.57:  # 20-90° — Searching zone
        # NEGATIVE reward — must improve!
        return -0.5 * (abs_error - 0.35)
    else:  # > 90° — Lost zone
        # Strong negative
        return -1.0 - 0.3 * (abs_error - 1.57)
```

**Result:**
| Error | Old Reward | New Reward |
|-------|------------|------------|
| 0° | +1.00 | +1.00 |
| 5° | +0.96 | +0.56 |
| 20° | +0.30 | 0.00 |
| 45° | +0.04 | -0.22 |
| 90° | ~0 | -0.61 |
| 180° | ~0 | -1.47 |

---

### 2. Excess Yaw Rate Penalty (not all yaw rate)

Only penalize EXCESSIVE yaw rate:

```python
def _compute_yaw_rate_penalty(self, yaw_rate: float, target_vel: float, yaw_error: float) -> float:
    """Penalize only excessive yaw rate, not necessary rotation."""
    # Required yaw rate = target velocity + correction for error
    required_rate = target_vel + 2.0 * yaw_error  # P-gain for error
    
    # Excess = how much faster than needed
    excess_rate = max(0, abs(yaw_rate) - abs(required_rate) - 0.3)  # 0.3 margin
    
    return -0.05 * excess_rate ** 2
```

---

### 3. Error Reduction Reward (Shaping)

Add immediate feedback for correct actions:

```python
# In step():
self._prev_yaw_error = yaw_error  # Store previous error

# In _compute_reward():
error_reduction = abs(self._prev_yaw_error) - abs(yaw_error)
reward += 0.5 * error_reduction  # Reward for reducing error
```

---

### 4. Continuous Tracking Bonus

Replace binary with progressive:

```python
# Proportional bonus for time on target
tracking_progress = min(1.0, self._time_on_target / cfg.sustained_tracking_time)
tracking_bonus = cfg.sustained_tracking_bonus * tracking_progress ** 0.5  # sqrt for early encouragement
reward += tracking_bonus
```

---

### 5. Conditional Alive Bonus

```python
# Bonus only when tracking reasonably
if abs(yaw_error) < 0.5:  # < 30°
    reward += cfg.alive_bonus * (1.0 - abs(yaw_error) / 0.5)
```

---

### 6. Velocity Matching Reward (NEW)

Encourage matching target speed:

```python
velocity_error = abs(target_angular_velocity - yaw_rate)
velocity_match = np.exp(-3.0 * velocity_error ** 2)
reward += 0.2 * velocity_match
```

---

### 7. Direction Alignment Bonus (NEW)

Encourage turning toward target:

```python
# If yaw_rate is in correct direction toward target
if np.sign(yaw_error) * np.sign(yaw_rate) > 0:
    reward += 0.1
```

---

## Implementation

### New Config Parameters

Add to `YawTrackingConfig`:

```python
# Reward weights (v2)
facing_reward_weight: float = 1.5          # Increased
error_reduction_weight: float = 0.5        # NEW
velocity_match_weight: float = 0.2         # NEW
direction_alignment_bonus: float = 0.1     # NEW
excess_yaw_rate_penalty: float = 0.05      # Renamed & reduced
action_rate_penalty_weight: float = 0.03   # Reduced
sustained_tracking_bonus: float = 0.3      # Reduced (now continuous)
alive_bonus: float = 0.05                  # Reduced & conditional
crash_penalty: float = 10.0                # Same

# Zone thresholds
on_target_threshold: float = 0.1           # 6°
close_tracking_threshold: float = 0.35     # 20°
searching_threshold: float = 1.57          # 90°
```

---

### Full New _compute_reward Implementation

```python
def _compute_reward(
    self,
    state: QuadrotorState,
    action: np.ndarray,
    terminated: bool,
) -> float:
    """Compute improved reward with shaping."""
    cfg = self.config
    reward = 0.0
    
    yaw_error = self._compute_yaw_error(state)
    yaw_rate = state.angular_velocity[2]
    target_vel = self._current_pattern.get_angular_velocity()
    abs_error = abs(yaw_error)
    
    # 1. Zone-based facing reward (replaces exponential)
    if abs_error < cfg.on_target_threshold:
        facing = 1.0 - 5.0 * abs_error
    elif abs_error < cfg.close_tracking_threshold:
        facing = 0.5 * (1.0 - (abs_error - cfg.on_target_threshold) / 
                        (cfg.close_tracking_threshold - cfg.on_target_threshold))
    elif abs_error < cfg.searching_threshold:
        facing = -0.5 * (abs_error - cfg.close_tracking_threshold)
    else:
        facing = -1.0 - 0.3 * (abs_error - cfg.searching_threshold)
    reward += cfg.facing_reward_weight * facing
    
    # 2. Error reduction (shaping)
    if hasattr(self, '_prev_yaw_error') and self._prev_yaw_error is not None:
        error_reduction = abs(self._prev_yaw_error) - abs_error
        reward += cfg.error_reduction_weight * error_reduction
    self._prev_yaw_error = yaw_error
    
    # 3. Velocity matching
    velocity_error = abs(target_vel - yaw_rate)
    velocity_match = np.exp(-3.0 * velocity_error ** 2)
    reward += cfg.velocity_match_weight * velocity_match
    
    # 4. Direction alignment bonus
    if abs_error > 0.05 and np.sign(yaw_error) * np.sign(yaw_rate) > 0:
        reward += cfg.direction_alignment_bonus
    
    # 5. Excess yaw rate penalty (not all yaw rate!)
    required_rate = target_vel + 2.0 * yaw_error
    excess = max(0, abs(yaw_rate) - abs(required_rate) - 0.3)
    reward -= cfg.excess_yaw_rate_penalty * excess ** 2
    
    # 6. Action smoothness penalty
    action_change = abs(float(action[0]) - self._previous_action)
    reward -= cfg.action_rate_penalty_weight * action_change ** 2
    
    # 7. Continuous tracking bonus
    tracking_progress = min(1.0, self._time_on_target / cfg.sustained_tracking_time)
    reward += cfg.sustained_tracking_bonus * np.sqrt(tracking_progress)
    
    # 8. Conditional alive bonus
    if abs_error < 0.5:
        reward += cfg.alive_bonus * (1.0 - abs_error / 0.5)
    
    # 9. Crash penalty
    if terminated:
        reward -= cfg.crash_penalty
    
    return reward
```

---

### Changes to reset()

Initialize `_prev_yaw_error`:

```python
def reset(self, ...):
    # ... existing code ...
    self._prev_yaw_error = None  # Add this line
    # ... rest of reset ...
```

---

## Files to Modify

1. `src/environments/yaw_tracking_env.py`
   - Update `YawTrackingConfig` dataclass
   - Replace `_compute_reward()` method
   - Update `reset()` to initialize `_prev_yaw_error`

2. `config/yaw_tracking.yaml` (optional)
   - Update default reward weights

3. `tests/test_environments.py`
   - Add unit tests for new reward function

---

## Acceptance Criteria

- [ ] New reward function implemented
- [ ] All new parameters added to `YawTrackingConfig`
- [ ] `_prev_yaw_error` initialized in `reset()`
- [ ] Unit test: reward is negative when error > 30°
- [ ] Unit test: error_reduction reward works correctly
- [ ] Integration test: P-controller shows >50% tracking
- [ ] Code passes `make check`
- [ ] Documentation updated

---

## Expected Results

| Metric | Before | Expected After |
|--------|--------|----------------|
| Tracking (<10°) | 3.6% | >60% |
| Mean error | 92° | <15° |
| Learning speed | Slow | 2-3x faster |
| Reward signal | Sparse, conflicting | Dense, consistent |

---

## Testing Commands

```bash
# Test reward function manually
python -c "
from src.environments.yaw_tracking_env import YawTrackingEnv, YawTrackingConfig
import numpy as np

env = YawTrackingEnv()
obs, info = env.reset()

# Test with P-controller
on_target = 0
for step in range(500):
    yaw_error = info.get('yaw_error', 0)
    action = np.array([np.clip(yaw_error * 2.0, -1, 1)])
    obs, reward, term, trunc, info = env.step(action)
    if abs(info.get('yaw_error', 0)) < 0.17:  # < 10°
        on_target += 1
    if term or trunc:
        break

print(f'Tracking: {100*on_target/(step+1):.1f}%')
# Should be >50% with new rewards
"

# Run full check
make check
```

---

## Notes

- This is a breaking change for existing trained models
- Models trained with old rewards will need retraining
- Consider keeping old reward function as `_compute_reward_v1()` for comparison
