# Issue #004: Config Parameter Mismatch (YAML vs Python)

**Date Discovered:** 2026-01-29  
**Severity:** High  
**Status:** Resolved  
**Related:** Issue #003 (hover-pid-instability)

---

## Summary

Critical mismatch between YAML configuration (`config/yaw_tracking.yaml`) and Python code defaults (`src/environments/yaw_tracking_env.py`) for 3 key parameters:

1. **crash_penalty**: 5x difference (10.0 vs 50.0)
2. **alive_bonus**: 2x difference (0.05 vs 0.1)  
3. **yaw_authority**: 6.67x difference (0.03 vs 0.20)

This caused **unpredictable training behavior** depending on whether configs were loaded from YAML or used Python defaults.

---

## Impact

### Bug 1: crash_penalty Mismatch

| Source | Value | Impact |
|--------|-------|--------|
| YAML | 10.0 | Too lenient - doesn't discourage crashes enough |
| Python | 50.0 | Correct - strongly penalizes crashes |

**Effect**: Models trained with YAML config may be **5x less risk-averse**, leading to more crashes during deployment.

### Bug 2: alive_bonus Mismatch

| Source | Value | Impact |
|--------|-------|--------|
| YAML | 0.05 | Lower survival reward |
| Python | 0.1 | Correct - better reward shaping |

**Effect**: YAML config provides **50% less survival incentive**, potentially biasing toward riskier strategies.

### Bug 3: yaw_authority Mismatch

| Source | Value | Max Yaw Rate | Impact |
|--------|-------|--------------|--------|
| YAML | 0.03 | ~0.05 rad/s | Too limited - can't track targets |
| Python | 0.20 | ~0.6 rad/s | Correct - effective tracking |

**Effect**: Most critical - YAML value makes **tracking physically impossible** for targets moving faster than 0.05 rad/s. This contradicts the documented fix in Issue #003.

---

## Root Cause

During debugging in Issue #003, the Python code was updated with new values after extensive testing:

```python
# src/environments/yaw_tracking_env.py (lines 542-543, 570)
crash_penalty: float = 50.0  # Increased to strongly discourage crashes
alive_bonus: float = 0.1     # Increased to reward survival
yaw_authority: float = 0.20  # Yaw torque authority (allows ~0.6 rad/s yaw)
```

However, the YAML config was **not updated** to match, leaving old values from earlier iterations.

---

## Detection

Discovered through code review when comparing:
- `config/yaw_tracking.yaml` lines 63-64, 99
- `src/environments/yaw_tracking_env.py` lines 542-543, 570

```bash
# YAML had old values
crash_penalty: 10.0
alive_bonus: 0.05
yaw_authority: 0.03

# Python had debugged values
crash_penalty: float = 50.0
alive_bonus: float = 0.1
yaw_authority: float = 0.20
```

---

## Fix

Updated `config/yaw_tracking.yaml` to match Python defaults:

### Before (Incorrect)
```yaml
crash_penalty: 10.0
alive_bonus: 0.05              # Reduced from 0.1 (now conditional)
yaw_authority: 0.03          # max yaw torque (balanced for tracking)
```

### After (Correct)
```yaml
crash_penalty: 50.0            # Strongly discourage crashes (updated from 10.0)
alive_bonus: 0.1               # Reward survival (updated from 0.05)
yaw_authority: 0.20          # max yaw torque (allows ~0.6 rad/s yaw, updated from 0.03)
```

---

## Verification

### Test 1: Config Loading

```python
from src.config import load_config

config = load_config("config/yaw_tracking.yaml")
assert config['environment']['rewards']['crash_penalty'] == 50.0
assert config['environment']['rewards']['alive_bonus'] == 0.1
assert config['environment']['stabilizer']['yaw_authority'] == 0.20
```

### Test 2: YawTrackingConfig Defaults

```python
from src.environments.yaw_tracking_env import YawTrackingConfig

cfg = YawTrackingConfig()
assert cfg.crash_penalty == 50.0
assert cfg.alive_bonus == 0.1
assert cfg.yaw_authority == 0.20
```

### Test 3: Training Consistency

```bash
# Train with YAML config
python scripts/train_yaw_tracker.py --config config/yaw_tracking.yaml

# Verify parameters in logs
# Should show crash_penalty=50.0, not 10.0
```

---

## Prevention

### 1. Add Config Validation

Create `tests/test_config_consistency.py`:

```python
def test_yaml_matches_python_defaults():
    """Ensure YAML configs match Python dataclass defaults."""
    from src.config import load_config
    from src.environments.yaw_tracking_env import YawTrackingConfig
    
    yaml_cfg = load_config("config/yaw_tracking.yaml")
    py_cfg = YawTrackingConfig()
    
    # Compare critical parameters
    assert yaml_cfg['environment']['rewards']['crash_penalty'] == py_cfg.crash_penalty
    assert yaml_cfg['environment']['rewards']['alive_bonus'] == py_cfg.alive_bonus
    assert yaml_cfg['environment']['stabilizer']['yaw_authority'] == py_cfg.yaw_authority
```

### 2. Documentation Standard

When updating parameters in Python code, **ALWAYS**:
1. Update corresponding YAML config
2. Add comment explaining change
3. Reference related issues
4. Update docs if needed

Example:
```python
# Updated from 0.03 to 0.20 (see Issue #003)
# Allows ~0.6 rad/s yaw rate for effective tracking
yaw_authority: float = 0.20
```

### 3. Pre-commit Hook

Add to `.pre-commit-config.yaml`:
```yaml
- id: check-config-consistency
  name: Check YAML-Python config consistency
  entry: python -m pytest tests/test_config_consistency.py
  language: system
  pass_filenames: false
```

---

## Lessons Learned

1. **Config files are code** - treat YAML with same rigor as Python
2. **Single source of truth** - either YAML or Python, not both
3. **Test config loading** - automated tests catch mismatches
4. **Document changes** - explain WHY values changed
5. **Cross-reference issues** - link to debugging history

---

## Related Issues

- **Issue #003**: Original PID tuning that led to yaw_authority increase
- **Training logs**: Models trained with old YAML likely underperformed

---

## Status

**RESOLVED** - All three parameters synchronized:
- ✅ crash_penalty: 50.0 (both YAML and Python)
- ✅ alive_bonus: 0.1 (both YAML and Python)
- ✅ yaw_authority: 0.20 (both YAML and Python)

---

## Commit

```
fix: synchronize YAML config with Python defaults

- Update crash_penalty: 10.0 → 50.0 (matches Python)
- Update alive_bonus: 0.05 → 0.1 (matches Python)
- Update yaw_authority: 0.03 → 0.20 (matches Python)

These values were debugged in Issue #003 but YAML was not updated.
Mismatch caused training inconsistency depending on config load method.

Fixes #004
```
