# Known Issues & Post-Mortems

This directory documents critical issues encountered during development, their root causes, solutions, and prevention guidelines.

## Purpose

1. **Prevent repeat issues** — Learn from past mistakes
2. **Onboarding** — Help new contributors understand common pitfalls
3. **Debugging reference** — Quick lookup when similar symptoms appear

## Issue Index

| # | Title | Severity | Status |
|---|-------|----------|--------|
| [001](001-yaw-sign-inversion.md) | Yaw Torque Sign Inversion | Critical | Resolved |
| [002](002-motor-mixing-stability.md) | Motor Mixing Changes Cause Stability Loss | Critical | Documented |
| [003](003-hover-pid-instability.md) | Hover PID Instability | Critical | Resolved |
| [004](004-config-parameter-mismatch.md) | Config Parameter Mismatch (YAML vs Python) | High | Resolved |
| [005](005-webots-sitl-protocol-mismatch.md) | Webots-SITL Protocol Mismatch | Critical | Resolved |

## Quick Diagnostic Guide

### Drone crashes immediately (step < 30)

1. Check hover stability: `action = [0.0]` should work
2. If fails → check PID gains and control frequency → see [Issue #003](003-hover-pid-instability.md)
3. If fails after PID changes → likely motor mixing issue → see [Issue #002](002-motor-mixing-stability.md)

### Model doesn't learn / reward decreases

1. Test with P-controller: `action = yaw_error * 2.0`
2. If tracking gets WORSE → yaw sign inverted → see [Issue #001](001-yaw-sign-inversion.md)
3. If tracking doesn't improve → check reward function → see [Ticket #001](../tickets/001-reward-system-upgrade.md)

### Yaw tracking poor but stable

1. Check yaw_authority (should be 0.01-0.03)
2. Check target speed vs drone capability
3. Review reward function incentives

### Webots simulation freezes at 0.00x speed

1. Check SITL is running: `ps aux | grep arducopter`
2. Check protocol compatibility → see [Issue #005](005-webots-sitl-protocol-mismatch.md)
3. Verify JSON format with debug output
4. Check for port conflicts on 9002

## Prevention Checklist

Before training a model, verify:

- [ ] `action = [0.0]` holds hover for 100+ steps
- [ ] `action = [1.0]` causes positive yaw rate
- [ ] `action = [-1.0]` causes negative yaw rate
- [ ] P-controller improves tracking (error decreases)
- [ ] Reward is negative when error > 30°

```bash
# Quick verification script
make check && python -c "
from src.environments.yaw_tracking_env import YawTrackingEnv
import numpy as np

env = YawTrackingEnv()

# Test 1: Hover
env.reset()
for _ in range(50):
    _, _, t, _, _ = env.step(np.array([0.0]))
    if t: exit('FAIL: Hover unstable')
print('✓ Hover stable')

# Test 2: Yaw signs
env.reset()
for _ in range(20):
    _, _, t, _, i = env.step(np.array([1.0]))
if i.get('yaw_rate', 0) <= 0: exit('FAIL: Yaw sign wrong')
print('✓ Yaw sign correct')

# Test 3: P-controller
env.reset()
errors = []
for _ in range(100):
    e = env._compute_yaw_error(env.sim.get_state())
    errors.append(abs(e))
    _, _, t, _, _ = env.step(np.array([np.clip(e * 2, -1, 1)]))
    if t: break
if errors[-1] > errors[0]: exit('FAIL: P-controller not improving')
print('✓ P-controller works')

print('All checks passed!')
"
```

## Template for New Issues

When documenting a new issue, include:

1. **Summary** — One sentence description
2. **Symptoms** — What did you observe?
3. **Root Cause** — Why did it happen?
4. **Solution** — How was it fixed?
5. **Prevention** — How to avoid in future?
6. **Related Files** — What code is involved?
