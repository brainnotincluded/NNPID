# Issue #002: Motor Mixing Changes Cause Stability Loss

**Date Discovered:** 2026-01-26  
**Severity:** Critical  
**Status:** Documented (Prevention)  
**Related:** Issue #001

---

## Summary

Изменение знаков в motor mixing matrix для исправления yaw приводило к **полной потере стабильности** — дрон падал на шаге 20-22 даже без yaw команды.

---

## Symptoms

1. Дрон стабилен с `action = 0` в оригинальном коде
2. После изменения yaw signs в motor mixing:
   - Roll раскачивается: 0° → -83° за 20 шагов
   - Pitch раскачивается: 0° → -32° за 20 шагов
   - Crash на шаге 20-22
3. Паттерн повторялся стабильно (не случайный)

---

## Attempted Change (WRONG)

```python
# WRONG: Changing motor mixing signs
# Original:
m1 = thrust + roll_torque - pitch_torque + yaw_torque  # CCW
m2 = thrust + roll_torque + pitch_torque - yaw_torque  # CW
m3 = thrust - roll_torque + pitch_torque + yaw_torque  # CCW
m4 = thrust - roll_torque - pitch_torque - yaw_torque  # CW

# Changed to (BREAKS STABILITY):
m1 = thrust + roll_torque - pitch_torque - yaw_torque  # CCW
m2 = thrust + roll_torque + pitch_torque + yaw_torque  # CW
m3 = thrust - roll_torque + pitch_torque - yaw_torque  # CCW
m4 = thrust - roll_torque - pitch_torque + yaw_torque  # CW
```

---

## Root Cause

### Motor Mixing Matrix Structure

Motor mixing для X-квадрокоптера имеет специфическую структуру где **все знаки взаимосвязаны**:

```
Motor Command = Thrust ± Roll ± Pitch ± Yaw
```

| Motor | Position | Roll | Pitch | Yaw |
|-------|----------|------|-------|-----|
| 1 (FL) | +X, +Y | + | - | + |
| 2 (BL) | -X, +Y | + | + | - |
| 3 (BR) | -X, -Y | - | + | + |
| 4 (FR) | +X, -Y | - | - | - |

### Почему изменение Yaw ломает Roll/Pitch

1. **Coupling через thrust distribution:**
   - Изменение yaw signs меняет распределение thrust между левой/правой и передней/задней парами
   - Это создаёт паразитные roll/pitch моменты

2. **Feedback loop:**
   - Паразитный roll вызывает коррекцию от PID
   - PID даёт roll_torque
   - Roll_torque с изменёнными yaw signs работает неправильно
   - Система становится нестабильной

3. **Step 20 pattern:**
   - ~10 шагов: ошибка накапливается медленно
   - ~10 шагов: PID начинает корректировать
   - Шаг 20: коррекция усиливает ошибку вместо уменьшения
   - Crash

---

## Correct Solution

**НЕ трогать motor mixing!** Вместо этого инвертировать знак в yaw controller:

```python
# В _compute_stabilizer_control():
yaw_rate_error = yaw_rate_cmd - omega[2]
yaw_torque = np.clip(
    -cfg.yaw_rate_kp * yaw_rate_error,  # ← Минус здесь, не в mixing!
    -cfg.yaw_authority, 
    cfg.yaw_authority
)
```

---

## Prevention Guidelines

### 1. НИКОГДА не менять motor mixing без симуляции

Motor mixing — это **физическая константа** для данной конфигурации дрона. Она определяется:
- Расположением моторов
- Направлением вращения
- Геометрией рамы

### 2. Если нужно изменить поведение — меняй controller, не mixing

| Проблема | Неправильно | Правильно |
|----------|-------------|-----------|
| Yaw инвертирован | Менять yaw в mixing | Инвертировать в controller |
| Roll инвертирован | Менять roll в mixing | Проверить систему координат |
| Слабый yaw | Увеличить yaw в mixing | Увеличить yaw_authority |

### 3. Тест стабильности перед любыми изменениями

```python
def test_hover_stability():
    """Drone should hover stable with zero action."""
    env = YawTrackingEnv()
    env.reset()
    
    for step in range(100):
        _, _, term, _, _ = env.step(np.array([0.0]))
        if term:
            pytest.fail(f"Drone crashed at step {step} with zero action")
    
    # Check final orientation
    state = env.sim.get_state()
    euler = Rotations.quaternion_to_euler(state.quaternion)
    roll, pitch = abs(euler[0]), abs(euler[1])
    
    assert roll < 0.1, f"Roll unstable: {np.rad2deg(roll):.1f}°"
    assert pitch < 0.1, f"Pitch unstable: {np.rad2deg(pitch):.1f}°"
```

### 4. При изменениях — инкрементальное тестирование

```bash
# После каждого изменения:
python -c "
from src.environments.yaw_tracking_env import YawTrackingEnv
import numpy as np

env = YawTrackingEnv()
env.reset()

for step in range(100):
    _, _, term, _, _ = env.step(np.array([0.0]))
    if term:
        print(f'FAIL: Crashed at step {step}')
        exit(1)

print('OK: Hover stable for 100 steps')
"
```

---

## Motor Mixing Reference

### X-Configuration (наша конфигурация)

```
    Front
   1     4
    \   /
     \ /
      X
     / \
    /   \
   2     3
    Back

Motor 1 (FL): CCW, position (+X, +Y)
Motor 2 (BL): CW,  position (-X, +Y)
Motor 3 (BR): CCW, position (-X, -Y)
Motor 4 (FR): CW,  position (+X, -Y)
```

### Mixing Matrix (DO NOT CHANGE)

```python
m1 = T + R - P + Y   # Front-Left,  CCW
m2 = T + R + P - Y   # Back-Left,   CW
m3 = T - R + P + Y   # Back-Right,  CCW
m4 = T - R - P - Y   # Front-Right, CW
```

### Физический смысл знаков

| Term | Положительный | Влияет на |
|------|---------------|-----------|
| Thrust (T) | Все моторы вверх | Высота |
| Roll (R) | Левые вверх, правые вниз | Крен влево |
| Pitch (P) | Задние вверх, передние вниз | Тангаж назад |
| Yaw (Y) | CCW вверх, CW вниз | Поворот влево* |

*Yaw требует негации в controller из-за реактивного момента

---

## Diagnostic Commands

### Проверка стабильности hover

```bash
python -c "
from src.environments.yaw_tracking_env import YawTrackingEnv
from src.utils.rotations import Rotations
import numpy as np

env = YawTrackingEnv()
env.reset()

print('Hover stability test (action=0):')
for step in range(50):
    _, _, term, _, _ = env.step(np.array([0.0]))
    
    state = env.sim.get_state()
    euler = Rotations.quaternion_to_euler(state.quaternion)
    roll, pitch = np.rad2deg(euler[0]), np.rad2deg(euler[1])
    
    if step % 10 == 0:
        print(f'  Step {step:2d}: Roll={roll:+5.1f}° Pitch={pitch:+5.1f}°')
    
    if term:
        print(f'FAIL: Crashed at step {step}')
        exit(1)

print('OK: Hover stable')
"
```

### Проверка motor mixing

```bash
python -c "
from src.environments.yaw_tracking_env import YawTrackingEnv
import numpy as np

env = YawTrackingEnv()
env.reset()

# Check motor commands for pure roll
print('Motor commands analysis:')
print('Pure thrust (T=0.6):')
print('  Expected: [0.6, 0.6, 0.6, 0.6]')

# This would require access to internal motor commands
# which we can verify by observing behavior
"
```

---

## Lessons Learned

1. **Motor mixing is physics, not software**
   - Определяется hardware конфигурацией
   - Изменения требуют изменения физической модели

2. **Coupling между осями**
   - Roll, pitch, yaw не независимы в mixing
   - Изменение одного влияет на другие

3. **Controller vs Mixing**
   - Поведенческие изменения → controller
   - Физические изменения → mixing (редко нужно)

4. **Step 20 pattern**
   - Типичный признак feedback instability
   - Система начинает осциллировать и diverges
