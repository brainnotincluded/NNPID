# Issue #001: Yaw Torque Sign Inversion

**Date Discovered:** 2026-01-26  
**Severity:** Critical  
**Status:** Resolved  
**Commit Fix:** `bdb96c1`

---

## Summary

Дрон поворачивался в **противоположном направлении** от команды yaw. Положительная команда yaw (повернуть влево) приводила к повороту вправо, и наоборот.

---

## Symptoms

1. **Модель после 20M шагов обучения показывала только 3.6% tracking**
2. Mean yaw error ~92° (почти перпендикулярно к цели)
3. P-контроллер `action = yaw_error * gain` ухудшал ситуацию вместо улучшения
4. Reward падал по мере обучения (с 77 до 32) — модель "оптимизировала" неправильное поведение

---

## Root Cause Analysis

### Цепочка событий в коде

```
action > 0 (want to yaw LEFT)
    ↓
yaw_rate_cmd = action * max_yaw_rate > 0
    ↓
yaw_rate_error = yaw_rate_cmd - omega[2] > 0  (assuming omega ≈ 0)
    ↓
yaw_torque = kp * yaw_rate_error > 0
    ↓
Motor mixing: positive yaw_torque → +m1, +m3 (CCW motors)
    ↓
CCW motors spin faster → CW reaction torque on body
    ↓
Body yaws RIGHT (OPPOSITE of intended!)
```

### Физика проблемы

В квадрокоптере X-конфигурации:
- **CCW моторы** (1, 3) при ускорении создают **CW реактивный момент** на корпус
- **CW моторы** (2, 4) при ускорении создают **CCW реактивный момент** на корпус

Текущий motor mixing:
```python
m1 = thrust + roll - pitch + yaw_torque  # CCW
m2 = thrust + roll + pitch - yaw_torque  # CW
m3 = thrust - roll + pitch + yaw_torque  # CCW
m4 = thrust - roll - pitch - yaw_torque  # CW
```

При `yaw_torque > 0`:
- m1, m3 (CCW) увеличиваются
- m2, m4 (CW) уменьшаются
- **Результат:** больше CW момент на корпус → корпус поворачивается CW (вправо)

Но мы хотели повернуть ВЛЕВО!

---

## Solution

### Вариант 1: Изменить motor mixing (НЕ РЕКОМЕНДУЕТСЯ)

```python
# Инвертировать знаки yaw_torque в mixing
m1 = thrust + roll - pitch - yaw_torque  # CCW
m2 = thrust + roll + pitch + yaw_torque  # CW
m3 = thrust - roll + pitch - yaw_torque  # CCW
m4 = thrust - roll - pitch + yaw_torque  # CW
```

❌ **Проблема:** Это сломало стабилизацию roll/pitch. Дрон падал на шаге 20.

### Вариант 2: Инвертировать yaw_torque calculation (ПРИМЕНЕНО)

```python
# В _compute_stabilizer_control():
yaw_rate_error = yaw_rate_cmd - omega[2]
yaw_torque = np.clip(
    -cfg.yaw_rate_kp * yaw_rate_error,  # ← Добавлен минус!
    -cfg.yaw_authority, 
    cfg.yaw_authority
)
```

✅ **Результат:** Yaw работает правильно, roll/pitch стабильны.

---

## Verification

### До исправления

```python
# P-контроллер: action = yaw_error * 2.0
Step   0: Error=142.2°
Step  40: Error=131.4°  # Ошибка РАСТЁТ!
Step  80: Error=125.6°
```

### После исправления

```python
# Тот же P-контроллер
Step   0: Error= 77.0°
Step  50: Error= 65.8°  # Ошибка УМЕНЬШАЕТСЯ!
Step 100: Error= 29.2°
Step 150: Error=  6.3°
Final:    Error=  3.3°  ✓
```

---

## Prevention Guidelines

### 1. Всегда проверять знак yaw вручную

Перед обучением модели выполнить тест:

```python
# Test: positive action should cause positive yaw rate
env = YawTrackingEnv()
env.reset()

for _ in range(20):
    obs, _, _, _, info = env.step(np.array([1.0]))  # Max positive
    
yaw_rate = info.get("yaw_rate", 0)
assert yaw_rate > 0, f"Yaw sign inverted! Got {yaw_rate}"
print("✓ Yaw sign correct")
```

### 2. Документировать motor physics

В комментариях к motor mixing ВСЕГДА указывать:
- Направление вращения каждого мотора (CCW/CW)
- Какой реактивный момент создаёт на корпус
- Как yaw_torque влияет на каждый мотор

```python
# === MOTOR MIXING (X configuration) ===
# Motor rotation and reaction torque:
#   Motor 1: CCW rotation → CW reaction torque on body
#   Motor 2: CW rotation  → CCW reaction torque on body
#   Motor 3: CCW rotation → CW reaction torque on body
#   Motor 4: CW rotation  → CCW reaction torque on body
#
# To yaw LEFT (CCW, positive ω_z):
#   Need CCW torque on body → increase CW motors (2,4)
#
# yaw_torque is NEGATED in the controller to account for this!
```

### 3. Unit test для yaw direction

Добавить в `tests/test_environments.py`:

```python
def test_yaw_direction():
    """Verify positive action causes positive yaw rate."""
    env = YawTrackingEnv()
    env.reset()
    
    # Apply positive yaw command
    for _ in range(30):
        _, _, term, _, info = env.step(np.array([1.0]))
        if term:
            pytest.fail("Drone crashed during yaw test")
    
    yaw_rate = info.get("yaw_rate", 0)
    assert yaw_rate > 0.01, f"Positive action should cause positive yaw rate, got {yaw_rate}"
    
    env.reset()
    
    # Apply negative yaw command
    for _ in range(30):
        _, _, term, _, info = env.step(np.array([-1.0]))
        if term:
            pytest.fail("Drone crashed during yaw test")
    
    yaw_rate = info.get("yaw_rate", 0)
    assert yaw_rate < -0.01, f"Negative action should cause negative yaw rate, got {yaw_rate}"
```

### 4. Проверять с P-контроллером перед обучением

Простой тест что окружение работает:

```bash
python -c "
from src.environments.yaw_tracking_env import YawTrackingEnv, YawTrackingConfig
import numpy as np

config = YawTrackingConfig(target_speed_min=0.0, target_speed_max=0.0)
env = YawTrackingEnv(config=config)
env.reset()

on_target = 0
for step in range(200):
    yaw_error = env._compute_yaw_error(env.sim.get_state())
    action = np.array([np.clip(yaw_error * 2.0, -1, 1)])
    _, _, term, _, info = env.step(action)
    if abs(info.get('yaw_error', 0)) < 0.1:
        on_target += 1
    if term:
        print(f'FAIL: Crashed at step {step}')
        break
else:
    print(f'OK: Tracking {100*on_target/200:.0f}% with stationary target')
"
```

Ожидаемый результат: `OK: Tracking >30% with stationary target`

---

## Related Files

- `src/environments/yaw_tracking_env.py` — основной файл с исправлением
- `models/quadrotor_x500.xml` — MuJoCo модель с описанием моторов
- `.cursorrules` — документация motor layout

---

## Lessons Learned

1. **Реактивный момент != направление вращения мотора**
   - CCW мотор создаёт CW момент на корпус (и наоборот)
   
2. **Не менять motor mixing без крайней необходимости**
   - Изменения в mixing могут сломать roll/pitch стабилизацию
   - Лучше инвертировать знак в контроллере
   
3. **Тестировать базовую физику перед обучением**
   - Простой P-контроллер должен работать
   - Если не работает — проблема в окружении, не в модели

4. **Падение reward во время обучения — красный флаг**
   - Если reward падает, модель может учить неправильное поведение
   - Остановить обучение и проверить окружение

---

## Timeline

| Time | Event |
|------|-------|
| 02:26 | Запущено mega training (20M steps) |
| 07:20 | Training завершено, reward упал с 77 до 32 |
| 07:30 | Evaluation показал 3.6% tracking, 92° error |
| 08:00 | Обнаружено что P-контроллер не работает |
| 08:30 | Диагностика: yaw sign inverted |
| 09:00 | Первая попытка fix (motor mixing) — сломало стабильность |
| 09:15 | Вторая попытка (yaw_torque negation) — успех |
| 09:20 | P-контроллер показал 13% tracking, 32° error |
| 09:30 | Fix закоммичен (bdb96c1) |
