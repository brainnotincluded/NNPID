# Использование обученной модели

## Быстрый старт

```python
from src.deployment.trained_yaw_tracker import TrainedYawTracker

# Загрузить модель
tracker = TrainedYawTracker.from_path("runs/analysis_20260126_150455/best_model")

# В вашем цикле управления
yaw_rate_cmd = tracker.predict(observation, deterministic=True)
```

## Описание класса

### `TrainedYawTracker`

Обертка для обученной нейросети, которая отслеживает цель по yaw.

#### Методы

##### `from_path(model_path, config=None) -> TrainedYawTracker`

**Статический метод** для загрузки модели из файла.

**Параметры:**
- `model_path` (str | Path): Путь к модели
  - Если директория: ищет `best_model.zip` или `final_model.zip`
  - Если файл: загружает напрямую
- `config` (YawTrackingConfig, optional): Конфигурация окружения

**Возвращает:**
- `TrainedYawTracker`: Загруженный контроллер

**Пример:**
```python
tracker = TrainedYawTracker.from_path("runs/best_model")
```

##### `predict(observation, deterministic=True) -> float`

Получить команду yaw rate из наблюдения.

**Параметры:**
- `observation` (np.ndarray): Вектор наблюдения (11 элементов)
  - `[0]` target_dir_x: X компонент направления к цели
  - `[1]` target_dir_y: Y компонент направления к цели
  - `[2]` target_angular_vel: Угловая скорость цели (rad/s)
  - `[3]` current_yaw_rate: Текущая скорость yaw (rad/s)
  - `[4]` yaw_error: Ошибка yaw (rad)
  - `[5]` roll: Угол roll (rad)
  - `[6]` pitch: Угол pitch (rad)
  - `[7]` altitude_error: Ошибка высоты (m)
  - `[8]` velocity_x: Скорость X (m/s)
  - `[9]` velocity_y: Скорость Y (m/s)
  - `[10]` previous_action: Предыдущее действие

- `deterministic` (bool): Использовать детерминированную политику (True для deployment)

**Возвращает:**
- `float`: Команда yaw rate в диапазоне [-1, 1]
  - `+1.0`: Максимальная положительная скорость yaw
  - `-1.0`: Максимальная отрицательная скорость yaw
  - `0.0`: Нет команды yaw

**Пример:**
```python
yaw_cmd = tracker.predict(obs, deterministic=True)
# Масштабировать до реальной скорости: yaw_rate = yaw_cmd * max_yaw_rate
```

##### `reset() -> None`

Сбросить внутреннее состояние (сейчас no-op, можно расширить).

##### `get_info() -> dict`

Получить информацию о модели.

**Возвращает:**
- `dict` с ключами:
  - `model_type`: Тип модели (например, "PPO")
  - `observation_space`: Размер вектора наблюдения
  - `has_normalization`: Загружен ли VecNormalize
  - `config`: Конфигурация окружения

## Пример интеграции

```python
from src.deployment.trained_yaw_tracker import TrainedYawTracker
import numpy as np

# 1. Загрузить модель один раз при старте
tracker = TrainedYawTracker.from_path("runs/best_model")

# 2. В вашем цикле управления
while running:
    # Получить текущее состояние
    state = get_current_state()
    
    # Построить вектор наблюдения (11 элементов)
    obs = build_observation(state, target)
    
    # Получить команду от обученной модели
    yaw_rate_cmd = tracker.predict(obs, deterministic=True)
    
    # Использовать команду с вашим стабилизатором
    # Масштабировать: yaw_rate_cmd в [-1, 1] -> реальная скорость в rad/s
    max_yaw_rate = 2.0  # rad/s
    actual_yaw_rate = yaw_rate_cmd * max_yaw_rate
    
    motors = stabilizer.compute_motors(
        state,
        yaw_rate_cmd=actual_yaw_rate,
        dt=dt
    )
    
    # Применить команды к моторам
    apply_motors(motors)
```

## Формат наблюдения

Модель ожидает вектор из 11 элементов:

```python
observation = np.array([
    target_dir_x,        # [0] X компонент направления к цели
    target_dir_y,        # [1] Y компонент направления к цели
    target_angular_vel,  # [2] Угловая скорость цели (rad/s)
    current_yaw_rate,    # [3] Текущая скорость yaw (rad/s)
    yaw_error,           # [4] Ошибка yaw (rad)
    roll,                # [5] Угол roll (rad)
    pitch,               # [6] Угол pitch (rad)
    altitude_error,      # [7] Ошибка высоты (m)
    velocity_x,          # [8] Скорость X (m/s)
    velocity_y,          # [9] Скорость Y (m/s)
    previous_action,     # [10] Предыдущее действие
], dtype=np.float32)
```

## Важные замечания

1. **Нормализация**: Модель автоматически применяет VecNormalize если он был сохранен при обучении
2. **Детерминизм**: Используйте `deterministic=True` для deployment, `False` для exploration
3. **Масштабирование**: Команда в [-1, 1] должна быть умножена на `max_yaw_rate` для получения реальной скорости
4. **Формат наблюдения**: Должен точно соответствовать формату обучения (11 элементов)

## Полный пример

См. `examples/use_trained_model.py` для полного рабочего примера.
