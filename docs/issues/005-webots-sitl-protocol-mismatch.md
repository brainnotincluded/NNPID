# Issue #005: Webots-SITL Protocol Mismatch

**Date Discovered:** 2026-01-29  
**Severity:** Critical  
**Status:** Resolved  
**Related:** ArduPilot JSON SITL Backend

---

## Summary

Webots controller не мог связаться с ArduPilot SITL из-за несовместимости протоколов. Симуляция зависала на 0.00x скорости, SITL показывал "No JSON sensor message received".

---

## Symptoms

1. Webots показывает **0.00x** скорость симуляции
2. SITL выводит: `No JSON sensor message received, resending servos`
3. MAVProxy показывает: `link 1 down`
4. Контроллер висит на "Connecting to ardupilot SITL..."
5. CPU usage контроллера ~100% (busy loop)

---

## Root Cause Analysis

### Проблема 1: Неправильный формат входных данных

**Было (неправильно):**
```python
# Отправляли бинарный struct
fdm_struct_format = 'd'*(1+3+3+3+3+3)  # 16 doubles
return struct.pack(self.fdm_struct_format, timestamp, gyro..., accel..., ...)
```

**Должно быть:**
```python
# SITL JSON backend ожидает JSON с переносами строк!
fdm = {
    "timestamp": time,
    "imu": {"gyro": [gx,gy,gz], "accel_body": [ax,ay,az]},
    "position": [north, east, down],  # NED метры, НЕ lat/lon!
    "attitude": [roll, pitch, yaw],
    "velocity": [vn, ve, vd]
}
json_str = "\n" + json.dumps(fdm) + "\n"  # Обязательно \n!
```

### Проблема 2: Неправильный формат выходных данных

**Ожидали (неправильно):**
```python
# Думали что SITL отправляет JSON
response = json.loads(data.decode('utf-8'))
servos = response['servos']
```

**На самом деле:**
```python
# SITL отправляет БИНАРНЫЕ данные (40 байт)!
# Format: uint16 magic + uint16 frame_rate + uint32 frame_count + uint16 pwm[16]
SITL_OUTPUT_FORMAT = '<HHI16H'  # little-endian
magic, frame_rate, frame_count, *pwm = struct.unpack(SITL_OUTPUT_FORMAT, data)
# magic должен быть 18458
```

### Проблема 3: Неправильная система координат

**Было:**
```python
# Отправляли позицию как [lat, lon, alt] или в Webots frame
"position": [gps_pos[0], gps_pos[1], gps_pos[2]]
```

**Должно быть:**
```python
# ArduPilot ожидает NED (North-East-Down) в метрах!
# Webots: X=East, Y=North, Z=Up -> NED: swap X/Y, negate Z
"position": [gps_pos[1], gps_pos[0], -gps_pos[2]]  # [North, East, Down]
```

### Проблема 4: Конфликт портов

**Было:**
```python
# Оба слушали на 9002
s.bind(('0.0.0.0', port))  # Webots
# SITL тоже слушает на 9002!
```

**Исправлено:**
```python
# Не биндимся к фиксированному порту
# Используем ephemeral port, SITL ответит на source port
sock.setblocking(False)
# НЕ вызываем bind()
```

### Проблема 5: Deadlock при установке соединения

**Было:**
```python
# Ждали данные от SITL перед отправкой
while not select.select([s], [], [], 0)[0]:
    pass  # Бесконечный цикл!
```

**Должно быть:**
```python
# Отправляем данные ПЕРВЫМИ
sock.sendto(json_data, (sitl_address, port))
# Потом ждём ответ
readable, _, _ = select.select([sock], [], [], 0.02)
```

---

## Protocol Specification

### SITL Input (Physics Backend → SITL)

**Format:** JSON с переносами строк

```
\n{"timestamp":T,"imu":{"gyro":[r,p,y],"accel_body":[x,y,z]},"position":[n,e,d],"attitude":[r,p,y],"velocity":[n,e,d]}\n
```

| Field | Type | Units | Frame |
|-------|------|-------|-------|
| timestamp | float | seconds | - |
| imu.gyro | [float, float, float] | rad/s | body NED |
| imu.accel_body | [float, float, float] | m/s² | body NED |
| position | [float, float, float] | meters | world NED |
| attitude | [float, float, float] | radians | roll, pitch, yaw |
| velocity | [float, float, float] | m/s | world NED |

### SITL Output (SITL → Physics Backend)

**Format:** Binary (little-endian), 40 bytes

```
struct sitl_output {
    uint16_t magic;       // 18458 (или 29569 для 32 каналов)
    uint16_t frame_rate;  // Hz
    uint32_t frame_count; // Счётчик кадров
    uint16_t pwm[16];     // PWM значения (1000-2000 μs)
};
```

**Python struct format:** `<HHI16H`

---

## Coordinate Frame Conversion

### Webots → ArduPilot NED

```
Webots World Frame (ENU-like):
  X = East
  Y = North  
  Z = Up

ArduPilot NED Frame:
  X = North
  Y = East
  Z = Down

Conversion:
  position_ned = [pos_webots[1], pos_webots[0], -pos_webots[2]]
  velocity_ned = [vel_webots[1], vel_webots[0], -vel_webots[2]]
  
Body Frame (gyro, accel):
  Webots: X=forward, Y=left, Z=up
  NED:    X=forward, Y=right, Z=down
  
  gyro_ned = [gyro[0], -gyro[1], -gyro[2]]
  accel_ned = [accel[0], -accel[1], -accel[2]]
```

---

## Fix Applied

Полностью переписан `controllers/ardupilot_vehicle_controller/webots_vehicle.py`:

### Ключевые изменения:

1. **JSON формат для sensor data:**
```python
def _get_sensor_json(self) -> bytes:
    fdm = {
        "timestamp": self.robot.getTime(),
        "imu": {
            "gyro": [gyro[0], -gyro[1], -gyro[2]],
            "accel_body": [accel[0], -accel[1], -accel[2]]
        },
        "position": [pos[1], pos[0], -pos[2]],
        "attitude": [rpy[0], -rpy[1], -rpy[2]],
        "velocity": [vel[1], vel[0], -vel[2]]
    }
    return ("\n" + json.dumps(fdm) + "\n").encode('utf-8')
```

2. **Binary парсинг servo output:**
```python
SITL_OUTPUT_FORMAT = '<HHI16H'
SITL_OUTPUT_SIZE = 40  # bytes

def _parse_sitl_output(self, data: bytes) -> bool:
    unpacked = struct.unpack(self.SITL_OUTPUT_FORMAT, data[:40])
    magic = unpacked[0]
    if magic != 18458:
        return False
    pwm = unpacked[3:19]
    motor_commands = [(p - 1000) / 1000.0 for p in pwm]
    self._apply_motor_commands(motor_commands)
    return True
```

3. **Правильная последовательность соединения:**
```python
# Отправляем JSON первыми
sock.sendto(json_data, (sitl_address, port))
# Ждём binary ответ
readable, _, _ = select.select([sock], [], [], 0.02)
if readable:
    data, addr = sock.recvfrom(1024)
    self._parse_sitl_output(data)
```

---

## Verification

### Успешное подключение выглядит так:

**Webots console:**
```
Connecting to ArduPilot SITL (I0) at 127.0.0.1:9002
Expected SITL output size: 40 bytes
First JSON packet sent (142 bytes)
Sample: b'\n{"timestamp":0.002,"imu":{"gyro":[0,0,0]...
First response received (40 bytes) from ('127.0.0.1', 9002)
Magic number: 18458 (expected: 18458)
Connected to ArduPilot SITL (I0)
```

**SITL (ArduCopter terminal):**
```
JSON received:
 timestamp
 gyro
 accel_body
 position
 attitude
 velocity
```

**MAVProxy:**
```
MAV> link 1 OK
```

### Тест скорости симуляции:

Webots должен показывать **1.00x** (или близко к этому), не **0.00x**.

---

## Prevention

### 1. Документация протокола

Всегда проверять официальную документацию ArduPilot:
- https://ardupilot.org/dev/docs/sitl-with-JSON.html
- https://github.com/ArduPilot/ardupilot/tree/master/libraries/SITL/examples/JSON

### 2. Debug output

Добавлять отладочный вывод при работе с сетевыми протоколами:
```python
print(f"Sent {len(data)} bytes")
print(f"Received {len(response)} bytes, magic={magic}")
```

### 3. Тестирование протокола

Перед интеграцией тестировать протокол отдельно:
```python
# Тест JSON формата
import json
test_fdm = {"timestamp": 0, "imu": {"gyro": [0,0,0], "accel_body": [0,0,-9.8]}, ...}
print(json.dumps(test_fdm))

# Тест binary парсинга
import struct
test_data = struct.pack('<HHI16H', 18458, 400, 0, *[1500]*16)
assert len(test_data) == 40
```

### 4. Reference implementation

Использовать примеры из ArduPilot репозитория как reference:
```bash
ls ~/ardupilot/libraries/SITL/examples/JSON/
# example.py  - Python пример
# readme.md   - Документация протокола
```

---

## Related Issues

- **Issue #003**: Hover PID Instability (косвенно связано - неправильные данные от симулятора)
- **Issue #004**: Config Parameter Mismatch

---

## References

- [ArduPilot SITL JSON Documentation](https://ardupilot.org/dev/docs/sitl-with-JSON.html)
- [ArduPilot JSON Examples](https://github.com/ArduPilot/ardupilot/tree/master/libraries/SITL/examples/JSON)
- [Webots Controller API](https://cyberbotics.com/doc/reference/robot)

---

## Commits

```
e8d0530 fix: rewrite Webots-SITL controller with correct JSON protocol
```
