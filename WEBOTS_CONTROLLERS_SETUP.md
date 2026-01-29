# Webots Controllers Setup

## Проблема

Webots сцена `iris_camera_human.wbt` требует контроллеры из ArduPilot:
- `ardupilot_vehicle_controller` - для дрона Iris
- `pedestrian` - для движения человека

## Решение 1: Использовать ArduPilot контроллеры (Рекомендуется)

### Скопировать из ArduPilot репозитория

Если у вас уже склонирован ArduPilot:

```bash
cd /Users/mac/projects/NNPID

# Скопировать контроллеры из ArduPilot
cp -r ~/ardupilot/libraries/SITL/examples/Webots_Python/controllers ./

# Проверить
ls controllers/
# Должно показать:
# ardupilot_vehicle_controller/
# pedestrian/
```

### Или скачать из GitHub

```bash
cd /Users/mac/projects/NNPID

# Клонировать ArduPilot (только контроллеры)
git clone --depth 1 https://github.com/ArduPilot/ardupilot.git temp_ap
cp -r temp_ap/libraries/SITL/examples/Webots_Python/controllers ./
rm -rf temp_ap

# Проверить
ls -la controllers/
```

## Решение 2: Запустить сцену из ArduPilot (Альтернатива)

Вместо копирования контроллеров, запустите сцену из папки ArduPilot:

```bash
# Перейти в ArduPilot Webots примеры
cd ~/ardupilot/libraries/SITL/examples/Webots_Python/worlds

# Запустить любую сцену (например iris_camera.wbt)
webots iris_camera.wbt
```

Затем в нашем проекте можно адаптировать эту сцену.

## Решение 3: Использовать встроенный контроллер Webots (Для тестирования)

Временно можно изменить контроллер в `.wbt` файле на `<none>` чтобы протестировать загрузку сцены:

```vrml
Iris {
    controller "<none>"  # Вместо "ardupilot_vehicle_controller"
    ...
}

Pedestrian {
    controller "<none>"  # Вместо "pedestrian"
    ...
}
```

## Структура контроллеров ArduPilot

```
controllers/
├── ardupilot_vehicle_controller/
│   ├── ardupilot_vehicle_controller.py  # Главный контроллер
│   ├── webots_vehicle.py                # Класс для ArduPilot интеграции
│   └── drone_interface.py               # MAVLink/JSON интерфейс
├── pedestrian/
│   └── pedestrian.py                    # Контроллер движения человека
└── (другие контроллеры...)
```

## Быстрая настройка (рекомендуется)

Самый простой способ - скопировать всю папку controllers из ArduPilot:

```bash
#!/bin/bash
cd /Users/mac/projects/NNPID

# Если ArduPilot уже установлен
if [ -d ~/ardupilot ]; then
    echo "Copying controllers from ArduPilot..."
    cp -r ~/ardupilot/libraries/SITL/examples/Webots_Python/controllers ./
    echo "✅ Controllers copied"
else
    echo "ArduPilot not found. Cloning..."
    git clone --depth 1 https://github.com/ArduPilot/ardupilot.git /tmp/ardupilot_temp
    cp -r /tmp/ardupilot_temp/libraries/SITL/examples/Webots_Python/controllers ./
    rm -rf /tmp/ardupilot_temp
    echo "✅ Controllers downloaded and copied"
fi

ls -la controllers/
```

Сохраните как `setup_controllers.sh` и запустите:
```bash
chmod +x setup_controllers.sh
./setup_controllers.sh
```

## Проверка установки

После копирования контроллеров, проверьте:

```bash
ls controllers/ardupilot_vehicle_controller/
# Должны быть: ardupilot_vehicle_controller.py, webots_vehicle.py, ...

ls controllers/pedestrian/
# Должен быть: pedestrian.py
```

## После установки контроллеров

1. Запустите Webots: `./run_webots.sh`
2. Проверьте console в Webots - контроллеры должны запуститься
3. Запустите SITL: `sim_vehicle.py -v ArduCopter -f webots-quad`
4. Запустите трекер: `python webots_human_tracker.py --model runs/best_model`

## Troubleshooting

### "ardupilot_vehicle_controller.py: No module named 'webots_vehicle'"

**Причина**: Не все файлы контроллера скопированы

**Решение**: Скопируйте всю папку `controllers/` из ArduPilot

### "pedestrian.py: command not found"

**Причина**: Webots не может найти контроллер

**Решение**: Убедитесь что `controllers/pedestrian/pedestrian.py` существует и имеет права на выполнение

### Контроллер не запускается

**Проверьте**:
1. Права на выполнение: `chmod +x controllers/*/. *.py`
2. Shebang в начале файла: `#!/usr/bin/env python3`
3. Python версия: контроллеры требуют Python 3.7+

## Ссылки

- [ArduPilot Webots GitHub](https://github.com/ArduPilot/ardupilot/tree/master/libraries/SITL/examples/Webots_Python)
- [Webots Controllers Documentation](https://cyberbotics.com/doc/guide/controller-programming)
- [ArduPilot SITL Documentation](https://ardupilot.org/dev/docs/sitl-with-webots-python.html)

---

После установки контроллеров вернитесь к [WEBOTS_QUICKSTART.md](WEBOTS_QUICKSTART.md)
