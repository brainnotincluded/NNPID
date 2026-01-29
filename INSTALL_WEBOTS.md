# Webots Installation Guide (macOS)

## Вариант 1: Уже установлен! ✅

У вас уже установлен **Webots R2025a** в `/Applications/Webots.app`

### Быстрый запуск

```bash
# Используйте наш launcher script
./run_webots.sh
```

Или добавьте alias в `~/.zshrc`:

```bash
# Добавьте эту строку в ~/.zshrc
alias webots='/Applications/Webots.app/Contents/MacOS/webots'

# Перезагрузите конфиг
source ~/.zshrc

# Теперь можно:
webots iris_camera_human.wbt
```

## Вариант 2: Установить через Homebrew

Если хотите управлять через brew:

```bash
# Установить Webots
brew install --cask webots

# Запустить
webots iris_camera_human.wbt
```

## Вариант 3: Скачать с сайта

1. Перейти на https://cyberbotics.com/
2. Скачать Webots для macOS
3. Перетащить в `/Applications/`
4. Использовать `run_webots.sh`

## Проверка установки

```bash
# Проверить версию
/Applications/Webots.app/Contents/MacOS/webots --version

# Должно показать:
# Webots version: R2025a (или новее)
```

## Python API (опционально)

Для использования Webots Python API в скриптах:

```bash
# Добавить в PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"

# Или добавьте в ~/.zshrc:
echo 'export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"' >> ~/.zshrc
source ~/.zshrc
```

Проверка:

```python
python3 -c "from controller import Supervisor; print('✅ Webots Python API OK')"
```

## Полный setup

```bash
# 1. Добавить alias
echo 'alias webots="/Applications/Webots.app/Contents/MacOS/webots"' >> ~/.zshrc

# 2. Добавить Python API
echo 'export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"' >> ~/.zshrc

# 3. Перезагрузить
source ~/.zshrc

# 4. Проверить
webots --version
python3 -c "from controller import Supervisor; print('✅ OK')"
```

## Запуск сцены

После настройки:

```bash
cd /Users/mac/projects/NNPID
webots iris_camera_human.wbt
```

Или используйте наш скрипт:

```bash
./run_webots.sh
```

## Troubleshooting

### "command not found: webots"

**Решение**: Используйте полный путь или добавьте alias (см. выше)

```bash
/Applications/Webots.app/Contents/MacOS/webots iris_camera_human.wbt
```

### "Cannot import Supervisor"

**Решение**: Добавьте Webots Python API в PYTHONPATH (см. выше)

### Webots не открывается

**Проверьте**:
1. Приложение установлено: `ls /Applications/Webots.app`
2. Права доступа: `ls -la /Applications/Webots.app/Contents/MacOS/webots`
3. Запустите из GUI: Finder → Applications → Webots

## Версия Webots

- **Текущая**: R2025a ✅
- **Минимальная**: R2023a
- **Рекомендуемая**: R2025a или новее

## Ресурсы

- 🌐 Официальный сайт: https://cyberbotics.com
- 📖 Документация: https://cyberbotics.com/doc/guide/index
- 🚁 ArduPilot + Webots: https://ardupilot.org/dev/docs/sitl-with-webots-python.html

---

✅ После установки вернитесь к [WEBOTS_QUICKSTART.md](WEBOTS_QUICKSTART.md)
