# Webots Installation Guide (macOS)

## Option 1: Webots already installed

If Webots is already installed at `/Applications/Webots.app`:

```bash
scripts/shell/run_webots.sh
```

Or add an alias in `~/.zshrc`:

```bash
alias webots="/Applications/Webots.app/Contents/MacOS/webots"
source ~/.zshrc
webots iris_camera_human.wbt
```

## Option 2: Install via Homebrew

```bash
brew install --cask webots
webots iris_camera_human.wbt
```

## Option 3: Download from the website

1. Go to https://cyberbotics.com/
2. Download Webots for macOS
3. Drag it to `/Applications/`
4. Use `scripts/shell/run_webots.sh` or the `webots` alias

## Verify installation

```bash
/Applications/Webots.app/Contents/MacOS/webots --version
```

## Python API (optional)

If you use the Webots Python API in scripts:

```bash
export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"
echo 'export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"' >> ~/.zshrc
source ~/.zshrc
```

Validation:

```bash
python -c "from controller import Supervisor; print('Webots Python API OK')"
```

## Full setup (one-time)

```bash
echo 'alias webots="/Applications/Webots.app/Contents/MacOS/webots"' >> ~/.zshrc
echo 'export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"' >> ~/.zshrc
source ~/.zshrc

webots --version
python -c "from controller import Supervisor; print('OK')"
```

## Launch the scene

```bash
cd /Users/mac/projects/NNPID
webots iris_camera_human.wbt
```

Or use the helper script:

```bash
scripts/shell/run_webots.sh
```

## Troubleshooting

### "command not found: webots"

Use the full path or add the alias:

```bash
/Applications/Webots.app/Contents/MacOS/webots iris_camera_human.wbt
```

### "Cannot import Supervisor"

Ensure the Webots Python API is on `PYTHONPATH`.

### Webots does not open

Check:
1. App exists: `ls /Applications/Webots.app`
2. Permissions: `ls -la /Applications/Webots.app/Contents/MacOS/webots`
3. Launch from GUI: Finder -> Applications -> Webots

## Recommended Webots version

- Minimum: R2023a
- Recommended: R2025a or newer

## Resources

- https://cyberbotics.com
- https://cyberbotics.com/doc/guide/index
- https://ardupilot.org/dev/docs/sitl-with-webots-python.html

---

After installation, return to `docs/QUICKSTART_HUMAN_TRACKING.md`.
