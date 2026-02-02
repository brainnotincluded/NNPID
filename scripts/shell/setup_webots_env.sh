#!/bin/bash
# Setup Webots environment for NNPID project

echo "🔧 Setting up Webots environment..."

# Check if Webots is installed
if [ ! -d "/Applications/Webots.app" ]; then
    echo "❌ Webots not found in /Applications/"
    echo "   Please install from: https://cyberbotics.com"
    exit 1
fi

echo "✅ Found Webots R2025a"

# Add alias to shell config
SHELL_CONFIG="$HOME/.zshrc"
if [ -n "$BASH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
fi

echo ""
echo "Adding Webots to your shell configuration..."

# Add webots alias
if ! grep -q "alias webots=" "$SHELL_CONFIG" 2>/dev/null; then
    echo "" >> "$SHELL_CONFIG"
    echo "# Webots alias (added by NNPID setup)" >> "$SHELL_CONFIG"
    echo 'alias webots="/Applications/Webots.app/Contents/MacOS/webots"' >> "$SHELL_CONFIG"
    echo "✅ Added 'webots' alias to $SHELL_CONFIG"
else
    echo "⏭️  'webots' alias already exists in $SHELL_CONFIG"
fi

# Add Python API to PYTHONPATH
if ! grep -q "Webots.app/Contents/lib/controller/python" "$SHELL_CONFIG" 2>/dev/null; then
    echo "" >> "$SHELL_CONFIG"
    echo "# Webots Python API (added by NNPID setup)" >> "$SHELL_CONFIG"
    echo 'export PYTHONPATH="${PYTHONPATH}:/Applications/Webots.app/Contents/lib/controller/python"' >> "$SHELL_CONFIG"
    echo "✅ Added Webots Python API to PYTHONPATH in $SHELL_CONFIG"
else
    echo "⏭️  Webots Python API already in $SHELL_CONFIG"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To apply changes, run:"
echo "  source $SHELL_CONFIG"
echo ""
echo "Or restart your terminal."
echo ""
echo "Then test with:"
echo "  webots --version"
echo "  python3 -c 'from controller import Supervisor; print(\"✅ OK\")'"
echo ""
echo "To start the scene:"
echo "  scripts/shell/run_webots.sh"
