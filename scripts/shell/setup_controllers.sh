#!/bin/bash
# Setup Webots controllers from ArduPilot

echo "🔧 Setting up Webots controllers..."

# Get project root (2 levels up from this script)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Check if controllers already exist
if [ -d "controllers/ardupilot_vehicle_controller" ] && [ -d "controllers/pedestrian" ]; then
    echo "✅ Controllers already exist"
    ls -la controllers/
    exit 0
fi

# Try to copy from local ArduPilot installation
if [ -d ~/ardupilot ]; then
    echo "📁 Found ArduPilot at ~/ardupilot"
    echo "   Copying controllers..."
    
    cp -r ~/ardupilot/libraries/SITL/examples/Webots_Python/controllers ./
    
    if [ $? -eq 0 ]; then
        echo "✅ Controllers copied successfully!"
        ls -la controllers/
        exit 0
    fi
fi

# If local copy failed, clone from GitHub
echo "📥 Downloading ArduPilot controllers from GitHub..."
TEMP_DIR="/tmp/ardupilot_webots_$$"

git clone --depth 1 https://github.com/ArduPilot/ardupilot.git "$TEMP_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to clone ArduPilot"
    echo "   Please install manually:"
    echo "   1. Clone: git clone https://github.com/ArduPilot/ardupilot.git"
    echo "   2. Copy: cp -r ardupilot/libraries/SITL/examples/Webots_Python/controllers ."
    exit 1
fi

echo "📦 Copying controllers..."
cp -r "$TEMP_DIR/libraries/SITL/examples/Webots_Python/controllers" ./

if [ $? -eq 0 ]; then
    echo "✅ Controllers installed successfully!"
    rm -rf "$TEMP_DIR"
    ls -la controllers/
    
    echo ""
    echo "📋 Installed controllers:"
    find controllers/ -name "*.py" -type f
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "Now you can:"
    echo "  1. Run Webots: scripts/shell/run_webots.sh"
    echo "  2. Run SITL: sim_vehicle.py -v ArduCopter -f webots-quad"
    echo "  3. Run tracker: python src/webots/webots_human_tracker.py --model runs/best_model"
else
    echo "❌ Failed to copy controllers"
    rm -rf "$TEMP_DIR"
    exit 1
fi
