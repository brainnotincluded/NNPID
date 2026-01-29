#!/bin/bash
# Quick launcher for Webots with our scene

WEBOTS="/Applications/Webots.app/Contents/MacOS/webots"
SCENE="iris_camera_human.wbt"

if [ ! -f "$WEBOTS" ]; then
    echo "❌ Webots not found at $WEBOTS"
    echo "Please install Webots from https://cyberbotics.com"
    exit 1
fi

if [ ! -f "$SCENE" ]; then
    echo "❌ Scene not found: $SCENE"
    echo "Run this script from the project root directory"
    exit 1
fi

echo "🚁 Starting Webots with $SCENE..."
"$WEBOTS" "$SCENE"
