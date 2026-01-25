#!/bin/bash
# Run true PX4 SITL + MuJoCo test

set -e

echo "Cleaning up..."
pkill -9 px4 2>/dev/null || true
pkill -9 python 2>/dev/null || true
sleep 2

echo "Activating conda..."
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate base

echo "Starting MuJoCo server..."
cd /Users/mac/projects/NNPID
python scripts/true_sitl.py &
MJ_PID=$!
echo "MuJoCo PID: $MJ_PID"

sleep 3

echo "Starting PX4 SITL..."
cd ~/PX4-Autopilot
make px4_sitl none_iris &
PX4_PID=$!
echo "PX4 PID: $PX4_PID"

echo "Waiting for simulation (45 seconds)..."
sleep 45

echo "Stopping..."
kill $MJ_PID 2>/dev/null || true
kill $PX4_PID 2>/dev/null || true
pkill -9 px4 2>/dev/null || true

echo "Done! Check /Users/mac/projects/NNPID/true_sitl_flight.mp4"
