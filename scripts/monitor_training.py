#!/usr/bin/env python3
"""Monitor mega training progress in real-time."""

import time
from pathlib import Path

import numpy as np


def format_time(seconds):
    """Format seconds to human readable time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def monitor_training(run_dir: str, interval: int = 30):
    """Monitor training progress.

    Args:
        run_dir: Path to training run directory
        interval: Update interval in seconds
    """
    run_path = Path(run_dir)
    eval_file = run_path / "eval_logs" / "evaluations.npz"

    print("=" * 80)
    print("  MEGA TRAINING MONITOR")
    print("=" * 80)
    print(f"Run directory: {run_path}")
    print("Target: 20,000,000 timesteps")
    print(f"Update interval: {interval}s")
    print("=" * 80)
    print()

    start_time = time.time()
    last_timesteps = 0

    while True:
        try:
            # Check if evaluation file exists
            if not eval_file.exists():
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for training to start...")
                time.sleep(interval)
                continue

            # Load evaluation data
            data = np.load(eval_file)
            timesteps = data["timesteps"]
            results = data["results"]
            ep_lengths = data["ep_lengths"]

            if len(timesteps) == 0:
                print(f"[{time.strftime('%H:%M:%S')}] No evaluations yet...")
                time.sleep(interval)
                continue

            # Get latest metrics
            current_steps = int(timesteps[-1])
            latest_reward = float(np.mean(results[-1]))
            latest_length = float(np.mean(ep_lengths[-1]))

            # Calculate progress
            progress_pct = (current_steps / 20_000_000) * 100

            # Calculate speed
            elapsed = time.time() - start_time
            if last_timesteps > 0:
                steps_since_last = current_steps - last_timesteps
                steps_per_sec = steps_since_last / interval
                eta_seconds = (
                    (20_000_000 - current_steps) / steps_per_sec if steps_per_sec > 0 else 0
                )
            else:
                steps_per_sec = current_steps / elapsed if elapsed > 0 else 0
                eta_seconds = (
                    (20_000_000 - current_steps) / steps_per_sec if steps_per_sec > 0 else 0
                )

            last_timesteps = current_steps

            # Find checkpoints
            checkpoint_dir = run_path / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("*.zip"))
                num_checkpoints = len(checkpoints)
            else:
                num_checkpoints = 0

            # Print status
            print(f"\n[{time.strftime('%H:%M:%S')}] Training Progress:")
            print(f"  Timesteps:     {current_steps:,} / 20,000,000  ({progress_pct:.2f}%)")
            print(f"  Speed:         {steps_per_sec:,.0f} steps/sec")
            print(f"  ETA:           {format_time(eta_seconds)}")
            print(f"  Latest Reward: {latest_reward:.2f}")
            print(f"  Avg Ep Length: {latest_length:.0f} steps")
            print(f"  Checkpoints:   {num_checkpoints}")
            print(f"  Elapsed:       {format_time(elapsed)}")

            # Progress bar
            bar_width = 50
            filled = int(bar_width * progress_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  Progress:      [{bar}]")

            # Check for best model updates
            best_model = run_path / "best_model" / "best_model.zip"
            if best_model.exists():
                mod_time = best_model.stat().st_mtime
                age_minutes = (time.time() - mod_time) / 60
                print(f"  Best Model:    Updated {age_minutes:.1f} min ago")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Find latest run
        runs_dir = Path("runs")
        if not runs_dir.exists():
            print("Error: runs/ directory not found")
            sys.exit(1)

        run_dirs = sorted(
            runs_dir.glob("yaw_tracking_*"), key=lambda x: x.stat().st_mtime, reverse=True
        )
        if not run_dirs:
            print("Error: No training runs found")
            sys.exit(1)

        run_dir = run_dirs[0]
    else:
        run_dir = Path(sys.argv[1])

    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    monitor_training(str(run_dir), interval)
