#!/usr/bin/env python3
"""Visualize simulation and telemetry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.controllers.base_controller import PIDController
from src.core.mujoco_sim import create_simulator


def run_interactive_viewer(
    model: str = "x500",
    controller: str = "pid",
    target: list | None = None,
):
    """Run interactive MuJoCo viewer with controller.

    Args:
        model: MuJoCo model name
        controller: Controller type (pid, hover)
        target: Target position [x, y, z]
    """
    try:
        import mujoco.viewer
    except ImportError:
        print("MuJoCo viewer not available")
        return

    print(f"Loading model: {model}")
    sim = create_simulator(model=model)

    # Setup controller
    if controller == "pid":
        ctrl = PIDController()
    else:
        ctrl = PIDController()  # Default to PID

    # Target position
    target_pos = np.array([0.0, 0.0, 1.0]) if target is None else np.array(target)

    print(f"Target position: {target_pos}")
    print("\nControls:")
    print("  - Drag to rotate view")
    print("  - Scroll to zoom")
    print("  - Double-click to reset camera")
    print("  - Esc to quit")

    # Reset simulation
    sim.reset(position=np.array([0.0, 0.0, 0.1]))

    dt = sim.timestep

    def controller_callback(model, data):
        """Controller callback for viewer."""
        state = sim.get_state()
        action = ctrl.compute_action(state, target_pos, dt)
        data.ctrl[:4] = action

    # Launch viewer with callback
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=None,
    ) as viewer:
        while viewer.is_running():
            # Compute control
            state = sim.get_state()
            action = ctrl.compute_action(state, target_pos, dt)

            # Step simulation
            sim.step(action)

            # Sync viewer
            viewer.sync()


def plot_telemetry(
    log_path: Path,
    show_plot: bool = True,
    save_path: Path | None = None,
):
    """Plot telemetry from logged data.

    Args:
        log_path: Path to telemetry file (.npz)
        show_plot: Whether to display plot
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting")
        return

    if not log_path.exists():
        print(f"File not found: {log_path}")
        return

    print(f"Loading telemetry: {log_path}")
    data = np.load(log_path)

    # Extract data
    t = data["timestamp"]
    pos = data["position"]
    vel = data["velocity"]
    euler = data["euler_angles"]
    motors = data["motor_commands"]

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Position
    ax = axes[0]
    ax.plot(t, pos[:, 0], label="X")
    ax.plot(t, pos[:, 1], label="Y")
    ax.plot(t, pos[:, 2], label="Z")
    ax.set_ylabel("Position (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Quadrotor Telemetry")

    # Velocity
    ax = axes[1]
    ax.plot(t, vel[:, 0], label="Vx")
    ax.plot(t, vel[:, 1], label="Vy")
    ax.plot(t, vel[:, 2], label="Vz")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Attitude
    ax = axes[2]
    ax.plot(t, np.degrees(euler[:, 0]), label="Roll")
    ax.plot(t, np.degrees(euler[:, 1]), label="Pitch")
    ax.plot(t, np.degrees(euler[:, 2]), label="Yaw")
    ax.set_ylabel("Attitude (deg)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Motor commands
    ax = axes[3]
    for i in range(4):
        ax.plot(t, motors[:, i], label=f"M{i + 1}")
    ax.set_ylabel("Motor Cmd")
    ax.set_xlabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")

    if show_plot:
        plt.show()


def plot_3d_trajectory(
    log_path: Path,
    show_plot: bool = True,
    save_path: Path | None = None,
):
    """Plot 3D trajectory from telemetry.

    Args:
        log_path: Path to telemetry file
        show_plot: Whether to display
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib required for plotting")
        return

    if not log_path.exists():
        print(f"File not found: {log_path}")
        return

    data = np.load(log_path)
    pos = data["position"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b-", linewidth=1, alpha=0.7)

    # Mark start and end
    ax.scatter(*pos[0], c="green", s=100, marker="o", label="Start")
    ax.scatter(*pos[-1], c="red", s=100, marker="x", label="End")

    # Target if available
    if "target_position" in data:
        target = data["target_position"]
        ax.scatter(*target[0], c="orange", s=200, marker="*", label="Target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    ax.set_title("3D Trajectory")

    # Equal aspect ratio
    max_range = (
        np.array(
            [
                pos[:, 0].max() - pos[:, 0].min(),
                pos[:, 1].max() - pos[:, 1].min(),
                pos[:, 2].max() - pos[:, 2].min(),
            ]
        ).max()
        / 2.0
    )

    mid_x = (pos[:, 0].max() + pos[:, 0].min()) / 2
    mid_y = (pos[:, 1].max() + pos[:, 1].min()) / 2
    mid_z = (pos[:, 2].max() + pos[:, 2].min()) / 2

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to: {save_path}")

    if show_plot:
        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Visualization tools")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Interactive viewer
    view_parser = subparsers.add_parser("view", help="Interactive viewer")
    view_parser.add_argument(
        "--model",
        type=str,
        default="x500",
        help="MuJoCo model",
    )
    view_parser.add_argument(
        "--controller",
        type=str,
        default="pid",
        help="Controller type",
    )
    view_parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=None,
        help="Target position (x y z)",
    )

    # Plot telemetry
    plot_parser = subparsers.add_parser("plot", help="Plot telemetry")
    plot_parser.add_argument(
        "log_file",
        type=Path,
        help="Path to telemetry file",
    )
    plot_parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to path",
    )
    plot_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot",
    )

    # 3D trajectory
    traj_parser = subparsers.add_parser("trajectory", help="3D trajectory plot")
    traj_parser.add_argument(
        "log_file",
        type=Path,
        help="Path to telemetry file",
    )
    traj_parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save figure to path",
    )

    args = parser.parse_args()

    if args.command == "view":
        run_interactive_viewer(
            model=args.model,
            controller=args.controller,
            target=args.target,
        )
    elif args.command == "plot":
        plot_telemetry(
            log_path=args.log_file,
            show_plot=not args.no_show,
            save_path=args.save,
        )
    elif args.command == "trajectory":
        plot_3d_trajectory(
            log_path=args.log_file,
            save_path=args.save,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
