#!/usr/bin/env python3
"""Run MuJoCo simulation with PX4 SITL connection."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.mujoco_sim import MuJoCoSimulator, create_simulator
from src.core.sensors import SensorSimulator, SensorConfig
from src.communication.mavlink_bridge import MAVLinkBridge, MAVLinkConfig
from src.communication.messages import HILSensorMessage, HILGPSMessage
from src.utils.transforms import CoordinateTransforms
from src.utils.logger import TelemetryLogger


# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global running
    print("\nShutdown requested...")
    running = False


def run_simulation(
    model: str = "x500",
    host: str = "127.0.0.1",
    port: int = 4560,
    duration: float = 60.0,
    visualize: bool = True,
    log_telemetry: bool = True,
):
    """Run MuJoCo simulation connected to PX4 SITL.
    
    Args:
        model: MuJoCo model name (x500 or generic)
        host: MAVLink server host
        port: MAVLink server port
        duration: Maximum simulation duration (seconds)
        visualize: Whether to show MuJoCo viewer
        log_telemetry: Whether to log telemetry data
    """
    global running
    
    print("=" * 60)
    print("NNPID SITL + MuJoCo Simulation")
    print("=" * 60)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create simulator
    print(f"\nLoading MuJoCo model: {model}")
    sim = create_simulator(model=model)
    print(f"  Timestep: {sim.timestep * 1000:.1f} ms")
    print(f"  Mass: {sim.mass:.2f} kg")
    
    # Create sensor simulator
    sensor_config = SensorConfig()
    sensors = SensorSimulator(config=sensor_config)
    
    # Create MAVLink bridge
    mavlink_config = MAVLinkConfig(host=host, port=port, lockstep=True)
    bridge = MAVLinkBridge(config=mavlink_config)
    
    # Create telemetry logger
    logger = None
    if log_telemetry:
        logger = TelemetryLogger(log_dir="logs/telemetry", auto_save=True)
    
    # Create viewer if visualization enabled
    viewer = None
    if visualize:
        try:
            import mujoco.viewer
            print("\nLaunching MuJoCo viewer...")
        except ImportError:
            print("MuJoCo viewer not available, running headless")
            visualize = False
    
    # Start MAVLink server and wait for PX4 connection
    print(f"\nStarting MAVLink server on {host}:{port}")
    print("Launch PX4 SITL with:")
    print(f"  make px4_sitl none_iris")
    print("  or")
    print(f"  ./Tools/simulation/mujoco/run.sh")
    print()
    
    if not bridge.start_server():
        print("Failed to connect to PX4 SITL")
        return
    
    print("\nPX4 connected! Starting simulation...")
    
    # Reset simulation
    sim.reset(position=np.array([0.0, 0.0, 0.1]))
    sensors.reset()
    
    # Timing
    start_time = time.time()
    last_print_time = start_time
    step_count = 0
    
    try:
        if visualize:
            # Run with viewer
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                while running and viewer.is_running():
                    step_count = simulation_step(
                        sim, sensors, bridge, logger, step_count
                    )
                    viewer.sync()
                    
                    # Print status periodically
                    now = time.time()
                    if now - last_print_time > 1.0:
                        print_status(sim, bridge, step_count)
                        last_print_time = now
                    
                    # Check duration
                    if sim.get_time() > duration:
                        print(f"\nReached duration limit: {duration}s")
                        break
        else:
            # Run headless
            while running:
                step_count = simulation_step(
                    sim, sensors, bridge, logger, step_count
                )
                
                # Print status periodically
                now = time.time()
                if now - last_print_time > 1.0:
                    print_status(sim, bridge, step_count)
                    last_print_time = now
                
                # Check duration
                if sim.get_time() > duration:
                    print(f"\nReached duration limit: {duration}s")
                    break
    
    except Exception as e:
        print(f"\nSimulation error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nShutting down...")
        
        if logger is not None:
            saved_path = logger.end_episode()
            if saved_path:
                print(f"Telemetry saved to: {saved_path}")
        
        bridge.stop()
        
        elapsed = time.time() - start_time
        print(f"\nSimulation complete:")
        print(f"  Steps: {step_count}")
        print(f"  Sim time: {sim.get_time():.2f} s")
        print(f"  Wall time: {elapsed:.2f} s")
        print(f"  Realtime factor: {sim.get_time() / elapsed:.2f}x")


def simulation_step(
    sim: MuJoCoSimulator,
    sensors: SensorSimulator,
    bridge: MAVLinkBridge,
    logger: Optional[TelemetryLogger],
    step_count: int,
) -> int:
    """Execute one simulation step.
    
    Args:
        sim: MuJoCo simulator
        sensors: Sensor simulator
        bridge: MAVLink bridge
        logger: Telemetry logger (optional)
        step_count: Current step count
        
    Returns:
        Updated step count
    """
    # Get motor commands from PX4
    motors = bridge.get_motor_commands(timeout=0.001)
    
    if motors is None:
        motors = np.zeros(4)  # No command, motors off
    
    # Step physics
    state = sim.step(motors)
    timestamp = sim.get_time()
    
    # Get sensor data
    gyro, accel = sim.get_imu_data()
    
    # Convert to PX4 frame (FRD body, NED world)
    gyro_frd = CoordinateTransforms.angular_velocity_mujoco_to_frd(gyro)
    accel_frd = CoordinateTransforms.acceleration_mujoco_to_frd(accel)
    pos_ned = CoordinateTransforms.position_mujoco_to_ned(state.position)
    vel_ned = CoordinateTransforms.velocity_mujoco_to_ned(state.velocity)
    quat_ned = CoordinateTransforms.quaternion_mujoco_to_ned(state.quaternion)
    
    # Add sensor noise
    imu_data = sensors.get_imu(gyro_frd, accel_frd, timestamp)
    
    # Create and send HIL_SENSOR message
    sensor_msg = HILSensorMessage.from_sensor_data(
        time_sec=timestamp,
        gyro=imu_data.gyro,
        accel=imu_data.accel,
        mag=np.array([0.21, 0.0, 0.42]),
        pressure=101325.0,
        temperature=20.0,
        altitude=-pos_ned[2],  # NED Z is down
    )
    bridge.send_hil_sensor(sensor_msg)
    
    # Send GPS at lower rate
    if bridge.lockstep.should_send_gps():
        gps_data = sensors.get_gps(pos_ned, vel_ned, timestamp)
        gps_msg = HILGPSMessage.from_gps_data(
            time_sec=timestamp,
            lat=gps_data.latitude,
            lon=gps_data.longitude,
            alt=gps_data.altitude,
            vel_ned=np.array([
                gps_data.velocity_north,
                gps_data.velocity_east,
                gps_data.velocity_down,
            ]),
        )
        bridge.send_hil_gps(gps_msg)
        sensors.update_timing("gps", timestamp)
    
    # Send heartbeat
    bridge.send_heartbeat()
    
    # Log telemetry
    if logger is not None:
        from src.utils.rotations import Rotations
        euler = np.array(Rotations.quaternion_to_euler(state.quaternion))
        
        logger.log_frame(
            timestamp=timestamp,
            position=state.position,
            velocity=state.velocity,
            quaternion=state.quaternion,
            angular_velocity=state.angular_velocity,
            motor_commands=motors,
            euler_angles=euler,
            gyro=gyro,
            accel=accel,
        )
    
    # Update lockstep
    bridge.lockstep.physics_stepped()
    bridge.lockstep.sensors_sent()
    
    return step_count + 1


def print_status(
    sim: MuJoCoSimulator,
    bridge: MAVLinkBridge,
    step_count: int,
):
    """Print simulation status."""
    state = sim.get_state()
    
    from src.utils.rotations import Rotations
    roll, pitch, yaw = Rotations.quaternion_to_euler(state.quaternion)
    
    print(
        f"t={sim.get_time():6.2f}s | "
        f"pos=[{state.position[0]:6.2f}, {state.position[1]:6.2f}, {state.position[2]:6.2f}] | "
        f"att=[{np.degrees(roll):5.1f}, {np.degrees(pitch):5.1f}, {np.degrees(yaw):5.1f}]° | "
        f"steps={step_count}"
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run MuJoCo simulation with PX4 SITL"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="x500",
        choices=["x500", "generic"],
        help="MuJoCo quadrotor model",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="MAVLink server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4560,
        help="MAVLink server port",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Maximum simulation duration (seconds)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable visualization",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable telemetry logging",
    )
    
    args = parser.parse_args()
    
    run_simulation(
        model=args.model,
        host=args.host,
        port=args.port,
        duration=args.duration,
        visualize=not args.no_viz,
        log_telemetry=not args.no_log,
    )


if __name__ == "__main__":
    main()
