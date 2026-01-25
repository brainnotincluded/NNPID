#!/usr/bin/env python3
"""Render MuJoCo drone flight video with SITL-style control."""

import numpy as np
import mujoco
import imageio

print("="*60)
print("    MUJOCO DRONE FLIGHT VIDEO")
print("    (SITL-style position control)")
print("="*60)

# Quadrotor model with nice visuals
xml = """
<mujoco model="quadrotor_video">
    <visual>
        <global offwidth="1280" offheight="720"/>
    </visual>
    <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
    
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0.05 0.1 0.2" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.35 0.25" rgb2="0.15 0.25 0.15" width="512" height="512"/>
        <material name="grid" texture="grid" texrepeat="15 15" reflectance="0.1"/>
    </asset>
    
    <worldbody>
        <light pos="0 -5 10" dir="0 0.5 -1" diffuse="1 1 1"/>
        <light pos="5 5 8" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.5"/>
        
        <!-- Ground -->
        <geom type="plane" size="20 20 0.1" material="grid"/>
        
        <!-- Landing pad -->
        <geom type="cylinder" pos="0 0 0.01" size="0.8 0.01" rgba="0.5 0.5 0.5 1"/>
        <geom type="box" pos="0 0 0.011" size="0.1 0.5 0.001" rgba="1 1 1 1"/>
        <geom type="box" pos="0 0 0.011" size="0.5 0.1 0.001" rgba="1 1 1 1"/>
        
        <!-- Waypoint markers -->
        <geom type="cylinder" pos="4 0 0.02" size="0.3 0.02" rgba="1 0.8 0 0.5"/>
        <geom type="cylinder" pos="4 4 0.02" size="0.3 0.02" rgba="1 0.8 0 0.5"/>
        <geom type="cylinder" pos="0 4 0.02" size="0.3 0.02" rgba="1 0.8 0 0.5"/>
        
        <!-- Drone -->
        <body name="drone" pos="0 0 0.15">
            <freejoint name="root"/>
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.008"/>
            
            <!-- Main body -->
            <geom type="box" size="0.06 0.06 0.02" rgba="0.2 0.2 0.2 1"/>
            
            <!-- Arms -->
            <geom type="capsule" fromto="0.04 0.04 0 0.14 0.14 0" size="0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="capsule" fromto="-0.04 0.04 0 -0.14 0.14 0" size="0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="capsule" fromto="-0.04 -0.04 0 -0.14 -0.14 0" size="0.01" rgba="0.3 0.3 0.3 1"/>
            <geom type="capsule" fromto="0.04 -0.04 0 0.14 -0.14 0" size="0.01" rgba="0.3 0.3 0.3 1"/>
            
            <!-- Motors -->
            <geom type="cylinder" pos="0.14 0.14 0.012" size="0.02 0.008" rgba="1 0.2 0.2 1"/>
            <geom type="cylinder" pos="-0.14 0.14 0.012" size="0.02 0.008" rgba="0.2 0.2 1 1"/>
            <geom type="cylinder" pos="-0.14 -0.14 0.012" size="0.02 0.008" rgba="1 0.2 0.2 1"/>
            <geom type="cylinder" pos="0.14 -0.14 0.012" size="0.02 0.008" rgba="0.2 0.2 1 1"/>
            
            <!-- Propeller discs -->
            <geom type="cylinder" pos="0.14 0.14 0.025" size="0.06 0.003" rgba="0.8 0.3 0.3 0.4"/>
            <geom type="cylinder" pos="-0.14 0.14 0.025" size="0.06 0.003" rgba="0.3 0.3 0.8 0.4"/>
            <geom type="cylinder" pos="-0.14 -0.14 0.025" size="0.06 0.003" rgba="0.8 0.3 0.3 0.4"/>
            <geom type="cylinder" pos="0.14 -0.14 0.025" size="0.06 0.003" rgba="0.3 0.3 0.8 0.4"/>
            
            <!-- Thrust sites -->
            <site name="m1" pos="0.14 0.14 0.015"/>
            <site name="m2" pos="-0.14 0.14 0.015"/>
            <site name="m3" pos="-0.14 -0.14 0.015"/>
            <site name="m4" pos="0.14 -0.14 0.015"/>
        </body>
    </worldbody>
    
    <actuator>
        <general name="t1" site="m1" gear="0 0 4 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general name="t2" site="m2" gear="0 0 4 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general name="t3" site="m3" gear="0 0 4 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
        <general name="t4" site="m4" gear="0 0 4 0 0 0" ctrllimited="true" ctrlrange="0 1"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=720, width=1280)

output_path = "/Users/mac/projects/NNPID/drone_sitl_flight.mp4"
writer = imageio.get_writer(output_path, fps=30, quality=8)

print(f"\nRendering to: {output_path}")

# Simple but stable position controller
def position_control(pos, vel, target, kp=0.3, kd=0.2):
    """Simple PD altitude control."""
    error = target - pos[2]
    thrust = 0.35 + kp * error - kd * vel[2]  # 0.35 is hover for 0.5kg with 4x4N
    return np.clip(thrust, 0.1, 0.9)

def xy_control(pos, vel, target_xy, kp=0.4, kd=0.3):
    """Simple XY position control via differential thrust."""
    error_xy = target_xy - pos[:2]
    cmd = kp * error_xy - kd * vel[:2]
    return np.clip(cmd, -0.15, 0.15)

# Flight plan
waypoints = [
    (np.array([0.0, 0.0, 3.0]), "Takeoff"),
    (np.array([4.0, 0.0, 3.0]), "Waypoint 1"),
    (np.array([4.0, 4.0, 3.0]), "Waypoint 2"),
    (np.array([0.0, 4.0, 3.0]), "Waypoint 3"),
    (np.array([0.0, 0.0, 3.0]), "Home"),
    (np.array([0.0, 0.0, 0.2]), "Land"),
]

print("\nFlight plan:")
for i, (wp, desc) in enumerate(waypoints):
    print(f"  {i+1}. {desc}: [{wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}]")

current_wp = 0
total_frames = 0

print("\nRendering...")

for step in range(12000):
    sim_time = step * model.opt.timestep
    
    pos = data.qpos[:3].copy()
    vel = data.qvel[:3].copy()
    
    # Current target
    if current_wp < len(waypoints):
        target, desc = waypoints[current_wp]
        dist = np.linalg.norm(target - pos)
        
        if dist < 0.4 and current_wp < len(waypoints) - 1:
            current_wp += 1
            print(f"  [{sim_time:.1f}s] Reached {desc}")
    else:
        target = waypoints[-1][0]
    
    # Altitude control
    base_thrust = position_control(pos, vel, target[2])
    
    # XY control
    xy_cmd = xy_control(pos, vel, target[:2])
    
    # Motor mixing
    m1 = base_thrust - xy_cmd[0] - xy_cmd[1]  # FR
    m2 = base_thrust + xy_cmd[0] - xy_cmd[1]  # FL
    m3 = base_thrust + xy_cmd[0] + xy_cmd[1]  # BL
    m4 = base_thrust - xy_cmd[0] + xy_cmd[1]  # BR
    
    data.ctrl[:] = np.clip([m1, m2, m3, m4], 0, 1)
    mujoco.mj_step(model, data)
    
    # Render every 16 steps (30fps)
    if step % 16 == 0:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [pos[0] * 0.5, pos[1] * 0.5, max(pos[2], 1.5)]
        cam.distance = 10.0
        cam.azimuth = 45 + sim_time * 3
        cam.elevation = -25
        
        renderer.update_scene(data, camera=cam)
        frame = renderer.render()
        writer.append_data(frame)
        total_frames += 1
        
        if total_frames % 60 == 0:
            print(f"    Frame {total_frames} | t={sim_time:.1f}s | Alt={pos[2]:.2f}m | Target: {waypoints[min(current_wp, len(waypoints)-1)][1]}")
    
    # End when landed
    if current_wp >= len(waypoints) - 1 and pos[2] < 0.3 and abs(vel[2]) < 0.1:
        for _ in range(45):
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
            total_frames += 1
        print(f"  [{sim_time:.1f}s] Landed!")
        break

writer.close()
print(f"\n{'='*60}")
print(f"Video saved: {output_path}")
print(f"Total frames: {total_frames} ({total_frames/30:.1f}s)")
print(f"{'='*60}")
