'''
This file implements a class that acts as a bridge between ArduPilot SITL and Webots

Protocol reference: https://ardupilot.org/dev/docs/sitl-with-JSON.html
https://github.com/ArduPilot/ardupilot/tree/master/libraries/SITL/examples/JSON

SITL OUTPUT (binary):
  uint16 magic = 18458
  uint16 frame_rate
  uint32 frame_count
  uint16 pwm[16]
  Total: 40 bytes

SITL INPUT (JSON with newlines):
  {"timestamp":T,"imu":{"gyro":[r,p,y],"accel_body":[x,y,z]},"position":[n,e,d],"attitude":[r,p,y],"velocity":[n,e,d]}
'''

import os
import sys
import time
import socket
import select
import struct
import json
import numpy as np
from threading import Thread
from typing import List, Union

# Setup Webots environment
if sys.platform.startswith("win"):
    WEBOTS_HOME = "C:\\Program Files\\Webots"
elif sys.platform.startswith("darwin"):
    WEBOTS_HOME = "/Applications/Webots.app"
elif sys.platform.startswith("linux"):
    WEBOTS_HOME = "/usr/local/webots"
else:
    raise Exception("Unsupported OS")

if os.environ.get("WEBOTS_HOME") is None:
    os.environ["WEBOTS_HOME"] = WEBOTS_HOME
else:
    WEBOTS_HOME = os.environ.get("WEBOTS_HOME")

os.environ["PYTHONIOENCODING"] = "UTF-8"
sys.path.append(f"{WEBOTS_HOME}/lib/controller/python")

from controller import Robot, Camera, RangeFinder, Supervisor  # noqa: E401, E402


class WebotsArduVehicle():
    """Class representing an ArduPilot controlled Webots Vehicle"""

    # SITL output format (binary): magic(2) + frame_rate(2) + frame_count(4) + pwm[16](32) = 40 bytes
    SITL_OUTPUT_MAGIC = 18458
    SITL_OUTPUT_MAGIC_32CH = 29569
    SITL_OUTPUT_FORMAT = '<HHI16H'  # little-endian: uint16, uint16, uint32, 16x uint16
    SITL_OUTPUT_SIZE = struct.calcsize(SITL_OUTPUT_FORMAT)  # 40 bytes

    def __init__(self,
                 motor_names: List[str],
                 accel_name: str = "accelerometer",
                 imu_name: str = "inertial unit",
                 gyro_name: str = "gyro",
                 gps_name: str = "gps",
                 camera_name: str = None,
                 camera_fps: int = 10,
                 camera_stream_port: int = None,
                 rangefinder_name: str = None,
                 rangefinder_fps: int = 10,
                 rangefinder_stream_port: int = None,
                 instance: int = 0,
                 motor_velocity_cap: float = float('inf'),
                 reversed_motors: List[int] = None,
                 bidirectional_motors: bool = False,
                 uses_propellers: bool = True,
                 sitl_address: str = "127.0.0.1",
                 tracking_stream_port: int = 9100):
        """WebotsArduVehicle constructor
        
        Args:
            tracking_stream_port: UDP port to stream drone + human position for NN tracking
        """
        
        self.motor_velocity_cap = motor_velocity_cap
        self._instance = instance
        self._reversed_motors = reversed_motors
        self._bidirectional_motors = bidirectional_motors
        self._uses_propellers = uses_propellers
        self._webots_connected = True
        self._last_frame_count = -1
        self._tracking_port = tracking_stream_port + instance

        # Setup Webots robot instance (use Supervisor for accessing other nodes)
        try:
            self.robot = Supervisor()
            self._is_supervisor = True
        except Exception:
            self.robot = Robot()
            self._is_supervisor = False
        self._timestep = int(self.robot.getBasicTimeStep())
        
        # Find pedestrian node for tracking
        self._pedestrian_node = None
        if self._is_supervisor:
            self._pedestrian_node = self.robot.getFromDef("PEDESTRIAN")
            if self._pedestrian_node:
                print(f"Found PEDESTRIAN node for tracking")

        # Init sensors
        self.accel = self.robot.getDevice(accel_name)
        self.imu = self.robot.getDevice(imu_name)
        self.gyro = self.robot.getDevice(gyro_name)
        self.gps = self.robot.getDevice(gps_name)

        self.accel.enable(self._timestep)
        self.imu.enable(self._timestep)
        self.gyro.enable(self._timestep)
        self.gps.enable(self._timestep)

        # Init camera
        if camera_name is not None:
            self.camera = self.robot.getDevice(camera_name)
            self.camera.enable(1000 // camera_fps)
            if camera_stream_port is not None:
                self._camera_thread = Thread(daemon=True,
                                             target=self._handle_image_stream,
                                             args=[self.camera, camera_stream_port])
                self._camera_thread.start()

        # Init rangefinder
        if rangefinder_name is not None:
            self.rangefinder = self.robot.getDevice(rangefinder_name)
            self.rangefinder.enable(1000 // rangefinder_fps)
            if rangefinder_stream_port is not None:
                self._rangefinder_thread = Thread(daemon=True,
                                                  target=self._handle_image_stream,
                                                  args=[self.rangefinder, rangefinder_stream_port])
                self._rangefinder_thread.start()

        # Init motors
        self._motors = [self.robot.getDevice(n) for n in motor_names]
        for m in self._motors:
            m.setPosition(float('inf'))
            m.setVelocity(0)

        # Start SITL communication thread
        self._sitl_thread = Thread(daemon=True, target=self._handle_sitl, 
                                   args=[sitl_address, 9002 + 10 * instance])
        self._sitl_thread.start()
        
        # Setup tracking data stream (UDP broadcast)
        self._tracking_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tracking_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        print(f"Tracking stream available on UDP port {self._tracking_port}")

    def _handle_sitl(self, sitl_address: str = "127.0.0.1", port: int = 9002):
        """Handle all communications with ArduPilot SITL using JSON protocol
        
        Protocol (from ArduPilot docs):
        1. Physics backend BINDS to port 9002 and LISTENS
        2. SITL SENDS binary PWM data to port 9002
        3. Backend RECEIVES PWM, applies to motors
        4. Backend RESPONDS with JSON sensor data to SITL's address
        """
        
        # Create UDP socket and BIND to port 9002 (listen for SITL)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', port))
        sock.settimeout(0.1)  # 100ms timeout for recv
        
        print(f"ArduPilot JSON backend listening on port {port} (I{self._instance})")
        print(f"Expected SITL output size: {self.SITL_OUTPUT_SIZE} bytes")
        print(f"Waiting for SITL to connect...")
        self.robot.step(self._timestep)  # Flush print

        # Main communication loop
        connected = False
        sitl_addr = None
        frame_count = 0
        last_print = 0
        
        while self._webots_connected:
            try:
                # RECEIVE binary PWM data from SITL
                data, addr = sock.recvfrom(1024)
                
                if len(data) < self.SITL_OUTPUT_SIZE:
                    continue
                
                # Parse SITL output (PWM commands)
                if not self._parse_sitl_output(data):
                    continue
                
                # Track connection
                if not connected:
                    sitl_addr = addr
                    print(f"✅ Connected to ArduPilot SITL at {addr}")
                    connected = True
                
                frame_count += 1
                
                # Build and RESPOND with JSON sensor data
                json_data = self._get_sensor_json()
                sock.sendto(json_data, addr)
                
                # Print status periodically
                if frame_count - last_print >= 1000:
                    print(f"Frame {frame_count}: Running at ~{1000/(self._timestep/1000):.0f} Hz")
                    last_print = frame_count
                
            except socket.timeout:
                # No data from SITL - this is normal when SITL isn't running
                pass
            except socket.error as e:
                if connected:
                    print(f"Socket error: {e}")

            # Send tracking data for external NN tracker
            self._send_tracking_data()

            # Step Webots simulation
            if self.robot.step(self._timestep) == -1:
                break

        sock.close()
        self._tracking_sock.close()
        self._webots_connected = False
        print(f"Disconnected from Webots (I{self._instance})")

    def _get_sensor_json(self) -> bytes:
        """Build JSON sensor packet for SITL
        
        Format: {"timestamp":T,"imu":{"gyro":[r,p,y],"accel_body":[x,y,z]},
                 "position":[n,e,d],"attitude":[r,p,y],"velocity":[n,e,d]}
        
        Coordinate frame: NED (North-East-Down)
        """
        # Get sensor data from Webots
        rpy = self.imu.getRollPitchYaw()  # [roll, pitch, yaw] radians
        gyro = self.gyro.getValues()       # [x, y, z] rad/s in body frame
        accel = self.accel.getValues()     # [x, y, z] m/s^2 in body frame
        pos = self.gps.getValues()         # [x, y, z] meters in Webots frame
        vel = self.gps.getSpeedVector()    # [vx, vy, vz] m/s in Webots frame
        
        # Webots uses ENU-like frame (X=East, Y=North, Z=Up for world)
        # But body frame is X=forward, Y=left, Z=up
        # ArduPilot uses NED (X=North, Y=East, Z=Down)
        
        # Convert position from Webots to NED
        # Webots world: X=East, Y=North, Z=Up -> NED: swap X/Y, negate Z
        pos_ned = [pos[1], pos[0], -pos[2]]  # [North, East, Down]
        
        # Convert velocity from Webots to NED
        vel_ned = [vel[1], vel[0], -vel[2]]  # [Vn, Ve, Vd]
        
        # Convert attitude - Webots uses X-forward body frame
        # For NED: roll same, pitch negated, yaw negated
        attitude = [rpy[0], -rpy[1], -rpy[2]]
        
        # Convert gyro (body frame angular rates)
        # Body X=forward, Y=left, Z=up -> NED body: X=fwd, Y=right, Z=down
        gyro_ned = [gyro[0], -gyro[1], -gyro[2]]
        
        # Convert accel (body frame)
        accel_ned = [accel[0], -accel[1], -accel[2]]
        
        # Build JSON packet
        fdm = {
            "timestamp": self.robot.getTime(),
            "imu": {
                "gyro": gyro_ned,
                "accel_body": accel_ned
            },
            "position": pos_ned,
            "attitude": attitude,
            "velocity": vel_ned
        }
        
        # JSON must be preceded and terminated with newline
        json_str = "\n" + json.dumps(fdm) + "\n"
        return json_str.encode('utf-8')

    def _parse_sitl_output(self, data: bytes) -> bool:
        """Parse binary servo output from SITL
        
        Format: uint16 magic, uint16 frame_rate, uint32 frame_count, uint16 pwm[16]
        
        Returns True if valid packet parsed
        """
        if len(data) < self.SITL_OUTPUT_SIZE:
            return False
        
        try:
            # Unpack binary data
            unpacked = struct.unpack(self.SITL_OUTPUT_FORMAT, data[:self.SITL_OUTPUT_SIZE])
            magic = unpacked[0]
            frame_rate = unpacked[1]
            frame_count = unpacked[2]
            pwm = unpacked[3:19]  # 16 PWM values
            
            # Verify magic number
            if magic != self.SITL_OUTPUT_MAGIC and magic != self.SITL_OUTPUT_MAGIC_32CH:
                print(f"Invalid magic: {magic}")
                return False
            
            # Check for new frame (avoid processing duplicates)
            if frame_count == self._last_frame_count:
                return True  # Valid but duplicate
            self._last_frame_count = frame_count
            
            # Convert PWM (1000-2000 us) to normalized (0-1)
            motor_commands = [(p - 1000) / 1000.0 for p in pwm[:len(self._motors)]]
            
            # Apply motor commands
            self._apply_motor_commands(motor_commands)
            
            return True
            
        except struct.error as e:
            print(f"Struct unpack error: {e}")
            return False

    def _get_tracking_data(self) -> dict:
        """Get tracking data (drone + human position) for external NN tracker.
        
        Returns:
            Dictionary with drone and human positions in NED frame
        """
        # Get drone position and attitude
        pos = self.gps.getValues()
        rpy = self.imu.getRollPitchYaw()
        vel = self.gps.getSpeedVector()
        gyro = self.gyro.getValues()
        
        # Convert to NED
        drone_pos_ned = [pos[1], pos[0], -pos[2]]  # Webots ENU -> NED
        drone_vel_ned = [vel[1], vel[0], -vel[2]]
        
        # Get human position if available
        human_pos_ned = None
        if self._is_supervisor and self._pedestrian_node:
            try:
                trans_field = self._pedestrian_node.getField("translation")
                if trans_field:
                    human_pos = trans_field.getSFVec3f()
                    human_pos_ned = [human_pos[1], human_pos[0], -human_pos[2]]
            except Exception:
                pass
        
        return {
            "timestamp": self.robot.getTime(),
            "drone": {
                "position": drone_pos_ned,
                "velocity": drone_vel_ned,
                "attitude": [rpy[0], -rpy[1], -rpy[2]],  # roll, pitch, yaw
                "angular_velocity": [gyro[0], -gyro[1], -gyro[2]],
            },
            "human": {
                "position": human_pos_ned,
            } if human_pos_ned else None,
        }
    
    def _send_tracking_data(self):
        """Send tracking data via UDP for external NN tracker."""
        try:
            data = self._get_tracking_data()
            json_str = json.dumps(data)
            self._tracking_sock.sendto(
                json_str.encode('utf-8'), 
                ('127.0.0.1', self._tracking_port)
            )
        except Exception:
            pass  # Silently ignore errors (no listener connected)

    def _apply_motor_commands(self, commands: list):
        """Apply motor commands to Webots motors
        
        Args:
            commands: list of motor values 0.0-1.0
        """
        # Clamp commands to valid range
        commands = [max(0.0, min(1.0, c)) for c in commands]
        
        # Scale for bidirectional motors
        if self._bidirectional_motors:
            commands = [c * 2 - 1 for c in commands]
        
        # Linearize propeller thrust
        if self._uses_propellers:
            # Thrust = k * omega^2, so omega = sqrt(thrust)
            commands = [np.sqrt(np.abs(c)) * np.sign(c) for c in commands]
        
        # Reverse motors if needed
        if self._reversed_motors:
            for m in self._reversed_motors:
                if m - 1 < len(commands):
                    commands[m - 1] *= -1
        
        # Set motor velocities
        for i, motor in enumerate(self._motors):
            if i < len(commands):
                vel = commands[i] * min(motor.getMaxVelocity(), self.motor_velocity_cap)
                motor.setVelocity(vel)

    def _handle_image_stream(self, camera: Union[Camera, RangeFinder], port: int):
        """Stream images over TCP"""
        if isinstance(camera, Camera):
            cam_sample_period = self.camera.getSamplingPeriod()
            cam_width = self.camera.getWidth()
            cam_height = self.camera.getHeight()
            print(f"Camera stream started at 127.0.0.1:{port} (I{self._instance}) "
                  f"({cam_width}x{cam_height} @ {1000/cam_sample_period:0.2f}fps)")
        elif isinstance(camera, RangeFinder):
            cam_sample_period = self.rangefinder.getSamplingPeriod()
            cam_width = self.rangefinder.getWidth()
            cam_height = self.rangefinder.getHeight()
            print(f"RangeFinder stream started at 127.0.0.1:{port} (I{self._instance}) "
                  f"({cam_width}x{cam_height} @ {1000/cam_sample_period:0.2f}fps)")
        else:
            print(f"Error: invalid camera type '{type(camera)}'", file=sys.stderr)
            return

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', port))
        server.listen(1)

        while self._webots_connected:
            conn, _ = server.accept()
            print(f"Camera client connected (I{self._instance})")
            try:
                while self._webots_connected:
                    start_time = self.robot.getTime()
                    
                    if isinstance(camera, Camera):
                        img = self.get_camera_gray_image()
                    else:
                        img = self.get_rangefinder_image()
                    
                    if img is None:
                        time.sleep(cam_sample_period / 1000)
                        continue
                    
                    header = struct.pack("=HH", cam_width, cam_height)
                    conn.sendall(header + img.tobytes())
                    
                    while self.robot.getTime() - start_time < cam_sample_period / 1000:
                        time.sleep(0.001)
            except (ConnectionResetError, BrokenPipeError):
                pass
            finally:
                conn.close()
                print(f"Camera client disconnected (I{self._instance})")

    def get_camera_gray_image(self) -> np.ndarray:
        """Get grayscale camera image"""
        img = self.get_camera_image()
        return np.average(img, axis=2).astype(np.uint8)

    def get_camera_image(self) -> np.ndarray:
        """Get RGB camera image"""
        img = self.camera.getImage()
        img = np.frombuffer(img, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
        return img[:, :, :3]

    def get_rangefinder_image(self, use_int16: bool = False) -> np.ndarray:
        """Get rangefinder depth image"""
        height = self.rangefinder.getHeight()
        width = self.rangefinder.getWidth()
        
        image_c_ptr = self.rangefinder.getRangeImage(data_type="buffer")
        img_arr = np.ctypeslib.as_array(image_c_ptr, (width * height,))
        img_floats = img_arr.reshape((height, width))
        
        range_range = self.rangefinder.getMaxRange() - self.rangefinder.getMinRange()
        img_normalized = (img_floats - self.rangefinder.getMinRange()) / range_range
        img_normalized[img_normalized == float('inf')] = 1
        
        if use_int16:
            return (img_normalized * 65535).astype(np.uint16)
        return (img_normalized * 255).astype(np.uint8)

    def stop_motors(self):
        """Stop all motors"""
        for m in self._motors:
            m.setPosition(float('inf'))
            m.setVelocity(0)

    def webots_connected(self) -> bool:
        """Check if Webots is connected"""
        return self._webots_connected
