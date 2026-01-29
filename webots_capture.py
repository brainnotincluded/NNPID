import socket
import struct
import sys
import numpy as np
import cv2

# Where Webots camera stream runs. Use 127.0.0.1 if Webots is on this machine,
# or the robot/sim host IP (e.g. 192.168.0.113 as in current.py) if remote.
WEBOTS_CAMERA = '127.0.0.1'
WEBOTS_CAMERA_PORT = 5599


def main():
    print(f"Connecting to {WEBOTS_CAMERA}:{WEBOTS_CAMERA_PORT} ...", flush=True)
    captureDevice = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        captureDevice.connect((WEBOTS_CAMERA, WEBOTS_CAMERA_PORT))
    except (ConnectionRefusedError, OSError) as e:
        print(f"Cannot connect to {WEBOTS_CAMERA}:{WEBOTS_CAMERA_PORT}: {e}", file=sys.stderr, flush=True)
        print("Ensure Webots is running with the camera stream server on that host/port.", file=sys.stderr, flush=True)
        sys.exit(1)
    print(f"Connected to {WEBOTS_CAMERA}:{WEBOTS_CAMERA_PORT}. Receiving frames (press 'q' to quit).", flush=True)
    webots_stream_header_size = struct.calcsize("=HH")
    cv2.namedWindow('Webots Camera', cv2.WINDOW_NORMAL)
    first_frame = True

    while True:
        header = captureDevice.recv(webots_stream_header_size)
        if not header:
            print("Connection closed by server.", file=sys.stderr, flush=True)
            break
        if len(header) != webots_stream_header_size:
            continue
        width, height = struct.unpack("=HH", header)
        bytes_to_read = width * height
        img = bytes()
        while len(img) < bytes_to_read:
            chunk = captureDevice.recv(min(bytes_to_read - len(img), 16384))
            if not chunk:
                print("Connection closed while reading frame.", file=sys.stderr, flush=True)
                break
            img += chunk
        if len(img) != bytes_to_read:
            break
        if first_frame:
            print(f"First frame: {width}x{height}", flush=True)
            first_frame = False
        img = np.frombuffer(img, np.uint8).reshape((height, width))
        cv2.imshow('Webots Camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()