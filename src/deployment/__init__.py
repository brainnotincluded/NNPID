"""Deployment module for trained neural network controllers.

This module provides utilities for:
- Exporting trained models to ONNX format
- Connecting to real drones via MAVLink
- Running inference on edge devices
- Deploying yaw tracking NN to ArduPilot SITL
"""

from .mavlink_client import DroneState, MAVLinkClient
from .model_export import ModelExporter, export_to_onnx
from .yaw_tracker_sitl import (
    TargetState,
    YawTrackerSITL,
    YawTrackerSITLConfig,
)

__all__ = [
    "ModelExporter",
    "export_to_onnx",
    "MAVLinkClient",
    "DroneState",
    "YawTrackerSITL",
    "YawTrackerSITLConfig",
    "TargetState",
]
