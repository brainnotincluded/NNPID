"""Deployment module for trained neural network controllers.

This module provides utilities for:
- Exporting trained models to ONNX format
- Connecting to real drones via MAVLink
- Running inference on edge devices
- Deploying yaw tracking NN to ArduPilot SITL
"""

from .model_export import ModelExporter, export_to_onnx
from .mavlink_client import MAVLinkClient, DroneState
from .yaw_tracker_sitl import (
    YawTrackerSITL,
    YawTrackerSITLConfig,
    TargetState,
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
