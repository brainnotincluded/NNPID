"""Deployment module for trained neural network controllers.

This module provides utilities for:
- Exporting trained models to ONNX format
- Connecting to real drones via MAVLink
- Running inference on edge devices
- Deploying yaw tracking NN to ArduPilot SITL
"""

from .mavlink_client import DroneState, MAVLinkClient
from .model_export import ModelExporter, export_to_onnx
from .model_loading import (
    find_vec_normalize_path,
    load_model_and_vecnormalize,
    load_sb3_model,
    resolve_model_path,
)
from .trained_yaw_tracker import TrainedYawTracker
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
    "load_model_and_vecnormalize",
    "load_sb3_model",
    "resolve_model_path",
    "find_vec_normalize_path",
    "TrainedYawTracker",
    "YawTrackerSITL",
    "YawTrackerSITLConfig",
    "TargetState",
]
