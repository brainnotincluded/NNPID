"""Gymnasium environments for drone control."""

from .base_drone_env import BaseDroneEnv, DroneEnvConfig
from .hover_env import HoverEnv, HoverEnvConfig
from .setpoint_base_env import SetpointBaseEnv, SetpointEnvConfig, SetpointMode
from .setpoint_env import (
    SetpointHoverConfig,
    SetpointHoverEnv,
    SetpointTrackingConfig,
    SetpointTrackingEnv,
    SetpointWaypointConfig,
    SetpointWaypointEnv,
)
from .sitl_env import SITLEnv, SITLEnvConfig
from .trajectory_env import TrajectoryEnv
from .waypoint_env import WaypointEnv
from .yaw_tracking_env import (
    CircularTarget,
    RandomTarget,
    SinusoidalTarget,
    StepTarget,
    TargetPatternType,
    YawTrackingConfig,
    YawTrackingEnv,
)

__all__ = [
    # Motor command environments
    "BaseDroneEnv",
    "DroneEnvConfig",
    "HoverEnv",
    "HoverEnvConfig",
    "WaypointEnv",
    "TrajectoryEnv",
    # Setpoint environments (fast training)
    "SetpointBaseEnv",
    "SetpointEnvConfig",
    "SetpointMode",
    "SetpointHoverEnv",
    "SetpointHoverConfig",
    "SetpointWaypointEnv",
    "SetpointWaypointConfig",
    "SetpointTrackingEnv",
    "SetpointTrackingConfig",
    # SITL environment
    "SITLEnv",
    "SITLEnvConfig",
    # Yaw tracking environment
    "YawTrackingEnv",
    "YawTrackingConfig",
    "TargetPatternType",
    "CircularTarget",
    "RandomTarget",
    "SinusoidalTarget",
    "StepTarget",
]
