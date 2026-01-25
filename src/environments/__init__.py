"""Gymnasium environments for drone control."""

from .base_drone_env import BaseDroneEnv, DroneEnvConfig
from .hover_env import HoverEnv, HoverEnvConfig
from .waypoint_env import WaypointEnv
from .trajectory_env import TrajectoryEnv
from .setpoint_base_env import SetpointBaseEnv, SetpointEnvConfig, SetpointMode
from .setpoint_env import (
    SetpointHoverEnv,
    SetpointHoverConfig,
    SetpointWaypointEnv,
    SetpointWaypointConfig,
    SetpointTrackingEnv,
    SetpointTrackingConfig,
)
from .sitl_env import SITLEnv, SITLEnvConfig
from .yaw_tracking_env import (
    YawTrackingEnv,
    YawTrackingConfig,
    TargetPatternType,
    CircularTarget,
    RandomTarget,
    SinusoidalTarget,
    StepTarget,
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
