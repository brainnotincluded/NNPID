"""
NNPID - Neural Network PID Replacement for Drone Target Tracking

A deep reinforcement learning system using Recurrent Soft Actor-Critic (RSAC)
with GRU networks for adaptive drone control.
"""

__version__ = "0.1.0"
__author__ = "NNPID Team"

from src.environment.simple_drone_sim import SimpleDroneSimulator
from src.utils.trajectory_generator import TrajectoryType, TrajectoryGenerator
from src.models.gru_networks import RSACSharedEncoder

__all__ = [
    "SimpleDroneSimulator",
    "TrajectoryType",
    "TrajectoryGenerator", 
    "RSACSharedEncoder",
]