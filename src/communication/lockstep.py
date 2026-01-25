"""Lockstep synchronization for deterministic simulation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class LockstepState(Enum):
    """States for lockstep synchronization."""
    WAITING_FOR_ACTUATORS = "waiting_for_actuators"
    STEPPING_PHYSICS = "stepping_physics"
    SENDING_SENSORS = "sending_sensors"


@dataclass
class TimingStats:
    """Statistics for lockstep timing."""
    
    total_steps: int = 0
    total_wait_time: float = 0.0
    total_physics_time: float = 0.0
    max_wait_time: float = 0.0
    missed_frames: int = 0
    
    @property
    def avg_wait_time(self) -> float:
        """Average time waiting for actuator commands."""
        if self.total_steps == 0:
            return 0.0
        return self.total_wait_time / self.total_steps
    
    @property
    def avg_physics_time(self) -> float:
        """Average physics step time."""
        if self.total_steps == 0:
            return 0.0
        return self.total_physics_time / self.total_steps


class LockstepController:
    """Controller for lockstep simulation synchronization.
    
    In lockstep mode:
    1. Simulator waits for HIL_ACTUATOR_CONTROLS from PX4
    2. Simulator steps physics with received commands
    3. Simulator sends HIL_SENSOR, HIL_GPS back to PX4
    4. Repeat
    
    This ensures deterministic simulation regardless of execution speed.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        timeout: float = 1.0,
        physics_timestep: float = 0.002,
    ):
        """Initialize lockstep controller.
        
        Args:
            enabled: Whether lockstep mode is enabled
            timeout: Maximum time to wait for actuator commands
            physics_timestep: Physics simulation timestep
        """
        self.enabled = enabled
        self.timeout = timeout
        self.physics_timestep = physics_timestep
        
        self._state = LockstepState.WAITING_FOR_ACTUATORS
        self._last_actuator_time: Optional[float] = None
        self._frame_count = 0
        self._sim_time = 0.0
        
        self._stats = TimingStats()
        self._step_start_time: Optional[float] = None
    
    def reset(self) -> None:
        """Reset lockstep state for new simulation."""
        self._state = LockstepState.WAITING_FOR_ACTUATORS
        self._last_actuator_time = None
        self._frame_count = 0
        self._sim_time = 0.0
        self._stats = TimingStats()
    
    def begin_step(self) -> None:
        """Mark beginning of a simulation step.
        
        Call this when starting to wait for actuator commands.
        """
        self._step_start_time = time.monotonic()
        self._state = LockstepState.WAITING_FOR_ACTUATORS
    
    def actuators_received(self, actuator_time_usec: int) -> bool:
        """Called when actuator commands are received.
        
        Args:
            actuator_time_usec: Timestamp from actuator message
            
        Returns:
            True if timing is valid, False if frame was missed
        """
        now = time.monotonic()
        
        if self._step_start_time is not None:
            wait_time = now - self._step_start_time
            self._stats.total_wait_time += wait_time
            self._stats.max_wait_time = max(self._stats.max_wait_time, wait_time)
        
        # Check for missed frames
        frame_valid = True
        if self._last_actuator_time is not None:
            expected_dt = self.physics_timestep * 1e6  # Convert to microseconds
            actual_dt = actuator_time_usec - self._last_actuator_time
            
            # Allow some tolerance (10%)
            if actual_dt > expected_dt * 1.1:
                self._stats.missed_frames += 1
                frame_valid = False
        
        self._last_actuator_time = actuator_time_usec
        self._state = LockstepState.STEPPING_PHYSICS
        
        return frame_valid
    
    def physics_stepped(self) -> None:
        """Called after physics simulation step completes."""
        now = time.monotonic()
        
        if self._step_start_time is not None:
            # Estimate physics time (total time minus wait time approximation)
            total_time = now - self._step_start_time
            physics_time = max(0, total_time - self._stats.avg_wait_time)
            self._stats.total_physics_time += physics_time
        
        self._state = LockstepState.SENDING_SENSORS
        self._frame_count += 1
        self._sim_time += self.physics_timestep
        self._stats.total_steps += 1
    
    def sensors_sent(self) -> None:
        """Called after sensor data is sent to PX4."""
        self._state = LockstepState.WAITING_FOR_ACTUATORS
        self._step_start_time = None
    
    def wait_for_actuators(
        self,
        check_func,
        poll_interval: float = 0.0001,
    ) -> bool:
        """Wait for actuator commands with timeout.
        
        Args:
            check_func: Function that returns True when actuators received
            poll_interval: How often to poll (seconds)
            
        Returns:
            True if actuators received, False if timeout
        """
        if not self.enabled:
            return True
        
        start_time = time.monotonic()
        
        while True:
            if check_func():
                return True
            
            elapsed = time.monotonic() - start_time
            if elapsed > self.timeout:
                return False
            
            time.sleep(poll_interval)
    
    @property
    def state(self) -> LockstepState:
        """Current lockstep state."""
        return self._state
    
    @property
    def frame_count(self) -> int:
        """Total number of completed frames."""
        return self._frame_count
    
    @property
    def simulation_time(self) -> float:
        """Current simulation time in seconds."""
        return self._sim_time
    
    @property
    def statistics(self) -> TimingStats:
        """Get timing statistics."""
        return self._stats
    
    def get_simulation_time_usec(self) -> int:
        """Get simulation time in microseconds."""
        return int(self._sim_time * 1e6)
    
    def should_send_gps(self, gps_rate: float = 10.0) -> bool:
        """Check if GPS should be sent this frame.
        
        Args:
            gps_rate: GPS update rate in Hz
            
        Returns:
            True if GPS should be sent
        """
        gps_period = 1.0 / gps_rate
        frames_per_gps = int(gps_period / self.physics_timestep)
        return self._frame_count % frames_per_gps == 0
    
    def should_send_baro(self, baro_rate: float = 50.0) -> bool:
        """Check if barometer should be sent this frame.
        
        Args:
            baro_rate: Barometer update rate in Hz
            
        Returns:
            True if barometer should be sent
        """
        baro_period = 1.0 / baro_rate
        frames_per_baro = int(baro_period / self.physics_timestep)
        return self._frame_count % max(1, frames_per_baro) == 0
