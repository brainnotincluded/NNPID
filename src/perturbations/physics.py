"""Physics perturbations for realistic physical variations."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

from .base import BasePerturbation, PhysicsConfig

if TYPE_CHECKING:
    from ..core.mujoco_sim import QuadrotorState


class MotorModel:
    """Model individual motor characteristics with variation.
    
    Simulates manufacturing variation, wear, and degradation
    of individual motors in a quadrotor.
    """
    
    def __init__(
        self,
        motor_id: int,
        thrust_factor: float = 1.0,
        response_factor: float = 1.0,
        degradation: float = 0.0,
    ):
        """Initialize motor model.
        
        Args:
            motor_id: Motor identifier (0-3)
            thrust_factor: Thrust multiplier (1.0 = nominal)
            response_factor: Response time multiplier
            degradation: Current degradation level (0-1)
        """
        self.motor_id = motor_id
        self.thrust_factor = thrust_factor
        self.response_factor = response_factor
        self.degradation = degradation
        
        # Internal state for dynamics
        self._current_thrust = 0.0
        self._failed = False
    
    def update(self, dt: float, command: float, degradation_rate: float = 0.0) -> float:
        """Update motor state and return actual thrust.
        
        Args:
            dt: Time step
            command: Commanded thrust (0-1)
            degradation_rate: Rate of degradation per second
            
        Returns:
            Actual thrust output (0-1)
        """
        if self._failed:
            return 0.0
        
        # Apply degradation
        self.degradation = min(1.0, self.degradation + degradation_rate * dt)
        
        # Calculate effective thrust
        effective = command * self.thrust_factor * (1.0 - self.degradation)
        
        # Motor dynamics (first-order response)
        tau = 0.02 * self.response_factor  # Time constant
        alpha = dt / (tau + dt)
        self._current_thrust = (1 - alpha) * self._current_thrust + alpha * effective
        
        return np.clip(self._current_thrust, 0.0, 1.0)
    
    def fail(self) -> None:
        """Simulate motor failure."""
        self._failed = True
        self._current_thrust = 0.0
    
    def reset(self) -> None:
        """Reset motor state."""
        self._current_thrust = 0.0
        self._failed = False
    
    @property
    def is_failed(self) -> bool:
        """Check if motor has failed."""
        return self._failed
    
    @property
    def effective_power(self) -> float:
        """Get effective power output ratio."""
        if self._failed:
            return 0.0
        return self.thrust_factor * (1.0 - self.degradation)


class GroundEffect:
    """Simulates ground effect - increased thrust efficiency near ground.
    
    Based on empirical models for multirotor ground effect.
    """
    
    def __init__(
        self,
        height_threshold: float = 0.5,
        strength: float = 0.3,
        thrust_multiplier_at_ground: float = 1.2,
        rotor_radius: float = 0.127,  # 5 inch prop
    ):
        """Initialize ground effect model.
        
        Args:
            height_threshold: Height below which ground effect is significant (m)
            strength: Overall effect strength (0-1)
            thrust_multiplier_at_ground: Thrust multiplier at ground level
            rotor_radius: Rotor radius for calculation (m)
        """
        self.height_threshold = height_threshold
        self.strength = strength
        self.thrust_multiplier_at_ground = thrust_multiplier_at_ground
        self.rotor_radius = rotor_radius
    
    def get_thrust_multiplier(self, altitude: float) -> float:
        """Get thrust multiplier based on altitude.
        
        Args:
            altitude: Height above ground (m)
            
        Returns:
            Thrust multiplier (>= 1.0)
        """
        if altitude >= self.height_threshold:
            return 1.0
        
        if altitude <= 0:
            return self.thrust_multiplier_at_ground
        
        # Empirical ground effect model
        # Based on Cheeseman and Bennett model
        z_R = altitude / self.rotor_radius
        
        if z_R < 0.1:
            z_R = 0.1  # Avoid singularity
        
        # Ground effect factor
        ge_factor = 1.0 / (1.0 - (self.rotor_radius / (4 * altitude)) ** 2)
        ge_factor = np.clip(ge_factor, 1.0, self.thrust_multiplier_at_ground)
        
        # Blend with altitude
        blend = (self.height_threshold - altitude) / self.height_threshold
        blend = np.clip(blend, 0.0, 1.0)
        
        multiplier = 1.0 + (ge_factor - 1.0) * blend * self.strength
        
        return multiplier
    
    def get_additional_force(self, altitude: float, current_thrust: float) -> np.ndarray:
        """Get additional upward force from ground effect.
        
        Args:
            altitude: Height above ground (m)
            current_thrust: Current total thrust (N)
            
        Returns:
            Additional force vector [fx, fy, fz]
        """
        multiplier = self.get_thrust_multiplier(altitude)
        additional_thrust = current_thrust * (multiplier - 1.0)
        
        # Purely vertical force
        return np.array([0.0, 0.0, additional_thrust])


class WallProximityEffect:
    """Simulates aerodynamic effects when flying near walls/obstacles."""
    
    def __init__(
        self,
        effect_distance: float = 1.0,
        force_coefficient: float = 0.1,
    ):
        """Initialize wall proximity effect.
        
        Args:
            effect_distance: Distance at which effect starts (m)
            force_coefficient: Force coefficient
        """
        self.effect_distance = effect_distance
        self.force_coefficient = force_coefficient
        
        # Wall definitions: list of (position, normal)
        self._walls: List[Tuple[np.ndarray, np.ndarray]] = []
    
    def add_wall(self, position: np.ndarray, normal: np.ndarray) -> None:
        """Add a wall to the environment.
        
        Args:
            position: Point on wall
            normal: Wall normal (pointing away from wall)
        """
        normal = normal / np.linalg.norm(normal)
        self._walls.append((position.copy(), normal))
    
    def clear_walls(self) -> None:
        """Remove all walls."""
        self._walls.clear()
    
    def get_force(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Calculate repulsive force from nearby walls.
        
        Args:
            position: Drone position
            velocity: Drone velocity
            
        Returns:
            Force vector
        """
        total_force = np.zeros(3)
        
        for wall_pos, wall_normal in self._walls:
            # Distance to wall
            to_wall = position - wall_pos
            distance = np.dot(to_wall, wall_normal)
            
            if 0 < distance < self.effect_distance:
                # Repulsive force proportional to 1/distance^2
                force_magnitude = self.force_coefficient / (distance ** 2)
                
                # Damping from velocity toward wall
                velocity_toward = -np.dot(velocity, wall_normal)
                if velocity_toward > 0:
                    force_magnitude *= (1.0 + velocity_toward * 0.5)
                
                # Force direction is wall normal (away from wall)
                total_force += force_magnitude * wall_normal
        
        return total_force


class PhysicsPerturbation(BasePerturbation):
    """Comprehensive physics perturbations.
    
    Components:
    - Center of mass offset: Asymmetric payload
    - Motor variation: Manufacturing differences
    - Motor degradation: Wear over time
    - Mass change: Variable payload
    - Ground effect: Increased thrust near ground
    - Wall proximity: Aerodynamic interaction with walls
    """
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        """Initialize physics perturbation.
        
        Args:
            config: Physics configuration. Uses defaults if None.
        """
        super().__init__(config or PhysicsConfig())
        self.physics_config: PhysicsConfig = self.config
        
        # Center of mass offset
        self._com_offset = np.zeros(3)
        
        # Motor models
        self._motors = [MotorModel(i) for i in range(4)]
        
        # Mass
        self._base_mass = 2.0  # kg
        self._current_mass = self._base_mass
        self._mass_multiplier = 1.0
        
        # Ground effect
        self._ground_effect = GroundEffect(
            height_threshold=self.physics_config.ground_effect_height,
            strength=self.physics_config.ground_effect_strength,
            thrust_multiplier_at_ground=self.physics_config.ground_effect_thrust_mult,
        )
        
        # Wall proximity
        self._wall_effect = WallProximityEffect(
            effect_distance=self.physics_config.proximity_distance,
            force_coefficient=self.physics_config.proximity_coefficient,
        )
        
        # Failure tracking
        self._motor_failure_checked = False
    
    def reset(self, rng: np.random.Generator) -> None:
        """Reset physics perturbation state."""
        super().reset(rng)
        
        cfg = self.physics_config
        
        # Reset center of mass
        if cfg.com_offset_enabled:
            if cfg.com_randomize:
                self._com_offset = rng.uniform(
                    -cfg.com_max_offset, cfg.com_max_offset, 3
                )
            else:
                self._com_offset = np.array(cfg.com_offset)
        else:
            self._com_offset = np.zeros(3)
        
        # Reset motors
        for i, motor in enumerate(self._motors):
            motor.reset()
            
            if cfg.motor_variation_enabled:
                if cfg.motor_per_motor:
                    motor.thrust_factor = 1.0 + rng.uniform(
                        -cfg.motor_thrust_variation, cfg.motor_thrust_variation
                    )
                    motor.response_factor = 1.0 + rng.uniform(
                        -cfg.motor_response_variation, cfg.motor_response_variation
                    )
                else:
                    # Same variation for all motors
                    common_thrust = 1.0 + rng.uniform(
                        -cfg.motor_thrust_variation, cfg.motor_thrust_variation
                    )
                    common_response = 1.0 + rng.uniform(
                        -cfg.motor_response_variation, cfg.motor_response_variation
                    )
                    motor.thrust_factor = common_thrust
                    motor.response_factor = common_response
            else:
                motor.thrust_factor = 1.0
                motor.response_factor = 1.0
            
            # Initial degradation
            if cfg.degradation_enabled:
                motor.degradation = cfg.degradation_initial[i]
            else:
                motor.degradation = 0.0
        
        # Reset mass
        if cfg.mass_change_enabled:
            self._mass_multiplier = rng.uniform(
                cfg.mass_variation_min, cfg.mass_variation_max
            )
        else:
            self._mass_multiplier = 1.0
        self._current_mass = self._base_mass * self._mass_multiplier
        
        # Motor failure check for this episode
        self._motor_failure_checked = False
        if cfg.degradation_enabled and cfg.degradation_failure_probability > 0:
            if rng.random() < cfg.degradation_failure_probability:
                # Random motor failure at random time
                # (handled in update)
                self._motor_failure_checked = True
    
    def _randomize_parameters(self) -> None:
        """Randomize physics parameters."""
        cfg = self.physics_config
        
        if cfg.com_randomize and cfg.com_offset_enabled:
            self._com_offset = self._rng.uniform(
                -cfg.com_max_offset, cfg.com_max_offset, 3
            )
    
    def update(self, dt: float, state: "QuadrotorState") -> None:
        """Update physics perturbation.
        
        Args:
            dt: Time step
            state: Current quadrotor state
        """
        if not self.enabled:
            self._current_force = np.zeros(3)
            self._current_torque = np.zeros(3)
            return
        
        self._time += dt
        cfg = self.physics_config
        
        # Update motor degradation
        if cfg.degradation_enabled:
            for motor in self._motors:
                motor.degradation = min(
                    1.0,
                    motor.degradation + cfg.degradation_rate * dt
                )
        
        # Dynamic mass change
        if cfg.mass_change_enabled and cfg.mass_dynamic_change:
            mass_delta = cfg.mass_change_rate * dt
            self._current_mass += mass_delta
            self._current_mass = np.clip(
                self._current_mass,
                self._base_mass * cfg.mass_variation_min,
                self._base_mass * cfg.mass_variation_max,
            )
        
        # Calculate forces
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        # Ground effect
        if cfg.ground_effect_enabled:
            altitude = state.position[2]
            # Estimate current thrust from motor speeds
            estimated_thrust = np.mean(state.motor_speeds) * 32.0  # 4 motors * 8N max
            ge_force = self._ground_effect.get_additional_force(altitude, estimated_thrust)
            total_force += ge_force * cfg.intensity
        
        # Wall proximity effect
        if cfg.proximity_enabled:
            wall_force = self._wall_effect.get_force(state.position, state.velocity)
            total_force += wall_force * cfg.intensity
        
        # Center of mass offset creates a torque under thrust
        if cfg.com_offset_enabled and np.linalg.norm(self._com_offset) > 0.001:
            # Torque = r x F (offset crossed with thrust)
            thrust_force = np.array([0, 0, np.mean(state.motor_speeds) * 32.0])
            com_torque = np.cross(self._com_offset, thrust_force)
            total_torque += com_torque * cfg.intensity * 0.1  # Scale down
        
        self._current_force = total_force
        self._current_torque = total_torque
    
    def apply_to_action(self, action: np.ndarray) -> np.ndarray:
        """Apply motor variations to action.
        
        Args:
            action: Original motor commands (0-1)
            
        Returns:
            Modified motor commands accounting for variations
        """
        if not self.enabled:
            return action
        
        cfg = self.physics_config
        result = action.copy()
        
        # Apply motor-specific effects
        if cfg.motor_variation_enabled or cfg.degradation_enabled:
            dt = 0.02  # Approximate control timestep
            for i, motor in enumerate(self._motors):
                if i < len(result):
                    result[i] = motor.update(
                        dt,
                        action[i],
                        degradation_rate=cfg.degradation_rate if cfg.degradation_enabled else 0.0,
                    )
        
        return result
    
    def set_base_mass(self, mass: float) -> None:
        """Set the base mass of the drone.
        
        Args:
            mass: Base mass in kg
        """
        self._base_mass = mass
        self._current_mass = mass * self._mass_multiplier
    
    def get_mass_multiplier(self) -> float:
        """Get current mass multiplier."""
        return self._mass_multiplier
    
    def get_motor_status(self) -> List[Dict[str, Any]]:
        """Get status of all motors.
        
        Returns:
            List of motor status dictionaries
        """
        return [
            {
                "id": motor.motor_id,
                "thrust_factor": motor.thrust_factor,
                "response_factor": motor.response_factor,
                "degradation": motor.degradation,
                "effective_power": motor.effective_power,
                "failed": motor.is_failed,
            }
            for motor in self._motors
        ]
    
    def fail_motor(self, motor_id: int) -> None:
        """Force a motor failure.
        
        Args:
            motor_id: Motor to fail (0-3)
        """
        if 0 <= motor_id < len(self._motors):
            self._motors[motor_id].fail()
    
    def add_wall(self, position: np.ndarray, normal: np.ndarray) -> None:
        """Add a wall for proximity effects.
        
        Args:
            position: Point on wall
            normal: Wall normal
        """
        self._wall_effect.add_wall(position, normal)
    
    def get_info(self) -> Dict[str, Any]:
        """Get physics perturbation information."""
        info = super().get_info()
        info.update({
            "com_offset": self._com_offset.tolist(),
            "current_mass": float(self._current_mass),
            "mass_multiplier": float(self._mass_multiplier),
            "motors": self.get_motor_status(),
            "ground_effect_active": self._current_force[2] > 0.01,
        })
        return info


# Convenience factory functions
def create_ideal_physics() -> PhysicsPerturbation:
    """Create ideal physics with no perturbations."""
    config = PhysicsConfig(
        enabled=True,
        intensity=1.0,
        com_offset_enabled=False,
        motor_variation_enabled=False,
        degradation_enabled=False,
        mass_change_enabled=False,
        ground_effect_enabled=False,
        proximity_enabled=False,
    )
    return PhysicsPerturbation(config)


def create_realistic_physics() -> PhysicsPerturbation:
    """Create realistic physics with typical variations."""
    config = PhysicsConfig(
        enabled=True,
        intensity=1.0,
        com_offset_enabled=True,
        com_randomize=True,
        com_max_offset=0.01,
        motor_variation_enabled=True,
        motor_thrust_variation=0.03,
        motor_response_variation=0.05,
        motor_per_motor=True,
        degradation_enabled=False,
        mass_change_enabled=True,
        mass_variation_min=0.95,
        mass_variation_max=1.05,
        ground_effect_enabled=True,
        ground_effect_height=0.5,
        ground_effect_strength=0.3,
    )
    return PhysicsPerturbation(config)


def create_worn_drone() -> PhysicsPerturbation:
    """Create physics for a worn/aged drone."""
    config = PhysicsConfig(
        enabled=True,
        intensity=1.0,
        com_offset_enabled=True,
        com_randomize=True,
        com_max_offset=0.02,
        motor_variation_enabled=True,
        motor_thrust_variation=0.1,
        motor_response_variation=0.15,
        motor_per_motor=True,
        degradation_enabled=True,
        degradation_rate=0.0001,
        degradation_initial=[0.05, 0.1, 0.03, 0.08],
        degradation_failure_probability=0.01,
        mass_change_enabled=True,
        mass_variation_min=0.9,
        mass_variation_max=1.15,
        ground_effect_enabled=True,
    )
    return PhysicsPerturbation(config)


def create_payload_variation() -> PhysicsPerturbation:
    """Create physics for variable payload scenarios."""
    config = PhysicsConfig(
        enabled=True,
        intensity=1.0,
        com_offset_enabled=True,
        com_offset=[0.02, 0.01, -0.01],  # Fixed offset
        com_randomize=False,
        motor_variation_enabled=True,
        motor_thrust_variation=0.02,
        degradation_enabled=False,
        mass_change_enabled=True,
        mass_variation_min=1.0,
        mass_variation_max=1.5,  # Up to 50% extra mass
        mass_dynamic_change=True,
        mass_change_rate=0.01,  # Slow mass change
        ground_effect_enabled=True,
    )
    return PhysicsPerturbation(config)
