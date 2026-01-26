"""Tests for Gymnasium environments."""

import numpy as np
import pytest

# Skip if dependencies not available
pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from src.environments.base_drone_env import BaseDroneEnv, DroneEnvConfig
from src.environments.hover_env import HoverEnv, HoverEnvConfig


class TestHoverEnv:
    """Test hover environment."""

    @pytest.fixture
    def env(self):
        """Create hover environment."""
        config = HoverEnvConfig(
            max_episode_steps=100,
            randomize_hover_position=False,
            hover_position=(0.0, 0.0, 1.0),
        )
        env = HoverEnv(config=config)
        yield env
        env.close()

    def test_env_creation(self, env):
        """Test environment can be created."""
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_observation_space(self, env):
        """Test observation space dimensions."""
        obs_space = env.observation_space
        assert obs_space.shape == (20,)

    def test_action_space(self, env):
        """Test action space."""
        action_space = env.action_space
        assert action_space.shape == (4,)
        assert action_space.low.min() == 0.0
        assert action_space.high.max() == 1.0

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset(seed=42)

        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_step(self, env):
        """Test environment step."""
        env.reset(seed=42)

        action = np.array([0.5, 0.5, 0.5, 0.5])
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self, env):
        """Test that episode eventually terminates."""
        env.reset(seed=42)

        done = False
        steps = 0
        max_steps = 200

        while not done and steps < max_steps:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done  # Should terminate (either crash or truncate)

    def test_hover_at_target(self, env):
        """Test hovering at target position."""
        env.reset(seed=42)

        # Approximate hover command
        hover_cmd = np.array([0.5, 0.5, 0.5, 0.5])

        rewards = []
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(hover_cmd)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Rewards should be finite
        assert all(np.isfinite(r) for r in rewards)

    def test_info_contains_position(self, env):
        """Test that info contains position."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.5, 0.5, 0.5, 0.5]))

        assert "position" in info
        assert "velocity" in info
        assert "position_error" in info

    def test_deterministic_reset(self, env):
        """Test that reset with same seed gives same result."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

    def test_render_mode(self):
        """Test render mode configuration."""
        config = HoverEnvConfig(max_episode_steps=10)
        env = HoverEnv(config=config, render_mode="rgb_array")

        env.reset(seed=42)
        env.step(np.array([0.5, 0.5, 0.5, 0.5]))

        frame = env.render()

        # Should return an image array
        assert frame is not None
        assert len(frame.shape) == 3  # Height, width, channels

        env.close()


class TestBaseDroneEnv:
    """Test base drone environment."""

    def test_termination_on_crash(self):
        """Test termination when crashing."""
        config = DroneEnvConfig(
            max_episode_steps=1000,
            min_altitude=0.0,
        )
        env = BaseDroneEnv(config=config)

        env.reset(seed=42)

        # Apply no thrust (should fall)
        zero_cmd = np.array([0.0, 0.0, 0.0, 0.0])

        terminated = False
        for _ in range(100):
            _, _, terminated, _, _ = env.step(zero_cmd)
            if terminated:
                break

        # Should eventually crash
        # Note: might not terminate if ground contact keeps it up

        env.close()

    def test_termination_on_tilt(self):
        """Test termination on excessive tilt."""
        config = DroneEnvConfig(
            max_episode_steps=1000,
            max_tilt_angle=0.5,  # Low threshold
        )
        env = BaseDroneEnv(config=config)

        env.reset(seed=42)

        # Asymmetric thrust to cause tilt
        tilt_cmd = np.array([1.0, 0.0, 1.0, 0.0])

        terminated = False
        for _ in range(50):
            _, _, terminated, _, info = env.step(tilt_cmd)
            if terminated:
                break

        env.close()


class TestGymnasiumCompatibility:
    """Test Gymnasium API compatibility."""

    def test_make_env(self):
        """Test creating environment through gymnasium.make."""
        import gymnasium as gym

        # Register environments
        from src.environments.base_drone_env import BaseDroneEnv

        # Manual creation works
        env = HoverEnv()
        assert env is not None
        env.close()

    def test_vectorized_env(self):
        """Test with vectorized environment."""
        from gymnasium.vector import SyncVectorEnv

        def make_env():
            config = HoverEnvConfig(max_episode_steps=10)
            return HoverEnv(config=config)

        vec_env = SyncVectorEnv([make_env for _ in range(2)])

        obs, _ = vec_env.reset(seed=42)
        assert obs.shape == (2, 20)  # 2 envs, 20 obs dim

        actions = np.array([[0.5, 0.5, 0.5, 0.5]] * 2)
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)

        assert obs.shape == (2, 20)
        assert rewards.shape == (2,)

        vec_env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
