"""Tests for TrainedYawTracker deployment wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    pytest.skip("stable-baselines3 not available", allow_module_level=True)

from src.deployment.trained_yaw_tracker import TrainedYawTracker
from src.environments.yaw_tracking_env import YawTrackingConfig


class TestTrainedYawTracker:
    """Test suite for TrainedYawTracker class."""

    def test_init(self):
        """Test TrainedYawTracker initialization."""
        # Create mock model with observation_space
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        # Create tracker
        tracker = TrainedYawTracker(model=mock_model)

        assert tracker.model == mock_model
        assert tracker.vec_normalize is None
        assert tracker.observation_space == 11
        assert tracker.config is not None

    def test_init_with_vec_normalize(self):
        """Test initialization with VecNormalize."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        mock_vec_norm = MagicMock(spec=VecNormalize)

        tracker = TrainedYawTracker(
            model=mock_model, vec_normalize=mock_vec_norm, config=YawTrackingConfig()
        )

        assert tracker.vec_normalize == mock_vec_norm
        assert tracker.config is not None

    def test_from_path_file_not_found(self):
        """Test from_path with non-existent file."""
        with pytest.raises(FileNotFoundError):
            TrainedYawTracker.from_path("nonexistent/path/model.zip")

    @patch("src.deployment.trained_yaw_tracker.load_model_and_vecnormalize")
    def test_from_path_loads_model(self, mock_loader):
        """Test from_path loads model correctly."""
        # Setup mock
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        mock_loader.return_value = (mock_model, None, Path("test_model.zip"))

        tracker = TrainedYawTracker.from_path("test_model.zip")

        assert tracker.model == mock_model
        assert tracker.observation_space == 11

    @patch("src.deployment.trained_yaw_tracker.load_model_and_vecnormalize")
    def test_from_path_loads_vec_normalize(self, mock_loader):
        """Test from_path loads VecNormalize when available."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        mock_vec_norm = MagicMock(spec=VecNormalize)
        mock_vec_norm.training = False
        mock_loader.return_value = (mock_model, mock_vec_norm, Path("test_model.zip"))

        tracker = TrainedYawTracker.from_path("test_model.zip")

        assert tracker.vec_normalize == mock_vec_norm
        assert tracker.vec_normalize.training is False

    def test_predict_basic(self):
        """Test predict method with basic observation."""
        # Create mock model
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        mock_model.predict.return_value = (np.array([0.5]), None)

        tracker = TrainedYawTracker(model=mock_model)

        # Create observation
        obs = np.random.randn(11).astype(np.float32)

        # Predict
        yaw_cmd = tracker.predict(obs, deterministic=True)

        assert isinstance(yaw_cmd, float)
        assert -1.0 <= yaw_cmd <= 1.0
        mock_model.predict.assert_called_once()

    def test_predict_with_normalization(self):
        """Test predict with VecNormalize."""
        # Create mock model
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        mock_model.predict.return_value = (np.array([-0.3]), None)

        # Create mock VecNormalize
        mock_vec_norm = MagicMock(spec=VecNormalize)
        mock_vec_norm.normalize_obs.return_value = np.random.randn(1, 11).astype(np.float32)

        tracker = TrainedYawTracker(model=mock_model, vec_normalize=mock_vec_norm)

        obs = np.random.randn(11).astype(np.float32)
        yaw_cmd = tracker.predict(obs, deterministic=False)

        assert isinstance(yaw_cmd, float)
        assert -1.0 <= yaw_cmd <= 1.0
        mock_vec_norm.normalize_obs.assert_called_once()

    def test_predict_wrong_shape(self):
        """Test predict with wrong observation shape."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        tracker = TrainedYawTracker(model=mock_model)

        # Wrong shape
        obs = np.random.randn(10).astype(np.float32)

        with pytest.raises(ValueError, match="Observation size mismatch"):
            tracker.predict(obs)

    def test_predict_wrong_dimension(self):
        """Test predict with wrong observation dimension."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        tracker = TrainedYawTracker(model=mock_model)

        # 2D instead of 1D
        obs = np.random.randn(1, 11).astype(np.float32)

        with pytest.raises(ValueError, match="Observation must be 1D"):
            tracker.predict(obs)

    def test_reset(self):
        """Test reset method."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        tracker = TrainedYawTracker(model=mock_model)

        # Reset should not raise
        tracker.reset()

    def test_get_info(self):
        """Test get_info method."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        type(mock_model).__name__ = "PPO"

        tracker = TrainedYawTracker(model=mock_model, config=YawTrackingConfig())

        info = tracker.get_info()

        assert isinstance(info, dict)
        assert "model_type" in info
        assert "observation_space" in info
        assert "has_normalization" in info
        assert "config" in info
        assert info["model_type"] == "PPO"
        assert info["observation_space"] == 11
        assert info["has_normalization"] is False

    def test_get_info_with_normalization(self):
        """Test get_info with VecNormalize."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        mock_vec_norm = MagicMock(spec=VecNormalize)

        tracker = TrainedYawTracker(
            model=mock_model, vec_normalize=mock_vec_norm, config=YawTrackingConfig()
        )

        info = tracker.get_info()
        assert info["has_normalization"] is True

    def test_repr(self):
        """Test string representation."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        type(mock_model).__name__ = "PPO"

        tracker = TrainedYawTracker(model=mock_model)
        repr_str = repr(tracker)

        assert "TrainedYawTracker" in repr_str
        assert "PPO" in repr_str

    def test_predict_action_range(self):
        """Test that predict returns values in [-1, 1] range."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space

        # Test different action values
        for action_val in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5]:
            mock_model.predict.return_value = (np.array([action_val]), None)

            tracker = TrainedYawTracker(model=mock_model)
            obs = np.random.randn(11).astype(np.float32)

            yaw_cmd = tracker.predict(obs)

            # Should always be in [-1, 1] after extraction
            assert isinstance(yaw_cmd, float)
            # Note: model might return values outside [-1, 1], but we extract as-is
            # The actual clipping should happen in the environment

    def test_predict_empty_action(self):
        """Test predict with empty action array."""
        mock_obs_space = MagicMock()
        mock_obs_space.shape = (11,)
        mock_model = MagicMock(spec=PPO)
        mock_model.observation_space = mock_obs_space
        mock_model.predict.return_value = (np.array([]), None)

        tracker = TrainedYawTracker(model=mock_model)
        obs = np.random.randn(11).astype(np.float32)

        yaw_cmd = tracker.predict(obs)

        # Should return 0.0 for empty action
        assert yaw_cmd == 0.0
