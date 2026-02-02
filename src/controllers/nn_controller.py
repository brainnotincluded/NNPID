"""Neural network controller for learned policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.mujoco_sim import QuadrotorState
from ..utils.logger import get_logger
from .base_controller import BaseController

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from stable_baselines3 import PPO, SAC, TD3

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    SAC = None
    TD3 = None


logger = get_logger(__name__)


class NNController(BaseController):
    """Neural network controller for learned policies.

    Supports loading policies from:
    - Stable-Baselines3 checkpoints
    - Raw PyTorch models
    - ONNX models (for deployment)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_type: str = "sb3",  # "sb3", "torch", "onnx"
        device: str = "cpu",
    ):
        """Initialize neural network controller.

        Args:
            model_path: Path to saved model checkpoint
            model_type: Type of model ("sb3", "torch", "onnx")
            device: Device to run inference on
        """
        super().__init__(name="NNController")

        self.model_path = Path(model_path) if model_path else None
        self.model_type = model_type
        self.device = device

        self._model = None
        self._policy = None

        # Observation normalization stats (from training)
        self._obs_mean: np.ndarray | None = None
        self._obs_std: np.ndarray | None = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str | Path) -> bool:
        """Load model from checkpoint.

        Args:
            model_path: Path to model file

        Returns:
            True if loaded successfully
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error("Model not found: %s", model_path)
            return False

        try:
            if self.model_type == "sb3":
                return self._load_sb3_model(model_path)
            elif self.model_type == "torch":
                return self._load_torch_model(model_path)
            elif self.model_type == "onnx":
                return self._load_onnx_model(model_path)
            else:
                logger.error("Unknown model type: %s", self.model_type)
                return False
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            return False

    def _load_sb3_model(self, model_path: Path) -> bool:
        """Load Stable-Baselines3 model."""
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for SB3 models")

        # Try different algorithms
        for algo_class in [PPO, SAC, TD3]:
            try:
                self._model = algo_class.load(str(model_path), device=self.device)
                self._policy = self._model.policy
                self._is_initialized = True
                logger.info("Loaded %s model from %s", algo_class.__name__, model_path)
                return True
            except Exception:
                continue

        logger.error("Could not load SB3 model from %s", model_path)
        return False

    def _load_torch_model(self, model_path: Path) -> bool:
        """Load raw PyTorch model."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for torch models")

        # Use weights_only=True for security (prevents arbitrary code execution)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Need to know architecture to load state dict
                logger.error("State dict loading requires model architecture")
                return False
            elif "policy" in checkpoint:
                self._policy = checkpoint["policy"]
            elif "model" in checkpoint:
                self._model = checkpoint["model"]
        else:
            # Assume it's a full model
            self._model = checkpoint

        if self._model is not None:
            self._model.eval()
        if self._policy is not None:
            self._policy.eval()

        self._is_initialized = True
        return True

    def _load_onnx_model(self, model_path: Path) -> bool:
        """Load ONNX model for deployment."""
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError("onnxruntime is required for ONNX models") from err

        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")

        self._model = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )

        self._is_initialized = True
        return True

    def compute_action(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Compute action using neural network policy.

        Args:
            state: Current quadrotor state
            target_position: Desired position
            dt: Time step

        Returns:
            Motor commands [0, 1]
        """
        if not self._is_initialized:
            # Return hover command if not initialized
            return np.array([0.5, 0.5, 0.5, 0.5])

        # Build observation vector
        obs = self._build_observation(state, target_position)

        # Normalize observation if stats available
        if self._obs_mean is not None and self._obs_std is not None:
            obs = (obs - self._obs_mean) / (self._obs_std + 1e-8)

        # Get action from policy
        action = self._predict(obs)

        # Ensure action is in [0, 1]
        action = np.clip(action, 0.0, 1.0)

        return action.astype(np.float32)

    def _build_observation(
        self,
        state: QuadrotorState,
        target_position: np.ndarray,
    ) -> np.ndarray:
        """Build observation vector matching training environment.

        Args:
            state: Current state
            target_position: Target position

        Returns:
            Observation array
        """
        # Match BaseDroneEnv observation format
        obs = np.concatenate(
            [
                state.position,
                state.velocity,
                state.quaternion,
                state.angular_velocity,
                target_position,
                state.motor_speeds,
            ]
        )

        return obs.astype(np.float32)

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        """Get action from model.

        Args:
            obs: Observation array

        Returns:
            Action array
        """
        if self.model_type == "sb3" and self._model is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            return action

        elif self.model_type == "torch":
            if not TORCH_AVAILABLE:
                return np.zeros(4)

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                if self._policy is not None:
                    action = self._policy(obs_tensor)
                elif self._model is not None:
                    action = self._model(obs_tensor)
                else:
                    return np.zeros(4)

                return action.cpu().numpy().squeeze()

        elif self.model_type == "onnx" and self._model is not None:
            input_name = self._model.get_inputs()[0].name
            output_name = self._model.get_outputs()[0].name

            result = self._model.run(
                [output_name],
                {input_name: obs.reshape(1, -1).astype(np.float32)},
            )

            return result[0].squeeze()

        return np.zeros(4)

    def set_normalization_stats(
        self,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ) -> None:
        """Set observation normalization statistics.

        Args:
            obs_mean: Mean of observations from training
            obs_std: Std of observations from training
        """
        self._obs_mean = obs_mean
        self._obs_std = obs_std

    def export_onnx(
        self,
        output_path: str | Path,
        obs_dim: int = 20,
    ) -> bool:
        """Export model to ONNX format for deployment.

        Args:
            output_path: Path to save ONNX model
            obs_dim: Observation dimension

        Returns:
            True if export successful
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch required for ONNX export")
            return False

        if self._policy is None and self._model is None:
            logger.error("No model to export")
            return False

        model = self._policy if self._policy is not None else self._model

        try:
            dummy_input = torch.randn(1, obs_dim)

            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["observation"],
                output_names=["action"],
                dynamic_axes={
                    "observation": {0: "batch_size"},
                    "action": {0: "batch_size"},
                },
            )

            logger.info("Exported model to %s", output_path)
            return True

        except Exception as e:
            logger.error("ONNX export failed: %s", e)
            return False

    def get_info(self) -> dict[str, Any]:
        """Get controller info."""
        info = super().get_info()
        info.update(
            {
                "model_type": self.model_type,
                "model_path": str(self.model_path) if self.model_path else None,
                "device": self.device,
                "has_normalization": self._obs_mean is not None,
            }
        )
        return info


class SimpleMLP(nn.Module):
    """Simple MLP policy network."""

    def __init__(
        self,
        obs_dim: int = 20,
        action_dim: int = 4,
        hidden_dims: list = None,
        activation: str = "tanh",
    ):
        """Initialize MLP.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SimpleMLP")

        super().__init__()

        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
        }.get(activation, nn.Tanh)

        layers = []
        in_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(obs)
