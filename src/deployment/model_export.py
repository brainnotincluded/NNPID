"""Export trained models to ONNX for deployment.

ONNX format enables running models on:
- Edge devices (Jetson, Raspberry Pi)
- Microcontrollers (with TensorRT/TFLite)
- Any platform with ONNX runtime
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import onnx
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None


@dataclass
class ExportConfig:
    """Configuration for model export."""

    # Input/output dimensions
    observation_dim: int = 19  # Setpoint env observation
    action_dim: int = 4  # Velocity setpoint

    # ONNX settings
    opset_version: int = 14
    dynamic_axes: bool = False  # True for variable batch size

    # Optimization
    optimize: bool = True
    quantize: bool = False  # INT8 quantization

    # Metadata
    model_name: str = "drone_controller"
    description: str = "Neural network drone controller"


class ModelExporter:
    """Export trained models to ONNX format.

    Supports:
    - Stable-Baselines3 policies
    - Raw PyTorch models
    - Custom policy wrappers
    """

    def __init__(self, config: ExportConfig | None = None):
        """Initialize exporter.

        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model export")

    def export_sb3_policy(
        self,
        model_path: str | Path,
        output_path: str | Path,
        algorithm: str = "PPO",
    ) -> Path:
        """Export a Stable-Baselines3 policy to ONNX.

        Args:
            model_path: Path to saved SB3 model (.zip)
            output_path: Path for ONNX output
            algorithm: SB3 algorithm name (PPO, SAC, TD3)

        Returns:
            Path to exported ONNX file
        """
        try:
            from stable_baselines3 import PPO, SAC, TD3
        except ImportError as err:
            raise ImportError("stable-baselines3 is required") from err

        # Load model
        algo_map = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
        if algorithm not in algo_map:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        model = algo_map[algorithm].load(model_path)

        # Extract policy network
        policy = model.policy

        # Create wrapper for consistent interface
        wrapper = SB3PolicyWrapper(policy, self.config.observation_dim)

        return self._export_pytorch_model(wrapper, output_path)

    def export_pytorch_model(
        self,
        model: nn.Module,
        output_path: str | Path,
    ) -> Path:
        """Export a PyTorch model to ONNX.

        Args:
            model: PyTorch model
            output_path: Path for ONNX output

        Returns:
            Path to exported ONNX file
        """
        return self._export_pytorch_model(model, output_path)

    def _export_pytorch_model(
        self,
        model: nn.Module,
        output_path: str | Path,
    ) -> Path:
        """Internal method to export PyTorch model.

        Args:
            model: PyTorch model
            output_path: Path for ONNX output

        Returns:
            Path to exported ONNX file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(1, self.config.observation_dim)

        # Dynamic axes for batch size
        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
            }

        # Export
        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes=dynamic_axes,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
        )

        print(f"Exported model to: {output_path}")

        # Verify
        if ONNX_AVAILABLE:
            self._verify_onnx(output_path)

        # Optimize
        if self.config.optimize and ONNX_AVAILABLE:
            self._optimize_onnx(output_path)

        return output_path

    def _verify_onnx(self, model_path: Path) -> bool:
        """Verify exported ONNX model.

        Args:
            model_path: Path to ONNX model

        Returns:
            True if valid
        """
        try:
            model = onnx.load(str(model_path))
            onnx.checker.check_model(model)
            print("ONNX model verification passed")
            return True
        except Exception as e:
            print(f"ONNX verification failed: {e}")
            return False

    def _optimize_onnx(self, model_path: Path) -> None:
        """Optimize ONNX model.

        Args:
            model_path: Path to ONNX model
        """
        try:
            from onnxruntime.transformers import optimizer

            optimized_path = model_path.with_suffix(".optimized.onnx")
            optimizer.optimize_model(
                str(model_path),
                str(optimized_path),
            )
            print(f"Optimized model saved to: {optimized_path}")
        except ImportError:
            print("onnxruntime-tools not available, skipping optimization")
        except Exception as e:
            print(f"Optimization failed: {e}")


class SB3PolicyWrapper(nn.Module):
    """Wrapper to extract action from SB3 policy."""

    def __init__(self, policy, obs_dim: int):
        """Initialize wrapper.

        Args:
            policy: SB3 policy
            obs_dim: Observation dimension
        """
        super().__init__()
        self.policy = policy
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from observation.

        Args:
            obs: Observation tensor

        Returns:
            Action tensor
        """
        # Get deterministic action
        with torch.no_grad():
            action, _, _ = self.policy(obs, deterministic=True)
        return action


class ONNXInference:
    """Run inference with ONNX model.

    For deployment on edge devices or real-time control.
    """

    def __init__(self, model_path: str | Path):
        """Initialize ONNX inference.

        Args:
            model_path: Path to ONNX model
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required for inference")

        self.model_path = Path(model_path)

        # Create session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Get shapes
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            observation: Observation array

        Returns:
            Action array
        """
        # Ensure correct shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        observation = observation.astype(np.float32)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: observation},
        )

        return outputs[0].squeeze()

    def benchmark(self, n_runs: int = 1000) -> dict[str, float]:
        """Benchmark inference speed.

        Args:
            n_runs: Number of runs for benchmark

        Returns:
            Dictionary with timing statistics
        """
        import time

        # Create dummy input
        dummy_input = np.random.randn(1, self.input_shape[1]).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(dummy_input)

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(dummy_input)
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # Convert to ms

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "hz": float(1000 / np.mean(times)),
        }


def export_to_onnx(
    model_path: str | Path,
    output_path: str | Path,
    algorithm: str = "PPO",
    observation_dim: int = 19,
    action_dim: int = 4,
) -> Path:
    """Convenience function to export SB3 model to ONNX.

    Args:
        model_path: Path to saved SB3 model
        output_path: Path for ONNX output
        algorithm: SB3 algorithm name
        observation_dim: Observation dimension
        action_dim: Action dimension

    Returns:
        Path to exported ONNX file
    """
    config = ExportConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
    )
    exporter = ModelExporter(config)
    return exporter.export_sb3_policy(model_path, output_path, algorithm)
