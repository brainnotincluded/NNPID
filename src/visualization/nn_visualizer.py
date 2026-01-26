"""Neural network visualization for real-time activation display.

This module provides visualization of neural network structure and activations,
drawing the network diagram with live activation values on rendered frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class NNVisualizerConfig:
    """Configuration for neural network visualizer."""

    # Position and size
    x: int = 10
    y: int = 200
    width: int = 300
    height: int = 250
    padding: int = 15

    # Node appearance
    node_radius: int = 12
    node_spacing_x: int = 80
    node_spacing_y: int = 25
    max_nodes_per_layer: int = 8

    # Colors (BGR for OpenCV)
    background_color: tuple = (30, 30, 30)
    border_color: tuple = (80, 80, 80)
    text_color: tuple = (200, 200, 200)
    label_color: tuple = (150, 150, 150)

    # Activation colors (gradient from negative to positive)
    activation_negative: tuple = (255, 100, 100)  # Blue-ish
    activation_zero: tuple = (80, 80, 80)  # Gray
    activation_positive: tuple = (100, 255, 100)  # Green

    # Connection appearance
    connection_alpha: float = 0.3
    connection_max_width: int = 3

    # Labels
    show_layer_labels: bool = True
    show_activation_values: bool = False
    title: str = "Neural Network"


class NetworkExtractor:
    """Extracts network architecture from stable-baselines3 models."""

    @staticmethod
    def extract_from_sb3(model) -> dict[str, Any]:
        """Extract network info from SB3 model.

        Args:
            model: Stable-baselines3 model (PPO, SAC, etc.)

        Returns:
            Dictionary with network architecture info
        """
        if not TORCH_AVAILABLE:
            return {"layers": [], "weights": []}

        try:
            # Get policy network
            policy = model.policy

            # Extract MLP layers
            if hasattr(policy, "mlp_extractor"):
                # PPO/A2C style
                extractor = policy.mlp_extractor
                if hasattr(extractor, "policy_net"):
                    layers = list(extractor.policy_net.children())
                elif hasattr(extractor, "shared_net"):
                    layers = list(extractor.shared_net.children())
                else:
                    layers = []
            elif hasattr(policy, "actor"):
                # SAC style
                layers = list(policy.actor.latent_pi.children())
            else:
                layers = []

            # Parse layer info
            layer_info = []
            weights = []

            for layer in layers:
                if isinstance(layer, torch.nn.Linear):
                    layer_info.append(
                        {
                            "type": "Linear",
                            "in_features": layer.in_features,
                            "out_features": layer.out_features,
                        }
                    )
                    weights.append(
                        {
                            "weight": layer.weight.detach().cpu().numpy(),
                            "bias": layer.bias.detach().cpu().numpy()
                            if layer.bias is not None
                            else None,
                        }
                    )
                elif isinstance(layer, (torch.nn.ReLU, torch.nn.Tanh)):
                    layer_info.append({"type": layer.__class__.__name__})

            return {"layers": layer_info, "weights": weights}

        except Exception as e:
            print(f"Error extracting network: {e}")
            return {"layers": [], "weights": []}

    @staticmethod
    def get_layer_sizes(model) -> list[int]:
        """Get sizes of each layer in the network.

        Args:
            model: SB3 model

        Returns:
            List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        info = NetworkExtractor.extract_from_sb3(model)
        sizes = []

        for layer in info["layers"]:
            if layer["type"] == "Linear":
                if not sizes:
                    sizes.append(layer["in_features"])
                sizes.append(layer["out_features"])

        # Add output layer (action dimension)
        if hasattr(model, "action_space"):
            action_dim = model.action_space.shape[0] if hasattr(model.action_space, "shape") else 1
            if sizes and sizes[-1] != action_dim:
                sizes.append(action_dim)

        return sizes if sizes else [11, 64, 64, 1]  # Default architecture


class ActivationTracker:
    """Tracks neural network activations during forward pass."""

    def __init__(self):
        """Initialize activation tracker."""
        self._activations: list[np.ndarray] = []
        self._hooks = []

    def register_hooks(self, model) -> None:
        """Register forward hooks on model layers.

        Args:
            model: SB3 model to track
        """
        if not TORCH_AVAILABLE:
            return

        self.clear_hooks()

        try:
            policy = model.policy

            # Find the MLP layers
            if hasattr(policy, "mlp_extractor"):
                extractor = policy.mlp_extractor
                if hasattr(extractor, "policy_net"):
                    target = extractor.policy_net
                elif hasattr(extractor, "shared_net"):
                    target = extractor.shared_net
                else:
                    return
            elif hasattr(policy, "actor"):
                target = policy.actor.latent_pi
            else:
                return

            # Register hooks on Linear layers
            for layer in target.children():
                if isinstance(layer, torch.nn.Linear):
                    hook = layer.register_forward_hook(self._hook_fn)
                    self._hooks.append(hook)

        except Exception as e:
            print(f"Error registering hooks: {e}")

    def _hook_fn(self, module, input_tensor, output):
        """Hook function to capture activations."""
        if TORCH_AVAILABLE:
            self._activations.append(output.detach().cpu().numpy())

    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear_activations(self) -> None:
        """Clear stored activations."""
        self._activations.clear()

    def get_activations(self) -> list[np.ndarray]:
        """Get stored activations.

        Returns:
            List of activation arrays
        """
        return self._activations.copy()

    def update_from_observation(self, model, observation: np.ndarray) -> list[np.ndarray]:
        """Run forward pass and capture activations.

        Args:
            model: SB3 model
            observation: Input observation

        Returns:
            List of activation arrays per layer
        """
        self.clear_activations()

        if not TORCH_AVAILABLE:
            return []

        try:
            # Convert observation to tensor
            torch.FloatTensor(observation).unsqueeze(0)

            # Run prediction (triggers hooks)
            with torch.no_grad():
                model.predict(observation, deterministic=True)

            return self.get_activations()

        except Exception:
            return []


class NNVisualizer:
    """Visualizes neural network structure and activations on frames."""

    def __init__(self, config: NNVisualizerConfig | None = None):
        """Initialize neural network visualizer.

        Args:
            config: Visualization configuration
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV required for NN visualization")

        self.config = config or NNVisualizerConfig()
        self._layer_sizes: list[int] = []
        self._activations: list[np.ndarray] = []
        self._activation_tracker = ActivationTracker()
        self._model = None
        self._input_values: np.ndarray | None = None

    def set_model(self, model) -> None:
        """Set the model to visualize.

        Args:
            model: SB3 model (PPO, SAC, etc.)
        """
        self._model = model
        self._layer_sizes = NetworkExtractor.get_layer_sizes(model)
        self._activation_tracker.register_hooks(model)

    def set_layer_sizes(self, sizes: list[int]) -> None:
        """Manually set layer sizes.

        Args:
            sizes: List of layer sizes [input, hidden1, ..., output]
        """
        self._layer_sizes = sizes

    def update(self, observation: np.ndarray, action: np.ndarray | None = None) -> None:
        """Update with new observation and optional action.

        Args:
            observation: Current observation
            action: Optional action output
        """
        self._input_values = observation.copy()

        if self._model is not None:
            self._activations = self._activation_tracker.update_from_observation(
                self._model, observation
            )
        else:
            # Generate fake activations for visualization without model
            self._activations = []
            for size in self._layer_sizes[1:]:
                self._activations.append(np.random.randn(1, size) * 0.5)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render neural network visualization on frame.

        Args:
            frame: Input frame to draw on

        Returns:
            Frame with NN visualization
        """
        if not self._layer_sizes:
            return frame

        cfg = self.config
        result = frame.copy()

        # Draw background panel
        panel_x = cfg.x
        panel_y = cfg.y
        panel_w = cfg.width
        panel_h = cfg.height

        # Semi-transparent background
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            cfg.background_color,
            -1,
        )
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            cfg.border_color,
            1,
        )
        result = cv2.addWeighted(overlay, 0.85, result, 0.15, 0)

        # Title
        cv2.putText(
            result,
            cfg.title,
            (panel_x + cfg.padding, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            cfg.text_color,
            1,
        )

        # Calculate node positions
        num_layers = len(self._layer_sizes)
        layer_x_start = panel_x + cfg.padding + cfg.node_radius
        layer_x_spacing = (panel_w - 2 * cfg.padding - 2 * cfg.node_radius) // max(
            1, num_layers - 1
        )

        # Collect all node positions
        node_positions: list[list[tuple[int, int]]] = []

        for layer_idx, layer_size in enumerate(self._layer_sizes):
            layer_x = layer_x_start + layer_idx * layer_x_spacing

            # Limit displayed nodes
            display_size = min(layer_size, cfg.max_nodes_per_layer)
            layer_height = display_size * cfg.node_spacing_y

            layer_y_start = panel_y + 40 + (panel_h - 60 - layer_height) // 2

            positions = []
            for node_idx in range(display_size):
                node_y = layer_y_start + node_idx * cfg.node_spacing_y
                positions.append((layer_x, node_y))

            node_positions.append(positions)

        # Draw connections (before nodes)
        for layer_idx in range(len(node_positions) - 1):
            from_positions = node_positions[layer_idx]
            to_positions = node_positions[layer_idx + 1]

            for from_pos in from_positions:
                for to_pos in to_positions:
                    # Thin gray lines
                    cv2.line(
                        result,
                        from_pos,
                        to_pos,
                        (60, 60, 60),
                        1,
                        cv2.LINE_AA,
                    )

        # Draw nodes
        for layer_idx, positions in enumerate(node_positions):
            # Get activations for this layer
            if layer_idx == 0:
                # Input layer - use observation values
                if self._input_values is not None:
                    layer_activations = self._input_values[: len(positions)]
                else:
                    layer_activations = np.zeros(len(positions))
            elif layer_idx - 1 < len(self._activations):
                layer_activations = self._activations[layer_idx - 1].flatten()
            else:
                layer_activations = np.zeros(len(positions))

            for node_idx, pos in enumerate(positions):
                # Get activation value
                if node_idx < len(layer_activations):
                    activation = layer_activations[node_idx]
                else:
                    activation = 0.0

                # Color based on activation
                color = self._activation_to_color(activation)

                # Draw node
                cv2.circle(result, pos, cfg.node_radius, color, -1, cv2.LINE_AA)
                cv2.circle(result, pos, cfg.node_radius, (100, 100, 100), 1, cv2.LINE_AA)

                # Show activation value
                if cfg.show_activation_values and abs(activation) > 0.01:
                    text = f"{activation:.1f}"
                    cv2.putText(
                        result,
                        text,
                        (pos[0] - 10, pos[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )

        # Layer labels
        if cfg.show_layer_labels:
            labels = ["In"] + [f"H{i + 1}" for i in range(len(self._layer_sizes) - 2)] + ["Out"]
            for layer_idx, positions in enumerate(node_positions):
                if positions:
                    label_x = positions[0][0] - 8
                    label_y = panel_y + panel_h - 5
                    cv2.putText(
                        result,
                        labels[layer_idx] if layer_idx < len(labels) else "",
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        cfg.label_color,
                        1,
                    )

        # Show layer sizes
        size_text = " -> ".join(str(s) for s in self._layer_sizes)
        cv2.putText(
            result,
            size_text,
            (panel_x + cfg.padding, panel_y + panel_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            cfg.label_color,
            1,
        )

        return result

    def _activation_to_color(self, activation: float) -> tuple:
        """Convert activation value to color.

        Args:
            activation: Activation value

        Returns:
            BGR color tuple
        """
        cfg = self.config

        # Clamp to reasonable range
        activation = np.clip(activation, -3.0, 3.0)

        if activation < 0:
            t = min(1.0, abs(activation) / 3.0)
            color = tuple(
                int(c1 * (1 - t) + c2 * t)
                for c1, c2 in zip(cfg.activation_zero, cfg.activation_negative, strict=True)
            )
        else:
            t = min(1.0, activation / 3.0)
            color = tuple(
                int(c1 * (1 - t) + c2 * t)
                for c1, c2 in zip(cfg.activation_zero, cfg.activation_positive, strict=True)
            )

        return color


def create_default_nn_visualizer() -> NNVisualizer:
    """Create NN visualizer with default settings."""
    return NNVisualizer()


def create_compact_nn_visualizer() -> NNVisualizer:
    """Create compact NN visualizer."""
    config = NNVisualizerConfig(
        width=200,
        height=180,
        node_radius=8,
        node_spacing_y=18,
        max_nodes_per_layer=6,
    )
    return NNVisualizer(config)
