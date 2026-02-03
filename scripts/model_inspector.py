#!/usr/bin/env python3
"""CLI tool for inspecting and visualizing neural network model weights.

This script provides various commands for analyzing trained stable-baselines3
models without running the simulation.

Usage:
    python scripts/model_inspector.py arch runs/model.zip
    python scripts/model_inspector.py weights runs/model.zip --layer 0 --heatmap
    python scripts/model_inspector.py activations runs/model.zip --episodes 5
    python scripts/model_inspector.py stats runs/model.zip --output stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.deployment.model_loading import load_sb3_model, resolve_model_path

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_model(model_path: str):
    """Load a stable-baselines3 model.

    Args:
        model_path: Path to model zip file

    Returns:
        Loaded model
    """
    try:
        resolved = resolve_model_path(model_path)
        model = load_sb3_model(resolved)
        print(f"Loaded model from {resolved}")
        return model
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)


def extract_layers(model) -> list[dict[str, Any]]:
    """Extract layer information from model.

    Args:
        model: SB3 model

    Returns:
        List of layer dictionaries
    """
    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available")
        return []

    layers = []
    policy = model.policy

    # Find MLP extractor
    if hasattr(policy, "mlp_extractor"):
        extractor = policy.mlp_extractor
        if hasattr(extractor, "policy_net"):
            net = extractor.policy_net
        elif hasattr(extractor, "shared_net"):
            net = extractor.shared_net
        else:
            net = None
    elif hasattr(policy, "actor"):
        net = policy.actor.latent_pi
    else:
        net = None

    if net is None:
        return layers

    for i, layer in enumerate(net.children()):
        layer_info = {
            "index": i,
            "type": layer.__class__.__name__,
        }

        if isinstance(layer, torch.nn.Linear):
            layer_info["in_features"] = layer.in_features
            layer_info["out_features"] = layer.out_features
            layer_info["has_bias"] = layer.bias is not None

            # Weight statistics
            weight = layer.weight.detach().cpu().numpy()
            layer_info["weight_shape"] = list(weight.shape)
            layer_info["weight_mean"] = float(np.mean(weight))
            layer_info["weight_std"] = float(np.std(weight))
            layer_info["weight_min"] = float(np.min(weight))
            layer_info["weight_max"] = float(np.max(weight))
            layer_info["weight_sparsity"] = float(np.mean(np.abs(weight) < 0.01))

            if layer.bias is not None:
                bias = layer.bias.detach().cpu().numpy()
                layer_info["bias_mean"] = float(np.mean(bias))
                layer_info["bias_std"] = float(np.std(bias))

        layers.append(layer_info)

    return layers


def cmd_arch(args):
    """Show model architecture."""
    model = load_model(args.model)
    layers = extract_layers(model)

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    # Model type
    print(f"\nModel Type: {model.__class__.__name__}")

    # Observation and action space
    if hasattr(model, "observation_space"):
        obs_shape = model.observation_space.shape
        print(f"Observation Space: {obs_shape}")

    if hasattr(model, "action_space"):
        act_shape = model.action_space.shape
        print(f"Action Space: {act_shape}")

    # Network architecture
    print("\n" + "-" * 40)
    print("NETWORK LAYERS")
    print("-" * 40)

    total_params = 0
    sizes = []

    for layer in layers:
        if layer["type"] == "Linear":
            in_f = layer["in_features"]
            out_f = layer["out_features"]
            params = in_f * out_f + (out_f if layer["has_bias"] else 0)
            total_params += params

            if not sizes:
                sizes.append(in_f)
            sizes.append(out_f)

            print(f"\n  Layer {layer['index']}: Linear")
            print(f"    Shape: {in_f} -> {out_f}")
            print(f"    Parameters: {params:,}")
            print(
                f"    Weight stats: mean={layer['weight_mean']:.4f}, std={layer['weight_std']:.4f}"
            )
            print(f"    Weight range: [{layer['weight_min']:.4f}, {layer['weight_max']:.4f}]")
            print(f"    Sparsity: {layer['weight_sparsity'] * 100:.1f}%")
        else:
            print(f"\n  Layer {layer['index']}: {layer['type']}")

    print("\n" + "-" * 40)
    print(f"Architecture: {' -> '.join(map(str, sizes))}")
    print(f"Total Parameters: {total_params:,}")
    print("=" * 60)

    # ASCII diagram
    if args.diagram:
        print("\n")
        print_ascii_network(sizes)


def print_ascii_network(sizes: list[int]):
    """Print ASCII diagram of network architecture."""
    max_size = max(sizes)
    height = min(10, max_size)
    scale = height / max_size

    print("Network Diagram:")
    print()

    # Draw layers
    layer_width = 8

    for row in range(height, 0, -1):
        line = ""
        for size in sizes:
            scaled_size = int(size * scale)
            if row <= scaled_size:
                line += "  ████  "
            else:
                line += "        "
        print(line)

    # Labels
    labels = [str(s).center(layer_width) for s in sizes]
    print("".join(labels))

    layer_labels = ["Input"] + [f"H{i + 1}" for i in range(len(sizes) - 2)] + ["Output"]
    layer_labels = [lbl.center(layer_width) for lbl in layer_labels]
    print("".join(layer_labels))


def cmd_weights(args):
    """Visualize model weights."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required for weight visualization")
        sys.exit(1)

    model = load_model(args.model)

    if not TORCH_AVAILABLE:
        print("Error: PyTorch not available")
        sys.exit(1)

    policy = model.policy

    # Get network
    if hasattr(policy, "mlp_extractor"):
        extractor = policy.mlp_extractor
        net = extractor.policy_net if hasattr(extractor, "policy_net") else extractor.shared_net
    elif hasattr(policy, "actor"):
        net = policy.actor.latent_pi
    else:
        print("Error: Could not find network in model")
        sys.exit(1)

    # Get linear layers
    linear_layers = [layer for layer in net.children() if isinstance(layer, torch.nn.Linear)]

    if args.layer is not None:
        if args.layer >= len(linear_layers):
            print(f"Error: Layer {args.layer} not found (max: {len(linear_layers) - 1})")
            sys.exit(1)
        layers_to_show = [(args.layer, linear_layers[args.layer])]
    else:
        layers_to_show = list(enumerate(linear_layers))

    # Create figure
    n_layers = len(layers_to_show)
    fig_height = 4 * n_layers if args.heatmap else 3 * n_layers

    if args.heatmap:
        fig, axes = plt.subplots(n_layers, 1, figsize=(12, fig_height))
        if n_layers == 1:
            axes = [axes]

        # Custom colormap
        cmap = LinearSegmentedColormap.from_list(
            "weight_cmap", [(0, "blue"), (0.5, "white"), (1, "red")]
        )

        for ax, (idx, layer) in zip(axes, layers_to_show, strict=True):
            weights = layer.weight.detach().cpu().numpy()

            # Normalize for visualization
            vmax = max(abs(weights.min()), abs(weights.max()))

            im = ax.imshow(
                weights,
                cmap=cmap,
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(f"Layer {idx}: {layer.in_features} -> {layer.out_features}")
            ax.set_xlabel("Input Features")
            ax.set_ylabel("Output Features")
            plt.colorbar(im, ax=ax)

    else:
        # Histogram mode
        fig, axes = plt.subplots(n_layers, 2, figsize=(12, fig_height))
        if n_layers == 1:
            axes = [axes]

        for (ax_w, ax_b), (idx, layer) in zip(axes, layers_to_show, strict=True):
            weights = layer.weight.detach().cpu().numpy().flatten()

            ax_w.hist(weights, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
            ax_w.set_title(f"Layer {idx} Weights ({layer.in_features} -> {layer.out_features})")
            ax_w.set_xlabel("Weight Value")
            ax_w.set_ylabel("Count")
            ax_w.axvline(0, color="red", linestyle="--", alpha=0.5)

            if layer.bias is not None:
                bias = layer.bias.detach().cpu().numpy()
                ax_b.hist(bias, bins=30, color="coral", edgecolor="black", alpha=0.7)
                ax_b.set_title(f"Layer {idx} Biases")
                ax_b.set_xlabel("Bias Value")
                ax_b.axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def cmd_activations(args):
    """Analyze activations on test episodes."""
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required for activation visualization")
        sys.exit(1)

    model = load_model(args.model)

    # Create environment
    try:
        from src.environments import YawTrackingEnv

        env = YawTrackingEnv(render_mode=None)
    except ImportError:
        print("Error: Could not import YawTrackingEnv")
        sys.exit(1)

    print(f"\nRunning {args.episodes} episodes to collect activations...")

    # Collect observations and actions
    all_observations: list = []
    all_actions: list = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            all_observations.append(obs)
            all_actions.append(action)

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        print(f"  Episode {ep + 1}: {len(all_observations)} samples total")

    env.close()

    # Convert to arrays
    observations = np.array(all_observations)
    actions = np.array(all_actions)

    print(f"\nCollected {len(observations)} samples")

    # Analyze observation-action correlations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Observation distribution
    ax = axes[0, 0]
    obs_means = np.mean(observations, axis=0)
    obs_stds = np.std(observations, axis=0)
    x = np.arange(len(obs_means))
    ax.bar(x, obs_means, yerr=obs_stds, capsize=3, color="steelblue", alpha=0.7)
    ax.set_title("Observation Statistics")
    ax.set_xlabel("Observation Dimension")
    ax.set_ylabel("Mean ± Std")

    # 2. Action distribution
    ax = axes[0, 1]
    if len(actions.shape) > 1 and actions.shape[1] > 1:
        for i in range(actions.shape[1]):
            ax.hist(actions[:, i], bins=50, alpha=0.5, label=f"Action {i}")
        ax.legend()
    else:
        ax.hist(actions.flatten(), bins=50, color="coral", alpha=0.7)
    ax.set_title("Action Distribution")
    ax.set_xlabel("Action Value")
    ax.set_ylabel("Count")

    # 3. Observation-Action correlation heatmap
    ax = axes[1, 0]
    combined = np.column_stack([observations, actions])
    corr = np.corrcoef(combined.T)
    n_obs = observations.shape[1]
    obs_act_corr = corr[:n_obs, n_obs:]
    im = ax.imshow(obs_act_corr.T, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_title("Observation-Action Correlation")
    ax.set_xlabel("Observation Dimension")
    ax.set_ylabel("Action Dimension")
    plt.colorbar(im, ax=ax)

    # 4. Action over time (sample episode)
    ax = axes[1, 1]
    sample_len = min(500, len(actions))
    for i in range(min(4, actions.shape[1] if len(actions.shape) > 1 else 1)):
        if len(actions.shape) > 1:
            ax.plot(actions[:sample_len, i], alpha=0.7, label=f"Action {i}")
        else:
            ax.plot(actions[:sample_len], alpha=0.7)
    ax.set_title("Actions Over Time (Sample)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Action Value")
    ax.legend()

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()


def cmd_stats(args):
    """Export model statistics to JSON."""
    model = load_model(args.model)
    layers = extract_layers(model)

    stats = {
        "model_type": model.__class__.__name__,
        "model_path": args.model,
        "observation_space": list(model.observation_space.shape)
        if hasattr(model, "observation_space")
        else None,
        "action_space": list(model.action_space.shape) if hasattr(model, "action_space") else None,
        "layers": layers,
        "total_parameters": sum(
            layer["in_features"] * layer["out_features"] + layer["out_features"]
            for layer in layers
            if layer["type"] == "Linear"
        ),
    }

    # Architecture summary
    sizes = []
    for layer in layers:
        if layer["type"] == "Linear":
            if not sizes:
                sizes.append(layer["in_features"])
            sizes.append(layer["out_features"])
    stats["architecture"] = sizes

    if args.output:
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {args.output}")
    else:
        print(json.dumps(stats, indent=2))


def cmd_compare(args):
    """Compare two models."""
    model1 = load_model(args.model1)
    model2 = load_model(args.model2)

    layers1 = extract_layers(model1)
    layers2 = extract_layers(model2)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"\nModel 1: {args.model1}")
    print(f"Model 2: {args.model2}")

    # Compare architectures
    sizes1 = [layer["out_features"] for layer in layers1 if layer["type"] == "Linear"]
    sizes2 = [layer["out_features"] for layer in layers2 if layer["type"] == "Linear"]

    print(f"\nArchitecture 1: {sizes1}")
    print(f"Architecture 2: {sizes2}")

    if sizes1 != sizes2:
        print("\n⚠ Architectures differ!")
    else:
        print("\n✓ Same architecture")

        # Compare weights
        if TORCH_AVAILABLE:
            print("\nWeight comparison:")
            policy1 = model1.policy
            policy2 = model2.policy

            if hasattr(policy1, "mlp_extractor"):
                net1 = policy1.mlp_extractor.policy_net
                net2 = policy2.mlp_extractor.policy_net
            else:
                net1 = policy1.actor.latent_pi
                net2 = policy2.actor.latent_pi

            linear1 = [lyr for lyr in net1.children() if isinstance(lyr, torch.nn.Linear)]
            linear2 = [lyr for lyr in net2.children() if isinstance(lyr, torch.nn.Linear)]

            for i, (l1, l2) in enumerate(zip(linear1, linear2, strict=True)):
                w1 = l1.weight.detach().cpu().numpy()
                w2 = l2.weight.detach().cpu().numpy()

                diff = np.abs(w1 - w2)
                print(f"\n  Layer {i}:")
                print(f"    Mean absolute difference: {np.mean(diff):.6f}")
                print(f"    Max absolute difference: {np.max(diff):.6f}")
                print(f"    Correlation: {np.corrcoef(w1.flatten(), w2.flatten())[0, 1]:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect and visualize neural network models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s arch runs/model.zip --diagram
  %(prog)s weights runs/model.zip --heatmap
  %(prog)s weights runs/model.zip --layer 0 --output weights.png
  %(prog)s activations runs/model.zip --episodes 5
  %(prog)s stats runs/model.zip --output stats.json
  %(prog)s compare runs/model1.zip runs/model2.zip
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # arch command
    arch_parser = subparsers.add_parser("arch", help="Show model architecture")
    arch_parser.add_argument("model", help="Path to model file")
    arch_parser.add_argument("--diagram", action="store_true", help="Show ASCII diagram")

    # weights command
    weights_parser = subparsers.add_parser("weights", help="Visualize model weights")
    weights_parser.add_argument("model", help="Path to model file")
    weights_parser.add_argument("--layer", type=int, help="Specific layer to show")
    weights_parser.add_argument(
        "--heatmap", action="store_true", help="Show heatmap instead of histogram"
    )
    weights_parser.add_argument("--output", "-o", help="Save to file instead of display")

    # activations command
    act_parser = subparsers.add_parser("activations", help="Analyze model activations")
    act_parser.add_argument("model", help="Path to model file")
    act_parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    act_parser.add_argument("--output", "-o", help="Save to file instead of display")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Export model statistics")
    stats_parser.add_argument("model", help="Path to model file")
    stats_parser.add_argument("--output", "-o", help="Output JSON file")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model1", help="First model file")
    compare_parser.add_argument("model2", help="Second model file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command
    commands = {
        "arch": cmd_arch,
        "weights": cmd_weights,
        "activations": cmd_activations,
        "stats": cmd_stats,
        "compare": cmd_compare,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
