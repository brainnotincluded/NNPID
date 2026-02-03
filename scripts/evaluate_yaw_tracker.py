#!/usr/bin/env python3
"""Evaluate trained yaw tracking neural network.

This script loads a trained model and evaluates it on the yaw tracking
task, computing metrics and optionally rendering videos.

Usage:
    python scripts/evaluate_yaw_tracker.py --model runs/yaw_tracking/best_model
    python scripts/evaluate_yaw_tracker.py --model runs/yaw_tracking/best_model --render --video
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed")


from src.deployment.model_loading import load_model_and_vecnormalize
from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv


@dataclass
class EpisodeMetrics:
    """Metrics for a single evaluation episode."""

    total_reward: float = 0.0
    episode_length: int = 0

    # Yaw tracking metrics
    mean_yaw_error: float = 0.0
    max_yaw_error: float = 0.0
    tracking_percentage: float = 0.0  # % time on target

    # Response metrics
    response_time: float = 0.0  # Time to first acquire target

    # Stability metrics
    mean_yaw_rate: float = 0.0
    mean_action_change: float = 0.0

    # Pattern info
    target_pattern: str = ""

    # Termination
    crashed: bool = False


def evaluate_episode(
    env: YawTrackingEnv,
    model: BaseAlgorithm,
    deterministic: bool = True,
    render: bool = False,
    pattern: str | None = None,
    tracking_threshold: float = 0.1,
    vec_normalize: VecNormalize | None = None,
) -> tuple[EpisodeMetrics, list[np.ndarray]]:
    """Run single evaluation episode.

    Args:
        env: Evaluation environment
        model: Trained model
        deterministic: Use deterministic actions
        render: Capture frames for video
        pattern: Force specific target pattern
        tracking_threshold: Threshold for "on target" in radians

    Returns:
        Tuple of (metrics, frames)
    """
    metrics = EpisodeMetrics()
    frames = []

    # Track metrics over time
    yaw_errors = []
    yaw_rates = []
    action_changes = []
    tracking_times = []

    # Reset
    options = {"pattern": pattern} if pattern else None
    obs, info = env.reset(options=options)
    metrics.target_pattern = pattern or "random"

    # Normalize initial observation if VecNormalize is available
    if vec_normalize is not None:
        obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))
        obs = np.array(obs_normalized[0], dtype=np.float32)

    prev_action = 0.0
    first_tracking_time = None

    done = False
    while not done:
        # Normalize observation if VecNormalize is available
        obs_normalized = obs
        if vec_normalize is not None:
            obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
            obs_normalized = np.array(obs_vec[0], dtype=np.float32)

        # Get action
        action, _ = model.predict(obs_normalized, deterministic=deterministic)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track metrics
        metrics.total_reward += reward
        metrics.episode_length += 1

        yaw_error = abs(info.get("yaw_error", 0.0))
        yaw_errors.append(yaw_error)

        yaw_rate = abs(info.get("yaw_rate", 0.0))
        yaw_rates.append(yaw_rate)

        action_change = abs(float(action[0]) - prev_action)
        action_changes.append(action_change)
        prev_action = float(action[0])

        is_tracking = yaw_error < tracking_threshold
        tracking_times.append(1.0 if is_tracking else 0.0)

        # Response time (first time acquiring target)
        if is_tracking and first_tracking_time is None:
            dt = 1.0 / getattr(env.config, "control_frequency", 50.0)
            first_tracking_time = metrics.episode_length * dt

        # Render
        if render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Check crash
        if terminated and not truncated:
            metrics.crashed = True

    # Compute aggregate metrics
    if yaw_errors:
        metrics.mean_yaw_error = float(np.mean(yaw_errors))
        metrics.max_yaw_error = float(np.max(yaw_errors))

    if tracking_times:
        metrics.tracking_percentage = float(np.mean(tracking_times) * 100)

    if yaw_rates:
        metrics.mean_yaw_rate = float(np.mean(yaw_rates))

    if action_changes:
        metrics.mean_action_change = float(np.mean(action_changes))

    metrics.response_time = first_tracking_time if first_tracking_time else float("inf")

    return metrics, frames


def save_video(
    frames: list[np.ndarray] | list[Any],
    output_path: Path,
    fps: int = 30,
) -> None:
    """Save frames as video.

    Args:
        frames: List of RGB frames
        output_path: Output path for video
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed, cannot save video")
        print("Install with: pip install imageio[ffmpeg]")
        return

    if not frames:
        print("Warning: No frames to save")
        return

    print(f"Saving video with {len(frames)} frames to {output_path}")
    imageio.mimwrite(str(output_path), frames, fps=fps)


def plot_metrics(
    all_metrics: list[EpisodeMetrics],
    output_path: Path | None = None,
) -> None:
    """Plot evaluation metrics.

    Args:
        all_metrics: List of episode metrics
        output_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, cannot plot")
        return

    # Prepare data
    patterns = list({m.target_pattern for m in all_metrics})

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Yaw Tracking Evaluation Metrics", fontsize=14)

    # 1. Mean yaw error by pattern
    ax = axes[0, 0]
    for pattern in patterns:
        pattern_metrics = [m for m in all_metrics if m.target_pattern == pattern]
        errors = [m.mean_yaw_error for m in pattern_metrics]
        ax.bar(
            patterns.index(pattern),
            np.mean(errors),
            yerr=np.std(errors) if len(errors) > 1 else 0,
            label=pattern,
            alpha=0.7,
        )
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, rotation=45)
    ax.set_ylabel("Mean Yaw Error (rad)")
    ax.set_title("Mean Yaw Error by Pattern")
    ax.axhline(y=0.1, color="r", linestyle="--", label="threshold")

    # 2. Tracking percentage
    ax = axes[0, 1]
    for pattern in patterns:
        pattern_metrics = [m for m in all_metrics if m.target_pattern == pattern]
        tracking = [m.tracking_percentage for m in pattern_metrics]
        ax.bar(
            patterns.index(pattern),
            np.mean(tracking),
            yerr=np.std(tracking) if len(tracking) > 1 else 0,
            label=pattern,
            alpha=0.7,
        )
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, rotation=45)
    ax.set_ylabel("Tracking %")
    ax.set_title("Time on Target by Pattern")
    ax.set_ylim(0, 100)

    # 3. Response time
    ax = axes[0, 2]
    response_times = [m.response_time for m in all_metrics if m.response_time < float("inf")]
    if response_times:
        ax.hist(response_times, bins=20, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count")
    ax.set_title("Response Time Distribution")

    # 4. Episode rewards
    ax = axes[1, 0]
    rewards = [m.total_reward for m in all_metrics]
    ax.hist(rewards, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(x=np.mean(rewards), color="r", linestyle="--", label=f"mean={np.mean(rewards):.1f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Episode Reward Distribution")
    ax.legend()

    # 5. Control smoothness
    ax = axes[1, 1]
    yaw_rates = [m.mean_yaw_rate for m in all_metrics]
    action_changes = [m.mean_action_change for m in all_metrics]
    ax.scatter(yaw_rates, action_changes, alpha=0.6)
    ax.set_xlabel("Mean Yaw Rate (rad/s)")
    ax.set_ylabel("Mean Action Change")
    ax.set_title("Control Smoothness")

    # 6. Success rate (not crashed)
    ax = axes[1, 2]
    crashed = sum(1 for m in all_metrics if m.crashed)
    not_crashed = len(all_metrics) - crashed
    ax.pie(
        [not_crashed, crashed],
        labels=["Success", "Crashed"],
        autopct="%1.1f%%",
        colors=["green", "red"],
    )
    ax.set_title("Episode Outcomes")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")

    plt.show()


def print_summary(all_metrics: list[EpisodeMetrics]) -> None:
    """Print evaluation summary.

    Args:
        all_metrics: List of episode metrics
    """
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)

    n = len(all_metrics)
    print(f"\nTotal episodes: {n}")

    # Overall metrics
    rewards = [m.total_reward for m in all_metrics]
    print("\nEpisode Reward:")
    print(f"  Mean: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min/Max: {np.min(rewards):.2f} / {np.max(rewards):.2f}")

    errors = [m.mean_yaw_error for m in all_metrics]
    print("\nMean Yaw Error (rad):")
    print(f"  Mean: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
    print(f"  Mean (deg): {np.degrees(np.mean(errors)):.2f}°")

    tracking = [m.tracking_percentage for m in all_metrics]
    print("\nTracking Percentage:")
    print(f"  Mean: {np.mean(tracking):.1f}% ± {np.std(tracking):.1f}%")

    response_times = [m.response_time for m in all_metrics if m.response_time < float("inf")]
    if response_times:
        print("\nResponse Time:")
        print(f"  Mean: {np.mean(response_times):.3f}s ± {np.std(response_times):.3f}s")

    crashes = sum(1 for m in all_metrics if m.crashed)
    print(f"\nCrash Rate: {crashes}/{n} ({100 * crashes / n:.1f}%)")

    # Per-pattern breakdown
    patterns = list({m.target_pattern for m in all_metrics})
    if len(patterns) > 1:
        print("\nPer-Pattern Results:")
        for pattern in patterns:
            pattern_metrics = [m for m in all_metrics if m.target_pattern == pattern]
            pm_errors = [m.mean_yaw_error for m in pattern_metrics]
            pm_tracking = [m.tracking_percentage for m in pattern_metrics]
            print(f"  {pattern}:")
            print(f"    Yaw Error: {np.degrees(np.mean(pm_errors)):.2f}°")
            print(f"    Tracking: {np.mean(pm_tracking):.1f}%")

    print("\n" + "=" * 60)


def evaluate(
    model_path: Path,
    n_episodes: int = 20,
    patterns: list[str] | None = None,
    deterministic: bool = True,
    render: bool = False,
    save_video_path: Path | None = None,
    save_plot_path: Path | None = None,
    seed: int = 42,
) -> list[EpisodeMetrics]:
    """Run full evaluation.

    Args:
        model_path: Path to trained model
        n_episodes: Number of evaluation episodes
        patterns: Specific patterns to test (None for all)
        deterministic: Use deterministic actions
        render: Render evaluation (for video)
        save_video_path: Path to save video
        save_plot_path: Path to save plot
        seed: Random seed

    Returns:
        List of episode metrics
    """
    if not SB3_AVAILABLE:
        print("Error: stable-baselines3 required to load models")
        return []

    # Create environment config (same defaults as training)
    config = YawTrackingConfig(
        target_patterns=["circular"] if patterns is None else patterns,
        target_speed_min=0.05,
        target_speed_max=0.1,
        control_frequency=100.0,
        action_smoothing=0.3,
        max_action_change=0.5,
        yaw_authority=0.20,
        yaw_rate_kp=5.0,
    )

    # Include all available patterns for evaluation if patterns not specified
    if patterns is None:
        all_patterns = [
            "circular",
            "random",
            "sinusoidal",
            "step",
            "figure8",
            "spiral",
            "evasive",
            "lissajous",
            "multi_frequency",
        ]
        config.target_patterns = all_patterns

    render_mode = "rgb_array" if render else None
    env = YawTrackingEnv(config=config, render_mode=render_mode)

    # Load model and VecNormalize
    try:
        model, vec_normalize, resolved = load_model_and_vecnormalize(
            model_path,
            env_factory=lambda: YawTrackingEnv(config=config),
        )
        print(f"Loaded model from {resolved}")
        if vec_normalize:
            print("Loaded VecNormalize")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        env.close()
        return []

    # Patterns to test
    test_patterns = patterns or all_patterns
    episodes_per_pattern = max(1, n_episodes // len(test_patterns))

    print(f"\nEvaluating on patterns: {test_patterns}")
    print(f"Episodes per pattern: {episodes_per_pattern}")

    all_metrics: list[EpisodeMetrics] = []
    all_frames: list[np.ndarray] = []

    for pattern in test_patterns:
        print(f"\n  Testing {pattern}...")

        for ep in range(episodes_per_pattern):
            env.reset(seed=seed + len(all_metrics))

            should_render = bool(render) and (len(all_frames) == 0 or save_video_path is not None)

            metrics, frames = evaluate_episode(
                env=env,
                model=model,
                deterministic=deterministic,
                render=should_render,
                pattern=pattern,
                vec_normalize=vec_normalize,
            )

            all_metrics.append(metrics)

            # Keep frames from first episode with rendering
            if frames and not all_frames:
                all_frames = frames

            print(
                f"    Episode {ep + 1}: reward={metrics.total_reward:.1f}, "
                f"error={np.degrees(metrics.mean_yaw_error):.1f}°, "
                f"tracking={metrics.tracking_percentage:.0f}%"
            )

    env.close()

    # Print summary
    print_summary(all_metrics)

    # Save video
    if save_video_path and all_frames:
        save_video(all_frames, save_video_path)

    # Save plot
    if True:  # Always show plot
        plot_path = save_plot_path or Path("yaw_tracking_eval.png")
        plot_metrics(all_metrics, plot_path if save_plot_path else None)

    return all_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate yaw tracking neural network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        choices=[
            "circular",
            "random",
            "sinusoidal",
            "step",
            "figure8",
            "spiral",
            "evasive",
            "lissajous",
            "multi_frequency",
        ],
        help="Target patterns to test",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Save video to path",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save plot to path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    deterministic = not args.stochastic

    evaluate(
        model_path=args.model,
        n_episodes=args.n_episodes,
        patterns=args.patterns,
        deterministic=deterministic,
        render=args.render or args.video is not None,
        save_video_path=args.video,
        save_plot_path=args.plot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
