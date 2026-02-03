#!/usr/bin/env python3
"""Unified MuJoCo visualization for trained yaw tracking models.

Supports interactive viewing, video recording, or both.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.model_loading import load_model_and_vecnormalize
from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv


def _init_display():
    try:
        import cv2  # type: ignore

        return "cv2", cv2
    except Exception:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            return "mpl", plt
        except Exception as exc:
            raise RuntimeError("Interactive display requires cv2 or matplotlib") from exc


def _show_frame(display_backend: str, backend: Any, frame: np.ndarray, window_name: str) -> bool:
    """Render a frame in an interactive window. Returns False if user quits."""
    if display_backend == "cv2":
        backend.imshow(window_name, frame[:, :, ::-1])
        key = backend.waitKey(1) & 0xFF
        return key != ord("q")

    # matplotlib fallback
    if not hasattr(_show_frame, "_fig"):
        backend.ion()
        _show_frame._fig, _show_frame._ax = backend.subplots(figsize=(8, 6))  # type: ignore[attr-defined]
        _show_frame._img = _show_frame._ax.imshow(frame)  # type: ignore[attr-defined]
        _show_frame._ax.axis("off")  # type: ignore[attr-defined]
        backend.tight_layout()
    _show_frame._img.set_data(frame)  # type: ignore[attr-defined]
    backend.pause(0.001)
    return True


def run_episode(
    env: YawTrackingEnv,
    model: Any,
    vec_normalize: Any | None,
    deterministic: bool,
    max_steps: int,
    render: bool,
    video_writer: Any | None,
    frame_skip: int,
    interactive: bool,
    display_backend: str | None,
    display_lib: Any | None,
    window_name: str,
    seed: int | None,
) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    tracking_time = 0
    total_steps = 0
    total_reward = 0.0

    start = time.time()

    for step in range(max_steps):
        obs_for_model = obs
        if vec_normalize is not None:
            obs_vec = vec_normalize.normalize_obs(obs.reshape(1, -1))
            obs_for_model = obs_vec[0]

        action, _ = model.predict(obs_for_model, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_steps += 1

        if info.get("is_tracking", False):
            tracking_time += 1

        if render and step % frame_skip == 0:
            frame = env.render()
            if frame is not None:
                if video_writer is not None:
                    video_writer.append_data(frame)
                if interactive and display_backend and display_lib:
                    if not _show_frame(display_backend, display_lib, frame, window_name):
                        break

        if terminated or truncated:
            break

    elapsed = time.time() - start
    tracking_pct = 100 * tracking_time / max(total_steps, 1)
    return {
        "steps": total_steps,
        "tracking_pct": tracking_pct,
        "reward": total_reward,
        "elapsed": elapsed,
        "crashed": float(total_steps < max_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize trained yaw tracking model in MuJoCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model dir or .zip")
    parser.add_argument(
        "--mode",
        choices=["interactive", "video", "both"],
        default="video",
        help="Visualization mode",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["circular"],
        help="Target patterns to use",
    )
    parser.add_argument("--target-speed-min", type=float, default=0.05, help="Min target speed")
    parser.add_argument("--target-speed-max", type=float, default=0.1, help="Max target speed")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--fps", type=int, default=30, help="Video fps")
    parser.add_argument("--frame-skip", type=int, default=2, help="Render every Nth frame")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path (required for video/both)",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable VecNormalize even if available",
    )

    args = parser.parse_args()
    deterministic = not args.stochastic

    record_video = args.mode in {"video", "both"}
    interactive = args.mode in {"interactive", "both"}
    render_mode = "rgb_array" if record_video or interactive else None

    if record_video and args.output is None:
        output_dir = Path("runs/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"mujoco_viz_{int(time.time())}.mp4"

    display_backend = None
    display_lib = None
    if interactive:
        display_backend, display_lib = _init_display()

    config = YawTrackingConfig(
        target_patterns=args.patterns,
        target_speed_min=args.target_speed_min,
        target_speed_max=args.target_speed_max,
    )
    env = YawTrackingEnv(config=config, render_mode=render_mode)

    model, vec_normalize, resolved = load_model_and_vecnormalize(
        args.model,
        env_factory=lambda: YawTrackingEnv(config=config),
    )
    if args.no_normalize and vec_normalize is not None:
        vec_normalize = None

    video_writer = None
    if record_video:
        try:
            import imageio

            video_writer = imageio.get_writer(str(args.output), fps=args.fps, quality=8)
            print(f"Recording video to: {args.output}")
        except Exception as exc:
            env.close()
            raise RuntimeError("Video recording requires imageio[ffmpeg]") from exc

    print("=" * 70)
    print("MuJoCo Visualization")
    print("=" * 70)
    print(f"Model: {resolved}")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Patterns: {args.patterns}")
    print(f"Target speed: {args.target_speed_min}-{args.target_speed_max} rad/s")
    vecnorm_label = "Yes" if vec_normalize else "No"
    if args.no_normalize:
        vecnorm_label = "No (disabled)"
    print(f"VecNormalize: {vecnorm_label}")
    print()

    metrics = []
    for ep in range(args.episodes):
        print(f"Episode {ep + 1}/{args.episodes}...", end=" ", flush=True)
        episode_metrics = run_episode(
            env=env,
            model=model,
            vec_normalize=vec_normalize,
            deterministic=deterministic,
            max_steps=args.steps,
            render=record_video or interactive,
            video_writer=video_writer,
            frame_skip=max(1, args.frame_skip),
            interactive=interactive,
            display_backend=display_backend,
            display_lib=display_lib,
            window_name="NNPID MuJoCo Viz (press q to quit)",
            seed=args.seed + ep,
        )
        metrics.append(episode_metrics)
        print(
            f"Tracking: {episode_metrics['tracking_pct']:.1f}% | "
            f"Reward: {episode_metrics['reward']:.1f} | "
            f"Steps: {episode_metrics['steps']}"
        )

    env.close()
    if video_writer is not None:
        video_writer.close()
    if display_backend == "cv2" and display_lib is not None:
        display_lib.destroyAllWindows()

    avg_tracking = np.mean([m["tracking_pct"] for m in metrics]) if metrics else 0.0
    avg_reward = np.mean([m["reward"] for m in metrics]) if metrics else 0.0
    crash_rate = 100 * np.mean([m["crashed"] for m in metrics]) if metrics else 0.0

    print("\nSummary")
    print(f"  Avg tracking: {avg_tracking:.1f}%")
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Crash rate: {crash_rate:.1f}%")


if __name__ == "__main__":
    main()
