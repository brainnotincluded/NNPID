#!/usr/bin/env python3
"""Training script with comprehensive logging for analysis.

This script trains a yaw tracking controller with detailed metrics
that can be analyzed to understand model behavior and identify improvements.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.yaw_tracking_env import YawTrackingConfig, YawTrackingEnv


@dataclass
class EpisodeMetrics:
    """Detailed metrics for a single episode."""

    # Basic stats
    episode_length: int = 0
    total_reward: float = 0.0
    terminated: bool = False  # Crashed?
    truncated: bool = False  # Time limit?

    # Tracking performance
    mean_yaw_error: float = 0.0
    max_yaw_error: float = 0.0
    min_yaw_error: float = 0.0
    std_yaw_error: float = 0.0
    time_on_target: float = 0.0  # % of time within threshold
    time_close: float = 0.0  # % of time within 20°

    # Control behavior
    mean_action: float = 0.0
    std_action: float = 0.0
    mean_action_change: float = 0.0  # Smoothness
    max_action: float = 0.0
    action_reversals: int = 0  # Number of direction changes

    # Stability
    mean_roll: float = 0.0
    mean_pitch: float = 0.0
    max_tilt: float = 0.0
    mean_altitude_error: float = 0.0

    # Reward breakdown (per step averages)
    reward_facing: float = 0.0
    reward_error_reduction: float = 0.0
    reward_velocity_match: float = 0.0
    reward_direction_bonus: float = 0.0
    reward_excess_penalty: float = 0.0
    reward_smoothness_penalty: float = 0.0
    reward_tracking_bonus: float = 0.0
    reward_alive_bonus: float = 0.0

    # Target info
    target_pattern: str = ""
    target_speed: float = 0.0


@dataclass
class TrainingAnalytics:
    """Aggregated analytics across training."""

    # Episode history
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    # Rolling averages (last 100 episodes)
    rolling_reward: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_tracking: deque = field(default_factory=lambda: deque(maxlen=100))
    rolling_crashes: deque = field(default_factory=lambda: deque(maxlen=100))

    # Improvement tracking
    best_tracking: float = 0.0
    best_reward: float = -float("inf")
    timesteps_to_50pct_tracking: int | None = None
    timesteps_to_10pct_crashes: int | None = None

    def add_episode(self, ep: EpisodeMetrics, timesteps: int) -> None:
        """Add episode and update rolling stats."""
        self.episodes.append(ep)
        self.rolling_reward.append(ep.total_reward)
        self.rolling_tracking.append(ep.time_on_target)
        self.rolling_crashes.append(1 if ep.terminated else 0)

        # Track milestones
        if ep.time_on_target > self.best_tracking:
            self.best_tracking = ep.time_on_target
        if ep.total_reward > self.best_reward:
            self.best_reward = ep.total_reward

        # Check milestones
        if len(self.rolling_tracking) >= 50:
            avg_tracking = np.mean(self.rolling_tracking)
            avg_crashes = np.mean(self.rolling_crashes)

            if self.timesteps_to_50pct_tracking is None and avg_tracking >= 0.5:
                self.timesteps_to_50pct_tracking = timesteps
                print(f"\n🎯 MILESTONE: 50% tracking achieved at {timesteps:,} steps!")

            if self.timesteps_to_10pct_crashes is None and avg_crashes <= 0.1:
                self.timesteps_to_10pct_crashes = timesteps
                print(f"\n✈️ MILESTONE: <10% crash rate at {timesteps:,} steps!")

    def get_summary(self, last_n: int = 100) -> dict[str, Any]:
        """Get summary statistics."""
        recent = self.episodes[-last_n:] if len(self.episodes) >= last_n else self.episodes

        if not recent:
            return {}

        return {
            "n_episodes": len(self.episodes),
            "avg_reward": np.mean([e.total_reward for e in recent]),
            "avg_tracking": np.mean([e.time_on_target for e in recent]),
            "avg_close": np.mean([e.time_close for e in recent]),
            "crash_rate": np.mean([1 if e.terminated else 0 for e in recent]),
            "avg_yaw_error_deg": np.mean([np.degrees(e.mean_yaw_error) for e in recent]),
            "avg_action_smoothness": np.mean([e.mean_action_change for e in recent]),
            "avg_max_tilt_deg": np.mean([np.degrees(e.max_tilt) for e in recent]),
            "best_tracking": self.best_tracking,
            "best_reward": self.best_reward,
        }


class DetailedLoggingCallback(BaseCallback):
    """Callback for comprehensive training analysis."""

    def __init__(
        self,
        log_dir: str,
        log_frequency: int = 1000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_frequency = log_frequency

        # Analytics
        self.analytics = TrainingAnalytics()

        # Per-episode tracking
        self._episode_data: list[dict] = []
        self._current_ep_actions: list[float] = []
        self._current_ep_yaw_errors: list[float] = []
        self._current_ep_rewards: list[float] = []
        self._current_ep_tilts: list[float] = []
        self._current_ep_alt_errors: list[float] = []

        # Timing
        self._start_time = time.time()
        self._last_log_time = time.time()

        # JSON log file
        self._json_log = self.log_dir / "training_analytics.jsonl"

    def _on_step(self) -> bool:
        """Called after each step."""
        # Get info from environment
        infos = self.locals.get("infos", [{}])
        actions = self.locals.get("actions", [[0]])
        rewards = self.locals.get("rewards", [0])
        dones = self.locals.get("dones", [False])

        for i, (info, action, reward, done) in enumerate(
            zip(infos, actions, rewards, dones, strict=False)
        ):
            # Collect step data
            if "yaw_error" in info:
                self._current_ep_yaw_errors.append(abs(info["yaw_error"]))
            if "roll" in info and "pitch" in info:
                tilt = np.sqrt(info["roll"] ** 2 + info["pitch"] ** 2)
                self._current_ep_tilts.append(tilt)
            if "altitude" in info:
                self._current_ep_alt_errors.append(abs(info["altitude"] - 1.0))

            self._current_ep_actions.append(float(action[0]) if len(action) > 0 else 0)
            self._current_ep_rewards.append(float(reward))

            # Episode finished
            if done:
                self._finish_episode(info, i)

        # Periodic logging
        if self.num_timesteps % self.log_frequency == 0:
            self._log_progress()

        return True

    def _finish_episode(self, info: dict, _env_idx: int) -> None:
        """Process finished episode."""
        if not self._current_ep_actions:
            return

        actions = np.array(self._current_ep_actions)
        yaw_errors = np.array(self._current_ep_yaw_errors)
        rewards = np.array(self._current_ep_rewards)
        tilts = np.array(self._current_ep_tilts) if self._current_ep_tilts else np.array([0])
        alt_errors = (
            np.array(self._current_ep_alt_errors) if self._current_ep_alt_errors else np.array([0])
        )

        # Action changes
        action_changes = np.abs(np.diff(actions)) if len(actions) > 1 else np.array([0])
        action_reversals = np.sum(np.diff(np.sign(actions)) != 0)

        # Tracking metrics
        on_target_threshold = 0.1  # ~6°
        close_threshold = 0.35  # ~20°
        time_on_target = np.mean(yaw_errors < on_target_threshold) if len(yaw_errors) > 0 else 0
        time_close = np.mean(yaw_errors < close_threshold) if len(yaw_errors) > 0 else 0

        # Create metrics
        ep = EpisodeMetrics(
            episode_length=len(actions),
            total_reward=float(np.sum(rewards)),
            terminated=info.get("TimeLimit.truncated", False) is False
            and info.get("terminal_observation") is not None,
            truncated=info.get("TimeLimit.truncated", False),
            mean_yaw_error=float(np.mean(yaw_errors)) if len(yaw_errors) > 0 else 0,
            max_yaw_error=float(np.max(yaw_errors)) if len(yaw_errors) > 0 else 0,
            min_yaw_error=float(np.min(yaw_errors)) if len(yaw_errors) > 0 else 0,
            std_yaw_error=float(np.std(yaw_errors)) if len(yaw_errors) > 0 else 0,
            time_on_target=float(time_on_target),
            time_close=float(time_close),
            mean_action=float(np.mean(actions)),
            std_action=float(np.std(actions)),
            mean_action_change=float(np.mean(action_changes)),
            max_action=float(np.max(np.abs(actions))),
            action_reversals=int(action_reversals),
            mean_roll=float(info.get("roll", 0)),
            mean_pitch=float(info.get("pitch", 0)),
            max_tilt=float(np.max(tilts)) if len(tilts) > 0 else 0,
            mean_altitude_error=float(np.mean(alt_errors)),
        )

        # Add to analytics
        self.analytics.add_episode(ep, self.num_timesteps)

        # Log to JSON
        log_entry = {
            "timesteps": self.num_timesteps,
            "episode": len(self.analytics.episodes),
            **asdict(ep),
        }
        with open(self._json_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Reset episode tracking
        self._current_ep_actions = []
        self._current_ep_yaw_errors = []
        self._current_ep_rewards = []
        self._current_ep_tilts = []
        self._current_ep_alt_errors = []

    def _log_progress(self) -> None:
        """Log training progress."""
        elapsed = time.time() - self._start_time
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        summary = self.analytics.get_summary(last_n=50)
        if not summary:
            return

        # Print progress
        print(f"\n{'=' * 70}")
        print(f"TRAINING PROGRESS - {self.num_timesteps:,} timesteps ({elapsed / 60:.1f} min)")
        print(f"{'=' * 70}")
        print(f"Episodes: {summary['n_episodes']} | FPS: {fps:.0f}")
        print(f"Avg Reward: {summary['avg_reward']:.2f} | Best: {summary['best_reward']:.2f}")
        print(
            f"Tracking (<6°): {summary['avg_tracking'] * 100:.1f}% | Best: {summary['best_tracking'] * 100:.1f}%"
        )
        print(f"Close (<20°): {summary['avg_close'] * 100:.1f}%")
        print(f"Crash Rate: {summary['crash_rate'] * 100:.1f}%")
        print(f"Avg Yaw Error: {summary['avg_yaw_error_deg']:.1f}°")
        print(f"Action Smoothness: {summary['avg_action_smoothness']:.3f}")
        print(f"Max Tilt: {summary['avg_max_tilt_deg']:.1f}°")

        # Diagnose issues
        self._diagnose_issues(summary)

    def _diagnose_issues(self, summary: dict) -> None:
        """Print diagnostic messages based on current performance."""
        issues = []

        if summary["crash_rate"] > 0.2:
            issues.append("⚠️ HIGH CRASH RATE - stabilizer may need tuning")

        if summary["avg_tracking"] < 0.1 and summary["n_episodes"] > 100:
            issues.append("⚠️ LOW TRACKING - model not learning to face target")

        if summary["avg_action_smoothness"] > 0.5:
            issues.append("⚠️ JERKY CONTROL - action changes too large")

        if summary["avg_yaw_error_deg"] > 90:
            issues.append("⚠️ LARGE YAW ERROR - model may be ignoring target")

        if summary["avg_max_tilt_deg"] > 30:
            issues.append("⚠️ EXCESSIVE TILT - stability issue")

        if issues:
            print("\nDIAGNOSTICS:")
            for issue in issues:
                print(f"  {issue}")

    def _on_training_end(self) -> None:
        """Called at end of training."""
        # Save final analytics
        final_summary = self.analytics.get_summary()
        with open(self.log_dir / "final_summary.json", "w") as f:
            json.dump(
                {
                    "total_episodes": len(self.analytics.episodes),
                    "total_timesteps": self.num_timesteps,
                    "training_time_seconds": time.time() - self._start_time,
                    "final_metrics": final_summary,
                    "milestones": {
                        "timesteps_to_50pct_tracking": self.analytics.timesteps_to_50pct_tracking,
                        "timesteps_to_10pct_crashes": self.analytics.timesteps_to_10pct_crashes,
                    },
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total timesteps: {self.num_timesteps:,}")
        print(f"Total episodes: {len(self.analytics.episodes)}")
        print(f"Training time: {(time.time() - self._start_time) / 60:.1f} minutes")
        print("\nFinal metrics (last 100 episodes):")
        for k, v in final_summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")


def make_env(config: YawTrackingConfig) -> YawTrackingEnv:
    """Create environment with config."""
    return YawTrackingEnv(config)


def main():
    """Main training function."""
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/analysis_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("YAW TRACKING TRAINING WITH DETAILED ANALYSIS")
    print("=" * 70)

    # Environment config - start with VERY easy settings for initial learning
    env_config = YawTrackingConfig(
        # Start with very slow targets for easier learning
        target_patterns=["circular"],  # Single simple pattern
        target_speed_min=0.05,  # Very slow (was 0.2)
        target_speed_max=0.1,  # Very slow (was 0.4)
        # Optimized stability settings
        control_frequency=100.0,
        max_episode_steps=2000,  # 20 seconds
        # Action smoothing (prevents exploration crashes)
        action_smoothing=0.3,  # Low-pass filter
        max_action_change=0.5,  # Limit jerky control
        # PID settings
        attitude_kp=15.0,
        attitude_ki=0.5,
        attitude_kd=5.0,
        yaw_authority=0.20,  # Higher for faster yaw (0.6 rad/s achievable)
        yaw_rate_kp=5.0,  # Higher for better tracking
        # Reward settings (v2)
        facing_reward_weight=1.5,
        error_reduction_weight=0.5,
        velocity_match_weight=0.2,
        direction_alignment_bonus=0.1,
        excess_yaw_rate_penalty=0.05,
        action_rate_penalty_weight=0.05,  # Increased to discourage jerky actions
        sustained_tracking_bonus=0.3,
        alive_bonus=0.1,
        crash_penalty=50.0,  # Increased
    )

    print("\nEnvironment config:")
    print(f"  Target patterns: {env_config.target_patterns}")
    print(f"  Target speed: {env_config.target_speed_min}-{env_config.target_speed_max} rad/s")
    print(f"  Control frequency: {env_config.control_frequency} Hz")
    print(
        f"  Episode length: {env_config.max_episode_steps} steps ({env_config.max_episode_steps / env_config.control_frequency}s)"
    )

    # Create vectorized environment
    n_envs = 8
    env = make_vec_env(
        lambda: make_env(env_config),
        n_envs=n_envs,
        seed=42,
    )

    # Normalize observations only (reward normalization can hide learning signal)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,  # Disabled to preserve reward gradient
        clip_obs=10.0,
    )

    print("\nTraining setup:")
    print(f"  Parallel environments: {n_envs}")
    print("  Observation normalization: enabled")
    print("  Reward normalization: enabled")

    # Model config
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]},
        },
        verbose=0,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=42,
    )

    print("\nModel: PPO")
    print("  Network: [256, 256, 128]")
    print("  Learning rate: 3e-4")
    print("  Batch size: 64")

    # Callbacks
    detailed_callback = DetailedLoggingCallback(
        log_dir=str(run_dir / "analytics"),
        log_frequency=10000,  # Log every 10k steps
    )

    # Eval environment
    eval_env = make_vec_env(
        lambda: make_env(env_config),
        n_envs=1,
        seed=123,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=25000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    # Training - 3M steps for 70%+ tracking
    total_timesteps = 3_000_000
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Logs: {run_dir}")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[detailed_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save final model
    model.save(run_dir / "final_model")
    env.save(run_dir / "vec_normalize.pkl")

    print(f"\nModels saved to: {run_dir}")
    print(f"Analytics: {run_dir / 'analytics'}")
    print(f"TensorBoard: tensorboard --logdir {run_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()
