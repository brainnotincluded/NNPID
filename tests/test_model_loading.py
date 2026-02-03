"""Tests for model loading path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.deployment.model_loading import resolve_model_path


def test_resolve_model_path_in_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best_model = run_dir / "best_model.zip"
    best_model.write_text("dummy")
    final_model = run_dir / "final_model.zip"
    final_model.write_text("dummy")

    resolved = resolve_model_path(run_dir)
    assert resolved == best_model


def test_resolve_model_path_in_best_model_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True)
    best_model = best_dir / "best_model.zip"
    best_model.write_text("dummy")

    resolved = resolve_model_path(run_dir)
    assert resolved == best_model


def test_resolve_model_path_file_without_suffix(tmp_path: Path) -> None:
    model_base = tmp_path / "model_checkpoint"
    model_zip = model_base.with_suffix(".zip")
    model_zip.write_text("dummy")

    resolved = resolve_model_path(model_base)
    assert resolved == model_zip


def test_resolve_model_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing_model"
    with pytest.raises(FileNotFoundError):
        resolve_model_path(missing)
