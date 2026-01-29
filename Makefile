# NNPID Project Makefile
# Code quality, testing, and automation

.PHONY: help lint format check security test clean install dev

# Default target
help:
	@echo "NNPID Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint       - Run all linters (ruff, mypy)"
	@echo "  make format     - Format code (ruff format)"
	@echo "  make check      - Full code review (lint + security + complexity)"
	@echo "  make security   - Security scan (bandit)"
	@echo "  make complexity - Code complexity analysis (radon)"
	@echo "  make deadcode   - Find unused code (vulture)"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make test-fast  - Run fast tests only"
	@echo ""
	@echo "Development:"
	@echo "  make install    - Install project dependencies"
	@echo "  make dev        - Install dev dependencies"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make pre-commit - Install pre-commit hooks"
	@echo ""
	@echo "Training:"
	@echo "  make train      - Start yaw tracker training"
	@echo "  make train-mega - Start mega training (20M steps)"
	@echo "  make status     - Check training status"
	@echo ""

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "🔍 Running ruff linter..."
	ruff check src/ scripts/ tests/
	@echo "🔍 Running ruff format check..."
	ruff format --check src/ scripts/ tests/
	@echo "✅ Lint passed!"

format:
	@echo "🎨 Formatting code with ruff..."
	ruff format src/ scripts/ tests/
	@echo "🔧 Auto-fixing issues..."
	ruff check --fix src/ scripts/ tests/
	@echo "✅ Formatting complete!"

mypy:
	@echo "🔍 Running mypy type checker..."
	mypy src/ --ignore-missing-imports
	@echo "✅ Type check passed!"

security:
	@echo "🔒 Running bandit security scan..."
	bandit -r src/ -ll -q
	@echo "✅ Security check passed!"

complexity:
	@echo "📊 Analyzing code complexity..."
	@echo ""
	@echo "=== Cyclomatic Complexity (C+ = needs attention) ==="
	radon cc src/ scripts/ -a -s --min C
	@echo ""
	@echo "=== Maintainability Index (B- = needs attention) ==="
	radon mi src/ -s --min B
	@echo "✅ Complexity analysis complete!"

deadcode:
	@echo "💀 Finding dead code..."
	vulture src/ scripts/ --min-confidence 80
	@echo "✅ Dead code analysis complete!"

# Full code review
check: lint security complexity
	@echo ""
	@echo "================================================"
	@echo "✅ ALL CODE QUALITY CHECKS PASSED!"
	@echo "================================================"

# Quick check (fast, for pre-commit)
quick-check:
	@ruff check src/ scripts/ --quiet
	@ruff format --check src/ scripts/ --quiet

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	@echo "🧪 Running fast tests..."
	pytest tests/ -v -m "not slow"

# =============================================================================
# Development Setup
# =============================================================================

install:
	@echo "📦 Installing dependencies..."
	pip install -e .

dev:
	@echo "📦 Installing dev dependencies..."
	pip install -e ".[dev]"
	pip install ruff mypy bandit vulture radon pyright
	@echo "🔗 Installing pre-commit hooks..."
	pre-commit install

pre-commit:
	@echo "🔗 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Pre-commit hooks installed!"

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete!"

# =============================================================================
# Training
# =============================================================================

train:
	@echo "🚀 Starting yaw tracker training..."
	python scripts/train_yaw_tracker.py --config config/yaw_tracking.yaml

train-mega:
	@echo "🚀 Starting MEGA training (20M steps)..."
	nohup python scripts/train_yaw_tracker.py --config config/yaw_tracking_mega.yaml > /tmp/training_mega.log 2>&1 &
	@echo "Training started in background. Check with: make status"

status:
	@echo "📊 Training Status"
	@echo "=================="
	@python3 -c "\
import numpy as np; \
from pathlib import Path; \
import glob; \
runs = sorted(glob.glob('runs/yaw_tracking_*/eval_logs/evaluations.npz'), key=lambda x: Path(x).stat().st_mtime, reverse=True); \
f = runs[0] if runs else None; \
print(f'Run: {Path(f).parent.parent.name}') if f else print('No runs found'); \
d = np.load(f) if f else None; \
print(f'Steps: {int(d[\"timesteps\"][-1]):,}') if d is not None and len(d['timesteps']) > 0 else None; \
print(f'Reward: {np.mean(d[\"results\"][-1]):.2f}') if d is not None and len(d['results']) > 0 else None; \
" 2>/dev/null || echo "No training data found"
	@ps aux | grep "train_yaw_tracker" | grep -v grep | head -1 || echo "No training process running"

# =============================================================================
# Visualization
# =============================================================================

viz:
	@echo "🎬 Starting visualization..."
	@LATEST=$$(ls -t runs/*/checkpoints/*.zip 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then echo "No models found"; exit 1; fi; \
	echo "Using model: $$LATEST"; \
	python scripts/run_mega_viz.py --model "$$LATEST" --viz-mode full --episodes 1 --max-steps 500 --record /tmp/viz_output.mp4
	@echo "Video saved to /tmp/viz_output.mp4"
	@open /tmp/viz_output.mp4 2>/dev/null || echo "Open: /tmp/viz_output.mp4"
