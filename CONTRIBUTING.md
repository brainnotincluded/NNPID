# Contributing to NNPID

Добро пожаловать! Этот документ описывает правила и стандарты для контрибуции в проект.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Code Standards](#code-standards)
5. [Commit Guidelines](#commit-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Code Review Checklist](#code-review-checklist)

---

## Code of Conduct

- Будьте уважительны и профессиональны
- Конструктивная критика приветствуется
- Фокус на качестве кода, а не на личностях

---

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/NNPID.git
cd NNPID
git remote add upstream https://github.com/brainnotincluded/NNPID.git
```

### 2. Setup Development Environment

```bash
# Install with dev dependencies
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

---

## Development Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Create    │───►│   Write     │───►│   Test      │───►│   Submit    │
│   Branch    │    │   Code      │    │   & Lint    │    │   PR        │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                   │
                          ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Follow     │    │  All tests  │
                   │  Standards  │    │  pass       │
                   └─────────────┘    └─────────────┘
```

### Before Starting

- [ ] Check existing issues/PRs for similar work
- [ ] Create issue describing your proposed change
- [ ] Get approval for significant changes

### While Coding

- [ ] Follow code standards (see below)
- [ ] Write tests for new functionality
- [ ] Update documentation if needed
- [ ] Keep commits small and focused

### Before Submitting

- [ ] Run full test suite
- [ ] Run linters and formatters
- [ ] Update CHANGELOG if applicable
- [ ] Rebase on latest main

---

## Code Standards

### Python Style Guide

#### General Rules

```python
# ✅ GOOD: Clear, typed, documented
def compute_yaw_error(
    drone_yaw: float,
    target_angle: float,
) -> float:
    """Compute normalized yaw error between drone and target.
    
    Args:
        drone_yaw: Current drone yaw in radians [-π, π]
        target_angle: Target direction in radians [-π, π]
        
    Returns:
        Normalized yaw error in radians [-π, π]
        
    Example:
        >>> compute_yaw_error(0.0, np.pi/2)
        1.5707963267948966
    """
    error = target_angle - drone_yaw
    # Normalize to [-π, π]
    while error > np.pi:
        error -= 2 * np.pi
    while error < -np.pi:
        error += 2 * np.pi
    return error


# ❌ BAD: No types, no docs, unclear name
def calc(a, b):
    e = b - a
    while e > 3.14159: e -= 6.28318
    while e < -3.14159: e += 6.28318
    return e
```

#### Naming Conventions

```python
# Classes: PascalCase
class YawTrackingEnv:
    pass

class QuadrotorState:
    pass

# Functions/Methods: snake_case
def compute_motor_commands():
    pass

def get_observation():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_YAW_RATE = 2.0
DEFAULT_HOVER_HEIGHT = 1.0

# Private: leading underscore
def _internal_helper():
    pass

class MyClass:
    def __init__(self):
        self._private_var = 42
```

#### Type Hints (Required)

```python
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# Simple types
def process(value: int, name: str) -> bool:
    ...

# Complex types
def get_state() -> Tuple[NDArray[np.float64], Dict[str, float]]:
    ...

# Optional values
def load_model(path: str, device: Optional[str] = None) -> nn.Module:
    ...

# Union types
def normalize(value: Union[float, NDArray]) -> NDArray:
    ...
```

#### Docstrings (Google Style)

```python
def train_model(
    env: gym.Env,
    config: TrainingConfig,
    callbacks: Optional[List[Callback]] = None,
) -> Tuple[Model, Dict[str, float]]:
    """Train a reinforcement learning model.
    
    This function trains a PPO/SAC model on the given environment
    using the specified configuration.
    
    Args:
        env: Gymnasium environment for training.
        config: Training configuration with hyperparameters.
        callbacks: Optional list of training callbacks.
            Defaults to None (no callbacks).
    
    Returns:
        Tuple containing:
            - Trained model ready for inference
            - Dictionary of training metrics (loss, reward, etc.)
    
    Raises:
        ValueError: If config contains invalid hyperparameters.
        RuntimeError: If training fails to converge.
    
    Example:
        >>> env = YawTrackingEnv()
        >>> config = TrainingConfig(timesteps=100000)
        >>> model, metrics = train_model(env, config)
        >>> print(f"Final reward: {metrics['mean_reward']}")
    
    Note:
        Training is deterministic if config.seed is set.
    """
    ...
```

#### Import Order

```python
# 1. Standard library
import os
import sys
from typing import Optional, Dict

# 2. Third-party packages
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO

# 3. Local imports
from src.core.mujoco_sim import MuJoCoSimulator
from src.environments import YawTrackingEnv
from src.utils.rotations import Rotations
```

### File Structure

```python
"""Module docstring explaining purpose.

This module provides X functionality for Y purpose.
It is used by Z components.
"""

from __future__ import annotations

# Imports (ordered as above)
import numpy as np
from typing import Optional

# Constants
MAX_EPISODES = 1000
DEFAULT_TIMEOUT = 30.0

# Type aliases
StateVector = np.ndarray
ActionVector = np.ndarray


# Classes
class MyClass:
    """Class docstring."""
    
    def __init__(self, param: int) -> None:
        """Initialize MyClass."""
        self.param = param
    
    def public_method(self) -> int:
        """Public method docstring."""
        return self._private_method()
    
    def _private_method(self) -> int:
        """Private helper."""
        return self.param * 2


# Functions
def main_function() -> None:
    """Main function docstring."""
    pass


def _helper_function() -> None:
    """Private helper function."""
    pass


# Module-level code (if needed)
if __name__ == "__main__":
    main_function()
```

### Error Handling

```python
# ✅ GOOD: Specific exceptions with context
def load_config(path: str) -> Dict:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e
    
    if "environment" not in config:
        raise KeyError("Config missing required 'environment' section")
    
    return config


# ❌ BAD: Catching all exceptions, no context
def load_config(path):
    try:
        with open(path) as f:
            return yaml.load(f)
    except:
        return {}
```

### Logging

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Levels:
logger.debug("Detailed info for debugging")      # Verbose
logger.info("Normal operation messages")          # Standard
logger.warning("Something unexpected but ok")     # Attention
logger.error("Operation failed")                  # Problems
logger.critical("System cannot continue")         # Fatal

# With context
logger.info(f"Training started: {config.timesteps} timesteps")
logger.error(f"Connection failed to {address}: {error}")
```

### Testing Standards

```python
import pytest
import numpy as np
from src.environments import YawTrackingEnv


class TestYawTrackingEnv:
    """Tests for YawTrackingEnv."""
    
    @pytest.fixture
    def env(self) -> YawTrackingEnv:
        """Create environment for testing."""
        return YawTrackingEnv()
    
    def test_reset_returns_valid_observation(self, env: YawTrackingEnv) -> None:
        """Reset should return observation matching space."""
        obs, info = env.reset(seed=42)
        
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
    
    def test_step_with_zero_action(self, env: YawTrackingEnv) -> None:
        """Zero action should maintain stability."""
        env.reset(seed=42)
        
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            
            if terminated:
                pytest.fail("Drone crashed with zero action")
        
        # Should still be flying
        assert info["altitude_error"] < 0.5
    
    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_deterministic_reset(self, env: YawTrackingEnv, seed: int) -> None:
        """Same seed should produce same initial state."""
        obs1, _ = env.reset(seed=seed)
        obs2, _ = env.reset(seed=seed)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    @pytest.mark.slow
    def test_full_episode(self, env: YawTrackingEnv) -> None:
        """Complete episode should run without errors."""
        obs, _ = env.reset(seed=42)
        total_reward = 0
        
        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        assert step > 0, "Episode ended immediately"
```

---

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code restructuring |
| `perf` | Performance improvement |
| `test` | Adding/fixing tests |
| `chore` | Maintenance tasks |

### Examples

```bash
# Feature
feat(environments): add sinusoidal target pattern

Add SinusoidalTarget class that moves the target in a
sine wave pattern. Configurable via target_patterns in YAML.

Closes #42

# Bug fix
fix(controllers): correct motor mixing signs for yaw

The yaw torque was applied with inverted sign, causing
the drone to turn in the wrong direction.

Fixes #37

# Documentation
docs(readme): add installation instructions for Windows

# Refactor
refactor(core): extract sensor logic to separate module

Move sensor-related code from mujoco_sim.py to new
sensors.py module for better separation of concerns.

# Performance
perf(training): parallelize environment creation

Use multiprocessing to create vectorized environments,
reducing startup time by 40%.
```

### Commit Rules

1. **One logical change per commit**
2. **Subject line ≤ 72 characters**
3. **Use imperative mood** ("add" not "added")
4. **Reference issues** when applicable
5. **No WIP commits** in main branch

---

## Pull Request Process

### PR Title Format

```
<type>(<scope>): <description>
```

Examples:
- `feat(env): add waypoint navigation environment`
- `fix(sitl): handle connection timeout gracefully`
- `docs: update training guide with curriculum learning`

### PR Description Template

```markdown
## Summary
Brief description of changes (2-3 sentences)

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing done

## Screenshots/Videos
(if applicable)

## Related Issues
Closes #XX
```

### PR Checklist

Before submitting:

- [ ] Code follows style guide
- [ ] All tests pass locally
- [ ] New code has tests
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] PR has clear description
- [ ] Linked to related issues

---

## Code Review Checklist

### For Reviewers

#### 🔍 Correctness

- [ ] Does the code do what it claims?
- [ ] Are edge cases handled?
- [ ] Are there any obvious bugs?
- [ ] Is error handling appropriate?

#### 📖 Readability

- [ ] Is the code easy to understand?
- [ ] Are names descriptive and consistent?
- [ ] Are complex parts commented?
- [ ] Is the code properly formatted?

#### 🏗️ Architecture

- [ ] Does it fit the project structure?
- [ ] Is there code duplication?
- [ ] Are dependencies appropriate?
- [ ] Is the abstraction level right?

#### ⚡ Performance

- [ ] Are there obvious inefficiencies?
- [ ] Is memory usage reasonable?
- [ ] Are there blocking operations?
- [ ] Is vectorization used where possible?

#### 🧪 Testing

- [ ] Are there sufficient tests?
- [ ] Do tests cover edge cases?
- [ ] Are tests readable and maintainable?
- [ ] Do tests run quickly?

#### 📚 Documentation

- [ ] Are public APIs documented?
- [ ] Are complex algorithms explained?
- [ ] Is README updated if needed?
- [ ] Are config options documented?

#### 🔒 Security

- [ ] No hardcoded secrets?
- [ ] Input validation present?
- [ ] Safe file operations?
- [ ] No unsafe dependencies?

### Review Feedback Guide

```python
# ✅ Good feedback - specific and actionable
# "Consider using numpy broadcasting here instead of the loop.
#  It would be ~10x faster for large arrays:
#  `result = arr1 * arr2` instead of the for loop."

# ❌ Bad feedback - vague
# "This could be better"

# ✅ Good - suggests alternative
# "This function is doing too much. Consider splitting into:
#  1. validate_input() - check parameters
#  2. compute_result() - main logic
#  3. format_output() - prepare return value"

# ❌ Bad - just criticism
# "This function is too long"
```

### Approval Requirements

| Change Type | Required Approvals |
|-------------|-------------------|
| Documentation | 1 |
| Bug fix | 1 |
| New feature | 2 |
| Architecture change | 2 + maintainer |
| Breaking change | 2 + maintainer |

---

## Pre-Commit Checks

The following checks run automatically:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      
  - repo: https://github.com/psf/black
    hooks:
      - id: black
        
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        additional_dependencies: [numpy, torch]
```

### Running Manually

```bash
# Run all checks
pre-commit run --all-files

# Run specific check
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run mypy --all-files

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    NNPID Contribution Quick Ref                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BEFORE CODING:                                                  │
│    □ Check existing issues/PRs                                   │
│    □ Create branch: feature/name or fix/name                     │
│    □ Read ARCHITECTURE.md                                        │
│                                                                  │
│  WHILE CODING:                                                   │
│    □ Type hints on all functions                                 │
│    □ Docstrings (Google style)                                   │
│    □ Tests for new code                                          │
│    □ No print(), use logger                                      │
│                                                                  │
│  BEFORE COMMIT:                                                  │
│    □ pre-commit run --all-files                                  │
│    □ pytest tests/ -v                                            │
│    □ git diff --check                                            │
│                                                                  │
│  COMMIT MESSAGE:                                                 │
│    type(scope): description                                      │
│    │                                                             │
│    ├─ feat: new feature                                          │
│    ├─ fix: bug fix                                               │
│    ├─ docs: documentation                                        │
│    ├─ test: tests                                                │
│    └─ refactor: restructure                                      │
│                                                                  │
│  PR CHECKLIST:                                                   │
│    □ Tests pass                                                  │
│    □ Docs updated                                                │
│    □ No conflicts                                                │
│    □ Clear description                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Getting Help

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue with reproduction steps
- **Features**: Open an Issue with use case description

Thank you for contributing! 🚀
