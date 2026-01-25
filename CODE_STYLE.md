# NNPID Code Style Guide

Quick reference for code style. See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## TL;DR

```bash
# Before every commit
pre-commit run --all-files
pytest tests/ -v
```

## Python Rules

### 1. Type Hints (Required)

```python
# ✅ Always use type hints
def compute_reward(state: QuadrotorState, action: np.ndarray) -> float:
    ...

# ❌ Never omit types
def compute_reward(state, action):
    ...
```

### 2. Docstrings (Google Style)

```python
def train(env: gym.Env, steps: int) -> Model:
    """Train a model on the environment.
    
    Args:
        env: Gymnasium environment.
        steps: Number of training steps.
        
    Returns:
        Trained model.
        
    Raises:
        ValueError: If steps <= 0.
    """
```

### 3. Naming

| Type | Style | Example |
|------|-------|---------|
| Class | PascalCase | `YawTrackingEnv` |
| Function | snake_case | `compute_reward` |
| Variable | snake_case | `yaw_error` |
| Constant | UPPER_SNAKE | `MAX_YAW_RATE` |
| Private | _leading | `_helper` |

### 4. Imports Order

```python
# 1. Standard library
import os
from typing import Optional

# 2. Third-party
import numpy as np
import torch

# 3. Local
from src.core import MuJoCoSimulator
```

### 5. Line Length

- **100 characters max**
- Break long lines logically

```python
# ✅ Good
result = some_function(
    arg1=value1,
    arg2=value2,
    arg3=value3,
)

# ❌ Bad - too long
result = some_function(arg1=value1, arg2=value2, arg3=value3, arg4=value4)
```

## Commit Messages

```
type(scope): description

feat(env): add new target pattern
fix(controller): correct motor mixing
docs(readme): update installation steps
test(env): add edge case tests
refactor(core): extract sensor logic
```

## Quick Checks

```bash
# Format code
black src/ scripts/ tests/

# Lint code
ruff check src/ --fix

# Type check
mypy src/

# Run tests
pytest tests/ -v

# All at once
pre-commit run --all-files
```

## File Template

```python
"""Module description.

Longer explanation if needed.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from src.utils import helper


CONSTANT = 42


class MyClass:
    """Class description."""
    
    def __init__(self, param: int) -> None:
        """Initialize."""
        self.param = param
    
    def method(self) -> int:
        """Method description."""
        return self.param


def function(arg: str) -> bool:
    """Function description."""
    return True
```

## Don'ts

- ❌ No `print()` - use `logger`
- ❌ No bare `except:` - catch specific exceptions
- ❌ No `import *`
- ❌ No mutable default arguments
- ❌ No hardcoded paths
- ❌ No TODO without issue reference
