# Code Review Guidelines

This document describes how to perform code quality checks on the NNPID project.

## Quick Start

```bash
# Full code review (recommended before commits)
make check

# Quick lint check
make lint

# Format code
make format
```

## Available Tools

### 1. Ruff - Linting & Formatting

Fast Python linter and formatter that replaces flake8, isort, and black.

```bash
# Check for issues
ruff check src/ scripts/

# Auto-fix issues
ruff check --fix src/ scripts/

# Format code
ruff format src/ scripts/
```

**Rules enabled:**
- `E` - pycodestyle errors
- `F` - pyflakes (unused imports, undefined names)
- `I` - isort (import sorting)
- `N` - pep8-naming
- `W` - pycodestyle warnings
- `UP` - pyupgrade (modern Python syntax)
- `B` - flake8-bugbear (common bugs)
- `C4` - flake8-comprehensions
- `SIM` - flake8-simplify

### 2. Bandit - Security Scanning

Static security analysis for Python code.

```bash
# Run security scan
bandit -r src/ -ll

# Full scan with all severities
bandit -r src/ scripts/ -l
```

**Common issues detected:**
- Hardcoded passwords/tokens
- Unsafe deserialization (pickle, yaml.load)
- SQL injection risks
- Use of insecure functions

### 3. Vulture - Dead Code Detection

Finds unused code (functions, variables, imports).

```bash
# Find unused code with high confidence
vulture src/ scripts/ --min-confidence 80

# Find all potentially unused code
vulture src/ scripts/ --min-confidence 60
```

### 4. Radon - Complexity Analysis

Analyzes code complexity and maintainability.

```bash
# Cyclomatic complexity (functions rated A-F)
radon cc src/ -a -s

# Only show complex functions (C rating or worse)
radon cc src/ -a -s --min C

# Maintainability index
radon mi src/ -s
```

**Complexity Ratings:**
| Rating | Score | Meaning |
|--------|-------|---------|
| A | 1-5 | Simple, easy to maintain |
| B | 6-10 | Low complexity |
| C | 11-20 | Moderate complexity |
| D | 21-30 | High complexity - consider refactoring |
| E | 31-40 | Very high complexity - refactor |
| F | 41+ | Extremely complex - must refactor |

### 5. Mypy - Type Checking

Static type checker for Python.

```bash
# Check types
mypy src/ --ignore-missing-imports

# Strict mode
mypy src/ --strict --ignore-missing-imports
```

### 6. Pyright - Advanced Type Checking

Alternative type checker with better inference.

```bash
pyright src/
```

## Pre-Commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

Hooks configured:
- `ruff` - linting
- `ruff-format` - formatting
- `mypy` - type checking
- `bandit` - security
- File checks (trailing whitespace, YAML validation, etc.)

## CI/CD Integration

### GitHub Actions Workflow

The project includes `.github/workflows/` for:
- Linting on every push/PR
- Tests on Python 3.10, 3.11, 3.12
- Security scanning

### Manual Checks

Before submitting a PR:

```bash
# 1. Format code
make format

# 2. Run full checks
make check

# 3. Run tests
make test

# 4. Verify no issues
git diff --stat  # Review changes
```

## Fixing Common Issues

### Import Sorting (I001)

```bash
ruff check --fix --select I src/
```

### Unused Imports (F401)

```bash
ruff check --fix --select F401 src/
```

### Type Annotations (UP035, UP045)

Replace old-style annotations:
```python
# Before
from typing import Dict, List, Optional
def foo(x: Optional[str]) -> Dict[str, List[int]]: ...

# After
def foo(x: str | None) -> dict[str, list[int]]: ...
```

### Bare Except (E722)

Always specify exception type:
```python
# Before
try:
    risky_operation()
except:
    pass

# After
try:
    risky_operation()
except Exception:
    pass
```

### Security: Unsafe torch.load (B614)

```python
# Before (unsafe - allows arbitrary code execution)
checkpoint = torch.load(model_path)

# After (safe - only loads tensor data)
checkpoint = torch.load(model_path, weights_only=True)
```

## Configuration

All tool configurations are in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]
ignore = ["E501", "E402", "B008", "N803", "N806"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true

[tool.bandit]
exclude_dirs = ["tests", "scripts"]
skips = ["B101", "B311"]
```

## Quality Thresholds

### Minimum Requirements

| Metric | Threshold |
|--------|-----------|
| Ruff errors | 0 |
| Bandit high severity | 0 |
| Complexity (avg) | < 10 |
| Test coverage | > 80% |

### Recommended

| Metric | Target |
|--------|--------|
| Ruff warnings | 0 |
| Bandit all | 0 |
| Complexity (max) | < 20 |
| Test coverage | > 90% |

## Summary

```bash
# Daily workflow
make format && make check && make test

# Before commit
pre-commit run

# Before PR
make check && make test-cov
```

For questions, see [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
