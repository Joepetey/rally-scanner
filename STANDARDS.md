# Market Rally Python Standards

Project-specific conventions for the rally phase-transition detector.

## Type Hints
- Use Python 3.11+ built-in types: `list[str]`, `dict[str, int]`, `tuple[int, ...]`, `X | None`
- Always annotate function signatures (parameters and return type)
- No `from __future__ import annotations` needed

## Import Organization
1. Standard library (`import json`, `from datetime import datetime`)
2. Third-party (`import pandas as pd`, `import numpy as np`)
3. Package-relative (`from .config import PARAMS`)

Separated by blank lines. Alphabetical within each group.

## Error Handling
- **Core modules** (features, model, trading, labels): let exceptions propagate
- **Orchestration** (orchestrator, scanner, retrain): catch and log with context
- Never silence exceptions with bare `except:` -- always catch specific types
- Use `logger.exception()` to capture tracebacks in orchestration code

## Logging
- Module-level logger: `logger = logging.getLogger(__name__)`
- `logger.info()` for operational messages (scan started, model saved)
- `logger.warning()` for degraded operations (VIX unavailable, stale model)
- `logger.error()` for failures that prevent normal operation
- Reserve `print()` for user-facing CLI output only (dashboard, backtest reports)

## Data Patterns
- DataFrames: `pd.DataFrame` with `DatetimeIndex`, tz-naive
- State files: JSON in `models/` directory
- History files: CSV in `models/` directory
- Model artifacts: joblib in `models/` directory

## Secrets
- Store in `.env` file (never in code or version control)
- Load via `python-dotenv` in entry points only
- Access via `os.environ.get(key, default)`

## Testing
- Test files mirror source: `rally/features.py` -> `tests/test_features.py`
- Use pytest fixtures for shared test data (see `tests/conftest.py`)
- Mock external APIs (yfinance) -- never call live APIs in tests
- Tests must run offline and complete in under 30 seconds total

## File Paths
- Use `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` in package modules
- Runtime directories (`models/`, `data_cache/`, `logs/`) live at project root
- All runtime directories are gitignored
