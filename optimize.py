#!/usr/bin/env python
"""Backward-compatibility shim. See src/rally/backtest/optimize.py."""
from rally.backtest.optimize import main

if __name__ == "__main__":
    main()
