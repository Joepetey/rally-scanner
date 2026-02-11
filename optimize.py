#!/usr/bin/env python
"""Backward-compatibility shim. See src/rally/optimize.py."""
from rally.optimize import main

if __name__ == "__main__":
    main()
