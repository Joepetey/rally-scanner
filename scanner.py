#!/usr/bin/env python
"""Backward-compatibility shim. See src/rally/live/scanner.py."""
from rally.live.scanner import main

if __name__ == "__main__":
    main()
