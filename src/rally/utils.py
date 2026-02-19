"""Shared utility functions."""

import json
import tempfile
from pathlib import Path


def atomic_json_write(path: Path, data: dict, **json_kwargs) -> None:
    """Write JSON atomically via temp file + rename.

    Writes to a temporary file in the same directory, then atomically
    replaces the target. This prevents partial writes on crash.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".tmp", prefix=path.stem + "_",
    )
    try:
        with open(fd, "w") as f:
            json.dump(data, f, indent=2, **json_kwargs)
        Path(tmp_path).replace(path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
