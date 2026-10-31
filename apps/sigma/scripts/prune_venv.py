#!/usr/bin/env python3
"""Dependency pruning for the SIGMA runtime venv.

Runs *after* `bootstrap.sh` and removes everything that is not needed to execute
the compiled LangGraph application:

1. Packages that only exist to run tests / build wheels (pytest, pygments, ...).
2. Bundled `tests/` directories shipped inside third-party distributions.
3. `__pycache__` bytecode caches (regenerated lazily on first import).

Everything removed here is rebuildable; nothing removed is imported at runtime.
The companion `audit_size.py` verifies the result against the 125 MB budget.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Removed unconditionally: they are build/test-time only and are imported by
# neither `sigma`, `langgraph`, `fastapi` nor `uvicorn` at runtime.
BUILD_AND_TEST_ONLY = (
    "pytest",
    "pluggy",
    "iniconfig",
    "pygments",
    "zstandard",  # langsmith tracing compression; unused when tracing is off
    "wheel",
)

# Byte-code caches and bundled test suites inside site-packages.
PRUNABLE_DIR_NAMES = {"__pycache__", "tests", "test", "_internal_testing"}


def _site_packages(venv: Path) -> Path:
    for candidate in sorted(venv.glob("lib/python*/site-packages")):
        return candidate
    raise SystemExit(f"no site-packages found under {venv}")


def prune(venv: Path, *, drop_pycache: bool = True) -> dict[str, int]:
    sp = _site_packages(venv)
    stats = {"uninstalled": 0, "dirs": 0, "bytes_freed": 0}

    existing = [
        pkg
        for pkg in BUILD_AND_TEST_ONLY
        if any(sp.glob(f"{pkg}*")) or (sp / pkg).exists()
    ]
    if existing:
        before = _dir_size(sp)
        subprocess.run(
            [str(venv / "bin" / "python"), "-m", "pip", "uninstall", "-y", "-q", *existing],
            check=False,
            capture_output=True,
        )
        stats["uninstalled"] = len(existing)
        stats["bytes_freed"] += max(0, before - _dir_size(sp))

    targets = {"__pycache__"} if drop_pycache else set()
    targets |= {"tests", "test", "_internal_testing"}
    for root, dirnames, _filenames in os.walk(sp):
        for name in list(dirnames):
            if name in targets:
                path = Path(root) / name
                stats["bytes_freed"] += _dir_size(path)
                shutil.rmtree(path, ignore_errors=True)
                stats["dirs"] += 1
                dirnames.remove(name)
    return stats


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = Path(root) / name
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    venv = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/sigma-venv")
    if not venv.is_dir():
        raise SystemExit(f"venv not found: {venv}")
    stats = prune(venv)
    mb = stats["bytes_freed"] / 1024 / 1024
    print(
        f"[prune] removed {stats['uninstalled']} build/test distributions and "
        f"{stats['dirs']} cache/test dirs -> {mb:.1f} MB freed"
    )
    print(f"[prune] venv is now {_dir_size(venv) / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
