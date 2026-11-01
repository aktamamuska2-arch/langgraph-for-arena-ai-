"""Isolated execution harness used by ``node_sandbox_tester``.

Isolation properties
--------------------
* fresh interpreter with ``-I`` (no user site-packages, no ``PYTHONPATH``,
  no environment-driven import hijacking) and ``-B`` (no bytecode writes)
* ``cwd`` pinned to the build directory, so relative writes stay contained
* CPU-time, address-space and file-descriptor rlimits via ``preexec_fn``
* wall-clock timeout with hard kill
* a scrubbed environment (no proxy variables, no credentials)
"""

from __future__ import annotations

import json
import os
import re
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RESULT_PREFIX = "__RESULT__"
TRACEBACK_TAIL_CHARS = 4000
MEMORY_LIMIT_BYTES = 3 * 1024 * 1024 * 1024
FILE_LIMIT = 256


@dataclass
class SandboxOutcome:
    exit_code: int
    timed_out: bool
    duration_seconds: float
    payload: dict[str, Any] | None
    stdout: str
    stderr: str
    traceback_summary: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.payload is not None and bool(self.payload.get("ok"))


def _limits(timeout: float):
    cpu = max(5, int(timeout) + 5)

    def _apply() -> None:  # pragma: no cover - runs in the child process
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (FILE_LIMIT, FILE_LIMIT))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


def _scrubbed_env() -> dict[str, str]:
    keep = {"PATH", "LANG", "LC_ALL", "TZ"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    return env


def _summarise_traceback(stderr: str) -> str:
    """Extract the raising exception line from a captured traceback."""

    if not stderr.strip():
        return ""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Warning|Interrupt)\b", line.strip()):
            return line.strip()[:400]
    return lines[-1][:400] if lines else ""


def execute_build(
    build_dir: Path,
    entrypoint: str = "generated_simulation.py",
    timeout: float = 90.0,
) -> SandboxOutcome:
    script = (build_dir / entrypoint).resolve()
    if not script.exists():
        return SandboxOutcome(
            exit_code=127,
            timed_out=False,
            duration_seconds=0.0,
            payload=None,
            stdout="",
            stderr=f"entrypoint not found: {script}",
            error="missing_entrypoint",
        )

    cmd = [sys.executable, "-I", "-B", str(script)]
    started = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(build_dir),
            env=_scrubbed_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limits(timeout),
            check=False,
        )
        stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        code = -9
    duration = time.perf_counter() - started

    payload: dict[str, Any] | None = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                payload = json.loads(line[len(RESULT_PREFIX) :])
            except json.JSONDecodeError:
                payload = None
            break

    return SandboxOutcome(
        exit_code=code,
        timed_out=timed_out,
        duration_seconds=duration,
        payload=payload,
        stdout=stdout[-TRACEBACK_TAIL_CHARS:],
        stderr=stderr[-TRACEBACK_TAIL_CHARS:],
        traceback_summary=_summarise_traceback(stderr),
    )
