"""Artifact size audit — the hard gate of PRD section 10.

One definition, used by three consumers: the CLI (`audit_size.py`), the HTTP
health endpoint, and the test suite (`tests/test_size.py`). All three must agree,
otherwise "under 125 MB" would be a claim about one thing and a measurement of
another.

What is counted (conservative — nothing is excluded because it is inconvenient):

1. **Runtime dependencies** — the whole `site-packages` of the running
   interpreter, *including* build tooling (`pip`, `setuptools`) and byte-code.
2. **Application code** — `src/` and `tests/`.
3. **Static assets** — `src/sigma/api/static`.
4. **Runtime data** — the checkpoint database at its worst allowed size.
"""

from __future__ import annotations

import sysconfig
from dataclasses import dataclass, field
from pathlib import Path

from sigma.config import SETTINGS
from sigma.graph.checkpointer import database_size_bytes

APP_ROOT = Path(__file__).resolve().parents[3]  # apps/sigma
MB = 1024 * 1024


@dataclass
class ArtifactReport:
    """Measured sizes in megabytes."""

    dependencies_mb: float = 0.0
    code_mb: float = 0.0
    assets_mb: float = 0.0
    database_mb: float = 0.0
    total_mb: float = 0.0
    budget_mb: float = SETTINGS.size_budget_mb
    passed: bool = True
    largest: list[tuple[str, float]] = field(default_factory=list)

    @property
    def headroom_mb(self) -> float:
        return round(self.budget_mb - self.total_mb, 2)

    @property
    def headroom_pct(self) -> float:
        return round(100.0 * self.headroom_mb / self.budget_mb, 1) if self.budget_mb else 0.0


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file() and not file_path.is_symlink():
            try:
                total += file_path.stat().st_size
            except OSError:  # pragma: no cover - transient files
                pass
    return total


def site_packages_dir() -> Path:
    """`site-packages` of the running interpreter (works inside and outside a venv)."""
    return Path(sysconfig.get_paths()["purelib"])


def measure(*, budget_mb: float | None = None,
            include_dependencies: bool = True) -> ArtifactReport:
    """Measure every artifact class and compare it against the budget."""
    limit = SETTINGS.size_budget_mb if budget_mb is None else budget_mb

    packages = site_packages_dir()
    dependencies = _dir_bytes(packages) if include_dependencies else 0
    code = _dir_bytes(APP_ROOT / "src") + _dir_bytes(APP_ROOT / "tests")
    assets = _dir_bytes(APP_ROOT / "src" / "sigma" / "api" / "static")
    database = database_size_bytes()

    largest: list[tuple[str, float]] = []
    if include_dependencies and packages.exists():
        entries = []
        for child in packages.iterdir():
            try:
                entries.append((child.name, _dir_bytes(child) if child.is_dir()
                                else child.stat().st_size))
            except OSError:  # pragma: no cover
                continue
        largest = sorted(((n, round(b / MB, 2)) for n, b in entries),
                         key=lambda item: item[1], reverse=True)[:8]

    total = dependencies + code + assets + database
    report = ArtifactReport(
        dependencies_mb=round(dependencies / MB, 2),
        code_mb=round(code / MB, 2),
        assets_mb=round(assets / MB, 2),
        database_mb=round(database / MB, 2),
        total_mb=round(total / MB, 2),
        budget_mb=limit,
        largest=largest,
    )
    report.passed = report.total_mb < limit
    return report


def render(report: ArtifactReport) -> str:
    """Human-readable audit output."""
    verdict = "PASS" if report.passed else "FAIL"
    lines = [
        "SIGMA artifact audit",
        "-" * 58,
        f"{'component':<28}{'size (MB)':>12}",
        f"{'runtime dependencies':<28}{report.dependencies_mb:>12.2f}",
        f"{'application code':<28}{report.code_mb:>12.2f}",
        f"{'static assets':<28}{report.assets_mb:>12.2f}",
        f"{'checkpoint database':<28}{report.database_mb:>12.2f}",
        "-" * 58,
        f"{'TOTAL':<28}{report.total_mb:>12.2f}",
        f"{'BUDGET':<28}{report.budget_mb:>12.2f}",
        f"{'HEADROOM':<28}{report.headroom_mb:>12.2f}  ({report.headroom_pct}%)",
        "-" * 58,
    ]
    if report.largest:
        lines.append("largest installed packages:")
        lines.extend(f"  {name:<26}{size:>8.2f} MB" for name, size in report.largest)
        lines.append("-" * 58)
    lines.append(f"RESULT: {verdict} — total {report.total_mb:.2f} MB "
                 f"{'<':>1} budget {report.budget_mb:.0f} MB"
                 if report.passed else
                 f"RESULT: {verdict} — total {report.total_mb:.2f} MB "
                 f"EXCEEDS budget {report.budget_mb:.0f} MB")
    return "\n".join(lines)


__all__ = ["APP_ROOT", "ArtifactReport", "measure", "render", "site_packages_dir"]
