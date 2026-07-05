#!/usr/bin/env python3
"""Run local quality checks for SubtitleTools."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_CHECK_DIRS: tuple[str, ...] = ("src", "tests", "scripts")
_VERIFY_SCRIPT = Path("scripts") / "verify.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _python_m(module: str, *module_args: str) -> list[str]:
    """Build a ``sys.executable -m module`` command (works on Windows and Unix)."""
    return [sys.executable, "-m", module, *module_args]


def _run_step(name: str, args: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess step; raise SystemExit on non-zero exit code."""
    print(f"==> {name}")
    result = subprocess.run(
        list(args),
        cwd=cwd if cwd is not None else _repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {name} (exit code {result.returncode})")


def _autopep8_args(*, fix: bool) -> list[str]:
    args = _python_m("autopep8", "--select=W291,W293", "-r", *_CHECK_DIRS)
    mode_flag = "--in-place" if fix else "--diff"
    args.insert(3, mode_flag)
    return args


def _isort_args(*, fix: bool) -> list[str]:
    args = _python_m("isort", *_CHECK_DIRS)
    if not fix:
        args.insert(3, "--check-only")
    return args


def _pytest_args() -> list[str]:
    """Build pytest command, enabling xdist when available."""
    args = _python_m("pytest", "-m", "not integration")
    if importlib.util.find_spec("xdist") is not None:
        args.extend(["-n", "auto"])
    return args


def main() -> None:
    """Execute formatting, linting, security, and test checks."""
    parser = argparse.ArgumentParser(description="Run SubtitleTools quality checks.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply autopep8 and isort fixes before running checks",
    )
    args = parser.parse_args()

    root = _repo_root()
    pylint_report = root / "pylint-report.txt"
    verify_script = str(_VERIFY_SCRIPT)

    steps: list[tuple[str, list[str]]] = [
        ("autopep8 (trailing whitespace)", _autopep8_args(fix=args.fix)),
        ("isort", _isort_args(fix=args.fix)),
        ("black", _python_m("black", "--check", *_CHECK_DIRS)),
        (
            "mypy",
            _python_m(
                "mypy",
                "src/subtitletools",
                "tests",
                verify_script,
                "scripts/generate_file_version_info.py",
            ),
        ),
        (
            "pylint (package)",
            _python_m("pylint", "src/subtitletools", f"--output={pylint_report}"),
        ),
        ("pylint (verify)", _python_m("pylint", verify_script)),
        (
            "bandit",
            _python_m(
                "bandit", "-r", "src/subtitletools", "-q", "-c", "pyproject.toml"
            ),
        ),
        (
            "pip-audit",
            _python_m("pip_audit", "-r", "requirements-dev.txt"),
        ),
        ("pytest", _pytest_args()),
    ]

    for name, step_args in steps:
        _run_step(name, step_args, cwd=root)

    print("All verification steps passed.")


if __name__ == "__main__":
    main()
