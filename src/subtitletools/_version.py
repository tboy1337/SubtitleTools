"""Package version resolution."""

import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE_NAME = "subtitletools"


def _pyproject_path() -> Path:
    """Return pyproject.toml for source trees or PyInstaller bundles."""
    if getattr(sys, "frozen", False):
        bundled = Path(getattr(sys, "_MEIPASS", "")) / "pyproject.toml"
        if bundled.is_file():
            return bundled
    return Path(__file__).resolve().parent.parent.parent / "pyproject.toml"


@lru_cache(maxsize=1)
def _fallback_version() -> str:
    """Read version from pyproject.toml when the package is not installed."""
    pyproject = _pyproject_path()
    if not pyproject.is_file():
        return "unknown"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version = "):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    return "unknown"


def get_version() -> str:
    """Return the package version, preferring pyproject.toml when available."""
    if _pyproject_path().is_file():
        pyproject_version = _fallback_version()
        if pyproject_version != "unknown":
            return pyproject_version
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _fallback_version()
