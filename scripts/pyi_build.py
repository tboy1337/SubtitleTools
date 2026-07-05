#!/usr/bin/env python3
"""Run PyInstaller for SubtitleTools with build-time noise reduced."""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "SubtitleTools.spec"


def main() -> int:
    """Execute PyInstaller against SubtitleTools.spec."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::DeprecationWarning,ignore::SyntaxWarning,ignore::UserWarning",
    )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--log-level",
        "ERROR",
        str(SPEC),
    ]
    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
