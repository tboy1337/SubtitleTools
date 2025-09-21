#!/usr/bin/env python3
"""Entry point script for SubtitleTools.

This script allows running SubtitleTools without installing it:
    python run.py [args...]

It delegates to the CLI module for argument parsing and command execution.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from subtitletools.cli import main
except ImportError as e:
    print(f"Error importing SubtitleTools: {e}", file=sys.stderr)
    print("Make sure you're running from the SubtitleTools directory and dependencies are installed.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
