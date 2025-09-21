#!/usr/bin/env python3
"""Main entry point for SubtitleTools package when run as a module.

This allows the package to be executed as:
    python -m subtitletools [args...]

It delegates to the CLI module for argument parsing and command execution.
"""

import sys
from typing import List, Optional

from .cli import main


def main_entry(args: Optional[List[str]] = None) -> int:
    """Main entry point for the SubtitleTools CLI.

    Args:
        args: Command line arguments. If None, uses sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    if args is None:
        args = sys.argv[1:]

    return main(args)


if __name__ == "__main__":
    sys.exit(main_entry())
