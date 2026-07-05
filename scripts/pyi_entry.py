"""PyInstaller entry point for the SubtitleTools Windows executable."""

import sys

from subtitletools.cli import main

if __name__ == "__main__":
    sys.exit(main())
