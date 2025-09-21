"""Utility modules for SubtitleTools.

This package contains utility functions and classes:
- audio: Audio extraction and processing utilities
- encoding: Text encoding conversion utilities
- postprocess: Subtitle post-processing utilities
- common: Common utility functions shared across modules
"""

from typing import List

__all__: List[str] = []

# Import utility functions for convenience
try:
    from .audio import extract_audio, find_ffmpeg
    from .encoding import convert_subtitle_encoding, detect_encoding
    from .postprocess import apply_subtitle_edit_postprocess
    from .common import format_timestamp, setup_logging

    __all__.extend([
        "extract_audio",
        "find_ffmpeg",
        "convert_subtitle_encoding",
        "detect_encoding",
        "apply_subtitle_edit_postprocess",
        "format_timestamp",
        "setup_logging",
    ])
except ImportError:
    # Allow partial imports during development
    pass
