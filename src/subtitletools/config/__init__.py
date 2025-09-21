"""Configuration modules for SubtitleTools.

This package contains configuration settings and constants:
- settings: Application settings and configuration
"""

from typing import List

__all__: List[str] = []

# Import configuration for convenience
try:
    from .settings import (
        DEFAULT_WHISPER_MODEL,
        DEFAULT_ENCODING,
        DEFAULT_TIMEOUT,
        SUPPORTED_VIDEO_EXTENSIONS,
        SUPPORTED_AUDIO_EXTENSIONS,
        SUPPORTED_SUBTITLE_FORMATS,
        get_config,
        set_config,
    )

    __all__.extend([
        "DEFAULT_WHISPER_MODEL",
        "DEFAULT_ENCODING",
        "DEFAULT_TIMEOUT",
        "SUPPORTED_VIDEO_EXTENSIONS",
        "SUPPORTED_AUDIO_EXTENSIONS",
        "SUPPORTED_SUBTITLE_FORMATS",
        "get_config",
        "set_config",
    ])
except ImportError:
    # Allow partial imports during development
    pass
