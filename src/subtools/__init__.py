"""SubtitleTools: Complete subtitle workflow tool.

This package provides comprehensive subtitle processing capabilities including:
- Audio/video transcription using OpenAI Whisper
- Subtitle translation between languages
- Encoding conversion for subtitle files
- Post-processing and formatting
- Batch processing workflows

Main components:
- core.transcription: Audio/video to subtitle transcription
- core.translation: Subtitle translation between languages
- core.subtitle: Subtitle file processing and manipulation
- core.workflow: End-to-end subtitle workflows
- utils: Utility functions for audio, encoding, and post-processing
"""

from typing import List

__version__ = "1.0.0"
__author__ = "tboy1337"
__email__ = "tboy1337.unchanged733@aleeas.com"

# Public API exports
__all__: List[str] = [
    "__version__",
    "__author__",
    "__email__",
]

# Import main functionality for convenience
try:
    from .core.transcription import SubWhisperTranscriber
    from .core.translation import SubtitleTranslator
    from .core.subtitle import SubtitleProcessor
    from .core.workflow import SubtitleWorkflow

    # Main package exports
    __all__.extend(["SubWhisperTranscriber", "SubtitleTranslator", "SubtitleProcessor", "SubtitleWorkflow"])
except ImportError:
    # Allow partial imports during development/testing
    pass
