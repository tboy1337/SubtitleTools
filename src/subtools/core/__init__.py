"""Core functionality modules for SubtitleTools.

This package contains the main processing modules:
- transcription: Audio/video to subtitle transcription using Whisper
- translation: Subtitle translation between languages
- subtitle: Subtitle file processing and manipulation
- workflow: End-to-end subtitle processing workflows
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

__all__: List[str] = []

# Import core classes for convenience
try:
    from .transcription import SubWhisperTranscriber
    from .translation import SubtitleTranslator
    from .subtitle import SubtitleProcessor
    from .workflow import SubtitleWorkflow

    # Export core processing classes
    __all__.extend([
        "SubWhisperTranscriber",  # Audio/video transcription
        "SubtitleTranslator",     # Language translation
        "SubtitleProcessor",      # Subtitle manipulation
        "SubtitleWorkflow",       # End-to-end workflows
    ])
except ImportError:
    # Allow partial imports during development
    logger.warning("Some core modules could not be imported")
