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

from typing import TYPE_CHECKING, List

from ._version import get_version

if TYPE_CHECKING:
    from .core.subtitle import SubtitleProcessor
    from .core.transcription import SubWhisperTranscriber
    from .core.translation import SubtitleTranslator
    from .core.workflow import SubtitleWorkflow

__version__ = get_version()
__author__ = "tboy1337"
__email__ = "tboy1337.unchanged733@aleeas.com"

# Public API exports
__all__: List[str] = [
    "__version__",
    "__author__",
    "__email__",
    "SubWhisperTranscriber",
    "SubtitleTranslator",
    "SubtitleProcessor",
    "SubtitleWorkflow",
]

_LAZY_EXPORTS = {
    "SubWhisperTranscriber": (".core.transcription", "SubWhisperTranscriber"),
    "SubtitleTranslator": (".core.translation", "SubtitleTranslator"),
    "SubtitleProcessor": (".core.subtitle", "SubtitleProcessor"),
    "SubtitleWorkflow": (".core.workflow", "SubtitleWorkflow"),
}


def __getattr__(name: str) -> object:
    """Lazy-load heavy core classes to avoid importing torch/whisper at import time."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        import importlib

        module = importlib.import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
