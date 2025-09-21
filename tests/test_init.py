"""Tests for main __init__.py module."""

from typing import Any
from unittest.mock import patch


class TestMainInit:
    """Test the main __init__.py module."""

    def test_version_info(self) -> None:
        """Test that version information is available."""
        import subtitletools

        assert hasattr(subtitletools, "__version__")
        assert hasattr(subtitletools, "__author__")
        assert hasattr(subtitletools, "__email__")
        assert subtitletools.__version__ == "1.0.0"
        assert subtitletools.__author__ == "tboy1337"

    def test_successful_imports(self) -> None:
        """Test successful imports of main classes."""
        import subtitletools

        # Check that classes are imported successfully
        assert hasattr(subtitletools, "SubWhisperTranscriber")
        assert hasattr(subtitletools, "SubtitleTranslator")
        assert hasattr(subtitletools, "SubtitleProcessor")
        assert hasattr(subtitletools, "SubtitleWorkflow")

        # Check that they're in __all__
        assert "SubWhisperTranscriber" in subtitletools.__all__
        assert "SubtitleTranslator" in subtitletools.__all__
        assert "SubtitleProcessor" in subtitletools.__all__
        assert "SubtitleWorkflow" in subtitletools.__all__

    @patch("subtitletools.core.transcription.SubWhisperTranscriber")
    def test_import_error_handling(self, mock_transcriber: Any) -> None:
        """Test that ImportError is handled gracefully."""
        # Make the import fail
        mock_transcriber.side_effect = ImportError("Module not found")

        # Reload the module to trigger the import error path
        import importlib

        import subtitletools

        importlib.reload(subtitletools)

        # Should still have basic attributes
        assert hasattr(subtitletools, "__version__")
        assert hasattr(subtitletools, "__author__")

        # But not the imported classes
        basic_all = ["__version__", "__author__", "__email__"]
        for item in basic_all:
            assert item in subtitletools.__all__


class TestCoreInit:
    """Test the core __init__.py module."""

    def test_core_init_imports(self) -> None:
        """Test core module imports."""
        from subtitletools.core import (
            SubtitleProcessor,
            SubtitleTranslator,
            SubtitleWorkflow,
            SubWhisperTranscriber,
        )

        # Should be able to import all classes
        assert SubWhisperTranscriber is not None
        assert SubtitleTranslator is not None
        assert SubtitleProcessor is not None
        assert SubtitleWorkflow is not None

    def test_core_init_all_attribute(self) -> None:
        """Test that __all__ contains expected classes."""
        from subtitletools import core

        expected_classes = [
            "SubWhisperTranscriber",
            "SubtitleTranslator",
            "SubtitleProcessor",
            "SubtitleWorkflow",
        ]

        for class_name in expected_classes:
            assert class_name in core.__all__


class TestUtilsInit:
    """Test the utils __init__.py module."""

    def test_utils_init_imports(self) -> None:
        """Test utils module imports."""
        from subtitletools.utils import (
            apply_subtitle_edit_postprocess,
            detect_encoding,
            extract_audio,
            setup_logging,
        )

        # Should be able to import all functions
        assert extract_audio is not None
        assert detect_encoding is not None
        assert setup_logging is not None
        assert apply_subtitle_edit_postprocess is not None

    def test_utils_init_all_attribute(self) -> None:
        """Test that __all__ contains expected functions."""
        from subtitletools import utils

        expected_functions = [
            "extract_audio",
            "detect_encoding",
            "setup_logging",
            "apply_subtitle_edit_postprocess",
        ]

        for func_name in expected_functions:
            assert func_name in utils.__all__


class TestConfigInit:
    """Test the config __init__.py module."""

    def test_config_init_imports(self) -> None:
        """Test config module imports."""
        from subtitletools.config import (
            DEFAULT_ENCODING,
            DEFAULT_WHISPER_MODEL,
            get_config,
            set_config,
        )

        # Should be able to import all functions and constants
        assert get_config is not None
        assert set_config is not None
        assert DEFAULT_ENCODING is not None
        assert DEFAULT_WHISPER_MODEL is not None

    def test_config_init_all_attribute(self) -> None:
        """Test that __all__ contains expected functions and constants."""
        from subtitletools import config

        expected_items = [
            "get_config",
            "set_config",
            "DEFAULT_ENCODING",
            "DEFAULT_WHISPER_MODEL",
            "SUPPORTED_VIDEO_EXTENSIONS",
        ]

        for item_name in expected_items:
            assert item_name in config.__all__
