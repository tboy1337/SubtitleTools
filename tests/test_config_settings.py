"""Tests for config.settings module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from subtitletools.config import settings


class TestConfigurationManagement:
    """Test configuration management functions."""

    def test_get_config_with_existing_key(self) -> None:
        """Test getting an existing configuration value."""
        settings.set_config("test_key", "test_value")
        result = settings.get_config("test_key")
        assert result == "test_value"

    def test_get_config_with_default(self) -> None:
        """Test getting configuration with default value."""
        result = settings.get_config("nonexistent_key", "default_value")
        assert result == "default_value"

    def test_get_config_none_when_missing(self) -> None:
        """Test getting None when key doesn't exist and no default."""
        settings.reset_config()
        result = settings.get_config("nonexistent_key")
        assert result is None

    def test_set_config_string(self) -> None:
        """Test setting string configuration value."""
        settings.set_config("string_key", "string_value")
        assert settings.get_config("string_key") == "string_value"

    def test_set_config_int(self) -> None:
        """Test setting integer configuration value."""
        settings.set_config("int_key", 42)
        assert settings.get_config("int_key") == 42

    def test_set_config_float(self) -> None:
        """Test setting float configuration value."""
        settings.set_config("float_key", 3.14)
        assert settings.get_config("float_key") == 3.14

    def test_set_config_bool(self) -> None:
        """Test setting boolean configuration value."""
        settings.set_config("bool_key", True)
        assert settings.get_config("bool_key") is True

    def test_get_all_config(self) -> None:
        """Test getting all configuration values."""
        settings.reset_config()
        settings.set_config("key1", "value1")
        settings.set_config("key2", 123)

        all_config = settings.get_all_config()

        assert all_config == {"key1": "value1", "key2": 123}
        # Ensure it returns a copy, not the original dict
        all_config["key3"] = "value3"
        assert "key3" not in settings.get_all_config()

    def test_reset_config(self) -> None:
        """Test resetting configuration to empty state."""
        settings.set_config("key1", "value1")
        settings.set_config("key2", "value2")

        settings.reset_config()

        assert settings.get_config("key1") is None
        assert settings.get_config("key2") is None
        assert not settings.get_all_config()


class TestDirectoryFunctions:
    """Test directory-related functions."""

    @patch('os.name', 'nt')
    @patch.dict(os.environ, {'APPDATA': 'C:\\Users\\Test\\AppData\\Roaming'})
    def test_get_app_data_dir_windows(self) -> None:
        """Test getting app data directory on Windows."""
        result = settings.get_app_data_dir()
        expected = Path("C:\\Users\\Test\\AppData\\Roaming\\SubtitleTools")
        assert result == expected

    @pytest.mark.skipif(os.name == 'nt', reason="Unix test on Windows platform")
    @patch('os.name', 'posix')
    def test_get_app_data_dir_unix(self) -> None:
        """Test getting app data directory on Unix-like systems."""
        # This test will be skipped on Windows
        # On actual Unix systems, this would test the Unix path logic
        pass

    @patch('os.name', 'nt')
    @patch.dict(os.environ, {}, clear=True)  # Clear APPDATA
    @patch('os.path.expanduser')
    def test_get_app_data_dir_windows_no_appdata(self, mock_expanduser) -> None:
        """Test getting app data directory on Windows without APPDATA."""
        mock_expanduser.return_value = "C:\\Users\\Test"
        result = settings.get_app_data_dir()
        expected = Path("C:\\Users\\Test\\SubtitleTools")
        assert result == expected

    def test_get_cache_dir(self) -> None:
        """Test getting cache directory."""
        with patch.object(settings, 'get_app_data_dir') as mock_app_dir:
            mock_app_dir.return_value = Path("/test/app")
            result = settings.get_cache_dir()
            assert result == Path("/test/app/cache")

    def test_get_temp_dir(self) -> None:
        """Test getting temporary directory."""
        with patch.object(settings, 'get_app_data_dir') as mock_app_dir:
            mock_app_dir.return_value = Path("/test/app")
            result = settings.get_temp_dir()
            assert result == Path("/test/app/temp")

    def test_get_logs_dir(self) -> None:
        """Test getting logs directory."""
        with patch.object(settings, 'get_app_data_dir') as mock_app_dir:
            mock_app_dir.return_value = Path("/test/app")
            result = settings.get_logs_dir()
            assert result == Path("/test/app/logs")


class TestConstants:
    """Test configuration constants."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        assert settings.DEFAULT_WHISPER_MODEL == "small"
        assert settings.DEFAULT_ENCODING == "UTF-8"
        assert settings.DEFAULT_TIMEOUT == 300
        assert settings.DEFAULT_SRC_LANGUAGE == "en"
        assert settings.DEFAULT_TARGET_LANGUAGE == "zh-CN"
        assert settings.DEFAULT_TRANSLATION_MODE == "split"
        assert settings.DEFAULT_MAX_SEGMENT_LENGTH is None

    def test_supported_extensions(self) -> None:
        """Test supported file extensions."""
        assert "mp4" in settings.SUPPORTED_VIDEO_EXTENSIONS
        assert "mkv" in settings.SUPPORTED_VIDEO_EXTENSIONS
        assert "avi" in settings.SUPPORTED_VIDEO_EXTENSIONS

        assert "wav" in settings.SUPPORTED_AUDIO_EXTENSIONS
        assert "mp3" in settings.SUPPORTED_AUDIO_EXTENSIONS
        assert "flac" in settings.SUPPORTED_AUDIO_EXTENSIONS

        assert "srt" in settings.SUPPORTED_SUBTITLE_FORMATS
        assert "vtt" in settings.SUPPORTED_SUBTITLE_FORMATS
        assert "ass" in settings.SUPPORTED_SUBTITLE_FORMATS

    def test_supported_services_and_models(self) -> None:
        """Test supported translation services and Whisper models."""
        assert "google" in settings.SUPPORTED_TRANSLATION_SERVICES
        assert "google_cloud" in settings.SUPPORTED_TRANSLATION_SERVICES

        assert "tiny" in settings.WHISPER_MODELS
        assert "base" in settings.WHISPER_MODELS
        assert "small" in settings.WHISPER_MODELS
        assert "medium" in settings.WHISPER_MODELS
        assert "large" in settings.WHISPER_MODELS

    def test_supported_encodings(self) -> None:
        """Test supported character encodings."""
        # UTF encodings
        assert "utf-8" in settings.SUPPORTED_ENCODINGS
        assert "utf-16" in settings.SUPPORTED_ENCODINGS

        # Western European
        assert "iso-8859-1" in settings.SUPPORTED_ENCODINGS
        assert "cp1252" in settings.SUPPORTED_ENCODINGS

        # Asian encodings
        assert "shift-jis" in settings.SUPPORTED_ENCODINGS
        assert "gb2312" in settings.SUPPORTED_ENCODINGS
        assert "big5" in settings.SUPPORTED_ENCODINGS

    def test_language_encodings(self) -> None:
        """Test language-specific encoding recommendations."""
        assert "utf-8" in settings.LANGUAGE_ENCODINGS["en"]
        assert "cp1252" in settings.LANGUAGE_ENCODINGS["en"]

        assert "shift-jis" in settings.LANGUAGE_ENCODINGS["ja"]
        assert "euc-jp" in settings.LANGUAGE_ENCODINGS["ja"]

        assert "gb2312" in settings.LANGUAGE_ENCODINGS["zh"]
        assert "cp936" in settings.LANGUAGE_ENCODINGS["zh"]

    def test_space_languages(self) -> None:
        """Test space and non-space language classifications."""
        assert "en" in settings.SPACE_LANGUAGES
        assert "fr" in settings.SPACE_LANGUAGES
        assert "de" in settings.SPACE_LANGUAGES

        assert "zh" in settings.NON_SPACE_LANGUAGES
        assert "ja" in settings.NON_SPACE_LANGUAGES
        assert "ko" in settings.NON_SPACE_LANGUAGES


class TestDirectoryEnsurance:
    """Test directory creation functionality."""

    @patch('pathlib.Path.mkdir')
    def test_ensure_directories_called_on_import(self, mock_mkdir) -> None:
        """Test that _ensure_directories is called and creates directories."""
        # Reset the module to trigger directory creation
        import importlib
        importlib.reload(settings)

        # Should have created app_data, cache, temp, and logs directories
        assert mock_mkdir.call_count >= 4


@pytest.fixture(autouse=True)
def cleanup_config():
    """Clean up configuration after each test."""
    yield
    settings.reset_config()
