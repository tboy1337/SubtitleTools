"""Tests for the common utilities module."""

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from subtitletools.utils.common import (
    ThreadSafeCounter,
    ensure_directory,
    format_timestamp,
    get_file_size_mb,
    get_system_info,
    is_audio_file,
    is_subtitle_file,
    is_video_file,
    safe_filename,
    setup_logging,
    validate_directory_exists,
    validate_file_exists,
)


class TestFormatTimestamp:
    """Test format_timestamp function."""

    def test_format_timestamp_zero(self) -> None:
        """Test formatting timestamp for zero seconds."""
        result = format_timestamp(0.0)
        assert result == "00:00:00,000"

    def test_format_timestamp_seconds(self) -> None:
        """Test formatting timestamp for seconds only."""
        result = format_timestamp(30.5)
        assert result == "00:00:30,500"

    def test_format_timestamp_minutes(self) -> None:
        """Test formatting timestamp with minutes."""
        result = format_timestamp(90.123)
        assert result == "00:01:30,123"

    def test_format_timestamp_hours(self) -> None:
        """Test formatting timestamp with hours."""
        result = format_timestamp(3661.456)  # 1 hour, 1 minute, 1 second, 456ms
        assert result == "01:01:01,456"

    def test_format_timestamp_large_time(self) -> None:
        """Test formatting timestamp for large time values."""
        result = format_timestamp(7322.789)  # 2:02:02,789
        assert result == "02:02:02,789"

    def test_format_timestamp_microseconds_rounding(self) -> None:
        """Test that microseconds are properly rounded to milliseconds."""
        result = format_timestamp(1.0006)  # 6000 microseconds = 6 milliseconds
        assert result == "00:00:01,000"  # Should round down

        result = format_timestamp(1.0009)  # 9000 microseconds = 9 milliseconds
        assert result == "00:00:01,000"  # Should round down


class TestSetupLogging:
    """Test setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test setup_logging with default parameters."""
        logger = setup_logging()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "subtitletools.utils.common"

    def test_setup_logging_with_level(self) -> None:
        """Test setup_logging with custom level."""
        logger = setup_logging(level=logging.DEBUG)

        # Check that root logger level was set
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_file(self) -> None:
        """Test setup_logging with file handler."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp:
            log_file = tmp.name

        try:
            logger = setup_logging(log_file=log_file)

            # Log something to test file handler
            logger.info("Test message")

            # Close all handlers to release file locks
            for handler in logging.getLogger().handlers:
                handler.close()
            logging.getLogger().handlers.clear()

            # Check file was created and has content
            assert Path(log_file).exists()
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "Test message" in content
        finally:
            Path(log_file).unlink(missing_ok=True)

    def test_setup_logging_with_custom_format(self) -> None:
        """Test setup_logging with custom format string."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logging(format_string=custom_format)

        # We can't easily test the format without capturing output,
        # but we can verify the function accepts the parameter
        assert isinstance(logger, logging.Logger)

    def test_setup_logging_pathlib_path(self) -> None:
        """Test setup_logging with pathlib Path for log file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp:
            log_file = Path(tmp.name)

        try:
            logger = setup_logging(log_file=log_file)
            logger.info("Test message")

            # Close all handlers to release file locks
            for handler in logging.getLogger().handlers:
                handler.close()
            logging.getLogger().handlers.clear()

            assert log_file.exists()
        finally:
            log_file.unlink(missing_ok=True)


class TestEnsureDirectory:
    """Test ensure_directory function."""

    def test_ensure_directory_new(self) -> None:
        """Test creating a new directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir = Path(tmp_dir) / "new_directory"

            result = ensure_directory(new_dir)

            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()

    def test_ensure_directory_existing(self) -> None:
        """Test with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_dir = Path(tmp_dir)

            result = ensure_directory(existing_dir)

            assert result == existing_dir
            assert existing_dir.exists()

    def test_ensure_directory_nested(self) -> None:
        """Test creating nested directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "level1" / "level2" / "level3"

            result = ensure_directory(nested_dir)

            assert result == nested_dir
            assert nested_dir.exists()
            assert nested_dir.is_dir()

    def test_ensure_directory_string_path(self) -> None:
        """Test with string path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_dir_str = os.path.join(tmp_dir, "string_dir")

            result = ensure_directory(new_dir_str)

            assert isinstance(result, Path)
            assert result.exists()


class TestGetFileSizeMb:
    """Test get_file_size_mb function."""

    def test_get_file_size_mb_small_file(self) -> None:
        """Test getting size of small file."""
        with tempfile.NamedTemporaryFile() as tmp:
            # Write some content
            content = b"Hello, world!"
            tmp.write(content)
            tmp.flush()

            size_mb = get_file_size_mb(tmp.name)

            # Size should be very small (bytes -> MB)
            expected_size = len(content) / (1024 * 1024)
            assert abs(size_mb - expected_size) < 0.001

    def test_get_file_size_mb_pathlib_path(self) -> None:
        """Test getting size with pathlib Path."""
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"test content")
            tmp.flush()

            size_mb = get_file_size_mb(Path(tmp.name))

            assert size_mb > 0

    def test_get_file_size_mb_nonexistent_file(self) -> None:
        """Test getting size of non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_file_size_mb("nonexistent_file.txt")


class TestFileTypeCheckers:
    """Test is_video_file, is_audio_file, is_subtitle_file functions."""

    def test_is_video_file_true(self) -> None:
        """Test video file detection with valid extensions."""
        assert is_video_file("movie.mp4") is True
        assert is_video_file("video.avi") is True
        assert is_video_file("clip.mov") is True
        assert is_video_file("test.mkv") is True
        assert is_video_file("VIDEO.MP4") is True  # Case insensitive

    def test_is_video_file_false(self) -> None:
        """Test video file detection with invalid extensions."""
        assert is_video_file("audio.mp3") is False
        assert is_video_file("subtitle.srt") is False
        assert is_video_file("document.txt") is False
        assert is_video_file("image.jpg") is False

    def test_is_video_file_pathlib_path(self) -> None:
        """Test video file detection with pathlib Path."""
        assert is_video_file(Path("movie.mp4")) is True
        assert is_video_file(Path("audio.mp3")) is False

    def test_is_audio_file_true(self) -> None:
        """Test audio file detection with valid extensions."""
        assert is_audio_file("song.mp3") is True
        assert is_audio_file("audio.wav") is True
        assert is_audio_file("music.flac") is True
        assert is_audio_file("sound.aac") is True
        assert is_audio_file("SONG.MP3") is True  # Case insensitive

    def test_is_audio_file_false(self) -> None:
        """Test audio file detection with invalid extensions."""
        assert is_audio_file("video.mp4") is False
        assert is_audio_file("subtitle.srt") is False
        assert is_audio_file("document.txt") is False

    def test_is_subtitle_file_true(self) -> None:
        """Test subtitle file detection with valid extensions."""
        assert is_subtitle_file("subtitle.srt") is True
        assert is_subtitle_file("captions.vtt") is True
        assert is_subtitle_file("text.ass") is True
        assert is_subtitle_file("SUBTITLE.SRT") is True  # Case insensitive

    def test_is_subtitle_file_false(self) -> None:
        """Test subtitle file detection with invalid extensions."""
        assert is_subtitle_file("video.mp4") is False
        assert is_subtitle_file("audio.mp3") is False
        assert is_subtitle_file("document.txt") is False

    def test_file_type_checkers_no_extension(self) -> None:
        """Test file type checkers with files that have no extension."""
        assert is_video_file("filename") is False
        assert is_audio_file("filename") is False
        assert is_subtitle_file("filename") is False

    def test_file_type_checkers_multiple_dots(self) -> None:
        """Test file type checkers with multiple dots in filename."""
        assert is_video_file("my.movie.file.mp4") is True
        assert is_audio_file("my.audio.file.mp3") is True
        assert is_subtitle_file("my.subtitle.file.srt") is True


class TestValidateFileExists:
    """Test validate_file_exists function."""

    def test_validate_file_exists_success(self) -> None:
        """Test validating existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = tmp.name

        try:
            result = validate_file_exists(tmp_path)

            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_file()
            assert str(result) == tmp_path
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_file_exists_pathlib_path(self) -> None:
        """Test validating file with pathlib Path."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            tmp_path = Path(tmp.name)

        try:
            result = validate_file_exists(tmp_path)
            assert result == tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_validate_file_exists_empty_path(self) -> None:
        """Test validating empty path."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            validate_file_exists("")

    def test_validate_file_exists_none_path(self) -> None:
        """Test validating None path."""
        with pytest.raises(ValueError, match="File path cannot be empty"):
            validate_file_exists(None)  # type: ignore[arg-type]

    def test_validate_file_exists_nonexistent(self) -> None:
        """Test validating non-existent file."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            validate_file_exists("nonexistent_file.txt")

    def test_validate_file_exists_directory(self) -> None:
        """Test validating directory instead of file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(ValueError, match="Path is not a file"):
                validate_file_exists(tmp_dir)


class TestValidateDirectoryExists:
    """Test validate_directory_exists function."""

    def test_validate_directory_exists_success(self) -> None:
        """Test validating existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = validate_directory_exists(tmp_dir)

            assert isinstance(result, Path)
            assert result.exists()
            assert result.is_dir()

    def test_validate_directory_exists_pathlib_path(self) -> None:
        """Test validating directory with pathlib Path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = validate_directory_exists(tmp_path)
            assert result == tmp_path

    def test_validate_directory_exists_empty_path(self) -> None:
        """Test validating empty path."""
        with pytest.raises(ValueError, match="Directory path cannot be empty"):
            validate_directory_exists("")

    def test_validate_directory_exists_nonexistent(self) -> None:
        """Test validating non-existent directory."""
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            validate_directory_exists("nonexistent_directory")

    def test_validate_directory_exists_file(self) -> None:
        """Test validating file instead of directory."""
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(ValueError, match="Path is not a directory"):
                validate_directory_exists(tmp.name)


class TestSafeFilename:
    """Test safe_filename function."""

    def test_safe_filename_clean(self) -> None:
        """Test safe filename with already clean name."""
        result = safe_filename("normal_filename.txt")
        assert result == "normal_filename.txt"

    def test_safe_filename_invalid_chars(self) -> None:
        """Test safe filename with invalid characters."""
        result = safe_filename("file<name>with:invalid/chars\\and|more?stuff*.txt")
        assert result == "file_name_with_invalid_chars_and_more_stuff_.txt"

    def test_safe_filename_custom_replacement(self) -> None:
        """Test safe filename with custom replacement character."""
        result = safe_filename("invalid:chars", replacement="-")
        assert result == "invalid-chars"

    def test_safe_filename_leading_trailing_whitespace(self) -> None:
        """Test safe filename removes leading/trailing whitespace and dots."""
        result = safe_filename("  ...filename...  ")
        assert result == "filename"

    def test_safe_filename_empty_result(self) -> None:
        """Test safe filename when result would be empty."""
        result = safe_filename("   ...   ")
        assert result == "unnamed"

        # Characters get replaced with underscores, not empty string
        result = safe_filename("<>|")
        assert result == "___"

        # Truly empty string becomes "unnamed"
        result = safe_filename("")
        assert result == "unnamed"

    def test_safe_filename_unicode(self) -> None:
        """Test safe filename with unicode characters."""
        result = safe_filename("файл中文.txt")
        assert result == "файл中文.txt"  # Unicode should be preserved


class TestThreadSafeCounter:
    """Test ThreadSafeCounter class."""

    def test_thread_safe_counter_init_default(self) -> None:
        """Test ThreadSafeCounter initialization with default value."""
        counter = ThreadSafeCounter()
        assert counter.get() == 0

    def test_thread_safe_counter_init_custom(self) -> None:
        """Test ThreadSafeCounter initialization with custom value."""
        counter = ThreadSafeCounter(10)
        assert counter.get() == 10

    def test_thread_safe_counter_increment(self) -> None:
        """Test counter increment."""
        counter = ThreadSafeCounter()

        result = counter.increment()
        assert result == 1
        assert counter.get() == 1

        result = counter.increment()
        assert result == 2
        assert counter.get() == 2

    def test_thread_safe_counter_decrement(self) -> None:
        """Test counter decrement."""
        counter = ThreadSafeCounter(5)

        result = counter.decrement()
        assert result == 4
        assert counter.get() == 4

        result = counter.decrement()
        assert result == 3
        assert counter.get() == 3

    def test_thread_safe_counter_set(self) -> None:
        """Test setting counter value."""
        counter = ThreadSafeCounter()

        counter.set(100)
        assert counter.get() == 100

        counter.set(-5)
        assert counter.get() == -5

    def test_thread_safe_counter_reset(self) -> None:
        """Test resetting counter."""
        counter = ThreadSafeCounter(42)

        counter.reset()
        assert counter.get() == 0

    def test_thread_safe_counter_threading(self) -> None:
        """Test counter thread safety."""
        counter = ThreadSafeCounter()
        results = []

        def worker():
            for _ in range(100):
                results.append(counter.increment())

        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Final counter value should be 500 (5 threads * 100 increments)
        assert counter.get() == 500
        # All increment results should be unique (no race conditions)
        assert len(set(results)) == 500


class TestGetSystemInfo:
    """Test get_system_info function."""

    def test_get_system_info_basic(self) -> None:
        """Test basic system info retrieval."""
        info = get_system_info()

        assert isinstance(info, dict)
        assert "platform" in info
        assert "python_version" in info
        assert "cwd" in info
        assert "torch_version" in info
        assert "cuda_available" in info

        assert info["platform"] == sys.platform
        assert info["python_version"] == sys.version
        assert info["cwd"] == os.getcwd()

    def test_get_system_info_with_torch(self) -> None:
        """Test system info with torch available."""
        # Mock the import to control torch behavior
        with patch.dict(sys.modules):
            mock_torch = Mock()
            mock_torch.__version__ = "1.12.0"
            mock_torch.cuda.is_available.return_value = True
            sys.modules['torch'] = mock_torch

            info = get_system_info()

            assert info["torch_version"] == "1.12.0"
            assert info["cuda_available"] is True

    def test_get_system_info_no_torch(self) -> None:
        """Test system info without torch (ImportError)."""
        # Mock import to raise ImportError
        with patch.dict(sys.modules, {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError("No module named 'torch'")):
                info = get_system_info()

                assert info["torch_version"] == "Not installed"
                assert info["cuda_available"] is False

    @patch('os.getcwd')
    def test_get_system_info_cwd_exception(self, mock_getcwd: Mock) -> None:
        """Test system info when getcwd raises an exception."""
        # This is more for completeness - the function doesn't handle exceptions
        # but we can test that os.getcwd is called
        mock_getcwd.return_value = "/test/directory"

        info = get_system_info()

        assert info["cwd"] == "/test/directory"
        mock_getcwd.assert_called_once()
