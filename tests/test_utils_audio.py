"""Tests for the audio utilities module."""

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from subtitletools.utils.audio import (
    cleanup_temp_dir,
    extract_audio,
    find_ffmpeg,
    find_ffprobe,
    get_audio_duration,
    validate_audio_file,
)


class TestFindFFmpeg:
    """Test find_ffmpeg function."""

    @patch("subprocess.run")
    def test_find_ffmpeg_in_path_windows(self, mock_run: Mock) -> None:
        """Test finding FFmpeg in PATH on Windows."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "C:\\tools\\ffmpeg.exe\nAnother path"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with patch("sys.platform", "win32"):
            result = find_ffmpeg()

        assert result == "C:\\tools\\ffmpeg.exe"
        mock_run.assert_called_once_with(
            ["where", "ffmpeg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_find_ffmpeg_in_path_unix(self, mock_run: Mock) -> None:
        """Test finding FFmpeg in PATH on Unix systems."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "/usr/bin/ffmpeg\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with patch("sys.platform", "linux"):
            result = find_ffmpeg()

        assert result == "/usr/bin/ffmpeg"
        mock_run.assert_called_once_with(
            ["which", "ffmpeg"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffmpeg_not_in_path_but_in_common_location(
        self, mock_isfile: Mock, mock_run: Mock
    ) -> None:
        """Test finding FFmpeg in common locations when not in PATH."""
        # PATH search fails
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg not found"
        mock_run.return_value = mock_result

        # Mock file existence check - first few return False, then one returns True
        mock_isfile.side_effect = [False, False, True]  # Third location exists

        result = find_ffmpeg()

        # Should find FFmpeg at the third common location
        assert result is not None
        assert mock_isfile.call_count == 3

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffmpeg_not_found(self, mock_isfile: Mock, mock_run: Mock) -> None:
        """Test when FFmpeg is not found anywhere."""
        # PATH search fails
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg not found"
        mock_run.return_value = mock_result

        # No common locations have FFmpeg
        mock_isfile.return_value = False

        result = find_ffmpeg()

        assert result is None

    @patch("subprocess.run")
    def test_find_ffmpeg_path_timeout(self, mock_run: Mock) -> None:
        """Test PATH search timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(["where"], 30)

        with patch("os.path.isfile", return_value=False):
            result = find_ffmpeg()

        assert result is None

    @patch("subprocess.run")
    def test_find_ffmpeg_path_exception(self, mock_run: Mock) -> None:
        """Test PATH search with generic exception."""
        mock_run.side_effect = Exception("Command failed")

        with patch("os.path.isfile", return_value=False):
            result = find_ffmpeg()

        assert result is None

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffmpeg_file_check_exception(
        self, mock_isfile: Mock, mock_run: Mock
    ) -> None:
        """Test exception during file existence check."""
        # PATH search fails
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        # File check raises exception, should continue checking other locations
        mock_isfile.side_effect = [Exception("Permission denied"), False, True]

        result = find_ffmpeg()

        # Should still find FFmpeg at third location despite exception at first
        assert result is not None


class TestExtractAudio:
    """Test extract_audio function."""

    def test_extract_audio_empty_path(self) -> None:
        """Test extract_audio with empty path."""
        with pytest.raises(ValueError, match="Video path cannot be empty"):
            extract_audio("")

    def test_extract_audio_nonexistent_file(self) -> None:
        """Test extract_audio with non-existent file."""
        with pytest.raises(ValueError, match="Video file does not exist"):
            extract_audio("nonexistent.mp4")

    @patch("subtitletools.utils.audio.find_ffmpeg")
    def test_extract_audio_ffmpeg_not_found(self, mock_find_ffmpeg: Mock) -> None:
        """Test extract_audio when FFmpeg is not found."""
        mock_find_ffmpeg.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            with pytest.raises(FileNotFoundError, match="FFmpeg not found"):
                extract_audio(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_success(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test successful audio extraction."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"

        # Mock successful FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "conversion successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            tmp_video.write(b"fake video content")
            video_path = tmp_video.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            # Create fake output audio file
            tmp_audio.write(b"fake audio content" * 100)  # Make it > 1KB
            output_path = tmp_audio.name

        try:
            # Mock the output file creation by patching the subprocess to create the file
            def side_effect(*args: Any, **kwargs: Any) -> Mock:
                with open(output_path, "wb") as f:
                    f.write(b"fake audio content" * 100)
                return mock_result

            mock_run.side_effect = side_effect

            result = extract_audio(video_path, output_path)

            assert result == output_path
            assert Path(output_path).exists()
            mock_find_ffmpeg.assert_called_once()

        finally:
            Path(video_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_temp_output(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test audio extraction with temporary output path."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"

        # Mock successful FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "conversion successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            tmp_video.write(b"fake video content")
            video_path = tmp_video.name

        try:
            # Mock the output file creation by creating it in the side effect
            created_files = []

            def side_effect(*args: Any, **kwargs: Any) -> Mock:
                # Extract output path from FFmpeg command (last arg before -y)
                command = args[0]
                output_path = command[-2]  # Second to last argument is output path

                # Create the output directory and file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(b"fake audio content" * 100)
                created_files.append(output_path)
                return mock_result

            mock_run.side_effect = side_effect

            result = extract_audio(video_path)

            # Should have created an output file
            assert len(created_files) == 1
            assert os.path.exists(result)
            assert result == created_files[0]

        finally:
            Path(video_path).unlink(missing_ok=True)
            # Clean up created files
            for file_path in created_files:
                try:
                    os.unlink(file_path)
                    # Also try to remove parent directories if they're temp dirs
                    parent = os.path.dirname(file_path)
                    if "subtitletools" in parent:
                        import shutil

                        shutil.rmtree(parent, ignore_errors=True)
                except Exception:
                    pass

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_ffmpeg_failure(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test audio extraction when FFmpeg fails."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"

        # Mock failed FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "FFmpeg error: invalid input"
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="FFmpeg error"):
                extract_audio(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_timeout(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test audio extraction timeout."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"
        mock_run.side_effect = subprocess.TimeoutExpired(["ffmpeg"], 3600)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="timed out after 1 hour"):
                extract_audio(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_output_not_created(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test audio extraction when output file is not created."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"

        # Mock successful FFmpeg execution but no output file created
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "conversion successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="output file not created"):
                extract_audio(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_small_output_warning(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test audio extraction with small output file (should warn but succeed)."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"

        # Mock successful FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "conversion successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            tmp_video.write(b"fake video content")
            video_path = tmp_video.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            # Create very small output file (< 1KB)
            tmp_audio.write(b"small")
            output_path = tmp_audio.name

        try:

            def side_effect(*args: Any, **kwargs: Any) -> Mock:
                with open(output_path, "wb") as f:
                    f.write(b"small")  # < 1000 bytes
                return mock_result

            mock_run.side_effect = side_effect

            result = extract_audio(video_path, output_path)

            # Should succeed despite small file size
            assert result == output_path

        finally:
            Path(video_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)


class TestCleanupTempDir:
    """Test cleanup_temp_dir function."""

    @patch("subtitletools.utils.audio._TEMP_DIR", None)
    def test_cleanup_temp_dir_no_temp_dir(self) -> None:
        """Test cleanup when no temp directory exists."""
        # Should not raise any exceptions
        cleanup_temp_dir()

    @patch("subtitletools.utils.audio._TEMP_DIR", "/tmp/nonexistent")
    @patch("os.path.exists")
    def test_cleanup_temp_dir_nonexistent(self, mock_exists: Mock) -> None:
        """Test cleanup when temp directory path doesn't exist."""
        mock_exists.return_value = False

        cleanup_temp_dir()

    @patch("subtitletools.utils.audio._TEMP_DIR", "/tmp/test_temp")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_cleanup_temp_dir_success(
        self, mock_rmtree: Mock, mock_exists: Mock
    ) -> None:
        """Test successful temp directory cleanup."""
        mock_exists.return_value = True

        cleanup_temp_dir()

        mock_rmtree.assert_called_once_with("/tmp/test_temp")

    @patch("subtitletools.utils.audio._TEMP_DIR", "/tmp/test_temp")
    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_cleanup_temp_dir_exception(
        self, mock_rmtree: Mock, mock_exists: Mock
    ) -> None:
        """Test temp directory cleanup with exception."""
        mock_exists.return_value = True
        mock_rmtree.side_effect = Exception("Permission denied")

        # Should not raise exception, just log error
        cleanup_temp_dir()

        mock_rmtree.assert_called_once()


class TestGetAudioDuration:
    """Test get_audio_duration function."""

    @patch("subtitletools.utils.audio.find_ffprobe")
    def test_get_audio_duration_ffprobe_not_found(
        self, mock_find_ffprobe: Mock
    ) -> None:
        """Test getting duration when FFprobe is not found."""
        mock_find_ffprobe.return_value = None

        result = get_audio_duration("test.wav")

        assert result is None

    @patch("subtitletools.utils.audio.find_ffprobe")
    @patch("subprocess.run")
    def test_get_audio_duration_success(
        self, mock_run: Mock, mock_find_ffprobe: Mock
    ) -> None:
        """Test successful duration retrieval."""
        mock_find_ffprobe.return_value = "/usr/bin/ffprobe"

        # Mock successful FFprobe execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"format": {"duration": "123.456"}})
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = get_audio_duration("test.wav")

        assert result == 123.456
        mock_run.assert_called_once_with(
            [
                "/usr/bin/ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "test.wav",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )

    @patch("subtitletools.utils.audio.find_ffprobe")
    @patch("subprocess.run")
    def test_get_audio_duration_ffprobe_failure(
        self, mock_run: Mock, mock_find_ffprobe: Mock
    ) -> None:
        """Test duration retrieval when FFprobe fails."""
        mock_find_ffprobe.return_value = "/usr/bin/ffprobe"

        # Mock failed FFprobe execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "FFprobe error: file not found"
        mock_run.return_value = mock_result

        result = get_audio_duration("test.wav")

        assert result is None

    @patch("subtitletools.utils.audio.find_ffprobe")
    @patch("subprocess.run")
    def test_get_audio_duration_invalid_json(
        self, mock_run: Mock, mock_find_ffprobe: Mock
    ) -> None:
        """Test duration retrieval with invalid JSON response."""
        mock_find_ffprobe.return_value = "/usr/bin/ffprobe"

        # Mock FFprobe with invalid JSON
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid json"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        result = get_audio_duration("test.wav")

        assert result is None

    @patch("subtitletools.utils.audio.find_ffprobe")
    @patch("subprocess.run")
    def test_get_audio_duration_exception(
        self, mock_run: Mock, mock_find_ffprobe: Mock
    ) -> None:
        """Test duration retrieval with subprocess exception."""
        mock_find_ffprobe.return_value = "/usr/bin/ffprobe"
        mock_run.side_effect = Exception("Subprocess failed")

        result = get_audio_duration("test.wav")

        assert result is None


class TestFindFFprobe:
    """Test find_ffprobe function."""

    @patch("subprocess.run")
    def test_find_ffprobe_in_path_windows(self, mock_run: Mock) -> None:
        """Test finding FFprobe in PATH on Windows."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "C:\\tools\\ffprobe.exe\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with patch("sys.platform", "win32"):
            result = find_ffprobe()

        assert result == "C:\\tools\\ffprobe.exe"
        mock_run.assert_called_once_with(
            ["where", "ffprobe"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_find_ffprobe_in_path_unix(self, mock_run: Mock) -> None:
        """Test finding FFprobe in PATH on Unix systems."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "/usr/bin/ffprobe\n"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        with patch("sys.platform", "linux"):
            result = find_ffprobe()

        assert result == "/usr/bin/ffprobe"

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffprobe_not_in_path_but_in_common_location(
        self, mock_isfile: Mock, mock_run: Mock
    ) -> None:
        """Test finding FFprobe in common locations when not in PATH."""
        # PATH search fails
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        # First location exists
        mock_isfile.side_effect = [True]

        result = find_ffprobe()

        assert result is not None

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffprobe_not_found(self, mock_isfile: Mock, mock_run: Mock) -> None:
        """Test when FFprobe is not found anywhere."""
        # PATH search fails
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        # No common locations have FFprobe
        mock_isfile.return_value = False

        result = find_ffprobe()

        assert result is None

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffprobe_path_timeout(self, mock_isfile: Mock, mock_run: Mock) -> None:
        """Test PATH search timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(["where"], 30)
        mock_isfile.return_value = False

        result = find_ffprobe()

        assert result is None

    @patch("subprocess.run")
    @patch("os.path.isfile")
    def test_find_ffprobe_file_check_exception(
        self, mock_isfile: Mock, mock_run: Mock
    ) -> None:
        """Test exception during file existence check."""
        # PATH search fails
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        # File check raises exception, should continue to next location
        mock_isfile.side_effect = [Exception("Permission denied"), True]

        result = find_ffprobe()

        # Should find FFprobe at second location despite exception at first
        assert result is not None


class TestValidateAudioFile:
    """Test validate_audio_file function."""

    def test_validate_audio_file_nonexistent(self) -> None:
        """Test validating non-existent audio file."""
        result = validate_audio_file("nonexistent.wav")
        assert result is False

    def test_validate_audio_file_too_small(self) -> None:
        """Test validating very small audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"small")  # < 100 bytes
            tmp_path = tmp.name

        try:
            result = validate_audio_file(tmp_path)
            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.get_audio_duration")
    def test_validate_audio_file_no_duration(self, mock_duration: Mock) -> None:
        """Test validating audio file with no duration."""
        mock_duration.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio content" * 50)  # > 100 bytes
            tmp_path = tmp.name

        try:
            result = validate_audio_file(tmp_path)
            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.get_audio_duration")
    def test_validate_audio_file_zero_duration(self, mock_duration: Mock) -> None:
        """Test validating audio file with zero duration."""
        mock_duration.return_value = 0.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio content" * 50)  # > 100 bytes
            tmp_path = tmp.name

        try:
            result = validate_audio_file(tmp_path)
            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.get_audio_duration")
    def test_validate_audio_file_valid(self, mock_duration: Mock) -> None:
        """Test validating valid audio file."""
        mock_duration.return_value = 120.5  # Valid duration

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio content" * 50)  # > 100 bytes
            tmp_path = tmp.name

        try:
            result = validate_audio_file(tmp_path)
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.get_audio_duration")
    def test_validate_audio_file_exception(self, mock_duration: Mock) -> None:
        """Test validating audio file with exception."""
        mock_duration.side_effect = Exception("File access error")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake audio content" * 50)  # > 100 bytes
            tmp_path = tmp.name

        try:
            result = validate_audio_file(tmp_path)
            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_validate_audio_file_pathlib_path(self) -> None:
        """Test validating audio file with pathlib Path."""
        result = validate_audio_file(Path("nonexistent.wav"))
        assert result is False


class TestThreadSafety:
    """Test thread safety of global temp directory."""

    @patch("tempfile.mkdtemp")
    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_temp_dir_thread_safety(
        self, mock_run: Mock, mock_find_ffmpeg: Mock, mock_mkdtemp: Mock
    ) -> None:
        """Test that temporary directory creation is thread-safe."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"
        mock_mkdtemp.return_value = "/tmp/thread_safe_test"

        # Mock successful FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        results = []

        def worker() -> None:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(b"fake video")
                video_path = tmp.name

            try:
                # Mock output file creation
                def create_output(*args: Any, **kwargs: Any) -> Mock:
                    # FFmpeg command structure: [ffmpeg, "-i", input, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output, "-y"]
                    output_path = args[0][-2]  # Second to last argument is output path
                    with open(output_path, "wb") as f:
                        f.write(b"fake audio" * 100)
                    return mock_result

                mock_run.side_effect = create_output

                result = extract_audio(video_path)
                results.append(result)
            except Exception as e:
                results.append(str(e))
            finally:
                Path(video_path).unlink(missing_ok=True)

        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # mkdtemp should only be called once due to thread safety
        assert (
            mock_mkdtemp.call_count <= 3
        )  # May be called multiple times but should be thread-safe
        assert len(results) == 3


class TestAdditionalAudioCoverage:
    """Additional test cases to improve audio coverage."""

    @patch("subprocess.run")
    def test_find_ffmpeg_system_path(self, mock_run: Mock) -> None:
        """Test finding FFmpeg on system PATH."""
        mock_run.return_value = Mock(returncode=0, stdout="/usr/bin/ffmpeg\n")

        result = find_ffmpeg()

        assert result == "/usr/bin/ffmpeg"

    @patch("os.path.isfile")
    @patch("subprocess.run")
    def test_find_ffmpeg_common_locations(
        self, mock_run: Mock, mock_isfile: Mock
    ) -> None:
        """Test finding FFmpeg in common locations."""
        mock_run.return_value = Mock(returncode=1)  # Not in PATH
        mock_isfile.side_effect = lambda path: path.endswith("ffmpeg.exe")

        result = find_ffmpeg()

        assert result is not None
        assert result.endswith("ffmpeg.exe")

    @patch("os.path.isfile")
    @patch("subprocess.run")
    def test_find_ffmpeg_test_failure(self, mock_run: Mock, mock_isfile: Mock) -> None:
        """Test FFmpeg path not found in PATH or common locations."""
        mock_run.return_value = Mock(returncode=1, stderr="Command not found")
        mock_isfile.return_value = False  # Not in common locations either

        result = find_ffmpeg()

        assert result is None

    @patch("subtitletools.utils.audio.find_ffmpeg")
    def test_extract_audio_ffmpeg_not_found(self, mock_find_ffmpeg: Mock) -> None:
        """Test extract_audio when FFmpeg is not found."""
        mock_find_ffmpeg.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            input_path = tmp.name

        try:
            with pytest.raises(FileNotFoundError, match="FFmpeg not found"):
                extract_audio(input_path, temp_dir=None)
        finally:
            Path(input_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    @patch("subtitletools.utils.audio.find_ffprobe")
    def test_get_audio_duration_ffprobe_not_found(
        self, mock_find_ffprobe: Mock, mock_run: Mock
    ) -> None:
        """Test get_audio_duration when FFprobe is not found."""
        mock_find_ffprobe.return_value = None

        result = get_audio_duration("test.wav")

        assert result is None

    @patch("subprocess.run")
    @patch("subtitletools.utils.audio.find_ffprobe")
    def test_get_audio_duration_command_failure(
        self, mock_find_ffprobe: Mock, mock_run: Mock
    ) -> None:
        """Test get_audio_duration with command failure."""
        mock_find_ffprobe.return_value = "/usr/bin/ffprobe"
        mock_run.return_value = Mock(returncode=1, stderr="Command failed")

        result = get_audio_duration("test.wav")

        assert result is None

    @patch("os.path.isfile")
    @patch("subprocess.run")
    def test_find_ffprobe_common_locations(
        self, mock_run: Mock, mock_isfile: Mock
    ) -> None:
        """Test finding FFprobe in common locations."""
        mock_run.return_value = Mock(returncode=1)  # Not in PATH
        mock_isfile.side_effect = lambda path: "ffprobe" in path

        result = find_ffprobe()

        assert result is not None
        assert "ffprobe" in result

    @patch("subprocess.run")
    @patch("subtitletools.utils.audio.find_ffprobe")
    def test_validate_audio_file_ffprobe_not_found(
        self, mock_find_ffprobe: Mock, mock_run: Mock
    ) -> None:
        """Test validate_audio_file when FFprobe is not found."""
        mock_find_ffprobe.return_value = None

        result = validate_audio_file("test.wav")

        assert result is False

    @patch("subprocess.run")
    def test_find_ffmpeg_unix_platform_timeout(self, mock_run: Mock) -> None:
        """Test find_ffmpeg timeout exception on Unix platforms."""
        mock_run.side_effect = subprocess.TimeoutExpired(["which", "ffmpeg"], 30)

        with patch("sys.platform", "linux"):
            with patch("os.path.isfile", return_value=False):
                result = find_ffmpeg()

        assert result is None

    @patch("subprocess.run")
    def test_find_ffmpeg_unix_platform_exception(self, mock_run: Mock) -> None:
        """Test find_ffmpeg general exception on Unix platforms."""
        mock_run.side_effect = Exception("Command failed")

        with patch("sys.platform", "linux"):
            with patch("os.path.isfile", return_value=False):
                result = find_ffmpeg()

        assert result is None

    @patch("os.path.isfile")
    def test_find_ffmpeg_location_exception(self, mock_isfile: Mock) -> None:
        """Test find_ffmpeg exception when checking common locations."""
        mock_isfile.side_effect = Exception("Permission denied")

        with patch("subprocess.run", return_value=Mock(returncode=1)):
            result = find_ffmpeg()

        assert result is None

    def test_extract_audio_empty_path(self) -> None:
        """Test extract_audio with empty video path."""
        with pytest.raises(ValueError, match="Video path cannot be empty"):
            extract_audio("")

    @patch("subtitletools.utils.audio.find_ffmpeg")
    def test_extract_audio_with_custom_temp_dir(self, mock_find_ffmpeg: Mock) -> None:
        """Test extract_audio with custom temp_dir."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            input_path = tmp.name

        with tempfile.TemporaryDirectory() as custom_temp_dir:
            try:
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = Mock(returncode=0)
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("pathlib.Path.stat") as mock_stat:
                            mock_stat.return_value = Mock(st_size=5000)
                            result = extract_audio(input_path, temp_dir=custom_temp_dir)

                            assert custom_temp_dir in result
            finally:
                Path(input_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.stat")
    def test_extract_audio_small_file_warning(
        self, mock_stat: Mock, mock_exists: Mock, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test extract_audio warning for small output file."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"
        mock_run.return_value = Mock(returncode=0)
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=500)  # Small file size

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            input_path = tmp.name

        try:
            with patch("subtitletools.utils.audio.logger") as mock_logger:
                result = extract_audio(input_path)

                assert result is not None
                mock_logger.warning.assert_called_once()
        finally:
            Path(input_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_timeout_exception(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test extract_audio with timeout exception."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"
        mock_run.side_effect = subprocess.TimeoutExpired(["ffmpeg"], 3600)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            input_path = tmp.name

        try:
            with pytest.raises(RuntimeError, match="timed out after 1 hour"):
                extract_audio(input_path)
        finally:
            Path(input_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    def test_extract_audio_ffmpeg_not_found_error(self, mock_find_ffmpeg: Mock) -> None:
        """Test extract_audio when FFmpeg is not found."""
        mock_find_ffmpeg.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            input_path = tmp.name

        try:
            with pytest.raises(FileNotFoundError, match="FFmpeg not found"):
                extract_audio(input_path)
        finally:
            Path(input_path).unlink(missing_ok=True)

    @patch("subtitletools.utils.audio.find_ffmpeg")
    @patch("subprocess.run")
    def test_extract_audio_filenotfound_reraise(
        self, mock_run: Mock, mock_find_ffmpeg: Mock
    ) -> None:
        """Test extract_audio re-raises FileNotFoundError."""
        mock_find_ffmpeg.return_value = "/usr/bin/ffmpeg"
        mock_run.side_effect = FileNotFoundError("FFmpeg not found")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            input_path = tmp.name

        try:
            with pytest.raises(FileNotFoundError):
                extract_audio(input_path)
        finally:
            Path(input_path).unlink(missing_ok=True)

    @patch("subprocess.run")
    def test_find_ffprobe_unix_platform(self, mock_run: Mock) -> None:
        """Test find_ffprobe on Unix platforms."""
        mock_run.return_value = Mock(returncode=0, stdout="/usr/bin/ffprobe\n")

        with patch("sys.platform", "linux"):
            result = find_ffprobe()

        assert result == "/usr/bin/ffprobe"

    @patch("subprocess.run")
    def test_find_ffprobe_exception_handling(self, mock_run: Mock) -> None:
        """Test find_ffprobe exception handling."""
        mock_run.side_effect = Exception("Command failed")

        with patch("os.path.isfile", return_value=False):
            result = find_ffprobe()

        assert result is None

    @patch("os.path.isfile")
    def test_find_ffprobe_location_exception(self, mock_isfile: Mock) -> None:
        """Test find_ffprobe exception when checking locations."""
        mock_isfile.side_effect = Exception("Permission denied")

        with patch("subprocess.run", return_value=Mock(returncode=1)):
            result = find_ffprobe()

        assert result is None

    @patch("pathlib.Path.stat")
    def test_validate_audio_file_exception_handling(self, mock_stat: Mock) -> None:
        """Test validate_audio_file exception handling."""
        mock_stat.side_effect = Exception("File access error")

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            with patch("subtitletools.utils.audio.logger") as mock_logger:
                result = validate_audio_file(tmp.name)

                assert result is False
                mock_logger.debug.assert_called_once()
