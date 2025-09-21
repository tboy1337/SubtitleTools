"""Test configuration and fixtures for SubtitleTools tests."""

import os
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import pytest
import srt


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Get the test_data directory, ensuring it exists and is clean after each test."""
    test_data_path = Path(__file__).parent / "test_data"
    test_data_path.mkdir(exist_ok=True)

    yield test_data_path

    # Clean up any files created during the test
    for file in test_data_path.glob("*"):
        if file.is_file():
            try:
                file.unlink()
            except (OSError, PermissionError):
                pass  # Ignore cleanup errors


@pytest.fixture
def sample_video_file(temp_dir: Path) -> Path:  # pylint: disable=redefined-outer-name
    """Create a sample video file for testing."""
    video_file = temp_dir / "test_video.mp4"
    video_file.write_bytes(b"fake video content")  # Minimal fake content
    return video_file


@pytest.fixture
def sample_audio_file(temp_dir: Path) -> Path:  # pylint: disable=redefined-outer-name
    """Create a sample audio file for testing."""
    audio_file = temp_dir / "test_audio.wav"
    audio_file.write_bytes(b"fake audio content")  # Minimal fake content
    return audio_file


@pytest.fixture
def sample_srt_content() -> str:
    """Sample SRT content for testing."""
    return """1
00:00:01,000 --> 00:00:03,000
Hello world

2
00:00:04,000 --> 00:00:06,000
This is a test subtitle

3
00:00:07,000 --> 00:00:10,000
With multiple lines
and formatting
"""


@pytest.fixture
def sample_srt_file(temp_dir: Path, sample_srt_content: str) -> Path:  # pylint: disable=redefined-outer-name
    """Create a sample SRT file for testing."""
    srt_file = temp_dir / "test.srt"
    srt_file.write_text(sample_srt_content, encoding="utf-8")
    return srt_file


@pytest.fixture
def sample_subtitles() -> list[srt.Subtitle]:
    """Create sample subtitle objects for testing."""
    return [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=1),
            end=timedelta(seconds=3),
            content="Hello world"
        ),
        srt.Subtitle(
            index=2,
            start=timedelta(seconds=4),
            end=timedelta(seconds=6),
            content="This is a test subtitle"
        ),
        srt.Subtitle(
            index=3,
            start=timedelta(seconds=7),
            end=timedelta(seconds=10),
            content="With multiple lines\nand formatting"
        ),
    ]


@pytest.fixture
def mock_env_clean() -> Generator[None, None, None]:
    """Clean environment variables for testing."""
    original_env = os.environ.copy()
    # Remove SubtitleTools-related environment variables
    for key in list(os.environ.keys()):
        if key.startswith("SUBTITLETOOLS_"):
            del os.environ[key]
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_whisper_model() -> Any:
    """Mock Whisper model for testing."""
    with patch('whisper.load_model') as mock_load:
        mock_model = mock_load.return_value
        mock_model.transcribe.return_value = {
            'text': 'Test transcription',
            'segments': [
                {
                    'start': 0.0,
                    'end': 3.0,
                    'text': 'Test transcription'
                }
            ]
        }
        yield mock_model


@pytest.fixture
def mock_requests() -> Any:
    """Mock requests for translation testing."""
    with patch('requests.Session') as mock_session:
        mock_response = mock_session.return_value.get.return_value
        mock_response.status_code = 200
        mock_response.text = 'mock response'
        yield mock_session


# Post-processing is now handled natively - no external dependencies needed


# Utility functions for tests
def create_test_subtitles(count: int = 3) -> list[srt.Subtitle]:
    """Create test subtitles for unit testing."""
    subtitles = []
    for i in range(1, count + 1):
        subtitle = srt.Subtitle(
            index=i,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 1),
            content=f"Test subtitle {i}"
        )
        subtitles.append(subtitle)
    return subtitles


def create_long_subtitle_line() -> str:
    """Create a long subtitle line for testing."""
    return "This is a very long subtitle line that should be split because it exceeds the maximum recommended length for subtitle display and readability purposes in most media players and subtitle rendering systems."


def create_malformed_srt() -> str:
    """Create malformed SRT content for error testing."""
    return """1
INVALID_TIMESTAMP
Hello world

2
00:00:04,000 --> INVALID_END
This is broken
"""


def get_test_data_path(filename: str) -> Path:
    """Get a path in the test_data directory for the given filename.

    This utility function ensures all test data files are created in the
    proper test_data directory instead of the project root.

    Args:
        filename: Name of the test file

    Returns:
        Path object pointing to the file in test_data directory
    """
    data_dir = Path(__file__).parent / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / filename
