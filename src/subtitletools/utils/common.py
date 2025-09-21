"""Common utility functions shared across SubtitleTools modules."""

import logging
import os
import sys
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

torch_module: Optional[object] = None
try:
    import torch as torch_module  # Optional dependency
except ImportError:
    torch_module = None

from ..config.settings import (
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_SUBTITLE_FORMATS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT format timestamp (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string in SRT format
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Optional custom format string

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    return logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Get file size in megabytes.

    Args:
        path: File path

    Returns:
        File size in MB
    """
    return Path(path).stat().st_size / (1024 * 1024)


def is_video_file(path: Union[str, Path]) -> bool:
    """Check if a file is a video file based on extension.

    Args:
        path: File path

    Returns:
        True if file appears to be a video file
    """
    extension = Path(path).suffix.lower().lstrip(".")
    return extension in SUPPORTED_VIDEO_EXTENSIONS


def is_audio_file(path: Union[str, Path]) -> bool:
    """Check if a file is an audio file based on extension.

    Args:
        path: File path

    Returns:
        True if file appears to be an audio file
    """
    extension = Path(path).suffix.lower().lstrip(".")
    return extension in SUPPORTED_AUDIO_EXTENSIONS


def is_subtitle_file(path: Union[str, Path]) -> bool:
    """Check if a file is a subtitle file based on extension.

    Args:
        path: File path

    Returns:
        True if file appears to be a subtitle file
    """
    extension = Path(path).suffix.lower().lstrip(".")
    return extension in SUPPORTED_SUBTITLE_FORMATS


def validate_file_exists(path: Union[str, Path]) -> Path:
    """Validate that a file exists and return Path object.

    Args:
        path: File path to validate

    Returns:
        Path object for the file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is empty or invalid
    """
    if not path:
        raise ValueError("File path cannot be empty")

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File does not exist: {path_obj}")

    if not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path_obj}")

    return path_obj


def validate_directory_exists(path: Union[str, Path]) -> Path:
    """Validate that a directory exists and return Path object.

    Args:
        path: Directory path to validate

    Returns:
        Path object for the directory

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If path is empty or invalid
    """
    if not path:
        raise ValueError("Directory path cannot be empty")

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Directory does not exist: {path_obj}")

    if not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path_obj}")

    return path_obj


def safe_filename(name: str, replacement: str = "_") -> str:
    """Create a safe filename by replacing invalid characters.

    Args:
        name: Original filename
        replacement: Character to use for replacements

    Returns:
        Safe filename string
    """
    # Characters that are invalid in filenames on Windows and Unix
    invalid_chars = '<>:"/\\|?*'

    safe_name = name
    for char in invalid_chars:
        safe_name = safe_name.replace(char, replacement)

    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(". ")

    # Ensure filename is not empty
    if not safe_name:
        safe_name = "unnamed"

    return safe_name


class ThreadSafeCounter:
    """Thread-safe counter for tracking progress."""

    def __init__(self, initial_value: int = 0):
        """Initialize counter.

        Args:
            initial_value: Initial counter value
        """
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self) -> int:
        """Increment counter and return new value.

        Returns:
            New counter value after increment
        """
        with self._lock:
            self._value += 1
            return self._value

    def decrement(self) -> int:
        """Decrement counter and return new value.

        Returns:
            New counter value after decrement
        """
        with self._lock:
            self._value -= 1
            return self._value

    def get(self) -> int:
        """Get current counter value.

        Returns:
            Current counter value
        """
        with self._lock:
            return self._value

    def set(self, value: int) -> None:
        """Set counter value.

        Args:
            value: New counter value
        """
        with self._lock:
            self._value = value

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


def get_system_info() -> Dict[str, Union[str, bool]]:
    """Get basic system information for logging.

    Returns:
        Dictionary with system information
    """
    info: Dict[str, Union[str, bool]] = {
        "platform": sys.platform,
        "python_version": sys.version,
        "cwd": os.getcwd(),
    }

    # Check for torch availability
    if torch_module is not None:
        info["torch_version"] = str(cast(Any, torch_module).__version__)
        info["cuda_available"] = bool(cast(Any, torch_module).cuda.is_available())
    else:
        info["torch_version"] = "Not installed"
        info["cuda_available"] = False

    return info
