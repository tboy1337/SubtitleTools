"""Audio processing utilities for SubtitleTools.

Contains functionality for audio extraction from video files using FFmpeg,
adapted from the original subwhisper implementation.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from subprocess import CompletedProcess
from typing import Optional, Union

# Global temporary directory for audio extraction
_TEMP_DIR: Optional[str] = None
_TEMP_DIR_LOCK = threading.Lock()

logger = logging.getLogger(__name__)


def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg executable in the system with comprehensive logging.

    Returns:
        Path to FFmpeg executable if found, None otherwise.

    Raises:
        No exceptions are raised; all errors are logged and handled gracefully.
    """
    logger.info("Searching for FFmpeg executable...")

    # Common locations for FFmpeg
    ffmpeg_locations = [
        # Windows Program Files and common installation locations
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        # Winget installation location - both versioned and essentials
        os.path.expanduser(
            r"~\AppData\Local\Microsoft\WinGet\Packages"
            + r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            + r"\ffmpeg-7.1.1-full_build\bin\ffmpeg.exe"
        ),
        os.path.expanduser(
            r"~\AppData\Local\Microsoft\WinGet\Packages"
            + r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
            + r"\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"
        ),
        # Default command (if in PATH)
        "ffmpeg",
    ]

    # Check if ffmpeg is in PATH first
    try:
        # Use appropriate command based on OS
        if sys.platform == "win32":
            check_cmd = ["where", "ffmpeg"]
        else:
            check_cmd = ["which", "ffmpeg"]

        logger.debug("Checking PATH for FFmpeg using command: %s", ' '.join(check_cmd))
        result: CompletedProcess[str] = subprocess.run(
            check_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip().split("\n")[0]
            logger.info("FFmpeg found in PATH: %s", ffmpeg_path)
            return ffmpeg_path
        logger.debug("FFmpeg not found in PATH, stderr: %s", result.stderr)
    except subprocess.TimeoutExpired:
        logger.warning("Timeout expired while searching for FFmpeg in PATH")
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Error while searching for FFmpeg in PATH: %s", e)

    # Check common locations
    logger.debug("Checking common FFmpeg installation locations...")
    for i, location in enumerate(ffmpeg_locations, 1):
        try:
            if os.path.isfile(location):
                logger.info(
                    "FFmpeg found at location %d/%d: %s",
                    i,
                    len(ffmpeg_locations),
                    location
                )
                return location

            logger.debug(
                "FFmpeg not found at location %d/%d: %s",
                i,
                len(ffmpeg_locations),
                location
            )
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug("Error checking FFmpeg location %s: %s", location, e)

    # If we get here, we couldn't find ffmpeg
    logger.error("FFmpeg executable not found in any known locations")
    return None


def extract_audio(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    temp_dir: Optional[str] = None
) -> str:
    """Extract audio from video file using ffmpeg with comprehensive error handling.

    Args:
        video_path: Path to the input video file.
        output_path: Optional output path for extracted audio.
            If None, creates temp file.
        temp_dir: Optional temporary directory to use. If None, creates one.

    Returns:
        Path to the extracted audio file.

    Raises:
        RuntimeError: If FFmpeg execution fails or audio extraction fails.
        FileNotFoundError: If FFmpeg executable is not found.
        ValueError: If input video path is invalid.
    """
    global _TEMP_DIR  # pylint: disable=global-statement

    # Validate input
    if not video_path:
        raise ValueError("Video path cannot be empty")

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise ValueError(f"Video file does not exist: {video_path}")

    logger.info("Starting audio extraction from: %s", video_path)

    # Thread-safe temporary directory creation
    if output_path is None:
        with _TEMP_DIR_LOCK:
            if temp_dir:
                working_temp_dir = temp_dir
            else:
                if _TEMP_DIR is None:
                    _TEMP_DIR = tempfile.mkdtemp(prefix="subtitletools_audio_")
                    logger.info("Created temporary directory: %s", _TEMP_DIR)
                working_temp_dir = _TEMP_DIR

        # Create a safe filename based on the input video name
        basename_no_ext = video_path_obj.stem
        output_path = os.path.join(working_temp_dir, f"{basename_no_ext}.wav")
        logger.debug("Generated output path: %s", output_path)
    else:
        logger.debug("Using provided output path: %s", output_path)

    try:
        # Find ffmpeg executable
        ffmpeg_path = find_ffmpeg()
        if ffmpeg_path is None:
            error_msg = (
                "FFmpeg not found. Please install FFmpeg and ensure it's in your PATH."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        command = [
            ffmpeg_path,
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_path),
            "-y",
        ]

        logger.info("Executing FFmpeg command: %s", ' '.join(command))
        start_time = time.time()

        result: CompletedProcess[str] = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3600,  # 1 hour timeout for large files
            check=False,
        )

        extraction_time = time.time() - start_time
        logger.info("Audio extraction completed in %.2f seconds", extraction_time)

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip() if result.stderr else "Unknown FFmpeg error"
            )
            logger.error(
                "FFmpeg failed with return code %d: %s",
                result.returncode,
                error_msg
            )
            raise RuntimeError(f"FFmpeg error (code {result.returncode}): {error_msg}")

        # Verify output file was created and has reasonable size
        output_path_obj = Path(output_path)
        if not output_path_obj.exists():
            raise RuntimeError(
                f"Audio extraction failed: output file not created at {output_path}"
            )

        file_size = output_path_obj.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious
            logger.warning(
                "Extracted audio file is very small (%d bytes), may indicate extraction issues",
                file_size
            )
        else:
            logger.info(
                "Successfully extracted audio file (%d bytes): %s",
                file_size,
                output_path
            )

        return str(output_path)

    except subprocess.TimeoutExpired:
        error_msg = f"Audio extraction timed out after 1 hour for file: {video_path}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from None
    except FileNotFoundError:
        logger.error("FFmpeg executable not found")
        raise
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("Unexpected error during audio extraction: %s", e)
        raise RuntimeError(f"Audio extraction failed: {e}") from e


def cleanup_temp_dir() -> None:
    """Clean up the temporary directory used for audio extraction."""
    global _TEMP_DIR  # pylint: disable=global-statement

    if _TEMP_DIR and os.path.exists(_TEMP_DIR):
        logger.info("Cleaning up temporary directory: %s", _TEMP_DIR)
        try:
            with _TEMP_DIR_LOCK:
                shutil.rmtree(_TEMP_DIR)
            logger.info("Temporary directory cleaned up successfully")
            _TEMP_DIR = None
        except (OSError, IOError) as e:
            logger.error("Failed to clean up temporary directory %s: %s", _TEMP_DIR, e)


def get_audio_duration(audio_path: Union[str, Path]) -> Optional[float]:
    """Get the duration of an audio file using FFprobe.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds, or None if unable to determine
    """
    try:
        ffprobe_path = find_ffprobe()
        if not ffprobe_path:
            logger.warning("FFprobe not found, cannot determine audio duration")
            return None

        command = [
            ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(audio_path)
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data["format"]["duration"])
            logger.debug("Audio duration: %.2f seconds", duration)
            return duration
        logger.warning("FFprobe failed: %s", result.stderr)

    except (subprocess.SubprocessError, ValueError, OSError) as e:
        logger.warning("Error getting audio duration: %s", e)

    return None


def find_ffprobe() -> Optional[str]:
    """Find FFprobe executable in the system.

    Returns:
        Path to FFprobe executable if found, None otherwise.
    """
    # Try common locations similar to FFmpeg
    ffprobe_locations = [
        # Windows Program Files and common installation locations
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffprobe.exe",
        # Default command (if in PATH)
        "ffprobe",
    ]

    # Check if ffprobe is in PATH first
    try:
        if sys.platform == "win32":
            check_cmd = ["where", "ffprobe"]
        else:
            check_cmd = ["which", "ffprobe"]

        result = subprocess.run(
            check_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        pass

    # Check common locations
    for location in ffprobe_locations:
        try:
            if os.path.isfile(location):
                return location
        except (OSError, TypeError):
            continue

    return None


def validate_audio_file(audio_path: Union[str, Path]) -> bool:
    """Validate that an audio file is readable and has content.

    Args:
        audio_path: Path to the audio file

    Returns:
        True if file appears to be valid audio file
    """
    try:
        audio_path_obj = Path(audio_path)

        # Check file exists and has reasonable size
        if not audio_path_obj.exists():
            return False

        file_size = audio_path_obj.stat().st_size
        if file_size < 100:  # Less than 100 bytes is suspicious
            return False

        # Try to get duration using FFprobe
        duration = get_audio_duration(audio_path)
        if duration is None or duration <= 0:
            return False

        return True

    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Audio file validation failed for %s: %s", audio_path, e)
        return False
