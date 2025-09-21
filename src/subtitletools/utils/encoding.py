"""Encoding conversion utilities for SubtitleTools.

This module provides functionality to convert subtitle files between different encodings
commonly used for subtitles in various languages and regions.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..config.settings import SUPPORTED_ENCODINGS, LANGUAGE_ENCODINGS

logger = logging.getLogger(__name__)


def detect_encoding(
    file_path: str, encodings_to_try: Optional[List[str]] = None
) -> Optional[str]:
    """Attempt to detect the encoding of a subtitle file by trying multiple encodings.

    Args:
        file_path: Path to the subtitle file
        encodings_to_try: List of encodings to try, defaults to SUPPORTED_ENCODINGS

    Returns:
        Detected encoding or None if detection fails
    """
    if encodings_to_try is None:
        encodings_to_try = SUPPORTED_ENCODINGS

    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        return None

    for encoding_name in encodings_to_try:
        try:
            with open(file_path, "r", encoding=encoding_name) as f:
                content = f.read()
                # If we can read at least 100 characters without error,
                # it's probably the right encoding
                if len(content) > 100:
                    logger.debug("Detected encoding: %s", encoding_name)
                    return encoding_name
        except UnicodeDecodeError:
            continue
        except (OSError, IOError) as e:
            logger.debug("Error trying encoding %s: %s", encoding_name, e)
            continue

    logger.error("Could not detect encoding for %s", file_path)
    return None


def convert_subtitle_encoding(
    input_file: str,
    output_file: str,
    target_encoding: str,
    source_encoding: Optional[str] = None,
) -> bool:
    """Convert subtitle file from source encoding to target encoding.

    Args:
        input_file: Path to the input subtitle file
        output_file: Path to save the converted subtitle file
        target_encoding: Target encoding to convert to
        source_encoding: Source encoding of input file (auto-detect if None)

    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Determine source encoding if not provided
        if source_encoding is None:
            source_encoding = detect_encoding(input_file)
            if source_encoding is None:
                return False

        # Read the source file
        with open(input_file, "r", encoding=source_encoding) as f:
            content = f.read()

        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write with target encoding
        with open(output_file, "wb") as f:
            # Add BOM if target is UTF-8 with BOM
            if target_encoding.lower() == "utf-8-sig":
                f.write(b"\xef\xbb\xbf")  # UTF-8 BOM
                f.write(content.encode("utf-8", errors="replace"))
            else:
                f.write(content.encode(target_encoding, errors="replace"))

        logger.info(
            "Converted %s from %s to %s -> %s",
            input_file,
            source_encoding,
            target_encoding,
            output_file,
        )
        return True
    except (OSError, IOError, UnicodeError, LookupError) as e:
        logger.error("Error converting %s to %s: %s", input_file, target_encoding, e)
        return False


def convert_to_multiple_encodings(
    input_file: str,
    output_dir: Optional[str] = None,
    target_encodings: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """Convert a subtitle file to multiple encodings.

    Args:
        input_file: Path to the input subtitle file
        output_dir: Directory to save converted files (defaults to input file directory)
        target_encodings: List of target encodings (defaults to a common subset)

    Returns:
        Dictionary mapping target encodings to conversion success status
    """
    if target_encodings is None:
        target_encodings = ["utf-8", "utf-8-sig", "cp874", "tis-620", "iso8859-11"]

    if not os.path.exists(input_file):
        logger.error("Input file not found: %s", input_file)
        return {encoding: False for encoding in target_encodings}

    if output_dir is None:
        output_dir = os.path.dirname(input_file) or "."

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine source file details
    source_path = Path(input_file)
    source_encoding = detect_encoding(input_file)
    if source_encoding is None:
        return {encoding: False for encoding in target_encodings}

    results = {}
    for target_encoding in target_encodings:
        # Create output filename with encoding as suffix
        stem = source_path.stem
        # Remove any existing encoding suffix
        for common_enc in SUPPORTED_ENCODINGS:
            suffix = f"-{common_enc}"
            if stem.lower().endswith(suffix.lower()):
                stem = stem[: -len(suffix)]
                break

        output_file = os.path.join(
            output_dir, f"{stem}-{target_encoding}{source_path.suffix}"
        )

        # Skip if the target encoding matches the source and same file
        try:
            if (target_encoding.lower() == source_encoding.lower()
                and os.path.samefile(input_file, output_file)):
                logger.info(
                    "Skipping conversion to %s as it matches source encoding",
                    target_encoding,
                )
                results[target_encoding] = True
                continue
        except OSError:
            # Files don't exist or can't be compared, continue with conversion
            pass

        # Convert the file
        conversion_success = convert_subtitle_encoding(
            input_file, output_file, target_encoding, source_encoding
        )
        results[target_encoding] = conversion_success

    return results


def get_recommended_encodings(language_code: str) -> List[str]:
    """Get recommended encodings for a specific language.

    Args:
        language_code: ISO language code (e.g., 'th', 'zh-CN', 'ja')

    Returns:
        List of recommended encodings for the language
    """
    # Default to UTF-8 and common Western encodings
    default_encodings = ["utf-8", "utf-8-sig", "cp1252", "iso8859-1", "iso8859-15"]

    # Get language code without region
    base_lang = language_code.split("-")[0]

    # Return recommended encodings or default
    return LANGUAGE_ENCODINGS.get(
        language_code, LANGUAGE_ENCODINGS.get(base_lang, default_encodings)
    )


def validate_encoding(encoding: str) -> bool:
    """Validate that an encoding is supported.

    Args:
        encoding: Encoding name to validate

    Returns:
        True if encoding is supported
    """
    try:
        # Try to encode/decode a test string
        test_string = "Test string with unicode: äöü中文"
        test_string.encode(encoding)
        return True
    except (LookupError, TypeError):
        return False


def get_file_encoding_info(file_path: str) -> Dict[str, Union[str, int, bool, None]]:
    """Get information about a file's encoding.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with encoding information
    """
    info: Dict[str, Union[str, int, bool, None]] = {
        "detected_encoding": None,
        "confidence": None,
        "file_size": None,
        "readable": False
    }

    if not os.path.exists(file_path):
        return info

    try:
        info["file_size"] = os.path.getsize(file_path)
        detected = detect_encoding(file_path)
        if detected:
            info["detected_encoding"] = detected
            info["readable"] = True
            # Simple confidence based on successful detection
            info["confidence"] = "high" if detected in ["utf-8", "utf-8-sig"] else "medium"
    except (OSError, UnicodeError, LookupError) as e:
        logger.debug("Error getting encoding info for %s: %s", file_path, e)

    return info


def normalize_encoding_name(encoding: str) -> str:
    """Normalize encoding name to a standard format.

    Args:
        encoding: Encoding name to normalize

    Returns:
        Normalized encoding name
    """
    # Common aliases and their standard names
    aliases = {
        "utf8": "utf-8",
        "utf-8-bom": "utf-8-sig",
        "utf-8-with-bom": "utf-8-sig",
        "windows-1252": "cp1252",
        "windows-1251": "cp1251",
        "windows-874": "cp874",
        "thai": "tis-620",
        "shift-jis": "shift_jis",
        "shiftjis": "shift_jis",
        "euc_jp": "euc-jp",
        "euc_kr": "euc-kr",
        "gb2312": "gb2312",
        "gbk": "cp936",
        "big-5": "big5",
    }

    normalized = encoding.lower().strip()
    return aliases.get(normalized, normalized)
