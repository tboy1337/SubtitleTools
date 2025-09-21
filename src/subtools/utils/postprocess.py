"""Post-processing utilities for SubtitleTools.

This module provides functionality for post-processing subtitle files,
including integration with Subtitle Edit CLI through Docker.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def apply_subtitle_edit_postprocess(
    subtitle_file: Union[str, Path],
    operations: List[str],
    output_format: str = "subrip",
    docker_image: str = "seconv:1.0",
) -> bool:
    """Apply post-processing operations using Subtitle Edit CLI via Docker.

    Args:
        subtitle_file: Path to the subtitle file
        operations: List of Subtitle Edit CLI operations
        output_format: Output format (subrip, ass, vtt, etc.)
        docker_image: Docker image name for Subtitle Edit CLI

    Returns:
        True if post-processing was successful
    """
    try:
        subtitle_path = Path(subtitle_file)
        if not subtitle_path.exists():
            logger.error("Subtitle file not found: %s", subtitle_file)
            return False

        # Build Docker command
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{subtitle_path.parent.absolute()}:/subtitles",
            docker_image,
            f"/subtitles/{subtitle_path.name}",
            output_format
        ]

        # Add operations
        for operation in operations:
            docker_cmd.append(operation)

        logger.info("Running post-processing: %s", ' '.join(docker_cmd))

        result = subprocess.run(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1800,  # 30 minute timeout
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            logger.error("Post-processing failed (code %s): %s", result.returncode, error_msg)
            return False

        logger.info("Post-processing completed successfully")
        if result.stdout.strip():
            logger.debug("Post-processing output: %s", result.stdout.strip())

        return True

    except subprocess.TimeoutExpired:
        logger.error("Post-processing timed out after 30 minutes")
        return False
    except (subprocess.SubprocessError, OSError, TimeoutError) as e:
        logger.error("Error during post-processing: %s", e)
        return False


def generate_docker_command(
    subtitle_file: Union[str, Path],
    *,  # Force keyword-only arguments
    fix_common_errors: bool = False,
    remove_hi: bool = False,
    auto_split_long_lines: bool = False,
    fix_punctuation: bool = False,
    ocr_fix: bool = False,
    convert_to: Optional[str] = None,
    docker_image: str = "seconv:1.0",
) -> Optional[str]:
    """Generate a Docker command for Subtitle Edit CLI post-processing.

    Args:
        subtitle_file: Path to the subtitle file
        fix_common_errors: Apply common error fixes
        remove_hi: Remove hearing impaired text
        auto_split_long_lines: Split long subtitle lines
        fix_punctuation: Fix punctuation issues
        ocr_fix: Apply OCR fixes
        convert_to: Convert to specified format
        docker_image: Docker image name

    Returns:
        Docker command string or None if no operations specified
    """
    operations = []

    if fix_common_errors:
        operations.append("/fixcommonerrors")
    if remove_hi:
        operations.append("/removetextforhi")
    if auto_split_long_lines:
        operations.append("/splitlonglines")
    if fix_punctuation:
        operations.append("/fixpunctuation")
    if ocr_fix:
        operations.append("/ocrfix")

    if not operations and not convert_to:
        return None

    subtitle_path = Path(subtitle_file)
    output_format = convert_to if convert_to else "subrip"

    docker_cmd = (
        f'docker run --rm -v "{subtitle_path.parent.absolute()}:/subtitles" '
        f'{docker_image} /subtitles/{subtitle_path.name} {output_format}'
    )

    for operation in operations:
        docker_cmd += f" {operation}"

    return docker_cmd


def check_docker_available() -> bool:
    """Check if Docker is available and running.

    Returns:
        True if Docker is available
    """
    try:
        result = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_subtitle_edit_image(docker_image: str = "seconv:1.0") -> bool:
    """Check if the Subtitle Edit CLI Docker image is available.

    Args:
        docker_image: Docker image name to check

    Returns:
        True if image is available
    """
    try:
        result = subprocess.run(
            ["docker", "images", "-q", docker_image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def validate_postprocess_environment() -> Dict[str, bool]:
    """Validate the post-processing environment.

    Returns:
        Dictionary with validation results
    """
    checks = {
        "docker_available": check_docker_available(),
        "subtitle_edit_image": False,
    }

    if checks["docker_available"]:
        checks["subtitle_edit_image"] = check_subtitle_edit_image()

    return checks


def get_available_operations() -> List[str]:
    """Get list of available post-processing operations.

    Returns:
        List of operation names
    """
    return [
        "fix_common_errors",
        "remove_hi",
        "auto_split_long_lines",
        "fix_punctuation",
        "ocr_fix",
    ]


def get_supported_output_formats() -> List[str]:
    """Get list of supported output formats for conversion.

    Returns:
        List of format names
    """
    return ["srt", "subrip", "ass", "ssa", "vtt", "stl", "smi"]


def apply_common_fixes(subtitle_file: Union[str, Path]) -> bool:
    """Apply common subtitle fixes using Subtitle Edit CLI.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if fixes were applied successfully
    """
    return apply_subtitle_edit_postprocess(
        subtitle_file,
        ["/fixcommonerrors"],
        "subrip"
    )


def remove_hearing_impaired(subtitle_file: Union[str, Path]) -> bool:
    """Remove hearing impaired text from subtitles.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_edit_postprocess(
        subtitle_file,
        ["/removetextforhi"],
        "subrip"
    )


def split_long_lines(subtitle_file: Union[str, Path]) -> bool:
    """Split long subtitle lines automatically.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_edit_postprocess(
        subtitle_file,
        ["/splitlonglines"],
        "subrip"
    )


def fix_punctuation(subtitle_file: Union[str, Path]) -> bool:
    """Fix punctuation issues in subtitles.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_edit_postprocess(
        subtitle_file,
        ["/fixpunctuation"],
        "subrip"
    )


def apply_ocr_fixes(subtitle_file: Union[str, Path]) -> bool:
    """Apply OCR error fixes to subtitles.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_edit_postprocess(
        subtitle_file,
        ["/ocrfix"],
        "subrip"
    )


def convert_subtitle_format(
    subtitle_file: Union[str, Path],
    output_format: str
) -> bool:
    """Convert subtitle to a different format.

    Args:
        subtitle_file: Path to the subtitle file
        output_format: Target format (srt, ass, vtt, etc.)

    Returns:
        True if conversion was successful
    """
    return apply_subtitle_edit_postprocess(
        subtitle_file,
        [],
        output_format
    )


def batch_postprocess(
    subtitle_files: List[Union[str, Path]],
    operations: List[str],
    output_format: str = "subrip",
) -> Dict[str, bool]:
    """Apply post-processing to multiple subtitle files.

    Args:
        subtitle_files: List of subtitle file paths
        operations: List of operations to apply
        output_format: Output format

    Returns:
        Dictionary mapping file paths to success status
    """
    results = {}

    for subtitle_file in subtitle_files:
        try:
            success = apply_subtitle_edit_postprocess(
                subtitle_file,
                operations,
                output_format
            )
            results[str(subtitle_file)] = success
        except (subprocess.SubprocessError, OSError) as e:
            logger.error("Error processing %s: %s", subtitle_file, e)
            results[str(subtitle_file)] = False

    return results
