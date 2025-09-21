"""Post-processing utilities for SubtitleTools.

This module provides functionality for post-processing subtitle files,
now using native Python implementations instead of external dependencies.
Previous SubtitleEdit-CLI functionality has been replaced with
equivalent Python implementations for better portability and performance.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from .format_converter import convert_subtitle_format, get_supported_formats
from .subtitle_fixes import apply_subtitle_fixes, batch_apply_subtitle_fixes

logger = logging.getLogger(__name__)


def apply_subtitle_edit_postprocess(
    subtitle_file: Union[str, Path],
    operations: List[str],
    output_format: str = "subrip",
) -> bool:
    """Apply post-processing operations using native Python implementation.

    Args:
        subtitle_file: Path to the subtitle file
        operations: List of post-processing operations
        output_format: Output format (subrip, ass, vtt, etc.)

    Returns:
        True if post-processing was successful
    """
    try:
        subtitle_path = Path(subtitle_file)
        if not subtitle_path.exists():
            logger.error("Subtitle file not found: %s", subtitle_file)
            return False

        logger.info("Running post-processing with native Python implementation")

        # Apply subtitle fixes using our Python implementation
        success = apply_subtitle_fixes(subtitle_path, operations)
        
        if not success:
            logger.error("Post-processing failed")
            return False

        # Convert format if requested and different from current
        if output_format and output_format.lower() not in ['subrip', 'srt']:
            # Map format names to our converter format names
            format_mapping = {
                'subrip': 'srt',
                'ass': 'ass', 
                'ssa': 'ssa',
                'vtt': 'vtt',
                'webvtt': 'vtt',
                'sami': 'sami',
                'smi': 'sami'
            }
            
            target_format = format_mapping.get(output_format.lower(), output_format.lower())
            
            if target_format in get_supported_formats():
                convert_success = convert_subtitle_format(subtitle_path, target_format)
                if not convert_success:
                    logger.warning("Format conversion failed, keeping original format")
            else:
                logger.warning("Unsupported output format: %s", output_format)

        logger.info("Post-processing completed successfully")
        return True

    except Exception as e:
        logger.error("Error during post-processing: %s", e)
        return False


def generate_processing_description(
    subtitle_file: Union[str, Path],
    *,  # Force keyword-only arguments
    fix_common_errors: bool = False,
    remove_hi: bool = False,
    auto_split_long_lines: bool = False,
    fix_punctuation: bool = False,
    ocr_fix: bool = False,
    convert_to: Optional[str] = None,
) -> Optional[str]:
    """Generate a processing description string.

    Args:
        subtitle_file: Path to the subtitle file
        fix_common_errors: Apply common error fixes
        remove_hi: Remove hearing impaired text
        auto_split_long_lines: Split long subtitle lines
        fix_punctuation: Fix punctuation issues
        ocr_fix: Apply OCR fixes
        convert_to: Convert to specified format

    Returns:
        Processing description string or None if no operations specified
    """
    operations = []

    if fix_common_errors:
        operations.append("fixcommonerrors")
    if remove_hi:
        operations.append("removetextforhi")
    if auto_split_long_lines:
        operations.append("splitlonglines")
    if fix_punctuation:
        operations.append("fixpunctuation")
    if ocr_fix:
        operations.append("ocrfix")

    if not operations and not convert_to:
        return None

    subtitle_path = Path(subtitle_file)
    output_format = convert_to if convert_to else "srt"

    # Return description of what will be processed with native implementation
    cmd_desc = f"Process {subtitle_path.name} with operations: {', '.join(operations)}"
    if convert_to:
        cmd_desc += f" and convert to {output_format}"
        
    return cmd_desc


# Legacy compatibility functions - deprecated


def check_postprocess_available() -> bool:
    """Check if post-processing is available.

    Returns:
        Always returns True since native implementation is always available
    """
    logger.debug("Post-processing availability check - native implementation always available")
    return True


def validate_postprocess_environment() -> Dict[str, bool]:
    """Validate the post-processing environment.

    Returns:
        Dictionary with validation results (always valid for native implementation)
    """
    checks = {
            "postprocess_available": True,   # Native implementation always available
    }

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
    # Use formats from our native converter
    native_formats = get_supported_formats()
    
    # Add legacy format names for compatibility
    legacy_formats = ["subrip", "stl"]
    
    all_formats = list(set(native_formats + legacy_formats))
    return sorted(all_formats)


def apply_common_fixes(subtitle_file: Union[str, Path]) -> bool:
    """Apply common subtitle fixes using native Python implementation.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if fixes were applied successfully
    """
    return apply_subtitle_fixes(subtitle_file, ["fixcommonerrors"])


def remove_hearing_impaired(subtitle_file: Union[str, Path]) -> bool:
    """Remove hearing impaired text from subtitles.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_fixes(subtitle_file, ["removetextforhi"])


def split_long_lines(subtitle_file: Union[str, Path]) -> bool:
    """Split long subtitle lines automatically.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_fixes(subtitle_file, ["splitlonglines"])


def fix_punctuation(subtitle_file: Union[str, Path]) -> bool:
    """Fix punctuation issues in subtitles.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_fixes(subtitle_file, ["fixpunctuation"])


def apply_ocr_fixes(subtitle_file: Union[str, Path]) -> bool:
    """Apply OCR error fixes to subtitles.

    Args:
        subtitle_file: Path to the subtitle file

    Returns:
        True if processing was successful
    """
    return apply_subtitle_fixes(subtitle_file, ["ocrfix"])


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
    from .format_converter import convert_subtitle_format as native_convert
    return native_convert(subtitle_file, output_format)


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
    # Use our native batch processing
    results = batch_apply_subtitle_fixes(subtitle_files, operations)
    
    # Apply format conversion if requested
    if output_format and output_format.lower() not in ['subrip', 'srt']:
        from .format_converter import batch_convert_subtitle_format
        
        # Only convert files that were successfully processed
        successful_files = [f for f, success in results.items() if success]
        if successful_files:
            convert_results = batch_convert_subtitle_format(successful_files, output_format)
            
            # Update results with conversion status
            for file_path, convert_success in convert_results.items():
                if not convert_success:
                    logger.warning("Format conversion failed for %s, keeping original format", file_path)

    return results
