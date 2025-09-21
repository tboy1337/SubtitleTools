"""Subtitle processing module for SubtitleTools.

This module provides functionality for parsing, manipulating, and saving subtitle files,
adapted from the original SubtranSlate implementation with additional features.
"""

import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, cast

import srt

from ..config.settings import DEFAULT_ENCODING
from ..utils.common import validate_file_exists, ensure_directory
from ..utils.encoding import detect_encoding

logger = logging.getLogger(__name__)

# Try importing jieba for Chinese segmentation
try:
    import jieba
    JIEBA_AVAILABLE = True
    logger.debug("Jieba library available for Chinese segmentation")
except ImportError:
    JIEBA_AVAILABLE = False
    logger.debug("Jieba library not found - Chinese segmentation may be limited")


class SubtitleLike(Protocol):
    """Protocol for subtitle objects with required attributes."""
    index: int
    start: timedelta
    end: timedelta
    content: str
    proprietary: str


# Type aliases
SubtitleList = List[SubtitleLike]


class SubtitleError(Exception):
    """Exception raised for subtitle processing errors."""


class Splitter:
    """Sentence splitter for text processing."""

    def __init__(self) -> None:
        # Improved regex pattern for sentence splitting
        self.pattern = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<!\b[A-Za-z]\.)(?<=\.|\?|\!)\s"
        )

    def split(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Use regex to split on sentence boundaries
        sentences = self.pattern.split(text)

        # Filter out empty strings and strip whitespace
        return [s.strip() for s in sentences if s.strip()]


class SubtitleProcessor:
    """Process subtitle files for various operations."""

    def __init__(self) -> None:
        self.splitter = Splitter()
        logger.debug("Initialized SubtitleProcessor")

    def parse_file(
        self,
        file_path: Union[str, Path],
        encoding: str = DEFAULT_ENCODING
    ) -> SubtitleList:
        """Parse an SRT file into subtitle objects.

        Args:
            file_path: Path to the SRT file
            encoding: File encoding

        Returns:
            List of srt.Subtitle objects

        Raises:
            SubtitleError: If file can't be parsed
        """
        file_path_obj = validate_file_exists(file_path)

        try:
            # Try to detect encoding if default fails
            detected_encoding = encoding
            if encoding == DEFAULT_ENCODING:
                detected = detect_encoding(str(file_path_obj))
                if detected:
                    detected_encoding = detected
                    logger.debug("Detected encoding: %s", detected_encoding)

            with open(file_path_obj, "r", encoding=detected_encoding) as f:
                content = f.read()

            subtitles_generator = cast(Any, srt.parse(content))
            subtitles = list(subtitles_generator)
            logger.info("Parsed %d subtitles from %s", len(subtitles), file_path_obj)

            if not subtitles:
                logger.warning("No subtitles found in %s", file_path_obj)

            return cast(List[SubtitleLike], subtitles)

        except UnicodeDecodeError as e:
            raise SubtitleError(
                f"Failed to decode subtitle file {file_path_obj} with encoding {detected_encoding}: {e}"
            ) from e
        except Exception as e:
            # Handle both SRT parse errors and other exceptions
            raise SubtitleError(f"Error parsing subtitle file {file_path_obj}: {e}") from e

    def save_file(
        self,
        subtitles: SubtitleList,
        file_path: Union[str, Path],
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        """Save subtitle objects to an SRT file.

        Args:
            subtitles: List of subtitle objects
            file_path: Output file path
            encoding: File encoding

        Raises:
            SubtitleError: If file can't be saved
        """
        file_path_obj = Path(file_path)

        try:
            # Ensure output directory exists
            ensure_directory(file_path_obj.parent)

            # Generate SRT content
            content = cast(str, srt.compose(subtitles))

            with open(file_path_obj, "w", encoding=encoding) as f:
                f.write(content)

            logger.info("Saved %d subtitles to %s", len(subtitles), file_path_obj)

        except Exception as e:
            raise SubtitleError(f"Error saving subtitle file {file_path_obj}: {e}") from e

    def extract_text(self, subtitles: SubtitleList) -> List[str]:
        """Extract text content from subtitle objects.

        Args:
            subtitles: List of subtitle objects

        Returns:
            List of text strings
        """
        return [sub.content for sub in subtitles]

    def split_sentences(
        self,
        subtitles: SubtitleList,
        language_code: Optional[str] = None
    ) -> List[str]:
        """Split subtitle content into sentences for better translation.

        Args:
            subtitles: List of subtitle objects
            language_code: Optional language code for language-specific processing

        Returns:
            List of sentences
        """
        if not subtitles:
            return []

        # Extract all text
        all_text = " ".join(sub.content for sub in subtitles)

        # Handle Chinese text segmentation if available
        if language_code and language_code.startswith("zh") and JIEBA_AVAILABLE:
            # Use jieba for Chinese sentence segmentation
            try:
                segments_generator = cast(Any, jieba.cut(all_text, cut_all=False))
                sentences = cast(List[str], list(segments_generator))
                # Filter meaningful segments
                filtered_sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 1]
                logger.debug("Chinese segmentation produced %d segments", len(filtered_sentences))
                return filtered_sentences
            except (ImportError, AttributeError, TypeError, Exception) as e:
                logger.warning("Chinese segmentation failed, using fallback: %s", e)

        # Use general sentence splitter
        sentences = self.splitter.split(all_text)
        logger.debug("Sentence splitting produced %d sentences", len(sentences))
        return sentences

    def reconstruct_subtitles(
        self,
        original_subtitles: SubtitleList,
        translated_sentences: List[str],
        space: bool = False,
        both: bool = True,
    ) -> SubtitleList:
        """Reconstruct subtitles with translated content.

        Args:
            original_subtitles: Original subtitle objects
            translated_sentences: List of translated sentences
            space: Whether target language uses spaces between words
            both: Whether to include both original and translated text

        Returns:
            List of reconstructed subtitle objects
        """
        if not original_subtitles:
            return []

        if not translated_sentences:
            logger.warning("No translated sentences provided")
            return original_subtitles

        # Join translated sentences
        translated_text = " ".join(translated_sentences) if space else "".join(translated_sentences)

        # Split translated text back into subtitle segments
        reconstructed: List[SubtitleLike] = []

        # Simple approach: distribute translated text proportionally
        total_original_length = sum(len(sub.content) for sub in original_subtitles)

        if total_original_length == 0:
            # Handle edge case of empty content
            return original_subtitles

        current_pos = 0

        for i, sub in enumerate(original_subtitles):
            original_length = len(sub.content)

            # Calculate proportion of translated text for this subtitle
            proportion = original_length / total_original_length
            translated_length = int(len(translated_text) * proportion)

            # Extract corresponding translated segment
            if i == len(original_subtitles) - 1:
                # Last subtitle gets remaining text
                translated_segment = translated_text[current_pos:].strip()
            else:
                translated_segment = translated_text[current_pos:current_pos + translated_length].strip()
                current_pos += translated_length

            # Create new subtitle content
            if both:
                # Include both original and translated text
                if sub.content.strip() and translated_segment:
                    new_content = f"{translated_segment}\n{sub.content}"
                else:
                    new_content = translated_segment or sub.content
            else:
                # Only translated text
                new_content = translated_segment or sub.content

            # Create new subtitle object
            new_subtitle = cast(SubtitleLike, srt.Subtitle(
                index=sub.index,
                start=sub.start,
                end=sub.end,
                content=new_content,
                proprietary=getattr(sub, 'proprietary', "")
            ))

            reconstructed.append(new_subtitle)

        logger.info("Reconstructed %d subtitles", len(reconstructed))
        return reconstructed

    def merge_subtitles(
        self,
        subtitles1: SubtitleList,
        subtitles2: SubtitleList,
        separator: str = "\n",
    ) -> SubtitleList:
        """Merge two subtitle lists by combining content.

        Args:
            subtitles1: First subtitle list
            subtitles2: Second subtitle list
            separator: Separator between merged content

        Returns:
            Merged subtitle list
        """
        if not subtitles1:
            return subtitles2
        if not subtitles2:
            return subtitles1

        min_length = min(len(subtitles1), len(subtitles2))
        merged: List[SubtitleLike] = []

        for i in range(min_length):
            sub1 = subtitles1[i]
            sub2 = subtitles2[i]

            # Use timing from first subtitle list
            merged_content = f"{sub1.content}{separator}{sub2.content}"

            merged_subtitle = cast(SubtitleLike, cast(Any, srt.Subtitle)(
                index=sub1.index,
                start=sub1.start,
                end=sub1.end,
                content=merged_content,
                proprietary=str(getattr(sub1, 'proprietary', ""))
            ))

            merged.append(merged_subtitle)

        # Add remaining subtitles from longer list
        if len(subtitles1) > min_length:
            merged.extend(subtitles1[min_length:])
        elif len(subtitles2) > min_length:
            for i in range(min_length, len(subtitles2)):
                sub2 = subtitles2[i]
                # Adjust index to continue sequence
                adjusted_subtitle = cast(SubtitleLike, cast(Any, srt.Subtitle)(
                    index=len(merged) + 1,
                    start=sub2.start,
                    end=sub2.end,
                    content=sub2.content,
                    proprietary=str(getattr(sub2, 'proprietary', ""))
                ))
                merged.append(adjusted_subtitle)

        logger.info("Merged %d and %d subtitles into %d", len(subtitles1), len(subtitles2), len(merged))
        return merged

    def filter_subtitles(
        self,
        subtitles: SubtitleList,
        *,  # Force keyword-only arguments
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        regex_pattern: Optional[str] = None,
    ) -> SubtitleList:
        """Filter subtitles based on various criteria.

        Args:
            subtitles: List of subtitle objects
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            regex_pattern: Regex pattern that content must match

        Returns:
            Filtered subtitle list
        """
        if not subtitles:
            return subtitles

        filtered = []

        regex = None
        if regex_pattern:
            try:
                regex = re.compile(regex_pattern)
            except re.error as e:
                logger.warning("Invalid regex pattern '%s': %s", regex_pattern, e)

        for sub in subtitles:
            # Duration check
            duration = (sub.end - sub.start).total_seconds()
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue

            # Length check
            text_length = len(sub.content)
            if min_length is not None and text_length < min_length:
                continue
            if max_length is not None and text_length > max_length:
                continue

            # Regex check
            if regex and not regex.search(sub.content):
                continue

            filtered.append(sub)

        logger.info("Filtered %d subtitles to %d", len(subtitles), len(filtered))
        return filtered

    def adjust_timing(
        self,
        subtitles: SubtitleList,
        offset_seconds: float = 0.0,
        speed_factor: float = 1.0,
    ) -> SubtitleList:
        """Adjust subtitle timing.

        Args:
            subtitles: List of subtitle objects
            offset_seconds: Time offset in seconds (positive = later, negative = earlier)
            speed_factor: Speed factor (>1 = faster, <1 = slower)

        Returns:
            Subtitles with adjusted timing
        """
        if not subtitles:
            return subtitles

        offset_delta = timedelta(seconds=offset_seconds)
        adjusted: List[SubtitleLike] = []

        for sub in subtitles:
            # Apply speed factor first
            if speed_factor != 1.0:
                start_seconds = sub.start.total_seconds() / speed_factor
                end_seconds = sub.end.total_seconds() / speed_factor
                new_start = timedelta(seconds=start_seconds)
                new_end = timedelta(seconds=end_seconds)
            else:
                new_start = sub.start
                new_end = sub.end

            # Apply offset
            new_start += offset_delta
            new_end += offset_delta

            # Ensure non-negative times
            new_start = max(new_start, timedelta(0))
            if new_end < new_start:
                new_end = new_start + timedelta(milliseconds=100)

            adjusted_subtitle = cast(SubtitleLike, cast(Any, srt.Subtitle)(
                index=sub.index,
                start=new_start,
                end=new_end,
                content=sub.content,
                proprietary=str(getattr(sub, 'proprietary', ""))
            ))

            adjusted.append(adjusted_subtitle)

        logger.info("Adjusted timing for %d subtitles", len(adjusted))
        return adjusted

    def get_statistics(self, subtitles: SubtitleList) -> Dict[str, Union[int, float]]:
        """Get statistics about subtitle file.

        Args:
            subtitles: List of subtitle objects

        Returns:
            Dictionary with statistics
        """
        if not subtitles:
            return {
                "count": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "total_characters": 0,
                "average_characters": 0.0,
                "longest_subtitle": 0,
                "shortest_subtitle": 0,
            }

        durations = [(sub.end - sub.start).total_seconds() for sub in subtitles]
        char_counts = [len(sub.content) for sub in subtitles]

        total_duration = sum(durations)
        total_characters = sum(char_counts)

        stats = {
            "count": len(subtitles),
            "total_duration": round(total_duration, 2),
            "average_duration": round(total_duration / len(subtitles), 2),
            "total_characters": total_characters,
            "average_characters": round(total_characters / len(subtitles), 2),
            "longest_subtitle": max(char_counts),
            "shortest_subtitle": min(char_counts),
            "longest_duration": round(max(durations), 2),
            "shortest_duration": round(min(durations), 2),
        }

        return stats

    def validate_subtitles(self, subtitles: SubtitleList) -> List[str]:
        """Validate subtitle list and return list of issues.

        Args:
            subtitles: List of subtitle objects

        Returns:
            List of validation issues (empty if no issues)
        """
        issues = []

        if not subtitles:
            issues.append("No subtitles found")
            return issues

        prev_end = timedelta(0)

        for i, sub in enumerate(subtitles):
            # Check index sequence
            if sub.index != i + 1:
                issues.append(f"Subtitle {i + 1}: Index mismatch (expected {i + 1}, got {sub.index})")

            # Check timing
            if sub.start >= sub.end:
                issues.append(f"Subtitle {i + 1}: Invalid timing (start >= end)")

            if sub.start < prev_end:
                issues.append(f"Subtitle {i + 1}: Overlaps with previous subtitle")

            prev_end = sub.end

            # Check content
            if not sub.content.strip():
                issues.append(f"Subtitle {i + 1}: Empty content")

        logger.info("Validated %d subtitles, found %d issues", len(subtitles), len(issues))
        return issues
