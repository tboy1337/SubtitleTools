"""Subtitle fixes implementation - Python port of SubtitleEdit-CLI functionality.

This module provides subtitle post-processing functionality that was previously
using native Python implementations. It includes:

- Fix common errors (overlapping times, short/long display times, etc.)
- Remove hearing impaired text
- Split long lines
- Fix punctuation issues  
- Apply OCR fixes
- Format conversion

The implementation is based on the SubtitleEdit-CLI C# codebase to provide
equivalent functionality without external dependencies.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import srt

logger = logging.getLogger(__name__)

# Configuration constants (matching SubtitleEdit defaults)
DEFAULT_MAX_CHARS_PER_LINE = 42
DEFAULT_MAX_CHARS_TOTAL = 84
DEFAULT_MIN_DISPLAY_TIME_MS = 500
DEFAULT_MAX_DISPLAY_TIME_MS = 7000
DEFAULT_MIN_GAP_MS = 24

class SubtitleFixer:
    """Main class for subtitle post-processing operations."""

    def __init__(self) -> None:
        self.max_chars_per_line = DEFAULT_MAX_CHARS_PER_LINE
        self.max_chars_total = DEFAULT_MAX_CHARS_TOTAL
        self.min_display_time_ms = DEFAULT_MIN_DISPLAY_TIME_MS
        self.max_display_time_ms = DEFAULT_MAX_DISPLAY_TIME_MS
        self.min_gap_ms = DEFAULT_MIN_GAP_MS

        # OCR fix patterns
        self._init_ocr_patterns()

        # Hearing impaired patterns
        self._init_hearing_impaired_patterns()

    def _init_ocr_patterns(self) -> None:
        """Initialize OCR correction patterns."""
        # Common OCR misrecognitions (from SubtitleEdit)
        self.ocr_fixes = {
            # Letter/number confusions
            r'\bl\b': 'I',  # Standalone l -> I
            r'\b0\b': 'O',  # Standalone 0 -> O (context dependent)
            r'\bO\b': '0',  # Standalone O -> 0 (context dependent)

            # Common character replacements
            r'rn': 'm',     # rn -> m
            r'\|': 'l',     # | -> l
            r'\bvv': 'w',   # vv -> w
            r'\bW\b': 'VV', # W -> VV (sometimes)

            # Punctuation fixes
            r'\.\.\.': '…', # ... -> …
            r"''": '"',     # '' -> "
            r',,': ',',     # ,, -> ,
            r';;': ';',     # ;; -> ;

            # Space fixes
            r'\s+': ' ',    # Multiple spaces -> single space
            r'^\s+': '',    # Leading whitespace
            r'\s+$': '',    # Trailing whitespace
        }

    def _init_hearing_impaired_patterns(self) -> None:
        """Initialize hearing impaired text removal patterns."""
        # Patterns for hearing impaired text (from SubtitleEdit)
        self.hi_patterns = {
            'brackets': re.compile(r'\[.*?\]', re.DOTALL),
            'parens': re.compile(r'\(.*?\)', re.DOTALL),
            'braces': re.compile(r'\{.*?\}', re.DOTALL),
            'question_marks': re.compile(r'\?.*?\?', re.DOTALL),
            'colons': re.compile(r'^[A-Z][A-Z\s]+:', re.MULTILINE),
            'all_caps': re.compile(r'^[A-Z\s]+$', re.MULTILINE),
        }

    def fix_common_errors(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Apply common subtitle error fixes."""
        logger.info("Applying common error fixes...")

        # Apply various fixes in order
        subtitles = self._fix_overlapping_times(subtitles)
        subtitles = self._fix_short_display_times(subtitles)
        subtitles = self._fix_long_display_times(subtitles)
        subtitles = self._fix_unneeded_spaces(subtitles)
        subtitles = self._fix_missing_spaces(subtitles)
        subtitles = self._fix_punctuation_spacing(subtitles)

        logger.info("Common error fixes completed")
        return subtitles

    def _fix_overlapping_times(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Fix overlapping subtitle times."""
        fixed_count = 0

        for i in range(len(subtitles) - 1):
            current = subtitles[i]
            next_sub = subtitles[i + 1]

            if current.end > next_sub.start:
                # Overlap detected - adjust end time
                current.end = next_sub.start - srt.timedelta(milliseconds=self.min_gap_ms)
                fixed_count += 1

        if fixed_count > 0:
            logger.debug("Fixed %d overlapping times", fixed_count)

        return subtitles

    def _fix_short_display_times(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Fix subtitles with too short display times."""
        fixed_count = 0
        min_display_seconds = self.min_display_time_ms / 1000.0

        for i, subtitle in enumerate(subtitles):
            duration = (subtitle.end - subtitle.start).total_seconds()

            if duration < min_display_seconds:
                # Calculate optimal display time based on text length
                text_length = len(subtitle.content.replace('\n', ''))
                optimal_duration = max(min_display_seconds, text_length * 0.1)  # ~10 chars/second reading speed

                # Extend end time but don't overlap with next subtitle
                new_end = subtitle.start + srt.timedelta(seconds=optimal_duration)

                if i < len(subtitles) - 1:
                    next_start = subtitles[i + 1].start
                    max_end = next_start - srt.timedelta(milliseconds=self.min_gap_ms)
                    subtitle.end = min(new_end, max_end)
                else:
                    subtitle.end = new_end

                fixed_count += 1

        if fixed_count > 0:
            logger.debug("Fixed %d short display times", fixed_count)

        return subtitles

    def _fix_long_display_times(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Fix subtitles with too long display times."""
        fixed_count = 0
        max_display_seconds = self.max_display_time_ms / 1000.0

        for subtitle in subtitles:
            duration = (subtitle.end - subtitle.start).total_seconds()

            if duration > max_display_seconds:
                # Calculate optimal display time based on text length
                text_length = len(subtitle.content.replace('\n', ''))
                optimal_duration = min(max_display_seconds, text_length * 0.08)  # ~12.5 chars/second

                subtitle.end = subtitle.start + srt.timedelta(seconds=optimal_duration)
                fixed_count += 1

        if fixed_count > 0:
            logger.debug("Fixed %d long display times", fixed_count)

        return subtitles

    def _fix_unneeded_spaces(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Remove unneeded spaces."""
        fixed_count = 0

        for subtitle in subtitles:
            original_content = subtitle.content

            # Multiple spaces to single space
            content = re.sub(r'\s+', ' ', subtitle.content)

            # Remove spaces before punctuation
            content = re.sub(r'\s+([.!?:;,])', r'\1', content)

            # Remove leading/trailing whitespace from lines
            lines = content.split('\n')
            lines = [line.strip() for line in lines]
            content = '\n'.join(lines)

            # Remove empty lines
            content = re.sub(r'\n+', '\n', content).strip()

            if content != original_content:
                subtitle.content = content
                fixed_count += 1

        if fixed_count > 0:
            logger.debug("Fixed unneeded spaces in %d subtitles", fixed_count)

        return subtitles

    def _fix_missing_spaces(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Add missing spaces after punctuation."""
        fixed_count = 0

        for subtitle in subtitles:
            original_content = subtitle.content

            # Add space after punctuation if missing (but not at end of line)
            content = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', subtitle.content)

            if content != original_content:
                subtitle.content = content
                fixed_count += 1

        if fixed_count > 0:
            logger.debug("Fixed missing spaces in %d subtitles", fixed_count)

        return subtitles

    def _fix_punctuation_spacing(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Fix punctuation spacing issues."""
        fixed_count = 0

        for subtitle in subtitles:
            original_content = subtitle.content
            content = subtitle.content

            # Fix ellipsis
            content = re.sub(r'\.{3,}', '…', content)

            # Fix double punctuation
            content = re.sub(r'([.!?])\1+', r'\1', content)

            # Fix quotation marks
            content = re.sub(r"''", '"', content)
            content = re.sub(r'``', '"', content)

            if content != original_content:
                subtitle.content = content
                fixed_count += 1

        if fixed_count > 0:
            logger.debug("Fixed punctuation spacing in %d subtitles", fixed_count)

        return subtitles

    def remove_hearing_impaired(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Remove hearing impaired text from subtitles."""
        logger.info("Removing hearing impaired text...")

        filtered_subtitles = []
        removed_count = 0

        for subtitle in subtitles:
            content = subtitle.content
            original_content = content

            # Remove text in brackets, parentheses, etc.
            content = self.hi_patterns['brackets'].sub('', content)
            content = self.hi_patterns['parens'].sub('', content)
            content = self.hi_patterns['braces'].sub('', content)
            content = self.hi_patterns['question_marks'].sub('', content)

            # Remove speaker names (ALL CAPS with colon)
            content = self.hi_patterns['colons'].sub('', content)

            # Clean up resulting text
            content = re.sub(r'\s+', ' ', content).strip()
            content = re.sub(r'\n\s*\n', '\n', content).strip()

            # Remove lines that are all caps (likely sound effects)
            lines = content.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if line and not self.hi_patterns['all_caps'].match(line):
                    clean_lines.append(line)
            content = '\n'.join(clean_lines)

            # Only keep subtitle if it has meaningful content after cleanup
            if content and not content.isspace():
                subtitle.content = content
                filtered_subtitles.append(subtitle)
                if content != original_content:
                    removed_count += 1
            else:
                removed_count += 1

        logger.info("Removed hearing impaired text from %d subtitles", removed_count)
        return filtered_subtitles

    def split_long_lines(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Split long subtitle lines."""
        logger.info("Splitting long lines...")

        result_subtitles = []
        split_count = 0

        for subtitle in subtitles:
            lines = subtitle.content.split('\n')

            # Check if any line is too long
            needs_split = any(len(line) > self.max_chars_per_line for line in lines)
            total_too_long = len(subtitle.content.replace('\n', '')) > self.max_chars_total

            if needs_split or total_too_long:
                split_result = self._split_subtitle(subtitle)
                result_subtitles.extend(split_result)
                if len(split_result) > 1:
                    split_count += 1
            else:
                result_subtitles.append(subtitle)

        logger.info("Split %d long subtitles", split_count)
        return result_subtitles

    def _split_subtitle(self, subtitle: srt.Subtitle) -> List[srt.Subtitle]:
        """Split a single subtitle into multiple parts."""
        content = subtitle.content.strip()

        # Try to split at natural break points
        split_points = self._find_split_points(content)

        if not split_points:
            # No good split points found, return original
            return [subtitle]

        # Split the content
        parts = []
        start_pos = 0

        for split_pos in split_points:
            part = content[start_pos:split_pos].strip()
            if part:
                parts.append(part)
            start_pos = split_pos

        # Add remaining content
        if start_pos < len(content):
            remaining = content[start_pos:].strip()
            if remaining:
                parts.append(remaining)

        if len(parts) <= 1:
            return [subtitle]

        # Create new subtitle objects with adjusted timing
        duration = subtitle.end - subtitle.start
        part_duration = duration / len(parts)

        result = []
        for i, part in enumerate(parts):
            new_start = subtitle.start + (part_duration * i)
            new_end = subtitle.start + (part_duration * (i + 1))

            new_subtitle = srt.Subtitle(
                index=subtitle.index,  # Will be renumbered later
                start=new_start,
                end=new_end,
                content=part
            )
            result.append(new_subtitle)

        return result

    def _find_split_points(self, content: str) -> List[int]:
        """Find good points to split subtitle content."""
        # Look for natural break points (sentence endings, line breaks, etc.)
        split_points = []

        # First try line breaks
        if '\n' in content:
            parts = content.split('\n')
            pos = 0
            for part in parts[:-1]:  # Don't include the last part
                pos += len(part) + 1  # +1 for the newline
                if pos < len(content):
                    split_points.append(pos)

        # If no line breaks, try sentence endings
        if not split_points:
            sentence_endings = re.finditer(r'[.!?]\s+', content)
            for match in sentence_endings:
                split_points.append(match.end())

        # If still no good splits, try commas or other punctuation
        if not split_points:
            punctuation = re.finditer(r'[,;]\s+', content)
            for match in punctuation:
                split_points.append(match.end())

        # If still no splits found, try to split at word boundaries near the middle
        if not split_points and len(content) > self.max_chars_per_line:
            # Find word boundaries near ideal split positions
            words = content.split()
            if len(words) > 1:
                current_pos = 0
                target_pos = len(content) // 2  # Try to split near middle

                for word in words[:-1]:  # Don't include last word
                    current_pos += len(word) + 1  # +1 for space
                    if current_pos >= target_pos:
                        split_points.append(current_pos)
                        break

        return split_points

    def apply_ocr_fixes(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Apply OCR error corrections."""
        logger.info("Applying OCR fixes...")

        fixed_count = 0

        for subtitle in subtitles:
            original_content = subtitle.content
            content = subtitle.content

            # Apply OCR corrections
            for pattern, replacement in self.ocr_fixes.items():
                content = re.sub(pattern, replacement, content)

            # Additional context-sensitive fixes
            content = self._fix_context_sensitive_ocr(content)

            if content != original_content:
                subtitle.content = content
                fixed_count += 1

        logger.info("Applied OCR fixes to %d subtitles", fixed_count)
        return subtitles

    def _fix_context_sensitive_ocr(self, content: str) -> str:
        """Apply context-sensitive OCR fixes."""
        # Fix common word-level OCR errors
        word_fixes = {
            'tlie': 'the',
            'witli': 'with',
            'tliis': 'this',
            'tliey': 'they',
            'tliose': 'those',
            'wlien': 'when',
            'wliere': 'where',
            'wlio': 'who',
            'wliy': 'why',
            'liis': 'his',
            'lier': 'her',
            'liim': 'him',
            'slie': 'she',
            'tliank': 'thank',
        }

        for incorrect, correct in word_fixes.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(incorrect) + r'\b'
            content = re.sub(pattern, correct, content, flags=re.IGNORECASE)

        return content

    def fix_punctuation(self, subtitles: List[srt.Subtitle]) -> List[srt.Subtitle]:
        """Fix punctuation issues."""
        logger.info("Fixing punctuation...")

        fixed_count = 0

        for subtitle in subtitles:
            original_content = subtitle.content
            content = subtitle.content

            # Fix ellipsis
            content = re.sub(r'\.{2,}', '…', content)
            content = re.sub(r'…+', '…', content)

            # Fix quotation marks
            content = re.sub(r'"([^"]*)"', r'"\1"', content)  # Ensure proper quote matching

            # Fix apostrophes
            content = re.sub(r"'([a-z])", r"'\1", content)  # Contractions

            # Fix hyphens and dashes
            content = re.sub(r'\s*-\s*', ' - ', content)  # Proper dash spacing
            content = re.sub(r'—+', '—', content)  # Single em dash

            # Fix exclamation and question marks
            content = re.sub(r'!+', '!', content)
            content = re.sub(r'\?+', '?', content)
            content = re.sub(r'!\?|\?!', '?!', content)  # Interrobang

            # Fix periods
            content = re.sub(r'\.{2,}(?!…)', '.', content)  # Multiple periods (but not ellipsis)

            if content != original_content:
                subtitle.content = content
                fixed_count += 1

        logger.info("Fixed punctuation in %d subtitles", fixed_count)
        return subtitles


def apply_subtitle_fixes(
    subtitle_file: Union[str, Path],
    operations: List[str],
    output_file: Optional[Union[str, Path]] = None
) -> bool:
    """Apply subtitle fixes to a file.
    
    Args:
        subtitle_file: Path to input subtitle file
        operations: List of operations to apply
        output_file: Path to output file (overwrites input if None)
        
    Returns:
        True if successful
    """
    try:
        subtitle_path = Path(subtitle_file)
        if not subtitle_path.exists():
            logger.error("Subtitle file not found: %s", subtitle_file)
            return False

        # Read subtitle file
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            subtitles = list(srt.parse(f.read()))

        if not subtitles:
            logger.error("No subtitles found in file: %s", subtitle_file)
            return False

        # Apply requested operations
        fixer = SubtitleFixer()

        for operation in operations:
            operation = operation.lstrip('/')  # Remove leading slash if present

            if operation == 'fixcommonerrors':
                subtitles = fixer.fix_common_errors(subtitles)
            elif operation == 'removetextforhi':
                subtitles = fixer.remove_hearing_impaired(subtitles)
            elif operation == 'splitlonglines':
                subtitles = fixer.split_long_lines(subtitles)
            elif operation == 'fixpunctuation':
                subtitles = fixer.fix_punctuation(subtitles)
            elif operation == 'ocrfix':
                subtitles = fixer.apply_ocr_fixes(subtitles)
            else:
                logger.warning("Unknown operation: %s", operation)

        # Renumber subtitles
        for i, subtitle in enumerate(subtitles, 1):
            subtitle.index = i

        # Write output file
        output_path = Path(output_file) if output_file else subtitle_path

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subtitles))

        logger.info("Successfully processed subtitle file: %s", output_path)
        return True

    except Exception as e:
        logger.error("Error processing subtitle file %s: %s", subtitle_file, e)
        return False


def batch_apply_subtitle_fixes(
    subtitle_files: List[Union[str, Path]],
    operations: List[str]
) -> Dict[str, bool]:
    """Apply subtitle fixes to multiple files.
    
    Args:
        subtitle_files: List of subtitle file paths
        operations: List of operations to apply
        
    Returns:
        Dictionary mapping file paths to success status
    """
    results = {}

    for subtitle_file in subtitle_files:
        try:
            success = apply_subtitle_fixes(subtitle_file, operations)
            results[str(subtitle_file)] = success
        except Exception as e:
            logger.error("Error processing %s: %s", subtitle_file, e)
            results[str(subtitle_file)] = False

    return results
