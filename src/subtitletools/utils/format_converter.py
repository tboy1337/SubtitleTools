"""Subtitle format converter - Python implementation.

This module provides subtitle format conversion functionality that was previously
handled by SubtitleEdit-CLI. It supports conversion between various subtitle formats
including SRT, ASS/SSA, VTT, and others.

The implementation is based on the SubtitleEdit-CLI format handlers to provide
equivalent functionality without external dependencies.
"""

import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
import srt

logger = logging.getLogger(__name__)

class FormatConverter:
    """Subtitle format converter."""

    def __init__(self) -> None:
        self.supported_formats: Dict[str, Callable[[List[srt.Subtitle]], str]] = {
            'srt': self._to_srt,
            'subrip': self._to_srt,
            'vtt': self._to_vtt,
            'webvtt': self._to_vtt,
            'ass': self._to_ass,
            'ssa': self._to_ssa,
            'smi': self._to_sami,
            'sami': self._to_sami,
        }

    def convert_subtitle_format(
        self,
        input_file: Union[str, Path],
        output_format: str,
        output_file: Optional[Union[str, Path]] = None
    ) -> bool:
        """Convert subtitle file to different format.
        
        Args:
            input_file: Path to input subtitle file
            output_format: Target format (srt, vtt, ass, etc.)
            output_file: Path to output file (auto-generated if None)
            
        Returns:
            True if conversion successful
        """
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                logger.error("Input file not found: %s", input_file)
                return False

            # Read and parse input file
            subtitles = self._read_subtitle_file(input_path)
            if not subtitles:
                logger.error("Could not parse subtitle file: %s", input_file)
                return False

            # Determine output file path
            if output_file:
                output_path = Path(output_file)
            else:
                output_path = input_path.with_suffix(f'.{output_format.lower()}')

            # Convert to target format
            output_format = output_format.lower()
            if output_format not in self.supported_formats:
                logger.error("Unsupported output format: %s", output_format)
                return False

            converter_func = self.supported_formats[output_format]
            content = converter_func(subtitles)

            # Write output file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info("Successfully converted %s to %s format: %s",
                       input_path.name, output_format.upper(), output_path.name)
            return True

        except Exception as e:
            logger.error("Error converting subtitle file %s: %s", input_file, e)
            return False

    def _read_subtitle_file(self, file_path: Path) -> Optional[List[srt.Subtitle]]:
        """Read subtitle file and return parsed subtitles."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Try to parse as SRT first (most common)
            try:
                return list(cast(List[srt.Subtitle], srt.parse(content)))
            except Exception:
                pass

            # Try other formats
            if file_path.suffix.lower() in ['.vtt', '.webvtt']:
                return self._parse_vtt(content)
            if file_path.suffix.lower() in ['.ass', '.ssa']:
                return self._parse_ass(content)
            if file_path.suffix.lower() in ['.smi', '.sami']:
                return self._parse_sami(content)
            # Default case - Try to auto-detect format
            return self._auto_detect_and_parse(content)

        except Exception as e:
            logger.error("Error reading subtitle file %s: %s", file_path, e)
            return None

    def _auto_detect_and_parse(self, content: str) -> Optional[List[srt.Subtitle]]:
        """Auto-detect subtitle format and parse."""
        content_lower = content.lower().strip()

        # WebVTT detection
        if content_lower.startswith('webvtt'):
            return self._parse_vtt(content)

        # ASS/SSA detection
        if '[script info]' in content_lower:
            return self._parse_ass(content)

        # SAMI detection
        if '<sami>' in content_lower:
            return self._parse_sami(content)

        # Default to SRT
        try:
            return list(cast(List[srt.Subtitle], srt.parse(content)))
        except Exception:
            logger.warning("Could not auto-detect subtitle format")
            return None

    def _extract_vtt_timestamp_line(self, lines: List[str], start_index: int) -> Tuple[Optional[str], int]:
        """Extract timestamp line from VTT content, handling cue identifiers.
        
        Returns:
            Tuple of (timestamp_line, actual_index) or (None, start_index) if not found
        """
        line = lines[start_index].strip()

        # If line starts with digit, it's likely a timestamp
        if line and line[0].isdigit():
            return line, start_index

        # Otherwise, look for timestamp in next few lines (might be after cue identifier)
        for j in range(start_index + 1, min(start_index + 3, len(lines))):
            if '-->' in lines[j]:
                return lines[j].strip(), j

        return None, start_index

    def _parse_vtt(self, content: str) -> List[srt.Subtitle]:
        """Parse WebVTT format."""
        subtitles = []
        lines = content.split('\n')

        i = 0
        subtitle_index = 1

        # Skip header and empty lines
        while i < len(lines):
            line = lines[i].strip()
            if line.lower().startswith('webvtt'):
                i += 1
                break
            i += 1

        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines
            if not line:
                i += 1
                continue

            # Check if this is a timestamp line
            if '-->' in line:
                # Parse timestamp
                try:
                    # Extract timestamp line, handling cue identifiers
                    timestamp_line, actual_i = self._extract_vtt_timestamp_line(lines, i)
                    if timestamp_line is None:
                        i += 1
                        continue

                    i = actual_i  # Update index to actual timestamp line position
                    start_str, end_str = timestamp_line.split('-->')
                    start_time = self._parse_vtt_timestamp(start_str.strip())
                    end_time = self._parse_vtt_timestamp(end_str.strip())

                    # Collect subtitle text
                    i += 1
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1

                    if text_lines:
                        # Clean VTT tags
                        text = '\n'.join(text_lines)
                        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                        text = re.sub(r'{[^}]*}', '', text)  # Remove style tags

                        subtitle = cast(srt.Subtitle, srt.Subtitle(
                            index=subtitle_index,
                            start=start_time,
                            end=end_time,
                            content=text
                        ))
                        subtitles.append(subtitle)
                        subtitle_index += 1

                except Exception as e:
                    logger.debug("Error parsing VTT timestamp: %s", e)

            i += 1

        return subtitles

    def _parse_vtt_timestamp(self, timestamp_str: str) -> timedelta:
        """Parse VTT timestamp format."""
        # VTT format: HH:MM:SS.mmm or MM:SS.mmm
        timestamp_str = timestamp_str.strip()

        # Remove any extra formatting
        timestamp_str = re.sub(r'\s+.*$', '', timestamp_str)

        if timestamp_str.count(':') == 2:
            # HH:MM:SS.mmm format
            hours_str, minutes_str, seconds_str = timestamp_str.split(':')
            hours = int(hours_str)
            minutes = int(minutes_str)
            seconds = seconds_str
        else:
            # MM:SS.mmm format
            hours = 0
            minutes_str, seconds_str = timestamp_str.split(':')
            minutes = int(minutes_str)
            seconds = seconds_str

        # Parse seconds and milliseconds
        if '.' in seconds:
            sec_str, ms_str = seconds.split('.')
            seconds_int = int(sec_str)
            milliseconds = int(ms_str.ljust(3, '0')[:3])  # Pad or truncate to 3 digits
        else:
            seconds_int = int(seconds)
            milliseconds = 0

        return timedelta(hours=hours, minutes=minutes, seconds=seconds_int, milliseconds=milliseconds)

    def _parse_ass(self, content: str) -> List[srt.Subtitle]:
        """Parse ASS/SSA format."""
        subtitles = []
        lines = content.split('\n')

        # Find the events section
        in_events = False
        subtitle_index = 1

        for line in lines:
            line = line.strip()

            if line.lower() == '[events]':
                in_events = True
                continue
            if line.startswith('[') and line.endswith(']'):
                in_events = False
                continue

            if in_events and line.startswith('Dialogue:'):
                try:
                    # Parse dialogue line
                    parts = line[9:].split(',', 9)  # Skip "Dialogue:" prefix
                    if len(parts) >= 10:
                        start_time = self._parse_ass_timestamp(parts[1])
                        end_time = self._parse_ass_timestamp(parts[2])
                        text = parts[9]

                        # Clean ASS formatting
                        text = re.sub(r'{[^}]*}', '', text)  # Remove ASS tags
                        text = text.replace('\\N', '\n')     # Convert line breaks
                        text = text.replace('\\n', '\n')

                        subtitle = cast(srt.Subtitle, srt.Subtitle(
                            index=subtitle_index,
                            start=start_time,
                            end=end_time,
                            content=text
                        ))
                        subtitles.append(subtitle)
                        subtitle_index += 1

                except Exception as e:
                    logger.debug("Error parsing ASS dialogue line: %s", e)

        return subtitles

    def _parse_ass_timestamp(self, timestamp_str: str) -> timedelta:
        """Parse ASS timestamp format (H:MM:SS.cc)."""
        timestamp_str = timestamp_str.strip()

        # Format: H:MM:SS.cc (centiseconds)
        time_part = timestamp_str.split('.')[0]
        centiseconds = int(timestamp_str.split('.')[1]) if '.' in timestamp_str else 0

        time_parts = time_part.split(':')
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
        milliseconds = centiseconds * 10

        return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

    def _parse_sami(self, content: str) -> List[srt.Subtitle]:
        """Parse SAMI format."""
        subtitles = []

        # Find all SYNC tags
        sync_pattern = re.compile(r'<SYNC\s+Start=(\d+)>([^<]*(?:<[^/][^>]*>[^<]*)*?)(?=<SYNC|$)', re.IGNORECASE | re.DOTALL)
        matches = cast(List[Tuple[str, str]], sync_pattern.findall(content))

        subtitle_index = 1

        for i, (start_ms_str, text_content) in enumerate(matches):
            try:
                start_time = timedelta(milliseconds=int(start_ms_str))

                # Determine end time (start of next subtitle or default duration)
                if i + 1 < len(matches):
                    end_time = timedelta(milliseconds=int(matches[i + 1][0]))
                else:
                    end_time = start_time + timedelta(seconds=3)  # Default 3 seconds

                # Clean HTML tags from text
                clean_text = re.sub(r'<[^>]+>', '', text_content)
                clean_text = clean_text.strip()

                if clean_text:
                    subtitle = cast(srt.Subtitle, srt.Subtitle(
                        index=subtitle_index,
                        start=start_time,
                        end=end_time,
                        content=clean_text
                    ))
                    subtitles.append(subtitle)
                    subtitle_index += 1

            except Exception as e:
                logger.debug("Error parsing SAMI sync: %s", e)

        return subtitles

    def _to_srt(self, subtitles: List[srt.Subtitle]) -> str:
        """Convert to SRT format."""
        return str(cast(str, srt.compose(subtitles)))

    def _to_vtt(self, subtitles: List[srt.Subtitle]) -> str:
        """Convert to WebVTT format."""
        lines = ['WEBVTT', '']

        for subtitle in subtitles:
            # Format timestamps
            start_str = self._format_vtt_timestamp(cast(Any, subtitle.start))
            end_str = self._format_vtt_timestamp(cast(Any, subtitle.end))

            lines.append(f'{start_str} --> {end_str}')
            lines.append(cast(str, subtitle.content))
            lines.append('')

        return '\n'.join(lines)

    def _format_vtt_timestamp(self, td: timedelta) -> str:
        """Format timestamp for VTT."""
        total_seconds = int(td.total_seconds())
        milliseconds = int(td.microseconds / 1000)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'

    def _to_ass(self, subtitles: List[srt.Subtitle]) -> str:
        """Convert to ASS format."""
        lines = [
            '[Script Info]',
            'Title: Converted Subtitle',
            'ScriptType: v4.00+',
            '',
            '[V4+ Styles]',
            'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding',
            'Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1',
            '',
            '[Events]',
            'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text'
        ]

        for subtitle in subtitles:
            start_str = self._format_ass_timestamp(cast(Any, subtitle.start))
            end_str = self._format_ass_timestamp(cast(Any, subtitle.end))

            # Convert line breaks to ASS format
            text = cast(str, subtitle.content).replace('\n', '\\N')

            line = f'Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}'
            lines.append(line)

        return '\n'.join(lines)

    def _format_ass_timestamp(self, td: timedelta) -> str:
        """Format timestamp for ASS."""
        total_seconds = int(td.total_seconds())
        centiseconds = int(td.microseconds / 10000)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f'{hours:01d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}'

    def _to_ssa(self, subtitles: List[srt.Subtitle]) -> str:
        """Convert to SSA format (similar to ASS but older version)."""
        # SSA is similar to ASS but with different header
        ass_content = self._to_ass(subtitles)

        # Replace ASS-specific headers with SSA equivalents
        ass_content = ass_content.replace('ScriptType: v4.00+', 'ScriptType: v4.00')
        ass_content = ass_content.replace('[V4+ Styles]', '[V4 Styles]')

        return ass_content

    def _to_sami(self, subtitles: List[srt.Subtitle]) -> str:
        """Convert to SAMI format."""
        lines = [
            '<SAMI>',
            '<HEAD>',
            '<TITLE>Converted Subtitle</TITLE>',
            '<STYLE TYPE="text/css">',
            '<!--',
            'P { font-family: Arial; font-size: 12pt; color: white; background-color: black; text-align: center; }',
            '-->',
            '</STYLE>',
            '</HEAD>',
            '<BODY>',
            ''
        ]

        for subtitle in subtitles:
            start_ms = int(cast(Any, subtitle.start).total_seconds() * 1000)

            # Add sync point
            lines.append(f'<SYNC Start={start_ms}>')
            lines.append('<P>' + cast(str, subtitle.content).replace('\n', '<BR>'))
            lines.append('')

        # Add final sync to end subtitles
        if subtitles:
            last_end_ms = int(cast(Any, subtitles[-1].end).total_seconds() * 1000)
            lines.append(f'<SYNC Start={last_end_ms}>')
            lines.append('<P>&nbsp;')

        lines.extend(['', '</BODY>', '</SAMI>'])

        return '\n'.join(lines)


def convert_subtitle_format(
    input_file: Union[str, Path],
    output_format: str,
    output_file: Optional[Union[str, Path]] = None
) -> bool:
    """Convert subtitle file to different format.
    
    Args:
        input_file: Path to input subtitle file
        output_format: Target format (srt, vtt, ass, etc.)
        output_file: Path to output file (auto-generated if None)
        
    Returns:
        True if conversion successful
    """
    converter = FormatConverter()
    return converter.convert_subtitle_format(input_file, output_format, output_file)


def batch_convert_subtitle_format(
    input_files: List[Union[str, Path]],
    output_format: str
) -> Dict[str, bool]:
    """Convert multiple subtitle files to different format.
    
    Args:
        input_files: List of input subtitle file paths
        output_format: Target format (srt, vtt, ass, etc.)
        
    Returns:
        Dictionary mapping file paths to success status
    """
    converter = FormatConverter()
    results = {}

    for input_file in input_files:
        try:
            success = converter.convert_subtitle_format(input_file, output_format)
            results[str(input_file)] = success
        except Exception as e:
            logger.error("Error converting %s: %s", input_file, e)
            results[str(input_file)] = False

    return results


def get_supported_formats() -> List[str]:
    """Get list of supported subtitle formats."""
    return list(cast(Dict[str, Any], FormatConverter().supported_formats).keys())
