"""Tests for format_converter module."""

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import srt

from subtitletools.utils.format_converter import (
    FormatConverter,
    convert_subtitle_format,
    batch_convert_subtitle_format,
    get_supported_formats,
)


@pytest.fixture
def sample_subtitles():
    """Create sample subtitles for testing."""
    return [
        srt.Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="Hello, world!"
        ),
        srt.Subtitle(
            index=2,
            start=timedelta(seconds=3),
            end=timedelta(seconds=5),
            content="This is a test\nwith multiple lines."
        ),
        srt.Subtitle(
            index=3,
            start=timedelta(seconds=6),
            end=timedelta(seconds=8),
            content="Final subtitle."
        ),
    ]


@pytest.fixture
def temp_srt_file(sample_subtitles):
    """Create a temporary SRT file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
        f.write(srt.compose(sample_subtitles))
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestFormatConverter:
    """Test FormatConverter class."""

    def test_init(self):
        """Test initialization."""
        converter = FormatConverter()
        assert 'srt' in converter.supported_formats
        assert 'vtt' in converter.supported_formats
        assert 'ass' in converter.supported_formats

    def test_convert_subtitle_format_srt_to_vtt(self, temp_srt_file):
        """Test converting SRT to VTT."""
        converter = FormatConverter()

        output_file = temp_srt_file.with_suffix('.vtt')

        try:
            success = converter.convert_subtitle_format(temp_srt_file, 'vtt', output_file)
            assert success
            assert output_file.exists()

            # Check VTT content
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()

            assert content.startswith('WEBVTT')
            assert '00:00:00.000 --> 00:00:02.000' in content
            assert 'Hello, world!' in content

        finally:
            if output_file.exists():
                output_file.unlink()

    def test_convert_subtitle_format_srt_to_ass(self, temp_srt_file):
        """Test converting SRT to ASS."""
        converter = FormatConverter()

        output_file = temp_srt_file.with_suffix('.ass')

        try:
            success = converter.convert_subtitle_format(temp_srt_file, 'ass', output_file)
            assert success
            assert output_file.exists()

            # Check ASS content
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()

            assert '[Script Info]' in content
            assert '[V4+ Styles]' in content
            assert '[Events]' in content
            assert 'Dialogue:' in content
            assert 'Hello, world!' in content

        finally:
            if output_file.exists():
                output_file.unlink()

    def test_convert_subtitle_format_auto_output_file(self, temp_srt_file):
        """Test conversion with auto-generated output filename."""
        converter = FormatConverter()

        success = converter.convert_subtitle_format(temp_srt_file, 'vtt')
        assert success

        # Should create .vtt file with same base name
        vtt_file = temp_srt_file.with_suffix('.vtt')
        try:
            assert vtt_file.exists()

        finally:
            if vtt_file.exists():
                vtt_file.unlink()

    def test_convert_nonexistent_file(self):
        """Test converting nonexistent file."""
        converter = FormatConverter()

        success = converter.convert_subtitle_format('nonexistent.srt', 'vtt')
        assert not success

    def test_convert_unsupported_format(self, temp_srt_file):
        """Test converting to unsupported format."""
        converter = FormatConverter()

        success = converter.convert_subtitle_format(temp_srt_file, 'unsupported')
        assert not success

    def test_read_subtitle_file_srt(self, temp_srt_file):
        """Test reading SRT file."""
        converter = FormatConverter()

        subtitles = converter._read_subtitle_file(temp_srt_file)
        assert subtitles is not None
        assert len(subtitles) == 3
        assert subtitles[0].content == "Hello, world!"

    def test_parse_vtt_timestamp(self):
        """Test parsing VTT timestamps."""
        converter = FormatConverter()

        # Test HH:MM:SS.mmm format
        td = converter._parse_vtt_timestamp("00:01:23.456")
        assert td == timedelta(hours=0, minutes=1, seconds=23, milliseconds=456)

        # Test MM:SS.mmm format
        td = converter._parse_vtt_timestamp("01:23.456")
        assert td == timedelta(minutes=1, seconds=23, milliseconds=456)

    def test_parse_ass_timestamp(self):
        """Test parsing ASS timestamps."""
        converter = FormatConverter()

        # Test H:MM:SS.cc format (centiseconds)
        td = converter._parse_ass_timestamp("1:23:45.67")
        assert td == timedelta(hours=1, minutes=23, seconds=45, milliseconds=670)

    def test_format_vtt_timestamp(self):
        """Test formatting VTT timestamps."""
        converter = FormatConverter()

        td = timedelta(hours=1, minutes=23, seconds=45, milliseconds=678)
        formatted = converter._format_vtt_timestamp(td)
        assert formatted == "01:23:45.678"

    def test_format_ass_timestamp(self):
        """Test formatting ASS timestamps."""
        converter = FormatConverter()

        td = timedelta(hours=1, minutes=23, seconds=45, milliseconds=670)
        formatted = converter._format_ass_timestamp(td)
        assert formatted == "1:23:45.67"

    def test_to_srt(self, sample_subtitles):
        """Test converting to SRT format."""
        converter = FormatConverter()

        result = converter._to_srt(sample_subtitles)

        assert "1\n00:00:00,000 --> 00:00:02,000\nHello, world!" in result
        assert "2\n00:00:03,000 --> 00:00:05,000\nThis is a test\nwith multiple lines." in result

    def test_to_vtt(self, sample_subtitles):
        """Test converting to VTT format."""
        converter = FormatConverter()

        result = converter._to_vtt(sample_subtitles)

        assert result.startswith('WEBVTT')
        assert '00:00:00.000 --> 00:00:02.000' in result
        assert 'Hello, world!' in result

    def test_to_ass(self, sample_subtitles):
        """Test converting to ASS format."""
        converter = FormatConverter()

        result = converter._to_ass(sample_subtitles)

        assert '[Script Info]' in result
        assert '[V4+ Styles]' in result
        assert '[Events]' in result
        assert 'Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello, world!' in result

    def test_to_ssa(self, sample_subtitles):
        """Test converting to SSA format."""
        converter = FormatConverter()

        result = converter._to_ssa(sample_subtitles)

        assert '[Script Info]' in result
        assert 'ScriptType: v4.00' in result  # SSA version
        assert '[V4 Styles]' in result        # SSA style section

    def test_to_sami(self, sample_subtitles):
        """Test converting to SAMI format."""
        converter = FormatConverter()

        result = converter._to_sami(sample_subtitles)

        assert '<SAMI>' in result
        assert '<HEAD>' in result
        assert '<BODY>' in result
        assert '<SYNC Start=0>' in result
        assert 'Hello, world!' in result


class TestVTTParsing:
    """Test VTT format parsing."""

    def test_parse_vtt_basic(self):
        """Test parsing basic VTT content."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.000
Hello, world!

00:00:03.000 --> 00:00:05.000
Second subtitle
"""
        converter = FormatConverter()
        subtitles = converter._parse_vtt(vtt_content)

        assert len(subtitles) == 2
        assert subtitles[0].content == "Hello, world!"
        assert subtitles[1].content == "Second subtitle"

    def test_parse_vtt_with_tags(self):
        """Test parsing VTT with style tags."""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.000
<i>Italic text</i>

00:00:03.000 --> 00:00:05.000
<b>Bold text</b> with {color:red}color{/color}
"""
        converter = FormatConverter()
        subtitles = converter._parse_vtt(vtt_content)

        assert len(subtitles) == 2
        assert subtitles[0].content == "Italic text"  # HTML tags removed
        assert subtitles[1].content == "Bold text with color"  # All tags removed


class TestASSParsing:
    """Test ASS format parsing."""

    def test_parse_ass_basic(self):
        """Test parsing basic ASS content."""
        ass_content = """[Script Info]
Title: Test

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello, world!
Dialogue: 0,0:00:03.00,0:00:05.00,Default,,0,0,0,,Second subtitle
"""
        converter = FormatConverter()
        subtitles = converter._parse_ass(ass_content)

        assert len(subtitles) == 2
        assert subtitles[0].content == "Hello, world!"
        assert subtitles[1].content == "Second subtitle"

    def test_parse_ass_with_formatting(self):
        """Test parsing ASS with formatting tags."""
        ass_content = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,{\\i1}Italic text{\\i0}
Dialogue: 0,0:00:03.00,0:00:05.00,Default,,0,0,0,,Line 1\\NLine 2
"""
        converter = FormatConverter()
        subtitles = converter._parse_ass(ass_content)

        assert len(subtitles) == 2
        assert subtitles[0].content == "Italic text"  # ASS tags removed
        assert subtitles[1].content == "Line 1\nLine 2"  # \\N converted to newline


class TestSAMIParsing:
    """Test SAMI format parsing."""

    def test_parse_sami_basic(self):
        """Test parsing basic SAMI content."""
        sami_content = """<SAMI>
<HEAD>
<TITLE>Test</TITLE>
</HEAD>
<BODY>

<SYNC Start=0>
<P>Hello, world!

<SYNC Start=3000>
<P>Second subtitle

<SYNC Start=6000>
<P>&nbsp;

</BODY>
</SAMI>"""
        converter = FormatConverter()
        subtitles = converter._parse_sami(sami_content)

        assert len(subtitles) == 2  # Third sync clears subtitles
        assert subtitles[0].content == "Hello, world!"
        assert subtitles[1].content == "Second subtitle"


class TestModuleFunctions:
    """Test module-level functions."""

    def test_convert_subtitle_format_function(self, temp_srt_file):
        """Test convert_subtitle_format function."""
        output_file = temp_srt_file.with_suffix('.vtt')

        try:
            success = convert_subtitle_format(temp_srt_file, 'vtt', output_file)
            assert success
            assert output_file.exists()

        finally:
            if output_file.exists():
                output_file.unlink()

    def test_batch_convert_subtitle_format_function(self, temp_srt_file):
        """Test batch_convert_subtitle_format function."""
        # Create a second temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
            temp_path2 = Path(f.name)

        try:
            files = [temp_srt_file, temp_path2]
            results = batch_convert_subtitle_format(files, 'vtt')

            assert len(results) == 2
            assert all(results.values())  # All should succeed

            # Check that VTT files were created
            for file_path in files:
                vtt_file = Path(file_path).with_suffix('.vtt')
                assert vtt_file.exists()
                vtt_file.unlink()  # Cleanup

        finally:
            if temp_path2.exists():
                temp_path2.unlink()

    def test_get_supported_formats_function(self):
        """Test get_supported_formats function."""
        formats = get_supported_formats()

        assert isinstance(formats, list)
        assert 'srt' in formats
        assert 'vtt' in formats
        assert 'ass' in formats


class TestFormatConverterEdgeCases:
    """Test edge cases and error conditions."""

    def test_auto_detect_unknown_format(self):
        """Test auto-detection with unknown format."""
        converter = FormatConverter()

        content = "This is not a valid subtitle format"
        result = converter._auto_detect_and_parse(content)

        # Should return None for unrecognizable content
        assert result is None

    def test_parse_vtt_malformed_timestamp(self):
        """Test parsing VTT with malformed timestamp."""
        vtt_content = """WEBVTT

invalid_timestamp --> 00:00:02.000
Hello, world!
"""
        converter = FormatConverter()
        subtitles = converter._parse_vtt(vtt_content)

        # Should handle error gracefully and return empty list
        assert len(subtitles) == 0

    def test_parse_ass_malformed_dialogue(self):
        """Test parsing ASS with malformed dialogue line."""
        ass_content = """[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: malformed_line
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Valid subtitle
"""
        converter = FormatConverter()
        subtitles = converter._parse_ass(ass_content)

        # Should skip malformed line and parse valid one
        assert len(subtitles) == 1
        assert subtitles[0].content == "Valid subtitle"

    def test_empty_subtitle_list(self):
        """Test format conversion with empty subtitle list."""
        converter = FormatConverter()

        result = converter._to_srt([])
        assert result.strip() == ""

        result = converter._to_vtt([])
        assert result.strip() == "WEBVTT"

        result = converter._to_ass([])
        assert "[Events]" in result  # Should have proper header even with no events

    def test_read_empty_file(self):
        """Test reading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = Path(f.name)

        try:
            converter = FormatConverter()
            subtitles = converter._read_subtitle_file(temp_path)

            # Should return empty list for empty file
            assert subtitles == []

        finally:
            if temp_path.exists():
                temp_path.unlink()
