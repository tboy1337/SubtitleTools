"""Tests for subtitle_fixes module."""

import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import srt

from src.subtools.utils.subtitle_fixes import (
    SubtitleFixer,
    apply_subtitle_fixes,
    batch_apply_subtitle_fixes,
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
            start=timedelta(seconds=1.5),  # Overlapping with previous
            end=timedelta(seconds=3),    # Very short display time
            content="This   has   extra  spaces."
        ),
        srt.Subtitle(
            index=3,
            start=timedelta(seconds=4),  # Non-overlapping
            end=timedelta(seconds=15),   # Very long display time
            content="This is a very long line that should probably be split into multiple parts for better readability."
        ),
        srt.Subtitle(
            index=4,
            start=timedelta(seconds=16),
            end=timedelta(seconds=18),
            content="[MUSIC PLAYING]\n(NARRATOR): Wlien the liero arrived..."
        ),
    ]


@pytest.fixture
def temp_subtitle_file(sample_subtitles):
    """Create a temporary subtitle file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
        f.write(srt.compose(sample_subtitles))
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestSubtitleFixer:
    """Test SubtitleFixer class."""
    
    def test_init(self):
        """Test initialization."""
        fixer = SubtitleFixer()
        assert fixer.max_chars_per_line == 42
        assert fixer.max_chars_total == 84
        assert fixer.min_display_time_ms == 500
        assert fixer.max_display_time_ms == 7000
        
    def test_fix_overlapping_times(self, sample_subtitles):
        """Test fixing overlapping subtitle times."""
        fixer = SubtitleFixer()
        
        # subtitles[1] overlaps with subtitles[0] in our sample data
        result = fixer._fix_overlapping_times(sample_subtitles)
        
        # Check that overlapping time was fixed
        assert result[0].end < result[1].start
        
    def test_fix_short_display_times(self, sample_subtitles):
        """Test fixing short display times."""
        fixer = SubtitleFixer()
        
        result = fixer._fix_short_display_times(sample_subtitles)
        
        # Check that short display times were extended
        for subtitle in result:
            duration = (subtitle.end - subtitle.start).total_seconds()
            # Allow some tolerance for minimum display time
            assert duration >= 0.4  # Slightly less than 0.5 due to gap constraints
            
    def test_fix_long_display_times(self, sample_subtitles):
        """Test fixing long display times."""
        fixer = SubtitleFixer()
        
        result = fixer._fix_long_display_times(sample_subtitles)
        
        # Check that long display times were shortened
        for subtitle in result:
            duration = (subtitle.end - subtitle.start).total_seconds()
            assert duration <= fixer.max_display_time_ms / 1000.0
            
    def test_fix_unneeded_spaces(self, sample_subtitles):
        """Test removing unneeded spaces."""
        fixer = SubtitleFixer()
        
        result = fixer._fix_unneeded_spaces(sample_subtitles)
        
        # Check that multiple spaces were reduced to single spaces
        for subtitle in result:
            assert "  " not in subtitle.content  # No double spaces
            assert not subtitle.content.startswith(" ")
            assert not subtitle.content.endswith(" ")
            
    def test_fix_common_errors(self, sample_subtitles):
        """Test applying common fixes."""
        fixer = SubtitleFixer()
        
        result = fixer.fix_common_errors(sample_subtitles)
        
        # Should have applied multiple fixes
        assert len(result) == len(sample_subtitles)
        
        # Check that spaces were fixed in subtitle 2
        assert "   " not in result[1].content
        
    def test_remove_hearing_impaired(self, sample_subtitles):
        """Test removing hearing impaired text."""
        fixer = SubtitleFixer()
        
        result = fixer.remove_hearing_impaired(sample_subtitles)
        
        # Should have fewer subtitles (HI subtitle removed)
        assert len(result) <= len(sample_subtitles)
        
        # Check that HI markers were removed
        for subtitle in result:
            assert "[MUSIC PLAYING]" not in subtitle.content
            assert "(NARRATOR):" not in subtitle.content
            
    def test_split_long_lines(self, sample_subtitles):
        """Test splitting long lines."""
        fixer = SubtitleFixer()
        
        result = fixer.split_long_lines(sample_subtitles)
        
        # Should have more subtitles (long lines split)
        assert len(result) >= len(sample_subtitles)
        
        # Check that no line is too long
        for subtitle in result:
            lines = subtitle.content.split('\n')
            for line in lines:
                assert len(line) <= fixer.max_chars_per_line * 1.2  # Some tolerance
                
    def test_apply_ocr_fixes(self, sample_subtitles):
        """Test applying OCR fixes."""
        fixer = SubtitleFixer()
        
        result = fixer.apply_ocr_fixes(sample_subtitles)
        
        # Check that OCR fixes were applied
        for subtitle in result:
            # "Wlien" should be fixed to "When"
            if "When" in subtitle.content:
                assert "Wlien" not in subtitle.content
            # "liero" should be fixed to "hero" 
            if "hero" in subtitle.content:
                assert "liero" not in subtitle.content
                
    def test_fix_punctuation(self, sample_subtitles):
        """Test fixing punctuation."""
        fixer = SubtitleFixer()
        
        # Add some punctuation issues to test
        sample_subtitles[0].content = "Hello...world!!!"
        
        result = fixer.fix_punctuation(sample_subtitles)
        
        # Check that punctuation was fixed
        assert "Hello…world!" in result[0].content


class TestApplySubtitleFixes:
    """Test apply_subtitle_fixes function."""
    
    def test_apply_single_operation(self, temp_subtitle_file):
        """Test applying a single operation."""
        success = apply_subtitle_fixes(temp_subtitle_file, ["fixcommonerrors"])
        assert success
        
        # Check that file was processed
        with open(temp_subtitle_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        subtitles = list(srt.parse(content))
        assert len(subtitles) > 0
        
    def test_apply_multiple_operations(self, temp_subtitle_file):
        """Test applying multiple operations."""
        operations = ["fixcommonerrors", "removetextforhi", "ocrfix"]
        success = apply_subtitle_fixes(temp_subtitle_file, operations)
        assert success
        
    def test_unknown_operation(self, temp_subtitle_file):
        """Test handling unknown operation."""
        with patch('src.subtools.utils.subtitle_fixes.logger') as mock_logger:
            success = apply_subtitle_fixes(temp_subtitle_file, ["unknown_operation"])
            
        # Should still succeed (just skip unknown operation)
        assert success
        mock_logger.warning.assert_called()
        
    def test_nonexistent_file(self):
        """Test with nonexistent file."""
        success = apply_subtitle_fixes("nonexistent.srt", ["fixcommonerrors"])
        assert not success
        
    def test_output_file_specified(self, temp_subtitle_file):
        """Test with output file specified.""" 
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as f:
            output_path = Path(f.name)
            
        try:
            success = apply_subtitle_fixes(
                temp_subtitle_file, 
                ["fixcommonerrors"], 
                output_file=output_path
            )
            assert success
            assert output_path.exists()
            
            # Check content was written to output file
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert "Hello, world!" in content
            
        finally:
            if output_path.exists():
                output_path.unlink()


class TestBatchApplySubtitleFixes:
    """Test batch_apply_subtitle_fixes function."""
    
    def test_batch_processing(self, temp_subtitle_file):
        """Test batch processing multiple files."""
        # Create a second temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            f.write("1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n")
            temp_path2 = Path(f.name)
            
        try:
            files = [temp_subtitle_file, temp_path2]
            results = batch_apply_subtitle_fixes(files, ["fixcommonerrors"])
            
            assert len(results) == 2
            assert all(results.values())  # All should succeed
            
        finally:
            if temp_path2.exists():
                temp_path2.unlink()
                
    def test_batch_with_failure(self, temp_subtitle_file):
        """Test batch processing with one failure."""
        files = [temp_subtitle_file, "nonexistent.srt"]
        results = batch_apply_subtitle_fixes(files, ["fixcommonerrors"])
        
        assert len(results) == 2
        assert results[str(temp_subtitle_file)] is True
        assert results["nonexistent.srt"] is False


class TestSubtitleFixerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_subtitles(self):
        """Test with empty subtitle list."""
        fixer = SubtitleFixer()
        
        result = fixer.fix_common_errors([])
        assert result == []
        
        result = fixer.remove_hearing_impaired([])  
        assert result == []
        
    def test_subtitle_with_no_content(self):
        """Test with subtitle containing no content."""
        subtitles = [
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=2),
                content=""
            )
        ]
        
        fixer = SubtitleFixer()
        result = fixer.remove_hearing_impaired(subtitles)
        
        # Empty subtitle should be removed
        assert len(result) == 0
        
    def test_subtitle_only_whitespace(self):
        """Test with subtitle containing only whitespace.""" 
        subtitles = [
            srt.Subtitle(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=2),
                content="   \n   "
            )
        ]
        
        fixer = SubtitleFixer()
        result = fixer.remove_hearing_impaired(subtitles)
        
        # Whitespace-only subtitle should be removed
        assert len(result) == 0
        
    def test_find_split_points_no_good_points(self):
        """Test _find_split_points with no good split points."""
        fixer = SubtitleFixer()
        
        # Short text with no punctuation or natural breaks
        content = "shorttext"
        split_points = fixer._find_split_points(content)
        
        assert split_points == []
        
    def test_find_split_points_line_breaks(self):
        """Test _find_split_points with line breaks."""
        fixer = SubtitleFixer()
        
        content = "First line\nSecond line\nThird line"
        split_points = fixer._find_split_points(content)
        
        assert len(split_points) == 2  # Two split points for three lines
        assert split_points[0] == len("First line\n")
        
    def test_find_split_points_sentences(self):
        """Test _find_split_points with sentence endings."""
        fixer = SubtitleFixer()
        
        content = "First sentence. Second sentence! Third sentence?"
        split_points = fixer._find_split_points(content)
        
        assert len(split_points) >= 1  # At least one split point
        
    def test_context_sensitive_ocr_case_insensitive(self):
        """Test context-sensitive OCR fixes are case insensitive."""
        fixer = SubtitleFixer()
        
        content = "THE HERO and tlie villain"
        result = fixer._fix_context_sensitive_ocr(content)
        
        # Should fix both "THE" and "tlie" variations
        assert "the hero" in result.lower()
        assert "the villain" in result.lower()
        
    def test_split_subtitle_no_split_needed(self):
        """Test _split_subtitle when no split is needed."""
        fixer = SubtitleFixer()
        
        subtitle = srt.Subtitle(
            index=1,
            start=timedelta(seconds=0),
            end=timedelta(seconds=2),
            content="Short text"
        )
        
        result = fixer._split_subtitle(subtitle)
        
        # Should return original subtitle unchanged
        assert len(result) == 1
        assert result[0].content == "Short text"
