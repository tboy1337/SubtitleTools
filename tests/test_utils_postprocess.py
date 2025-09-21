"""Tests for the postprocess utilities module (native implementation)."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from subtitletools.utils.postprocess import (
    apply_common_fixes,
    apply_ocr_fixes,
    apply_subtitle_edit_postprocess,
    batch_postprocess,
    check_postprocess_available,
    convert_subtitle_format,
    fix_punctuation,
    generate_processing_description,
    get_available_operations,
    get_supported_output_formats,
    remove_hearing_impaired,
    split_long_lines,
    validate_postprocess_environment,
)


class TestApplySubtitleEditPostprocess:
    """Test apply_subtitle_edit_postprocess function with native implementation."""

    def test_apply_subtitle_edit_postprocess_success(self) -> None:
        """Test successful post-processing with native implementation."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest subtitle\n")
            tmp_path = tmp.name

        try:
            result = apply_subtitle_edit_postprocess(
                tmp_path, ["fixcommonerrors"]
            )

            assert result is True
            # Verify file still exists after processing
            assert Path(tmp_path).exists()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_apply_subtitle_edit_postprocess_file_not_found(self) -> None:
        """Test post-processing with non-existent file."""
        result = apply_subtitle_edit_postprocess(
            "nonexistent.srt", ["/fixcommonerrors"]
        )

        assert result is False

    def test_apply_subtitle_edit_postprocess_multiple_operations(self) -> None:
        """Test post-processing with multiple operations."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\n[MUSIC] Test subtitle\n")
            tmp_path = tmp.name

        try:
            result = apply_subtitle_edit_postprocess(
                tmp_path, 
                ["fixcommonerrors", "removetextforhi", "ocrfix"]
            )

            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestGenerateProcessingDescription:
    """Test generate_processing_description function."""

    def test_generate_processing_description_no_operations(self) -> None:
        """Test generating description with no operations."""
        result = generate_processing_description("test.srt")
        assert result is None

    def test_generate_processing_description_fix_common_errors(self) -> None:
        """Test generating description with fix common errors."""
        result = generate_processing_description("test.srt", fix_common_errors=True)

        assert result is not None
        assert "fixcommonerrors" in result
        assert "test.srt" in result

    def test_generate_processing_description_remove_hi(self) -> None:
        """Test generating description with remove hearing impaired."""
        result = generate_processing_description("test.srt", remove_hi=True)

        assert result is not None
        assert "removetextforhi" in result

    def test_generate_processing_description_all_options(self) -> None:
        """Test generating description with all options."""
        result = generate_processing_description(
            "test.srt",
            fix_common_errors=True,
            remove_hi=True,
            auto_split_long_lines=True,
            fix_punctuation=True,
            ocr_fix=True,
            convert_to="vtt"
        )

        assert result is not None
        assert "fixcommonerrors" in result
        assert "removetextforhi" in result
        assert "splitlonglines" in result
        assert "fixpunctuation" in result
        assert "ocrfix" in result
        assert "vtt" in result

    def test_generate_processing_description_pathlib_path(self) -> None:
        """Test generating description with pathlib Path."""
        subtitle_path = Path("test.srt")
        result = generate_processing_description(subtitle_path, fix_common_errors=True)

        assert result is not None
        assert "test.srt" in result


class TestCheckPostprocessAvailable:
    """Test check_postprocess_available function."""

    def test_check_postprocess_available_success(self) -> None:
        """Test post-processing availability check (always succeeds with native implementation)."""
        result = check_postprocess_available()
        assert result is True


class TestValidatePostprocessEnvironment:
    """Test validate_postprocess_environment function."""

    def test_validate_postprocess_environment_all_available(self) -> None:
        """Test environment validation (always valid with native implementation)."""
        result = validate_postprocess_environment()

        expected = {
            "postprocess_available": True,
        }

        assert result == expected


class TestGetAvailableOperations:
    """Test get_available_operations function."""

    def test_get_available_operations(self) -> None:
        """Test getting list of available operations."""
        operations = get_available_operations()

        assert isinstance(operations, list)
        assert "fix_common_errors" in operations
        assert "remove_hi" in operations
        assert "auto_split_long_lines" in operations
        assert "fix_punctuation" in operations
        assert "ocr_fix" in operations


class TestGetSupportedOutputFormats:
    """Test get_supported_output_formats function."""

    def test_get_supported_output_formats(self) -> None:
        """Test getting list of supported output formats."""
        formats = get_supported_output_formats()

        assert isinstance(formats, list)
        assert "srt" in formats
        assert "vtt" in formats
        assert "ass" in formats


class TestIndividualOperations:
    """Test individual operation functions."""

    def test_apply_common_fixes(self) -> None:
        """Test apply_common_fixes function."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest subtitle\n")
            tmp_path = tmp.name

        try:
            result = apply_common_fixes(tmp_path)
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_remove_hearing_impaired(self) -> None:
        """Test remove_hearing_impaired function."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\n[MUSIC] Test subtitle\n")
            tmp_path = tmp.name

        try:
            result = remove_hearing_impaired(tmp_path)
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_split_long_lines(self) -> None:
        """Test split_long_lines function."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nThis is a very long subtitle line that should be split\n")
            tmp_path = tmp.name

        try:
            result = split_long_lines(tmp_path)
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_fix_punctuation(self) -> None:
        """Test fix_punctuation function."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest subtitle...\n")
            tmp_path = tmp.name

        try:
            result = fix_punctuation(tmp_path)
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_apply_ocr_fixes(self) -> None:
        """Test apply_ocr_fixes function."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nWlien the liero arrived\n")
            tmp_path = tmp.name

        try:
            result = apply_ocr_fixes(tmp_path)
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_convert_subtitle_format(self) -> None:
        """Test convert_subtitle_format function."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest subtitle\n")
            tmp_path = tmp.name

        try:
            result = convert_subtitle_format(tmp_path, "vtt")
            assert result is True
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestBatchPostprocess:
    """Test batch_postprocess function."""

    def test_batch_postprocess_success(self) -> None:
        """Test successful batch post-processing."""
        # Create two temp files
        temp_files = []
        for i in range(2):
            tmp = tempfile.NamedTemporaryFile(suffix='.srt', delete=False)
            tmp.write(f"1\n00:00:0{i+1},000 --> 00:00:0{i+2},000\nTest subtitle {i+1}\n".encode())
            tmp.close()
            temp_files.append(tmp.name)

        try:
            results = batch_postprocess(
                temp_files,
                ["fixcommonerrors"]
            )

            assert len(results) == 2
            assert all(results.values())
        finally:
            for tmp_path in temp_files:
                Path(tmp_path).unlink(missing_ok=True)

    def test_batch_postprocess_with_failure(self) -> None:
        """Test batch post-processing with one failure."""
        # Create one valid file and one invalid path
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest subtitle\n")
            valid_file = tmp.name

        invalid_file = "nonexistent.srt"

        try:
            results = batch_postprocess(
                [valid_file, invalid_file],
                ["fixcommonerrors"]
            )

            assert len(results) == 2
            assert results[valid_file] is True
            assert results[invalid_file] is False
        finally:
            Path(valid_file).unlink(missing_ok=True)
