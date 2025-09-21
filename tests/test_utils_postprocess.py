"""Tests for the postprocess utilities module."""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from subtitletools.utils.postprocess import (
    apply_common_fixes,
    apply_ocr_fixes,
    apply_subtitle_edit_postprocess,
    batch_postprocess,
    check_docker_available,
    check_subtitle_edit_image,
    convert_subtitle_format,
    fix_punctuation,
    generate_docker_command,
    get_available_operations,
    get_supported_output_formats,
    remove_hearing_impaired,
    split_long_lines,
    validate_postprocess_environment,
)


class TestApplySubtitleEditPostprocess:
    """Test apply_subtitle_edit_postprocess function."""

    def test_apply_subtitle_edit_postprocess_file_not_found(self) -> None:
        """Test post-processing with non-existent file."""
        result = apply_subtitle_edit_postprocess(
            "nonexistent.srt", ["/fixcommonerrors"]
        )
        assert result is False

    @patch('subprocess.run')
    def test_apply_subtitle_edit_postprocess_success(self, mock_run: Mock) -> None:
        """Test successful post-processing."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            tmp_path = tmp.name

        try:
            # Mock successful subprocess result
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = "Processing completed"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = apply_subtitle_edit_postprocess(
                tmp_path, ["/fixcommonerrors"], "subrip"
            )

            assert result is True
            mock_run.assert_called_once()

            # Check the Docker command structure
            call_args = mock_run.call_args[0][0]
            assert "docker" in call_args
            assert "run" in call_args
            assert "--rm" in call_args
            assert "/fixcommonerrors" in call_args
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('subprocess.run')
    def test_apply_subtitle_edit_postprocess_failure(self, mock_run: Mock) -> None:
        """Test post-processing failure."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            tmp_path = tmp.name

        try:
            # Mock failed subprocess result
            mock_result = Mock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "Docker error"
            mock_run.return_value = mock_result

            result = apply_subtitle_edit_postprocess(
                tmp_path, ["/fixcommonerrors"]
            )

            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('subprocess.run')
    def test_apply_subtitle_edit_postprocess_timeout(self, mock_run: Mock) -> None:
        """Test post-processing timeout."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            tmp_path = tmp.name

        try:
            # Mock timeout
            mock_run.side_effect = subprocess.TimeoutExpired(['docker'], 1800)

            result = apply_subtitle_edit_postprocess(
                tmp_path, ["/fixcommonerrors"]
            )

            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('subprocess.run')
    def test_apply_subtitle_edit_postprocess_exception(self, mock_run: Mock) -> None:
        """Test post-processing with generic exception."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            tmp_path = tmp.name

        try:
            # Mock exception
            mock_run.side_effect = Exception("Unexpected error")

            result = apply_subtitle_edit_postprocess(
                tmp_path, ["/fixcommonerrors"]
            )

            assert result is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('subprocess.run')
    def test_apply_subtitle_edit_postprocess_custom_docker_image(self, mock_run: Mock) -> None:
        """Test post-processing with custom Docker image."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            tmp_path = tmp.name

        try:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = apply_subtitle_edit_postprocess(
                tmp_path, ["/fixcommonerrors"], "subrip", "custom:image"
            )

            assert result is True

            # Check that custom image was used
            call_args = mock_run.call_args[0][0]
            assert "custom:image" in call_args
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('subprocess.run')
    def test_apply_subtitle_edit_postprocess_multiple_operations(self, mock_run: Mock) -> None:
        """Test post-processing with multiple operations."""
        with tempfile.NamedTemporaryFile(suffix='.srt', delete=False) as tmp:
            tmp.write(b"1\n00:00:01,000 --> 00:00:02,000\nTest\n")
            tmp_path = tmp.name

        try:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            operations = ["/fixcommonerrors", "/removetextforhi", "/splitlonglines"]
            result = apply_subtitle_edit_postprocess(tmp_path, operations)

            assert result is True

            # Check all operations are in the command
            call_args = mock_run.call_args[0][0]
            for op in operations:
                assert op in call_args
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestGenerateDockerCommand:
    """Test generate_docker_command function."""

    def test_generate_docker_command_no_operations(self) -> None:
        """Test generating command with no operations."""
        result = generate_docker_command("test.srt")
        assert result is None

    def test_generate_docker_command_fix_common_errors(self) -> None:
        """Test generating command with fix common errors."""
        result = generate_docker_command("test.srt", fix_common_errors=True)

        assert result is not None
        assert "docker run" in result
        assert "--rm" in result
        assert "/fixcommonerrors" in result
        assert "test.srt" in result
        assert "subrip" in result

    def test_generate_docker_command_remove_hi(self) -> None:
        """Test generating command with remove hearing impaired."""
        result = generate_docker_command("test.srt", remove_hi=True)

        assert result is not None
        assert "/removetextforhi" in result

    def test_generate_docker_command_auto_split_long_lines(self) -> None:
        """Test generating command with auto split long lines."""
        result = generate_docker_command("test.srt", auto_split_long_lines=True)

        assert result is not None
        assert "/splitlonglines" in result

    def test_generate_docker_command_fix_punctuation(self) -> None:
        """Test generating command with fix punctuation."""
        result = generate_docker_command("test.srt", fix_punctuation=True)

        assert result is not None
        assert "/fixpunctuation" in result

    def test_generate_docker_command_ocr_fix(self) -> None:
        """Test generating command with OCR fix."""
        result = generate_docker_command("test.srt", ocr_fix=True)

        assert result is not None
        assert "/ocrfix" in result

    def test_generate_docker_command_convert_to(self) -> None:
        """Test generating command with convert to format."""
        result = generate_docker_command("test.srt", convert_to="ass")

        assert result is not None
        assert "ass" in result

    def test_generate_docker_command_all_options(self) -> None:
        """Test generating command with all options."""
        result = generate_docker_command(
            "test.srt",
            fix_common_errors=True,
            remove_hi=True,
            auto_split_long_lines=True,
            fix_punctuation=True,
            ocr_fix=True,
            convert_to="vtt",
            docker_image="custom:image"
        )

        assert result is not None
        assert "custom:image" in result
        assert "vtt" in result
        assert "/fixcommonerrors" in result
        assert "/removetextforhi" in result
        assert "/splitlonglines" in result
        assert "/fixpunctuation" in result
        assert "/ocrfix" in result

    def test_generate_docker_command_pathlib_path(self) -> None:
        """Test generating command with pathlib Path."""
        subtitle_path = Path("test.srt")
        result = generate_docker_command(subtitle_path, fix_common_errors=True)

        assert result is not None
        assert "test.srt" in result


class TestCheckDockerAvailable:
    """Test check_docker_available function."""

    @patch('subprocess.run')
    def test_check_docker_available_success(self, mock_run: Mock) -> None:
        """Test successful Docker availability check."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = check_docker_available()

        assert result is True
        mock_run.assert_called_once_with(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )

    @patch('subprocess.run')
    def test_check_docker_available_failure(self, mock_run: Mock) -> None:
        """Test Docker availability check failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        result = check_docker_available()

        assert result is False

    @patch('subprocess.run')
    def test_check_docker_available_timeout(self, mock_run: Mock) -> None:
        """Test Docker availability check timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(['docker'], 10)

        result = check_docker_available()

        assert result is False

    @patch('subprocess.run')
    def test_check_docker_available_file_not_found(self, mock_run: Mock) -> None:
        """Test Docker availability check with Docker not installed."""
        mock_run.side_effect = FileNotFoundError("docker not found")

        result = check_docker_available()

        assert result is False


class TestCheckSubtitleEditImage:
    """Test check_subtitle_edit_image function."""

    @patch('subprocess.run')
    def test_check_subtitle_edit_image_available(self, mock_run: Mock) -> None:
        """Test Subtitle Edit image availability check success."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123def456\n"  # Image ID
        mock_run.return_value = mock_result

        result = check_subtitle_edit_image()

        assert result is True
        mock_run.assert_called_once_with(
            ["docker", "images", "-q", "seconv:1.0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    @patch('subprocess.run')
    def test_check_subtitle_edit_image_not_available(self, mock_run: Mock) -> None:
        """Test Subtitle Edit image not available."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""  # No output means image not found
        mock_run.return_value = mock_result

        result = check_subtitle_edit_image()

        assert result is False

    @patch('subprocess.run')
    def test_check_subtitle_edit_image_command_failed(self, mock_run: Mock) -> None:
        """Test Subtitle Edit image check with command failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        result = check_subtitle_edit_image()

        assert result is False

    @patch('subprocess.run')
    def test_check_subtitle_edit_image_custom_image(self, mock_run: Mock) -> None:
        """Test checking custom Subtitle Edit image."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "image123\n"
        mock_run.return_value = mock_result

        result = check_subtitle_edit_image("custom:image")

        assert result is True

        # Check custom image was used in command
        call_args = mock_run.call_args[0][0]
        assert "custom:image" in call_args

    @patch('subprocess.run')
    def test_check_subtitle_edit_image_timeout(self, mock_run: Mock) -> None:
        """Test Subtitle Edit image check timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(['docker'], 30)

        result = check_subtitle_edit_image()

        assert result is False

    @patch('subprocess.run')
    def test_check_subtitle_edit_image_file_not_found(self, mock_run: Mock) -> None:
        """Test Subtitle Edit image check with Docker not installed."""
        mock_run.side_effect = FileNotFoundError("docker not found")

        result = check_subtitle_edit_image()

        assert result is False


class TestValidatePostprocessEnvironment:
    """Test validate_postprocess_environment function."""

    @patch('subtitletools.utils.postprocess.check_docker_available')
    @patch('subtitletools.utils.postprocess.check_subtitle_edit_image')
    def test_validate_postprocess_environment_all_available(
        self, mock_check_image: Mock, mock_check_docker: Mock
    ) -> None:
        """Test environment validation with all components available."""
        mock_check_docker.return_value = True
        mock_check_image.return_value = True

        result = validate_postprocess_environment()

        expected = {
            "docker_available": True,
            "subtitle_edit_image": True,
        }
        assert result == expected

        mock_check_docker.assert_called_once()
        mock_check_image.assert_called_once()

    @patch('subtitletools.utils.postprocess.check_docker_available')
    @patch('subtitletools.utils.postprocess.check_subtitle_edit_image')
    def test_validate_postprocess_environment_docker_not_available(
        self, mock_check_image: Mock, mock_check_docker: Mock
    ) -> None:
        """Test environment validation with Docker not available."""
        mock_check_docker.return_value = False

        result = validate_postprocess_environment()

        expected = {
            "docker_available": False,
            "subtitle_edit_image": False,
        }
        assert result == expected

        mock_check_docker.assert_called_once()
        mock_check_image.assert_not_called()  # Should not check image if Docker unavailable

    @patch('subtitletools.utils.postprocess.check_docker_available')
    @patch('subtitletools.utils.postprocess.check_subtitle_edit_image')
    def test_validate_postprocess_environment_image_not_available(
        self, mock_check_image: Mock, mock_check_docker: Mock
    ) -> None:
        """Test environment validation with Docker available but image not available."""
        mock_check_docker.return_value = True
        mock_check_image.return_value = False

        result = validate_postprocess_environment()

        expected = {
            "docker_available": True,
            "subtitle_edit_image": False,
        }
        assert result == expected


class TestGetFunctions:
    """Test getter functions for operations and formats."""

    def test_get_available_operations(self) -> None:
        """Test getting available operations."""
        result = get_available_operations()

        expected = [
            "fix_common_errors",
            "remove_hi",
            "auto_split_long_lines",
            "fix_punctuation",
            "ocr_fix",
        ]
        assert result == expected

    def test_get_supported_output_formats(self) -> None:
        """Test getting supported output formats."""
        result = get_supported_output_formats()

        expected = ["srt", "subrip", "ass", "ssa", "vtt", "stl", "smi"]
        assert result == expected


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_apply_common_fixes(self, mock_apply: Mock) -> None:
        """Test apply_common_fixes wrapper."""
        mock_apply.return_value = True

        result = apply_common_fixes("test.srt")

        assert result is True
        mock_apply.assert_called_once_with("test.srt", ["/fixcommonerrors"], "subrip")

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_remove_hearing_impaired(self, mock_apply: Mock) -> None:
        """Test remove_hearing_impaired wrapper."""
        mock_apply.return_value = True

        result = remove_hearing_impaired("test.srt")

        assert result is True
        mock_apply.assert_called_once_with("test.srt", ["/removetextforhi"], "subrip")

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_split_long_lines(self, mock_apply: Mock) -> None:
        """Test split_long_lines wrapper."""
        mock_apply.return_value = True

        result = split_long_lines("test.srt")

        assert result is True
        mock_apply.assert_called_once_with("test.srt", ["/splitlonglines"], "subrip")

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_fix_punctuation(self, mock_apply: Mock) -> None:
        """Test fix_punctuation wrapper."""
        mock_apply.return_value = True

        result = fix_punctuation("test.srt")

        assert result is True
        mock_apply.assert_called_once_with("test.srt", ["/fixpunctuation"], "subrip")

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_apply_ocr_fixes(self, mock_apply: Mock) -> None:
        """Test apply_ocr_fixes wrapper."""
        mock_apply.return_value = True

        result = apply_ocr_fixes("test.srt")

        assert result is True
        mock_apply.assert_called_once_with("test.srt", ["/ocrfix"], "subrip")

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_convert_subtitle_format(self, mock_apply: Mock) -> None:
        """Test convert_subtitle_format wrapper."""
        mock_apply.return_value = True

        result = convert_subtitle_format("test.srt", "ass")

        assert result is True
        mock_apply.assert_called_once_with("test.srt", [], "ass")


class TestBatchPostprocess:
    """Test batch_postprocess function."""

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_batch_postprocess_success(self, mock_apply: Mock) -> None:
        """Test successful batch post-processing."""
        mock_apply.return_value = True

        files = ["file1.srt", "file2.srt", "file3.srt"]
        operations = ["/fixcommonerrors", "/removetextforhi"]

        result = batch_postprocess(files, operations)

        expected = {
            "file1.srt": True,
            "file2.srt": True,
            "file3.srt": True,
        }
        assert result == expected
        assert mock_apply.call_count == 3

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_batch_postprocess_mixed_results(self, mock_apply: Mock) -> None:
        """Test batch post-processing with mixed results."""
        # First call succeeds, second fails, third succeeds
        mock_apply.side_effect = [True, False, True]

        files = ["file1.srt", "file2.srt", "file3.srt"]
        operations = ["/fixcommonerrors"]

        result = batch_postprocess(files, operations)

        expected = {
            "file1.srt": True,
            "file2.srt": False,
            "file3.srt": True,
        }
        assert result == expected

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_batch_postprocess_with_exception(self, mock_apply: Mock) -> None:
        """Test batch post-processing with exception."""
        mock_apply.side_effect = [True, Exception("Processing error"), True]

        files = ["file1.srt", "file2.srt", "file3.srt"]
        operations = ["/fixcommonerrors"]

        result = batch_postprocess(files, operations)

        expected = {
            "file1.srt": True,
            "file2.srt": False,
            "file3.srt": True,
        }
        assert result == expected

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_batch_postprocess_custom_format(self, mock_apply: Mock) -> None:
        """Test batch post-processing with custom output format."""
        mock_apply.return_value = True

        files = ["file1.srt", "file2.srt"]
        operations = ["/fixcommonerrors"]

        result = batch_postprocess(files, operations, "vtt")

        # Check that custom format was passed to each call
        assert mock_apply.call_count == 2
        for call in mock_apply.call_args_list:
            assert call[0][2] == "vtt"  # output_format argument

    @patch('subtitletools.utils.postprocess.apply_subtitle_edit_postprocess')
    def test_batch_postprocess_pathlib_paths(self, mock_apply: Mock) -> None:
        """Test batch post-processing with pathlib Path objects."""
        mock_apply.return_value = True

        files = [Path("file1.srt"), Path("file2.srt")]
        operations = ["/fixcommonerrors"]

        result = batch_postprocess(files, operations)

        expected = {
            "file1.srt": True,
            "file2.srt": True,
        }
        assert result == expected
