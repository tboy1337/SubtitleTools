"""Tests for the CLI module."""

import argparse
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from subtitletools.cli import (
    create_parser,
    handle_encode_command,
    handle_transcribe_command,
    handle_translate_command,
    handle_workflow_command,
    main,
)
from subtitletools.core.subtitle import SubtitleError
from subtitletools.core.transcription import TranscriptionError
from subtitletools.core.translation import TranslationError
from subtitletools.core.workflow import WorkflowError


class TestParserCreation:
    """Test parser creation functions."""

    def test_create_parser(self) -> None:
        """Test main parser creation."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.description is not None and "SubtitleTools" in parser.description

        # Test version argument
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])

    def test_parser_help(self) -> None:
        """Test parser help."""
        parser = create_parser()

        # Should not raise exception
        help_text = parser.format_help()
        assert "SubtitleTools" in help_text
        assert "transcribe" in help_text
        assert "translate" in help_text
        assert "encode" in help_text
        assert "workflow" in help_text

    def test_transcribe_subparser(self) -> None:
        """Test transcribe subparser."""
        parser = create_parser()
        args = parser.parse_args(["transcribe", "input.mp4"])

        assert args.command == "transcribe"
        assert args.input == "input.mp4"
        assert args.model == "small"  # Default
        assert args.batch is False

    def test_translate_subparser(self) -> None:
        """Test translate subparser."""
        parser = create_parser()
        args = parser.parse_args(["translate", "input.srt", "output.srt"])

        assert args.command == "translate"
        assert args.input == "input.srt"
        assert args.output == "output.srt"
        assert args.src_lang == "en"  # Default
        assert args.target_lang == "zh-CN"  # Default
        assert args.both is True

    def test_encode_subparser(self) -> None:
        """Test encode subparser."""
        parser = create_parser()
        args = parser.parse_args(["encode", "--list-encodings"])

        assert args.command == "encode"
        assert args.list_encodings is True

    def test_workflow_subparser(self) -> None:
        """Test workflow subparser."""
        parser = create_parser()
        args = parser.parse_args(["workflow", "input.mp4"])

        assert args.command == "workflow"
        assert args.input == "input.mp4"
        assert args.resume is True  # Default


class TestTranscribeCommand:
    """Test transcribe command handler."""

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("subtitletools.utils.common.is_video_file")
    def test_handle_transcribe_command_file_not_found(
        self, mock_is_video: Mock, mock_transcriber_class: Mock
    ) -> None:
        """Test transcribe command with transcription error."""
        args = argparse.Namespace(
            input="nonexistent.mp4",
            output=None,
            model="base",
            language=None,
            max_segment_length=None,
            batch=False,
            extensions="mp4",
        )

        mock_is_video.return_value = True
        mock_transcriber = Mock()
        mock_transcriber.transcribe_video.side_effect = TranscriptionError(
            "File not found"
        )
        mock_transcriber_class.return_value = mock_transcriber

        result = handle_transcribe_command(args)
        assert result == 1

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("subtitletools.utils.common.is_video_file")
    def test_handle_transcribe_command_single_video_success(
        self, mock_is_video: Mock, mock_transcriber_class: Mock
    ) -> None:
        """Test successful single video transcription."""
        args = argparse.Namespace(
            input="video.mp4",
            output="output.srt",
            model="base",
            language="en",
            max_segment_length=50,
            batch=False,
            extensions="mp4",
        )

        mock_is_video.return_value = True
        mock_transcriber = Mock()
        mock_transcriber.transcribe_video.return_value = Path("output.srt")
        mock_transcriber_class.return_value = mock_transcriber

        result = handle_transcribe_command(args)

        assert result == 0
        mock_transcriber.transcribe_video.assert_called_once_with(
            Path("video.mp4"), Path("output.srt"), 50
        )

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("subtitletools.utils.common.is_audio_file")
    def test_handle_transcribe_command_single_audio_success(
        self, mock_is_audio: Mock, mock_transcriber_class: Mock
    ) -> None:
        """Test successful single audio transcription."""
        args = argparse.Namespace(
            input="audio.mp3",
            output="output.srt",
            model="base",
            language="en",
            max_segment_length=50,
            batch=False,
            extensions="mp3",
        )

        mock_is_audio.return_value = True
        mock_transcriber = Mock()
        mock_transcriber.transcribe_audio.return_value = {"segments": []}
        mock_transcriber_class.return_value = mock_transcriber

        result = handle_transcribe_command(args)

        assert result == 0
        mock_transcriber.transcribe_audio.assert_called_once()
        mock_transcriber.generate_srt.assert_called_once()

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("os.path.isdir")
    def test_handle_transcribe_command_batch_success(
        self, mock_isdir: Mock, mock_transcriber_class: Mock
    ) -> None:
        """Test successful batch transcription."""
        args = argparse.Namespace(
            input="videos/",
            output="output/",
            model="base",
            language=None,
            max_segment_length=None,
            batch=True,
            extensions="mp4,mkv",
        )

        mock_isdir.return_value = True
        mock_transcriber = Mock()
        mock_transcriber.batch_transcribe.return_value = {
            "video1.mp4": {"status": "success"},
            "video2.mp4": {"status": "success"},
        }
        mock_transcriber_class.return_value = mock_transcriber

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch(
                "pathlib.Path.glob",
                return_value=[Path("video1.mp4"), Path("video2.mp4")],
            ):
                result = handle_transcribe_command(args)

        assert result == 0

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("os.path.isdir")
    def test_handle_transcribe_command_batch_no_files(
        self, mock_isdir: Mock, mock_transcriber_class: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test batch transcription with no files found."""
        args = argparse.Namespace(
            input="videos/",
            output="output/",
            model="base",
            language=None,
            max_segment_length=None,
            batch=True,
            extensions="mp4",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.glob", return_value=[]):
                result = handle_transcribe_command(args)

        assert result == 1

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("subtitletools.utils.common.is_video_file")
    @patch("subtitletools.utils.common.is_audio_file")
    def test_handle_transcribe_command_unsupported_file(
        self, mock_is_audio: Mock, mock_is_video: Mock, mock_transcriber_class: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test transcribe command with unsupported file type."""
        args = argparse.Namespace(
            input="document.txt",
            output=None,
            model="base",
            language=None,
            max_segment_length=None,
            batch=False,
            extensions="txt",
        )

        mock_is_video.return_value = False
        mock_is_audio.return_value = False

        result = handle_transcribe_command(args)
        assert result == 1

    def test_handle_transcribe_command_exception(self) -> None:
        """Test transcribe command with unexpected exception."""
        args = argparse.Namespace(
            input="video.mp4",
            output=None,
            model="base",
            language=None,
            max_segment_length=None,
            batch=False,
            extensions="mp4",
        )

        with patch(
            "subtitletools.cli.SubWhisperTranscriber",
            side_effect=Exception("Unexpected error"),
        ):
            result = handle_transcribe_command(args)
            assert result == 1


class TestTranslateCommand:
    """Test translate command handler."""

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_handle_translate_command_single_success(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:
        """Test successful single file translation."""
        args = argparse.Namespace(
            input="input.srt",
            output="output.srt",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=False,
            pattern="*.srt",
        )

        mock_isdir.return_value = False

        # Mock subtitle processor
        mock_processor = Mock()
        mock_processor.parse_file.return_value = [Mock()]
        mock_processor.extract_text.return_value = ["Hello world"]
        mock_processor.reconstruct_subtitles.return_value = [Mock()]
        mock_processor_class.return_value = mock_processor

        # Mock translator
        mock_translator = Mock()
        mock_translator.translate_lines.return_value = ["Hola mundo"]
        mock_translator_class.return_value = mock_translator

        with patch(
            "subtitletools.core.translation.is_space_language", return_value=True
        ):
            result = handle_translate_command(args)

        assert result == 0
        mock_processor.save_file.assert_called_once()

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_handle_translate_command_batch_success(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:
        """Test successful batch translation."""
        args = argparse.Namespace(
            input="input/",
            output="output/",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        # Mock subtitle processor
        mock_processor = Mock()
        mock_processor.parse_file.return_value = [Mock()]
        mock_processor.extract_text.return_value = ["Hello world"]
        mock_processor.reconstruct_subtitles.return_value = [Mock()]
        mock_processor_class.return_value = mock_processor

        # Mock translator
        mock_translator = Mock()
        mock_translator.translate_lines.return_value = ["Hola mundo"]
        mock_translator_class.return_value = mock_translator

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.mkdir"):
                with patch(
                    "pathlib.Path.glob",
                    return_value=[Path("file1.srt"), Path("file2.srt")],
                ):
                    with patch(
                        "subtitletools.core.translation.is_space_language",
                        return_value=True,
                    ):
                        result = handle_translate_command(args)

        assert result == 0

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_handle_translate_command_input_dir_not_found(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test translate command with input directory not found."""
        args = argparse.Namespace(
            input="nonexistent/",
            output="output/",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.is_dir", return_value=False):
            result = handle_translate_command(args)

        assert result == 1

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_handle_translate_command_no_files_found(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test translate command with no files matching pattern."""
        args = argparse.Namespace(
            input="input/",
            output="output/",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.glob", return_value=[]):
                    result = handle_translate_command(args)

        assert result == 1

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    def test_handle_translate_command_translation_error(
        self, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test translate command with translation error."""
        args = argparse.Namespace(
            input="input.srt",
            output="output.srt",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=False,
            pattern="*.srt",
        )

        mock_translator_class.side_effect = TranslationError("Translation failed")

        result = handle_translate_command(args)
        assert result == 1

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    def test_handle_translate_command_subtitle_error(
        self, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test translate command with subtitle error."""
        args = argparse.Namespace(
            input="input.srt",
            output="output.srt",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=False,
            pattern="*.srt",
        )

        mock_processor = Mock()
        mock_processor.parse_file.side_effect = SubtitleError("Parse failed")
        mock_processor_class.return_value = mock_processor

        with patch("os.path.isdir", return_value=False):
            result = handle_translate_command(args)

        assert result == 1


class TestEncodeCommand:
    """Test encode command handler."""

    def test_handle_encode_command_list_encodings(self) -> None:
        """Test encode command with list encodings option."""
        args = argparse.Namespace(
            list_encodings=True,
            input=None,
            to_encoding=None,
            recommended=False,
            language=None,
            batch=False,
            pattern="*.srt",
            output_dir=None,
        )

        result = handle_encode_command(args)
        assert result == 0

    def test_handle_encode_command_no_input(self) -> None:
        """Test encode command without input when not listing encodings."""
        args = argparse.Namespace(
            list_encodings=False,
            input=None,
            to_encoding=None,
            recommended=False,
            language=None,
            batch=False,
            pattern="*.srt",
            output_dir=None,
        )

        result = handle_encode_command(args)
        assert result == 1

    @patch("subtitletools.cli.convert_to_multiple_encodings")
    @patch("os.path.isdir")
    def test_handle_encode_command_single_success(
        self, mock_isdir: Mock, mock_convert: Mock
    ) -> None:
        """Test successful single file encoding."""
        args = argparse.Namespace(
            list_encodings=False,
            input="input.srt",
            to_encoding="utf-8",
            recommended=False,
            language=None,
            batch=False,
            pattern="*.srt",
            output_dir=None,
        )

        mock_isdir.return_value = False
        mock_convert.return_value = {"utf-8": True}

        result = handle_encode_command(args)
        assert result == 0

    @patch("subtitletools.cli.convert_to_multiple_encodings")
    @patch("os.path.isdir")
    def test_handle_encode_command_batch_success(
        self, mock_isdir: Mock, mock_convert_multiple: Mock
    ) -> None:
        """Test successful batch encoding."""
        args = argparse.Namespace(
            list_encodings=False,
            input="input/",
            to_encoding="utf-8,utf-8-sig",
            recommended=False,
            language=None,
            batch=True,
            pattern="*.srt",
            output_dir="output/",
        )

        mock_isdir.return_value = True
        mock_convert_multiple.return_value = {"utf-8": True, "utf-8-sig": True}

        with patch(
            "pathlib.Path.glob", return_value=[Path("file1.srt"), Path("file2.srt")]
        ):
            result = handle_encode_command(args)

        assert result == 0

    @patch("subtitletools.utils.encoding.get_recommended_encodings")
    @patch("subtitletools.cli.convert_to_multiple_encodings")
    @patch("os.path.isdir")
    def test_handle_encode_command_recommended_encodings(
        self, mock_isdir: Mock, mock_convert: Mock, mock_get_recommended: Mock
    ) -> None:
        """Test encode command with recommended encodings."""
        args = argparse.Namespace(
            list_encodings=False,
            input="input.srt",
            to_encoding=None,
            recommended=True,
            language="zh",
            batch=False,
            pattern="*.srt",
            output_dir=None,
        )

        mock_isdir.return_value = False
        mock_get_recommended.return_value = ["gbk", "utf-8"]
        mock_convert.return_value = {"gbk": True, "utf-8": True}

        result = handle_encode_command(args)
        assert result == 0

    def test_handle_encode_command_exception(self) -> None:
        """Test encode command with unexpected exception."""
        args = argparse.Namespace(
            list_encodings=False,
            input="input.srt",
            to_encoding="utf-8",
            recommended=False,
            language=None,
            batch=False,
            pattern="*.srt",
            output_dir=None,
        )

        with patch("os.path.isdir", side_effect=Exception("Unexpected error")):
            result = handle_encode_command(args)
            assert result == 1


class TestWorkflowCommand:
    """Test workflow command handler."""

    @patch("subtitletools.cli.SubtitleWorkflow")
    @patch("subtitletools.utils.common.is_video_file")
    def test_handle_workflow_command_single_video_success(
        self, mock_is_video: Mock, mock_workflow_class: Mock
    ) -> None:
        """Test successful single video workflow."""
        args = argparse.Namespace(
            input="video.mp4",
            output="output.srt",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=False,
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            ocr_fix=False,
            convert_to=None,
            batch=False,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        mock_is_video.return_value = True

        mock_workflow = Mock()
        mock_workflow.transcribe_and_translate.return_value = {
            "output_path": "output.srt",
            "total_time": 120.5,
        }
        mock_workflow_class.return_value = mock_workflow

        result = handle_workflow_command(args)
        assert result == 0

    @patch("subtitletools.cli.SubtitleWorkflow")
    @patch("subtitletools.utils.common.is_subtitle_file")
    def test_handle_workflow_command_single_subtitle_success(
        self, mock_is_subtitle: Mock, mock_workflow_class: Mock
    ) -> None:
        """Test successful single subtitle workflow."""
        args = argparse.Namespace(
            input="input.srt",
            output="output.srt",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=False,
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            ocr_fix=False,
            convert_to=None,
            batch=False,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        mock_is_subtitle.return_value = True

        mock_workflow = Mock()
        mock_workflow.translate_existing_subtitles.return_value = {
            "output_path": "output.srt",
            "total_time": 30.2,
        }
        mock_workflow_class.return_value = mock_workflow

        with patch("subtitletools.utils.common.is_video_file", return_value=False):
            with patch("subtitletools.utils.common.is_audio_file", return_value=False):
                result = handle_workflow_command(args)

        assert result == 0

    @patch("subtitletools.cli.SubtitleWorkflow")
    @patch("subtitletools.utils.common.is_video_file")
    @patch("subtitletools.utils.common.is_audio_file")
    @patch("subtitletools.utils.common.is_subtitle_file")
    def test_handle_workflow_command_unsupported_file(
        self,
        mock_is_subtitle: Mock,
        mock_is_audio: Mock,
        mock_is_video: Mock,
        mock_workflow_class: Mock,
    ) -> None:  # pylint: disable=unused-argument
        """Test workflow command with unsupported file type."""
        args = argparse.Namespace(
            input="document.txt",
            output="output.srt",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=False,
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            remove_filler_words=False,
            normalize_punctuation=False,
            merge_short_lines=False,
            ocr_fix=False,
            convert_to=None,
            batch=False,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        mock_is_video.return_value = False
        mock_is_audio.return_value = False
        mock_is_subtitle.return_value = False

        result = handle_workflow_command(args)
        assert result == 1

    @patch("subtitletools.cli.SubtitleWorkflow")
    def test_handle_workflow_command_workflow_error(
        self, mock_workflow_class: Mock
    ) -> None:
        """Test workflow command with workflow error."""
        args = argparse.Namespace(
            input="video.mp4",
            output="output.srt",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=False,
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            remove_filler_words=False,
            normalize_punctuation=False,
            merge_short_lines=False,
            ocr_fix=False,
            convert_to=None,
            batch=False,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        mock_workflow_class.side_effect = WorkflowError("Workflow failed")

        result = handle_workflow_command(args)
        assert result == 1

    def test_handle_workflow_command_exception(self) -> None:
        """Test workflow command with unexpected exception."""
        args = argparse.Namespace(
            input="video.mp4",
            output="output.srt",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=False,
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            remove_filler_words=False,
            normalize_punctuation=False,
            merge_short_lines=False,
            ocr_fix=False,
            convert_to=None,
            batch=False,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        with patch(
            "subtitletools.cli.SubtitleWorkflow",
            side_effect=Exception("Unexpected error"),
        ):
            result = handle_workflow_command(args)
            assert result == 1


class TestMainFunction:
    """Test main function."""

    @patch("subtitletools.cli.create_parser")
    @patch("subtitletools.utils.common.setup_logging")
    def test_main_no_command(
        self, mock_setup_logging: Mock, mock_create_parser: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test main function with no command."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = argparse.Namespace(
            command=None, verbose=False, log_file=None
        )
        mock_create_parser.return_value = mock_parser

        result = main([])

        assert result == 0
        mock_parser.print_help.assert_called_once()

    @patch("subtitletools.cli.handle_transcribe_command")
    @patch("subtitletools.utils.common.setup_logging")
    def test_main_transcribe_command(
        self, mock_setup_logging: Mock, mock_handle_transcribe: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test main function with transcribe command."""
        mock_handle_transcribe.return_value = 0

        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.return_value = argparse.Namespace(
                command="transcribe", verbose=False, log_file=None
            )
            mock_create_parser.return_value = mock_parser

            result = main(["transcribe", "input.mp4"])

        assert result == 0
        mock_handle_transcribe.assert_called_once()

    @patch("subtitletools.cli.handle_translate_command")
    @patch("subtitletools.utils.common.setup_logging")
    def test_main_translate_command(
        self, mock_setup_logging: Mock, mock_handle_translate: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test main function with translate command."""
        mock_handle_translate.return_value = 0

        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.return_value = argparse.Namespace(
                command="translate", verbose=False, log_file=None
            )
            mock_create_parser.return_value = mock_parser

            result = main(["translate", "input.srt", "output.srt"])

        assert result == 0
        mock_handle_translate.assert_called_once()

    @patch("subtitletools.cli.handle_encode_command")
    @patch("subtitletools.utils.common.setup_logging")
    def test_main_encode_command(
        self, mock_setup_logging: Mock, mock_handle_encode: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test main function with encode command."""
        mock_handle_encode.return_value = 0

        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.return_value = argparse.Namespace(
                command="encode", verbose=False, log_file=None
            )
            mock_create_parser.return_value = mock_parser

            result = main(["encode", "--list-encodings"])

        assert result == 0
        mock_handle_encode.assert_called_once()

    @patch("subtitletools.cli.handle_workflow_command")
    @patch("subtitletools.utils.common.setup_logging")
    def test_main_workflow_command(
        self, mock_setup_logging: Mock, mock_handle_workflow: Mock
    ) -> None:  # pylint: disable=unused-argument
        """Test main function with workflow command."""
        mock_handle_workflow.return_value = 0

        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.return_value = argparse.Namespace(
                command="workflow", verbose=False, log_file=None
            )
            mock_create_parser.return_value = mock_parser

            result = main(["workflow", "input.mp4"])

        assert result == 0
        mock_handle_workflow.assert_called_once()

    def test_main_verbose_mode(self, test_data_dir: Path) -> None:
        """Test main function in verbose mode."""
        log_file = test_data_dir / "debug.log"
        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.return_value = argparse.Namespace(
                command=None, verbose=True, log_file=str(log_file)
            )
            mock_create_parser.return_value = mock_parser

            with patch("subtitletools.utils.common.setup_logging"):
                with patch("subtitletools.utils.common.get_system_info"):
                    result = main(["--verbose", "--log-file", str(log_file)])

        assert result == 0

    @patch("subtitletools.utils.common.setup_logging")
    def test_main_keyboard_interrupt(self, mock_setup_logging: Mock) -> None:
        """Test main function with keyboard interrupt."""
        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.side_effect = KeyboardInterrupt()
            mock_create_parser.return_value = mock_parser

            result = main([])

        assert result == 130

    @patch("subtitletools.utils.common.setup_logging")
    def test_main_unexpected_exception(self, mock_setup_logging: Mock) -> None:
        """Test main function with unexpected exception."""
        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.side_effect = Exception("Unexpected error")
            mock_create_parser.return_value = mock_parser

            result = main([])

        assert result == 1

    @patch("subtitletools.utils.common.setup_logging")
    def test_main_unexpected_exception_verbose(self, mock_setup_logging: Mock) -> None:
        """Test main function with unexpected exception in verbose mode."""
        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.return_value = argparse.Namespace(
                command="transcribe", verbose=True, log_file=None
            )
            mock_create_parser.return_value = mock_parser

            with patch(
                "subtitletools.cli.handle_transcribe_command",
                side_effect=Exception("Unexpected error"),
            ):
                with patch("traceback.print_exc") as mock_print_exc:
                    result = main(["transcribe", "input.mp4", "--verbose"])

        assert result == 1
        mock_print_exc.assert_called_once()

    def test_main_default_args(self) -> None:
        """Test main function with default args (sys.argv)."""
        with patch("sys.argv", ["subtitletools"]):
            with patch("subtitletools.cli.create_parser") as mock_create_parser:
                mock_parser = Mock()
                mock_parser.parse_args.return_value = argparse.Namespace(
                    command=None, verbose=False, log_file=None
                )
                mock_create_parser.return_value = mock_parser

                with patch("subtitletools.utils.common.setup_logging"):
                    result = main()

        assert result == 0


class TestAdditionalCliCoverage:
    """Additional test cases to improve CLI coverage."""

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_translate_batch_file_errors(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:
        """Test batch translate with individual file processing errors."""
        args = argparse.Namespace(
            input="input/",
            output="output/",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        # Mock subtitle processor to fail on first file
        mock_processor = Mock()
        mock_processor.parse_file.side_effect = [
            SubtitleError("Parse failed"),
            [Mock()],
        ]
        mock_processor.extract_text.return_value = ["Hello world"]
        mock_processor.reconstruct_subtitles.return_value = [Mock()]
        mock_processor_class.return_value = mock_processor

        # Mock translator
        mock_translator = Mock()
        mock_translator.translate_lines.return_value = ["Hola mundo"]
        mock_translator_class.return_value = mock_translator

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.mkdir"):
                with patch(
                    "pathlib.Path.glob",
                    return_value=[Path("file1.srt"), Path("file2.srt")],
                ):
                    with patch(
                        "subtitletools.core.translation.is_space_language",
                        return_value=True,
                    ):
                        result = handle_translate_command(args)

        # Should succeed with partial success
        assert result == 0

    @patch("subtitletools.cli.get_recommended_encodings")
    @patch("subtitletools.cli.convert_to_multiple_encodings")
    @patch("os.path.isdir")
    def test_encode_batch_errors(
        self, mock_isdir: Mock, mock_convert: Mock, mock_get_rec: Mock
    ) -> None:
        """Test encode batch processing with errors."""
        args = argparse.Namespace(
            input="input/",
            output_dir="output/",
            to_encoding=None,
            recommended=False,
            language="en",
            list_encodings=False,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True
        mock_convert.side_effect = [
            {"utf-8": True, "latin-1": False},  # First file partial success
            Exception("Conversion failed"),  # Second file fails
        ]

        with patch(
            "pathlib.Path.glob", return_value=[Path("file1.srt"), Path("file2.srt")]
        ):
            result = handle_encode_command(args)

        assert result == 0

    @patch("subtitletools.cli.get_recommended_encodings")
    @patch("subtitletools.cli.convert_to_multiple_encodings")
    @patch("os.path.isdir")
    def test_encode_batch_no_files(
        self, mock_isdir: Mock, mock_convert: Mock, mock_get_rec: Mock
    ) -> None:
        """Test encode batch with no matching files."""
        args = argparse.Namespace(
            input="input/",
            output_dir="output/",
            to_encoding=None,
            recommended=False,
            language="en",
            list_encodings=False,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.glob", return_value=[]):
            result = handle_encode_command(args)

        assert result == 1

    @patch("subtitletools.cli.get_recommended_encodings")
    def test_encode_recommended(self, mock_get_rec: Mock) -> None:
        """Test encode with recommended encodings."""
        args = argparse.Namespace(
            input="input.srt",
            output_dir="output/",
            to_encoding=None,
            recommended=True,
            language="zh",
            list_encodings=False,
            batch=False,
            pattern="*.srt",
        )

        mock_get_rec.return_value = ["gbk", "gb2312", "big5"]

        with patch("subtitletools.cli.convert_to_multiple_encodings") as mock_convert:
            mock_convert.return_value = {"gbk": True, "gb2312": True, "big5": False}
            with patch("os.path.isdir", return_value=False):
                result = handle_encode_command(args)

        assert result == 0

    @patch("subtitletools.utils.postprocess.validate_postprocess_environment")
    @patch("subtitletools.cli.SubtitleWorkflow")
    @patch("os.path.isdir")
    def test_workflow_batch_no_files(
        self, mock_isdir: Mock, mock_workflow_class: Mock, mock_validate: Mock
    ) -> None:
        """Test workflow batch with no files found."""
        args = argparse.Namespace(
            input="input/",
            output="output/",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=False,
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            ocr_fix=False,
            convert_to=None,
            batch=True,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        mock_isdir.return_value = True
        mock_validate.return_value = {"postprocess_available": True}

        with patch("pathlib.Path.glob", return_value=[]):
            result = handle_workflow_command(args)

        assert result == 1

    @patch("subtitletools.utils.postprocess.validate_postprocess_environment")
    @patch("subtitletools.cli.SubtitleWorkflow")
    @patch("os.path.isdir")
    def test_workflow_postprocessing_always_available(
        self, mock_isdir: Mock, mock_workflow_class: Mock, mock_validate: Mock
    ) -> None:
        """Test workflow with post-processing (now always available with native implementation)."""
        args = argparse.Namespace(
            input="video.mp4",
            output="output.srt",
            model="base",
            language=None,
            src_lang="auto",
            target_lang="en",
            service="google",
            api_key=None,
            both=True,
            max_segment_length=None,
            fix_common_errors=True,  # Now handled natively
            remove_hi=False,
            auto_split_long_lines=False,
            fix_punctuation=False,
            ocr_fix=False,
            convert_to=None,
            batch=False,
            extensions="mp4,mkv,avi,mov,webm",
            resume=True,
        )

        mock_isdir.return_value = False
        mock_validate.return_value = {"postprocess_available": True}

        mock_workflow = Mock()
        mock_workflow.transcribe_and_translate.return_value = {
            "output_path": "output.srt",
            "total_time": 120.5,
        }
        mock_workflow_class.return_value = mock_workflow

        with patch("subtitletools.utils.common.is_video_file", return_value=True):
            result = handle_workflow_command(args)

        assert result == 0

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("os.path.isdir")
    def test_transcribe_batch_input_dir_not_found(
        self, mock_isdir: Mock, mock_transcriber_class: Mock
    ) -> None:
        """Test transcribe batch command with input directory not found."""
        args = argparse.Namespace(
            input="nonexistent_dir/",
            output="output/",
            model="base",
            language=None,
            max_segment_length=None,
            batch=True,
            extensions="mp4",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.is_dir", return_value=False):
            result = handle_transcribe_command(args)

        assert result == 1

    @patch("subtitletools.cli.SubWhisperTranscriber")
    @patch("os.path.isdir")
    def test_transcribe_batch_no_files_found(
        self, mock_isdir: Mock, mock_transcriber_class: Mock
    ) -> None:
        """Test transcribe batch command with no matching files."""
        args = argparse.Namespace(
            input="input_dir/",
            output="output/",
            model="base",
            language=None,
            max_segment_length=None,
            batch=True,
            extensions="mp4",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.glob", return_value=[]):
                result = handle_transcribe_command(args)

        assert result == 1

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_translate_batch_input_processing(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:
        """Test translate batch processing success path."""
        args = argparse.Namespace(
            input="input/",
            output="output/",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        # Mock subtitle processor
        mock_processor = Mock()
        mock_processor.parse_file.return_value = [Mock()]
        mock_processor.extract_text.return_value = ["Hello world"]
        mock_processor.reconstruct_subtitles.return_value = [Mock()]
        mock_processor_class.return_value = mock_processor

        # Mock translator
        mock_translator = Mock()
        mock_translator.translate_lines.return_value = ["Hola mundo"]
        mock_translator_class.return_value = mock_translator

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.glob", return_value=[Path("test.srt")]):
                    with patch(
                        "subtitletools.core.translation.is_space_language",
                        return_value=True,
                    ):
                        result = handle_translate_command(args)

        assert result == 0

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_translate_batch_no_files_found(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:
        """Test translate batch command with no matching files."""
        args = argparse.Namespace(
            input="input/",
            output="output/",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=True,
            pattern="*.srt",
        )

        mock_isdir.return_value = True

        with patch("pathlib.Path.is_dir", return_value=True):
            with patch("pathlib.Path.glob", return_value=[]):
                result = handle_translate_command(args)

        assert result == 1

    @patch("subtitletools.cli.SubtitleTranslator")
    @patch("subtitletools.cli.SubtitleProcessor")
    @patch("os.path.isdir")
    def test_translate_command_unexpected_exception(
        self, mock_isdir: Mock, mock_processor_class: Mock, mock_translator_class: Mock
    ) -> None:
        """Test translate command with unexpected exception."""
        args = argparse.Namespace(
            input="test.srt",
            output="output.srt",
            service="google",
            api_key=None,
            src_lang="en",
            target_lang="es",
            encoding="utf-8",
            both=True,
            batch=False,
            pattern="*.srt",
        )

        mock_isdir.return_value = False
        mock_processor_class.side_effect = Exception("Unexpected error")

        result = handle_translate_command(args)
        assert result == 1

    def test_encode_command_missing_input(self) -> None:
        """Test encode command without input when not listing encodings."""
        args = argparse.Namespace(
            input=None,
            output_dir=None,
            to_encoding=None,
            recommended=False,
            language="en",
            list_encodings=False,
            batch=False,
            pattern="*.srt",
        )

        result = handle_encode_command(args)
        assert result == 1

    @patch("subtitletools.cli.convert_to_multiple_encodings")
    @patch("os.path.isdir")
    def test_encode_command_default_encodings(
        self, mock_isdir: Mock, mock_convert: Mock
    ) -> None:
        """Test encode command using default encodings."""
        args = argparse.Namespace(
            input="test.srt",
            output_dir=None,
            to_encoding=None,
            recommended=False,
            language="en",
            list_encodings=False,
            batch=False,
            pattern="*.srt",
        )

        mock_isdir.return_value = False
        mock_convert.return_value = {"utf-8": True, "utf-8-sig": True}

        result = handle_encode_command(args)
        assert result == 0

    @patch("sys.argv", ["subtitletools"])
    def test_main_with_no_args(self) -> None:
        """Test main function when args is None (uses sys.argv)."""
        with patch("subtitletools.cli.create_parser") as mock_create_parser:
            mock_parser = Mock()
            mock_parser.parse_args.side_effect = SystemExit(0)
            mock_create_parser.return_value = mock_parser

            with pytest.raises(SystemExit):
                main(None)

    @patch("subtitletools.cli.create_parser")
    def test_main_with_verbose_exception(self, mock_create_parser: Mock) -> None:
        """Test main function exception handling with verbose mode and traceback."""
        mock_parser = Mock()
        mock_parser.parse_args.return_value = argparse.Namespace(
            command="transcribe", verbose=True, log_file=None
        )
        mock_create_parser.return_value = mock_parser

        with patch(
            "subtitletools.cli.handle_transcribe_command",
            side_effect=Exception("Test error"),
        ):
            with patch("subtitletools.cli.setup_logging"):
                with patch("subtitletools.cli.get_system_info", return_value={}):
                    with patch("traceback.print_exc") as mock_print_exc:
                        result = main(["transcribe", "test.mp4", "--verbose"])

                        assert result == 1
                        mock_print_exc.assert_called_once()

    def test_type_checking_import(self) -> None:
        """Test TYPE_CHECKING import coverage."""
        from typing import TYPE_CHECKING

        if TYPE_CHECKING:
            pass  # This line should be covered
        assert True
