"""Tests for the workflow module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from subtitletools.core.workflow import (
    CheckpointManager,
    SubtitleWorkflow,
    WorkflowError,
)
from subtitletools.core.subtitle import SubtitleError
from subtitletools.core.transcription import TranscriptionError
from subtitletools.core.translation import TranslationError


class TestWorkflowError:
    """Test WorkflowError exception class."""

    def test_workflow_error_creation(self) -> None:
        """Test WorkflowError creation."""
        error = WorkflowError("Workflow failed")
        assert str(error) == "Workflow failed"
        assert isinstance(error, Exception)

    def test_workflow_error_inheritance(self) -> None:
        """Test that WorkflowError inherits from Exception."""
        error = WorkflowError("Test")
        assert isinstance(error, Exception)


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_init(self) -> None:
        """Test CheckpointManager initialization."""
        with patch('subtitletools.core.workflow.get_cache_dir') as mock_cache:
            with patch('subtitletools.core.workflow.ensure_directory'):
                mock_cache.return_value = Path("/cache")

                manager = CheckpointManager("test_workflow")

                assert manager.workflow_id == "test_workflow"
                assert manager.checkpoint_dir == Path("/cache/checkpoints")
                assert manager.checkpoint_file == Path("/cache/checkpoints/test_workflow.json")

    def test_save_checkpoint_success(self) -> None:
        """Test successful checkpoint saving."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("test_workflow")

                test_data = {"step": "transcription", "progress": 50}
                manager.save_checkpoint(test_data)

                assert manager.checkpoint_file.exists()

                # Verify saved content
                with open(manager.checkpoint_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)

                assert saved_data["workflow_id"] == "test_workflow"
                assert saved_data["data"] == test_data
                assert "timestamp" in saved_data

    def test_save_checkpoint_error(self) -> None:
        """Test checkpoint saving with file error."""
        with patch('subtitletools.core.workflow.get_cache_dir') as mock_cache:
            with patch('subtitletools.core.workflow.ensure_directory'):
                mock_cache.return_value = Path("/invalid/path")

                manager = CheckpointManager("test_workflow")

                # Should not raise exception, just log warning
                manager.save_checkpoint({"test": "data"})

    def test_load_checkpoint_success(self) -> None:
        """Test successful checkpoint loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("test_workflow")

                # Save a checkpoint first
                test_data = {"step": "translation", "progress": 75}
                manager.save_checkpoint(test_data)

                # Load the checkpoint
                loaded_data = manager.load_checkpoint()

                assert loaded_data == test_data

    def test_load_checkpoint_no_file(self) -> None:
        """Test loading checkpoint when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("nonexistent_workflow")

                loaded_data = manager.load_checkpoint()

                assert loaded_data is None

    def test_load_checkpoint_invalid_json(self) -> None:
        """Test loading checkpoint with invalid JSON."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("test_workflow")

                # Create invalid JSON file
                with open(manager.checkpoint_file, 'w', encoding='utf-8') as f:
                    f.write("invalid json content")

                loaded_data = manager.load_checkpoint()

                assert loaded_data is None

    def test_clear_checkpoint_success(self) -> None:
        """Test successful checkpoint clearing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("test_workflow")

                # Create a checkpoint
                manager.save_checkpoint({"test": "data"})
                assert manager.checkpoint_file.exists()

                # Clear the checkpoint
                manager.clear_checkpoint()
                assert not manager.checkpoint_file.exists()

    def test_clear_checkpoint_no_file(self) -> None:
        """Test clearing checkpoint when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("test_workflow")

                # Should not raise exception
                manager.clear_checkpoint()

    def test_clear_checkpoint_error(self) -> None:
        """Test clearing checkpoint with file error."""
        with patch('subtitletools.core.workflow.get_cache_dir') as mock_cache:
            with patch('subtitletools.core.workflow.ensure_directory'):
                mock_cache.return_value = Path("/tmp")
                manager = CheckpointManager("test_workflow")

                # Mock exists and unlink methods
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.unlink', side_effect=OSError("Permission denied")):
                        # Should not raise exception, just log warning
                        manager.clear_checkpoint()


class TestSubtitleWorkflow:
    """Test SubtitleWorkflow class."""

    def test_init_default(self) -> None:
        """Test SubtitleWorkflow initialization with defaults."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber:
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator:
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor:

                    workflow = SubtitleWorkflow()

                    mock_transcriber.assert_called_once_with("small")  # DEFAULT_WHISPER_MODEL
                    mock_translator.assert_called_once_with("google", None)
                    mock_processor.assert_called_once()

                    assert workflow.transcriber is not None
                    assert workflow.translator is not None
                    assert workflow.subtitle_processor is not None

    def test_init_custom_params(self) -> None:
        """Test SubtitleWorkflow initialization with custom parameters."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber:
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator:
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    SubtitleWorkflow(
                        whisper_model="large",
                        translation_service="google_cloud",
                        api_key="test_key"
                    )

                    mock_transcriber.assert_called_once_with("large")
                    mock_translator.assert_called_once_with("google_cloud", "test_key")

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_video_file')
    @patch('subtitletools.core.workflow.get_file_size_mb')
    def test_transcribe_and_translate_video_success(self, mock_size: Mock, mock_is_video: Mock, mock_validate: Mock) -> None:
        """Test successful transcribe and translate workflow for video."""
        # Setup mocks
        input_path = Path("test.mp4")
        mock_validate.return_value = input_path
        mock_is_video.return_value = True
        mock_size.return_value = 100.5

        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator_class:
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:
                    with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:
                        with patch('subtitletools.core.workflow.is_space_language', return_value=True):

                            # Mock instances
                            mock_transcriber = Mock()
                            mock_transcriber.transcribe_video.return_value = "temp.srt"
                            mock_transcriber_class.return_value = mock_transcriber

                            mock_translator = Mock()
                            mock_translator_class.return_value = mock_translator

                            mock_processor = Mock()
                            mock_processor.parse_file.return_value = [Mock()]
                            mock_processor.reconstruct_subtitles.return_value = [Mock()]
                            mock_processor_class.return_value = mock_processor

                            mock_checkpoint = Mock()
                            mock_checkpoint.load_checkpoint.return_value = None
                            mock_checkpoint_class.return_value = mock_checkpoint

                            workflow = SubtitleWorkflow()
                            workflow._translate_subtitles = Mock(return_value=[Mock()])

                            result = workflow.transcribe_and_translate(
                                input_path="test.mp4",
                                src_lang="en",
                                target_lang="es"
                            )

                            assert result["input_path"] == str(input_path)
                            assert result["src_lang"] == "en"
                            assert result["target_lang"] == "es"
                            assert result["file_size_mb"] == 100.5
                            assert "total_time" in result
                            assert "steps_completed" in result

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_video_file')
    @patch('subtitletools.core.workflow.is_audio_file')
    @patch('subtitletools.core.workflow.get_file_size_mb')
    def test_transcribe_and_translate_audio_success(self, mock_size: Mock, mock_is_audio: Mock, mock_is_video: Mock, mock_validate: Mock) -> None:
        """Test successful transcribe and translate workflow for audio."""
        # Setup mocks
        input_path = Path("test.mp3")
        mock_validate.return_value = input_path
        mock_is_video.return_value = False
        mock_is_audio.return_value = True
        mock_size.return_value = 50.2

        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:
                    with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:
                        with patch('subtitletools.core.workflow.is_space_language', return_value=True):

                            # Mock instances
                            mock_transcriber = Mock()
                            mock_transcriber.transcribe_audio.return_value = {"segments": []}
                            mock_transcriber_class.return_value = mock_transcriber

                            mock_processor = Mock()
                            mock_processor_class.return_value = mock_processor

                            mock_checkpoint = Mock()
                            mock_checkpoint.load_checkpoint.return_value = None
                            mock_checkpoint_class.return_value = mock_checkpoint

                            workflow = SubtitleWorkflow()
                            workflow._translate_subtitles = Mock(return_value=[])

                            with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
                                tmp_path = Path(tmp.name)

                                # Mock the temporary SRT file creation
                                with patch('pathlib.Path.with_suffix') as mock_with_suffix:
                                    mock_with_suffix.return_value = tmp_path

                                    result = workflow.transcribe_and_translate(
                                        input_path="test.mp3",
                                        output_path=tmp_path,
                                        src_lang="en",
                                        target_lang="fr"
                                    )

                            assert result["input_path"] == str(input_path)
                            assert result["file_size_mb"] == 50.2

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.get_file_size_mb')
    def test_transcribe_and_translate_invalid_file(self, mock_size: Mock, mock_validate: Mock) -> None:
        """Test transcribe and translate with invalid file type."""
        input_path = Path("test.txt")
        mock_validate.return_value = input_path
        mock_size.return_value = 10.0  # Mock file size

        with patch('subtitletools.core.workflow.is_video_file', return_value=False):
            with patch('subtitletools.core.workflow.is_audio_file', return_value=False):
                with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
                    with patch('subtitletools.core.workflow.SubtitleTranslator'):
                        with patch('subtitletools.core.workflow.SubtitleProcessor'):
                            with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:

                                mock_checkpoint = Mock()
                                mock_checkpoint.load_checkpoint.return_value = None
                                mock_checkpoint_class.return_value = mock_checkpoint

                                workflow = SubtitleWorkflow()

                                with pytest.raises(WorkflowError, match="Unsupported file type"):
                                    workflow.transcribe_and_translate(input_path="test.txt")

    @patch('subtitletools.core.workflow.validate_file_exists')
    def test_transcribe_and_translate_with_resume(self, mock_validate: Mock) -> None:
        """Test transcribe and translate workflow with checkpoint resumption."""
        input_path = Path("test.mp4")
        mock_validate.return_value = input_path

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:
                    with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:
                        with patch('subtitletools.core.workflow.is_video_file', return_value=True):
                            with patch('subtitletools.core.workflow.get_file_size_mb', return_value=100):
                                with patch('subtitletools.core.workflow.is_space_language', return_value=True):

                                    # Mock checkpoint with translation step
                                    mock_checkpoint = Mock()
                                    mock_checkpoint.load_checkpoint.return_value = {
                                        "step": "translation",
                                        "temp_srt_path": "temp.srt",
                                        "results": {
                                            "transcription_time": 30.0,
                                            "steps_completed": ["transcription"]
                                        }
                                    }
                                    mock_checkpoint_class.return_value = mock_checkpoint

                                    # Mock subtitle processor
                                    mock_processor = Mock()
                                    mock_processor.parse_file.return_value = [Mock()]
                                    mock_processor_class.return_value = mock_processor

                                    workflow = SubtitleWorkflow()
                                    workflow._translate_subtitles = Mock(return_value=[Mock()])

                                    result = workflow.transcribe_and_translate(
                                        input_path="test.mp4",
                                        resume=True
                                    )

                                    # Should skip transcription step
                                    mock_checkpoint.load_checkpoint.assert_called_once()
                                    assert "total_time" in result

    @patch('subtitletools.core.workflow.validate_file_exists')
    def test_transcribe_and_translate_transcription_error(self, mock_validate: Mock) -> None:
        """Test transcribe and translate workflow with transcription error."""
        input_path = Path("test.mp4")
        mock_validate.return_value = input_path

        with patch('subtitletools.core.workflow.is_video_file', return_value=True):
            with patch('subtitletools.core.workflow.get_file_size_mb', return_value=100):
                with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
                    with patch('subtitletools.core.workflow.SubtitleTranslator'):
                        with patch('subtitletools.core.workflow.SubtitleProcessor'):
                            with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:

                                # Mock transcriber to raise error
                                mock_transcriber = Mock()
                                mock_transcriber.transcribe_video.side_effect = TranscriptionError("Transcription failed")
                                mock_transcriber_class.return_value = mock_transcriber

                                mock_checkpoint = Mock()
                                mock_checkpoint.load_checkpoint.return_value = None
                                mock_checkpoint_class.return_value = mock_checkpoint

                                workflow = SubtitleWorkflow()

                                with pytest.raises(WorkflowError, match="Workflow failed"):
                                    workflow.transcribe_and_translate(input_path="test.mp4")

    def test_translate_subtitles_same_language(self) -> None:
        """Test _translate_subtitles with same source and target language."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()
                    subtitles = [Mock()]

                    result = workflow._translate_subtitles(
                        subtitles, "en", "en"
                    )

                    assert result == subtitles

    def test_translate_subtitles_success(self) -> None:
        """Test successful _translate_subtitles."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator_class:
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock subtitle processor
                    mock_processor = Mock()
                    mock_processor.extract_text.return_value = ["Hello"]
                    mock_processor.reconstruct_subtitles.return_value = [Mock()]
                    mock_processor_class.return_value = mock_processor

                    # Mock translator
                    mock_translator = Mock()
                    mock_translator.translate_lines.return_value = ["Hola"]
                    mock_translator_class.return_value = mock_translator

                    workflow = SubtitleWorkflow()
                    subtitles = [Mock()]

                    result = workflow._translate_subtitles(
                        subtitles, "en", "es"
                    )

                    mock_processor.extract_text.assert_called_once_with(subtitles)
                    mock_translator.translate_lines.assert_called_once()
                    mock_processor.reconstruct_subtitles.assert_called_once()
                    assert len(result) == 1

    def test_translate_subtitles_error(self) -> None:
        """Test _translate_subtitles with translation error."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock processor to raise error
                    mock_processor = Mock()
                    mock_processor.extract_text.side_effect = Exception("Extract failed")
                    mock_processor_class.return_value = mock_processor

                    workflow = SubtitleWorkflow()

                    with pytest.raises(TranslationError, match="Subtitle translation failed"):
                        workflow._translate_subtitles([Mock()], "en", "es")

    @patch('subtitletools.core.workflow.validate_postprocess_environment')
    def test_apply_postprocessing_native_implementation(self, mock_validate_env: Mock) -> None:
        """Test _apply_postprocessing with native implementation (always available)."""
        mock_validate_env.return_value = {
            "postprocess_available": True
        }

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    result = workflow._apply_postprocessing(
                        Path("test.srt"), ["fix-common-errors"]
                    )

                    assert result is False

    @patch('subtitletools.core.workflow.validate_postprocess_environment')
    def test_apply_postprocessing_no_image(self, mock_validate_env: Mock) -> None:
        """Test _apply_postprocessing when Subtitle Edit image is not available."""
        mock_validate_env.return_value = {
            "postprocess_available": True
        }

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    result = workflow._apply_postprocessing(
                        Path("test.srt"), ["fix-common-errors"]
                    )

                    assert result is False

    @patch('subtitletools.core.workflow.validate_postprocess_environment')
    @patch('subtitletools.core.workflow.apply_subtitle_edit_postprocess')
    def test_apply_postprocessing_success(self, mock_apply: Mock, mock_validate_env: Mock) -> None:
        """Test successful _apply_postprocessing."""
        mock_validate_env.return_value = {
            "postprocess_available": True
        }
        mock_apply.return_value = True

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    result = workflow._apply_postprocessing(
                        Path("test.srt"), ["fix-common-errors"]
                    )

                    assert result is True
                    mock_apply.assert_called_once_with(
                        Path("test.srt"), ["fix-common-errors"], "subrip"
                    )

    @patch('subtitletools.core.workflow.validate_postprocess_environment')
    def test_apply_postprocessing_exception(self, mock_validate_env: Mock) -> None:
        """Test _apply_postprocessing with exception."""
        mock_validate_env.side_effect = Exception("Validation error")

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    result = workflow._apply_postprocessing(
                        Path("test.srt"), ["fix-common-errors"]
                    )

                    assert result is False

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_space_language')
    def test_translate_existing_subtitles_success(self, mock_space: Mock, mock_validate: Mock) -> None:
        """Test successful translate_existing_subtitles."""
        input_path = Path("input.srt")
        output_path = Path("output.srt")
        mock_validate.return_value = input_path
        mock_space.return_value = True

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock subtitle processor
                    mock_processor = Mock()
                    mock_processor.parse_file.return_value = [Mock(), Mock()]
                    mock_processor_class.return_value = mock_processor

                    workflow = SubtitleWorkflow()
                    workflow._translate_subtitles = Mock(return_value=[Mock(), Mock(), Mock()])

                    result = workflow.translate_existing_subtitles(
                        input_path="input.srt",
                        output_path="output.srt",
                        src_lang="en",
                        target_lang="es"
                    )

                    assert result["input_path"] == str(input_path)
                    assert result["output_path"] == str(output_path)
                    assert result["src_lang"] == "en"
                    assert result["target_lang"] == "es"
                    assert result["original_segments"] == 2
                    assert result["translated_segments"] == 3
                    assert result["status"] == "completed"
                    assert "total_time" in result

                    mock_processor.save_file.assert_called_once()

    @patch('subtitletools.core.workflow.validate_file_exists')
    def test_translate_existing_subtitles_error(self, mock_validate: Mock) -> None:
        """Test translate_existing_subtitles with error."""
        input_path = Path("input.srt")
        mock_validate.return_value = input_path

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock processor to raise error
                    mock_processor = Mock()
                    mock_processor.parse_file.side_effect = SubtitleError("Parse failed")
                    mock_processor_class.return_value = mock_processor

                    workflow = SubtitleWorkflow()

                    with pytest.raises(WorkflowError, match="Subtitle translation workflow failed"):
                        workflow.translate_existing_subtitles("input.srt", "output.srt")

    @patch('subtitletools.core.workflow.ensure_directory')
    def test_batch_process_success(self, mock_ensure: Mock) -> None:
        """Test successful batch_process."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    # Simplify test - just test that it processes files and returns results
                    workflow.transcribe_and_translate = Mock(return_value={"status": "completed", "total_time": 120})

                    with patch('subtitletools.core.workflow.is_video_file', return_value=True):
                        with patch('subtitletools.core.workflow.is_audio_file', return_value=False):
                            result = workflow.batch_process(
                                input_paths=["video.mp4"],
                                output_dir="output/"
                            )

                    # Should return a dictionary result structure
                    assert isinstance(result, dict)

    @patch('subtitletools.core.workflow.ensure_directory')
    def test_batch_process_with_error(self, mock_ensure: Mock) -> None:
        """Test batch_process with some files failing."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    # Mock to always fail to test error handling
                    workflow.transcribe_and_translate = Mock(side_effect=Exception("Processing failed"))

                    with patch('subtitletools.core.workflow.is_video_file', return_value=True):
                        with patch('subtitletools.core.workflow.is_audio_file', return_value=False):
                            result = workflow.batch_process(
                                input_paths=["video1.mp4"],
                                output_dir="output/"
                            )

                    assert len(result) >= 1
                    # Should have at least one result entry
                    assert isinstance(result, dict)

    def test_get_workflow_info(self) -> None:
        """Test get_workflow_info method."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator_class:
                with patch('subtitletools.core.workflow.SubtitleProcessor'):
                    # Native processing is always available

                    # Mock instances
                    mock_transcriber = Mock()
                    mock_transcriber.get_model_info.return_value = {"model": "small", "device": "cpu"}
                    mock_transcriber_class.return_value = mock_transcriber

                    mock_translator = Mock()
                    mock_translator.get_service_info.return_value = {"service": "google", "has_api_key": False}
                    mock_translator_class.return_value = mock_translator

                    workflow = SubtitleWorkflow()

                    info = workflow.get_workflow_info()

                    assert "transcriber" in info
                    assert "translator" in info
                    assert "postprocess_available" in info
                    assert info["postprocess_available"] is True
                    assert info["transcriber"]["model"] == "small"
                    assert info["translator"]["service"] == "google"


class TestWorkflowMissingCoverage:
    """Test cases specifically targeting missing coverage in workflow module."""

    def test_checkpoint_manager_save_checkpoint_error_handling(self) -> None:
        """Test CheckpointManager.save_checkpoint with file writing errors."""
        with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path("/invalid/path")):
            with patch('subtitletools.core.workflow.ensure_directory'):
                manager = CheckpointManager("test_workflow")

                # Should handle the error gracefully without raising
                manager.save_checkpoint({"test": "data"})

    def test_checkpoint_manager_load_checkpoint_json_error(self) -> None:
        """Test CheckpointManager.load_checkpoint with JSON parsing error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path(tmp_dir)):
                manager = CheckpointManager("test_workflow")

                # Create invalid JSON file
                with open(manager.checkpoint_file, 'w', encoding='utf-8') as f:
                    f.write("invalid json content {")

                result = manager.load_checkpoint()
                assert result is None

    def test_checkpoint_manager_load_checkpoint_file_error(self) -> None:
        """Test CheckpointManager.load_checkpoint with file reading error."""
        with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path("/tmp")):
            with patch('subtitletools.core.workflow.ensure_directory'):
                manager = CheckpointManager("test_workflow")

                # Mock file exists but reading fails
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('builtins.open', side_effect=OSError("Permission denied")):
                        result = manager.load_checkpoint()
                        assert result is None

    def test_checkpoint_manager_clear_checkpoint_error_handling(self) -> None:
        """Test CheckpointManager.clear_checkpoint with file deletion errors."""
        with patch('subtitletools.core.workflow.get_cache_dir', return_value=Path("/tmp")):
            with patch('subtitletools.core.workflow.ensure_directory'):
                manager = CheckpointManager("test_workflow")

                # Mock file exists but deletion fails
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.unlink', side_effect=OSError("Permission denied")):
                        # Should handle the error gracefully without raising
                        manager.clear_checkpoint()

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_video_file')
    @patch('subtitletools.core.workflow.is_audio_file')
    def test_transcribe_and_translate_unsupported_file_type(self, mock_is_audio: Mock, mock_is_video: Mock, mock_validate: Mock) -> None:
        """Test transcribe_and_translate with unsupported file type."""
        input_path = Path("test.txt")
        mock_validate.return_value = input_path
        mock_is_video.return_value = False
        mock_is_audio.return_value = False

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):
                    with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:
                        with patch('subtitletools.core.workflow.get_file_size_mb', return_value=1.0):

                            mock_checkpoint = Mock()
                            mock_checkpoint.load_checkpoint.return_value = None
                            mock_checkpoint_class.return_value = mock_checkpoint

                            workflow = SubtitleWorkflow()

                            with pytest.raises(WorkflowError, match="Unsupported file type"):
                                workflow.transcribe_and_translate(input_path="test.txt")

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_video_file')
    @patch('subtitletools.core.workflow.get_file_size_mb')
    def test_transcribe_and_translate_audio_with_temp_file_cleanup(self, mock_size: Mock, mock_is_video: Mock, mock_validate: Mock) -> None:
        """Test transcribe_and_translate with audio file and temporary file handling."""
        input_path = Path("test.mp3")
        mock_validate.return_value = input_path
        mock_is_video.return_value = False
        mock_size.return_value = 10.0

        with patch('subtitletools.core.workflow.is_audio_file', return_value=True):
            with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
                with patch('subtitletools.core.workflow.SubtitleTranslator'):
                    with patch('subtitletools.core.workflow.SubtitleProcessor'):
                        with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:
                            with patch('subtitletools.core.workflow.is_space_language', return_value=True):

                                # Mock transcriber
                                mock_transcriber = Mock()
                                mock_transcriber.transcribe_audio.return_value = {"segments": []}
                                mock_transcriber_class.return_value = mock_transcriber

                                mock_checkpoint = Mock()
                                mock_checkpoint.load_checkpoint.return_value = None
                                mock_checkpoint_class.return_value = mock_checkpoint

                                workflow = SubtitleWorkflow()
                                workflow._translate_subtitles = Mock(return_value=[])

                                with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
                                    tmp_path = Path(tmp.name)

                                    with patch('pathlib.Path.with_suffix', return_value=tmp_path):
                                        result = workflow.transcribe_and_translate(
                                            input_path="test.mp3",
                                            output_path=tmp_path
                                        )

                                assert result["input_path"] == str(input_path)

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_video_file')
    @patch('subtitletools.core.workflow.get_file_size_mb')
    def test_transcribe_and_translate_progress_callback(self, mock_size: Mock, mock_is_video: Mock, mock_validate: Mock) -> None:
        """Test transcribe_and_translate with progress callback."""
        input_path = Path("test.mp4")
        mock_validate.return_value = input_path
        mock_is_video.return_value = True
        mock_size.return_value = 100.0

        progress_calls = []
        def mock_progress_callback(step: str, progress: float) -> None:
            progress_calls.append((step, progress))

        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):
                    with patch('subtitletools.core.workflow.CheckpointManager') as mock_checkpoint_class:
                        with patch('subtitletools.core.workflow.is_space_language', return_value=True):

                            mock_transcriber = Mock()
                            mock_transcriber.transcribe_video.return_value = "temp.srt"
                            mock_transcriber_class.return_value = mock_transcriber

                            mock_checkpoint = Mock()
                            mock_checkpoint.load_checkpoint.return_value = None
                            mock_checkpoint_class.return_value = mock_checkpoint

                            workflow = SubtitleWorkflow()
                            workflow._translate_subtitles = Mock(return_value=[])

                            workflow.transcribe_and_translate(
                                input_path="test.mp4",
                                progress_callback=mock_progress_callback
                            )

                            # Should have called progress callback multiple times
                            assert len(progress_calls) >= 3  # Transcribing, Translating, Completed
                            progress_messages = [call[0] for call in progress_calls]
                            assert any("transcrib" in msg.lower() for msg in progress_messages)
                            assert any("translat" in msg.lower() for msg in progress_messages)
                            assert any("completed" in msg.lower() for msg in progress_messages)

    def test_translate_subtitles_translation_error(self) -> None:
        """Test _translate_subtitles with translation error."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock processor to raise error
                    mock_processor = Mock()
                    mock_processor.extract_text.side_effect = Exception("Extract failed")
                    mock_processor_class.return_value = mock_processor

                    workflow = SubtitleWorkflow()

                    with pytest.raises(TranslationError, match="Subtitle translation failed"):
                        workflow._translate_subtitles([Mock()], "en", "es")

    def test_translate_subtitles_translator_error(self) -> None:
        """Test _translate_subtitles with translator error."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator_class:
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock processor
                    mock_processor = Mock()
                    mock_processor.extract_text.return_value = ["Hello"]
                    mock_processor_class.return_value = mock_processor

                    # Mock translator to raise error
                    mock_translator = Mock()
                    mock_translator.translate_lines.side_effect = Exception("Translation failed")
                    mock_translator_class.return_value = mock_translator

                    workflow = SubtitleWorkflow()

                    with pytest.raises(TranslationError, match="Subtitle translation failed"):
                        workflow._translate_subtitles([Mock()], "en", "es")

    @patch('subtitletools.core.workflow.validate_postprocess_environment')
    @patch('subtitletools.core.workflow.apply_subtitle_edit_postprocess')
    def test_apply_postprocessing_apply_error(self, mock_apply: Mock, mock_validate_env: Mock) -> None:
        """Test _apply_postprocessing when apply function fails."""
        mock_validate_env.return_value = {
            "postprocess_available": True
        }
        mock_apply.side_effect = Exception("Processing error")

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    result = workflow._apply_postprocessing(
                        Path("test.srt"), ["fix-common-errors"]
                    )

                    assert result is False

    @patch('subtitletools.core.workflow.validate_file_exists')
    @patch('subtitletools.core.workflow.is_space_language')
    def test_translate_existing_subtitles_processor_error(self, mock_space: Mock, mock_validate: Mock) -> None:
        """Test translate_existing_subtitles with subtitle processor error."""
        input_path = Path("input.srt")
        mock_validate.return_value = input_path
        mock_space.return_value = True

        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor') as mock_processor_class:

                    # Mock processor to raise error
                    mock_processor = Mock()
                    mock_processor.parse_file.side_effect = SubtitleError("Parse failed")
                    mock_processor_class.return_value = mock_processor

                    workflow = SubtitleWorkflow()

                    with pytest.raises(WorkflowError, match="Subtitle translation workflow failed"):
                        workflow.translate_existing_subtitles("input.srt", "output.srt")

    @patch('subtitletools.core.workflow.ensure_directory')
    def test_batch_process_mixed_success_failure(self, mock_ensure: Mock) -> None:
        """Test batch_process with some files succeeding and some failing."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    # Mock to succeed on first file, fail on second
                    def mock_transcribe_and_translate(*args, **kwargs):
                        if "video1" in str(args[0]):
                            return {"status": "completed", "total_time": 120}
                        raise RuntimeError("Processing failed")

                    workflow.transcribe_and_translate = Mock(side_effect=mock_transcribe_and_translate)

                    with patch('subtitletools.core.workflow.is_video_file', return_value=True):
                        with patch('subtitletools.core.workflow.is_audio_file', return_value=False):
                            result = workflow.batch_process(
                                input_paths=["video1.mp4", "video2.mp4"],
                                output_dir="output/"
                            )

                    # Should have both results now that the bug is fixed
                    assert len(result) == 2
                    # First should succeed
                    assert result["video1.mp4"]["status"] == "completed"
                    # Second should fail
                    assert result["video2.mp4"]["status"] == "failed"
                    assert "error" in result["video2.mp4"]

    @patch('subtitletools.core.workflow.ensure_directory')
    def test_batch_process_subtitle_file_workflow(self, mock_ensure: Mock) -> None:
        """Test batch_process with subtitle files (translation-only workflow)."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber'):
            with patch('subtitletools.core.workflow.SubtitleTranslator'):
                with patch('subtitletools.core.workflow.SubtitleProcessor'):

                    workflow = SubtitleWorkflow()

                    workflow.translate_existing_subtitles = Mock(return_value={"status": "completed", "total_time": 30})

                    with patch('subtitletools.core.workflow.is_video_file', return_value=False):
                        with patch('subtitletools.core.workflow.is_audio_file', return_value=False):
                            result = workflow.batch_process(
                                input_paths=["subtitle.srt"],
                                output_dir="output/"
                            )

                    assert len(result) >= 1
                    workflow.translate_existing_subtitles.assert_called_once()

    def test_get_workflow_info_native_processing(self) -> None:
        """Test get_workflow_info with native processing (always available)."""
        with patch('subtitletools.core.workflow.SubWhisperTranscriber') as mock_transcriber_class:
            with patch('subtitletools.core.workflow.SubtitleTranslator') as mock_translator_class:
                with patch('subtitletools.core.workflow.SubtitleProcessor'):
                    with patch('subtitletools.core.workflow.validate_postprocess_environment') as mock_validate:
                        # Mock native processing only (no external postprocessing tools)
                        mock_validate.return_value = {"postprocess_available": False}

                        # Mock instances
                        mock_transcriber = Mock()
                        mock_transcriber.get_model_info.return_value = {"model": "large", "device": "cuda"}
                        mock_transcriber_class.return_value = mock_transcriber

                        mock_translator = Mock()
                        mock_translator.get_service_info.return_value = {"service": "google_cloud", "has_api_key": True}
                        mock_translator_class.return_value = mock_translator

                        workflow = SubtitleWorkflow()

                        info = workflow.get_workflow_info()

                        assert "transcriber" in info
                        assert "translator" in info
                        assert "postprocess_available" in info
                        assert info["postprocess_available"] is False
                        assert info["transcriber"]["model"] == "large"
                        assert info["translator"]["service"] == "google_cloud"
