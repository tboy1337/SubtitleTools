"""Workflow orchestration module for SubtitleTools.

This module provides end-to-end subtitle workflows that combine transcription,
translation, and post-processing operations.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union, cast

from ..config.settings import (
    DEFAULT_ENCODING,
    DEFAULT_SRC_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    get_cache_dir,
)
from ..utils.common import (
    ensure_directory,
    get_file_size_mb,
    is_audio_file,
    is_video_file,
    validate_file_exists,
)
from ..utils.postprocess import (
    apply_subtitle_edit_postprocess,
    validate_postprocess_environment,
)
from .subtitle import SubtitleError, SubtitleProcessor
from .transcription import SubWhisperTranscriber, TranscriptionError
from .translation import SubtitleTranslator, TranslationError, is_space_language

logger = logging.getLogger(__name__)


class CheckpointData(TypedDict):
    """Type definition for checkpoint data."""

    workflow_id: str
    timestamp: float
    data: Dict[str, Union[str, float, int, bool, List[str]]]


class WorkflowResults(TypedDict, total=False):
    """Type definition for workflow results."""

    status: str
    output_path: str
    input_path: str
    total_time: float
    transcription_time: Optional[float]
    translation_time: Optional[float]
    postprocessing_time: Optional[float]
    postprocessing_success: Optional[bool]
    original_segments: Optional[int]
    translated_segments: Optional[int]
    steps_completed: List[str]
    # Allow additional fields for flexibility
    file_size_mb: Optional[float]
    src_lang: Optional[str]
    target_lang: Optional[str]
    error: Optional[str]


class WorkflowError(Exception):
    """Exception raised for workflow errors."""


class CheckpointManager:
    """Manages workflow checkpoints for resumability."""

    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.checkpoint_dir = get_cache_dir() / "checkpoints"
        ensure_directory(self.checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / f"{workflow_id}.json"

    def save_checkpoint(
        self, data: Dict[str, Union[str, float, int, bool, List[str]]]
    ) -> None:
        """Save workflow checkpoint."""
        try:
            checkpoint_data: CheckpointData = {
                "workflow_id": self.workflow_id,
                "timestamp": time.time(),
                "data": data,
            }

            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.debug("Saved checkpoint: %s", self.checkpoint_file)

        except (OSError, IOError, ValueError) as e:
            logger.warning("Failed to save checkpoint: %s", e)

    def load_checkpoint(
        self,
    ) -> Optional[Dict[str, Union[str, float, int, bool, List[str]]]]:
        """Load workflow checkpoint."""
        try:
            if not self.checkpoint_file.exists():
                return None

            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = cast(Dict[str, Any], json.load(f))

            logger.debug("Loaded checkpoint: %s", self.checkpoint_file)
            return cast(
                Optional[Dict[str, Union[str, float, int, bool, List[str]]]],
                checkpoint_data.get("data"),
            )

        except (OSError, IOError, ValueError, KeyError) as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return None

    def clear_checkpoint(self) -> None:
        """Clear workflow checkpoint."""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.debug("Cleared checkpoint: %s", self.checkpoint_file)
        except (OSError, IOError) as e:
            logger.warning("Failed to clear checkpoint: %s", e)


class SubtitleWorkflow:
    """Main workflow orchestrator for subtitle processing."""

    def __init__(
        self,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        translation_service: str = "google",
        api_key: Optional[str] = None,
    ):
        """Initialize the workflow.

        Args:
            whisper_model: Whisper model to use for transcription
            translation_service: Translation service to use
            api_key: Optional API key for translation service
        """
        self.transcriber = SubWhisperTranscriber(whisper_model)
        self.translator = SubtitleTranslator(translation_service, api_key)
        self.subtitle_processor = SubtitleProcessor()

        logger.info(
            "Initialized workflow with model: %s, service: %s",
            whisper_model,
            translation_service,
        )

    def transcribe_and_translate(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        *,  # Force keyword-only arguments
        src_lang: str = DEFAULT_SRC_LANGUAGE,
        target_lang: str = DEFAULT_TARGET_LANGUAGE,
        max_segment_length: Optional[int] = None,
        both: bool = True,
        resume: bool = True,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        postprocess_operations: Optional[List[str]] = None,
        **kwargs: Union[str, int, float, bool],
    ) -> WorkflowResults:
        """Complete workflow: transcribe video/audio and translate subtitles.

        Args:
            input_path: Path to input video/audio file
            output_path: Optional output path for final subtitles
            src_lang: Source language for translation
            target_lang: Target language for translation
            max_segment_length: Maximum segment length in characters
            both: Whether to include both original and translated text
            resume: Whether to resume from checkpoint
            progress_callback: Optional progress callback
            postprocess_operations: Optional list of post-processing operations
            **kwargs: Additional options

        Returns:
            Dictionary with workflow results

        Raises:
            WorkflowError: If workflow fails
        """
        input_path_obj = validate_file_exists(input_path)

        if output_path is None:
            output_path = input_path_obj.with_suffix(".srt")
        else:
            output_path = Path(output_path)

        # Create checkpoint manager
        workflow_id = f"transcribe_translate_{input_path_obj.stem}_{int(time.time())}"
        checkpoint_manager = CheckpointManager(workflow_id)

        # Check for existing checkpoint
        checkpoint_data = None
        if resume:
            checkpoint_data = checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                logger.info(
                    "Resuming workflow from checkpoint: %s",
                    checkpoint_data.get("step", "unknown"),
                )

        try:
            results: WorkflowResults = {
                "status": "in_progress",
                "input_path": str(input_path_obj),
                "output_path": str(output_path),
                "total_time": 0.0,
                "transcription_time": None,
                "translation_time": None,
                "postprocessing_time": None,
                "postprocessing_success": None,
                "translated_segments": None,
                "steps_completed": [],
                "file_size_mb": get_file_size_mb(input_path_obj),
                "src_lang": src_lang,
                "target_lang": target_lang,
            }

            start_time = time.time()

            # Step 1: Transcription
            if not checkpoint_data or checkpoint_data.get("step") == "transcription":
                if progress_callback:
                    progress_callback("Transcribing audio/video...", 0.1)

                logger.info("Step 1: Transcribing audio/video")
                step_start = time.time()

                if is_video_file(input_path_obj):
                    transcribe_kwargs = {}
                    if src_lang != "auto":
                        transcribe_kwargs["language"] = src_lang
                    temp_srt_path_str = self.transcriber.transcribe_video(
                        input_path_obj,
                        max_segment_length=max_segment_length,
                        **transcribe_kwargs,
                    )
                    temp_srt_path = Path(temp_srt_path_str)
                elif is_audio_file(input_path_obj):
                    transcribe_kwargs = {}
                    if src_lang != "auto":
                        transcribe_kwargs["language"] = src_lang
                    transcription_result = self.transcriber.transcribe_audio(
                        input_path_obj,
                        max_segment_length=max_segment_length,
                        **transcribe_kwargs,
                    )
                    temp_srt_path = input_path_obj.with_suffix("_temp.srt")
                    self.transcriber.generate_srt(
                        transcription_result["segments"], temp_srt_path
                    )
                else:
                    raise WorkflowError(f"Unsupported file type: {input_path_obj}")

                step_time = time.time() - step_start
                results["transcription_time"] = step_time
                results["steps_completed"].append("transcription")

                # Save checkpoint
                checkpoint_manager.save_checkpoint(
                    {
                        "step": "translation",
                        "temp_srt_path": str(temp_srt_path),
                        "results": results,  # type: ignore[dict-item]
                    }
                )

                logger.info("Transcription completed in %.2f seconds", step_time)

            else:
                # Resume from checkpoint
                temp_srt_path = Path(str(checkpoint_data["temp_srt_path"]))
                checkpoint_results = cast(
                    Dict[str, Union[str, float, int, bool, List[str]]],
                    checkpoint_data["results"],
                )
                # Update results with checkpoint data
                for key, value in checkpoint_results.items():
                    if key in results:
                        results[key] = value  # type: ignore[literal-required]
                logger.info("Resumed: Transcription already completed")

            # Step 2: Translation
            if not checkpoint_data or checkpoint_data.get("step") in [
                "transcription",
                "translation",
            ]:
                if progress_callback:
                    progress_callback("Translating subtitles...", 0.5)

                logger.info("Step 2: Translating subtitles")
                step_start = time.time()

                # Parse transcribed subtitles
                original_subtitles = self.subtitle_processor.parse_file(temp_srt_path)

                # Determine if target language uses spaces
                space = (
                    is_space_language(target_lang)
                    if target_lang != src_lang
                    else bool(kwargs.get("space", False))
                )

                # Translate subtitles
                translated_subtitles = self._translate_subtitles(
                    original_subtitles,
                    src_lang,
                    target_lang,
                    space=space,
                    both=both,
                    progress_callback=lambda current, total, msg: (
                        progress_callback(
                            f"Translating: {msg}", 0.5 + 0.3 * (current / total)
                        )
                        if progress_callback
                        else None
                    ),
                )

                # Save translated subtitles
                self.subtitle_processor.save_file(translated_subtitles, output_path)

                step_time = time.time() - step_start
                results["translation_time"] = step_time
                results["steps_completed"].append("translation")
                results["translated_segments"] = len(translated_subtitles)

                # Save checkpoint
                checkpoint_manager.save_checkpoint(
                    {
                        "step": "postprocessing",
                        "results": results,  # type: ignore[dict-item]
                    }
                )

                logger.info("Translation completed in %.2f seconds", step_time)

            # Step 3: Post-processing (optional)
            if postprocess_operations:
                if progress_callback:
                    progress_callback("Post-processing subtitles...", 0.8)

                logger.info("Step 3: Post-processing subtitles")
                step_start = time.time()

                success = self._apply_postprocessing(
                    output_path, postprocess_operations
                )

                step_time = time.time() - step_start
                results["postprocessing_time"] = step_time
                results["postprocessing_success"] = success
                results["steps_completed"].append("postprocessing")

                logger.info("Post-processing completed in %.2f seconds", step_time)

            # Finalize results
            total_time = time.time() - start_time
            results["total_time"] = total_time
            results["status"] = "completed"

            if progress_callback:
                progress_callback("Workflow completed!", 1.0)

            # Clear checkpoint
            checkpoint_manager.clear_checkpoint()

            # Clean up temporary files
            if "temp_srt_path" in locals() and Path(temp_srt_path).exists():
                try:
                    Path(temp_srt_path).unlink()
                    logger.debug("Cleaned up temporary file: %s", temp_srt_path)
                except (OSError, IOError) as e:
                    logger.warning("Failed to clean up temporary file: %s", e)

            logger.info("Workflow completed successfully in %.2f seconds", total_time)
            return results

        except (
            WorkflowError,
            TranscriptionError,
            TranslationError,
            SubtitleError,
        ) as e:
            # Save error checkpoint
            error_data: Dict[str, Union[str, float, int, bool, List[str]]] = {
                "step": "error",
                "error": str(e),
            }
            checkpoint_manager.save_checkpoint(error_data)

            raise WorkflowError(f"Workflow failed: {e}") from e

    def _translate_subtitles(
        self,
        subtitles: List["srt.Subtitle"],  # type: ignore[name-defined]
        src_lang: str,
        target_lang: str,
        *,  # Force keyword-only arguments
        space: bool = False,
        both: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List["srt.Subtitle"]:  # type: ignore[name-defined]
        """Translate subtitles using the translation service."""
        if src_lang == target_lang:
            logger.info(
                "Source and target languages are the same, skipping translation"
            )
            return subtitles

        try:
            # Extract text for translation
            text_lines = self.subtitle_processor.extract_text(subtitles)

            # Translate text
            translated_lines = self.translator.translate_lines(
                text_lines,
                src_lang,
                target_lang,
                progress_callback,
            )

            # Reconstruct subtitles with translated content
            translated_subtitles = self.subtitle_processor.reconstruct_subtitles(
                subtitles,
                translated_lines,
                space=space,
                both=both,
            )

            return translated_subtitles

        except Exception as e:
            raise TranslationError(f"Subtitle translation failed: {e}") from e

    def _apply_postprocessing(
        self,
        subtitle_path: Path,
        operations: List[str],
    ) -> bool:
        """Apply post-processing operations to subtitles."""
        try:
            # Check if post-processing environment is available
            env_check = validate_postprocess_environment()
            # Native implementation is always available
            if not env_check.get("postprocess_available", True):
                logger.debug("Post-processing available via native implementation")

            # Apply operations
            success = apply_subtitle_edit_postprocess(
                subtitle_path, operations, "subrip"
            )

            if success:
                logger.info("Post-processing completed successfully")
            else:
                logger.warning("Post-processing failed")

            return success
        except Exception as e:
            logger.error("Post-processing failed with error: %s", e)
            return False

    def translate_existing_subtitles(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        *,  # Force keyword-only arguments
        src_lang: str = DEFAULT_SRC_LANGUAGE,
        target_lang: str = DEFAULT_TARGET_LANGUAGE,
        both: bool = True,
        encoding: str = DEFAULT_ENCODING,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> WorkflowResults:
        """Translate existing subtitle file.

        Args:
            input_path: Path to input subtitle file
            output_path: Path to output subtitle file
            src_lang: Source language code
            target_lang: Target language code
            both: Whether to include both original and translated text
            encoding: File encoding
            progress_callback: Optional progress callback

        Returns:
            Dictionary with translation results
        """
        input_path_obj = validate_file_exists(input_path)
        output_path_obj = Path(output_path)

        try:
            logger.info(
                "Translating subtitles: %s -> %s", input_path_obj, output_path_obj
            )
            start_time = time.time()

            # Parse input subtitles
            subtitles = self.subtitle_processor.parse_file(input_path_obj, encoding)

            # Determine if target language uses spaces
            space = is_space_language(target_lang)

            # Translate subtitles
            translated_subtitles = self._translate_subtitles(
                subtitles,
                src_lang,
                target_lang,
                space=space,
                both=both,
                progress_callback=progress_callback,
            )

            # Save translated subtitles
            self.subtitle_processor.save_file(
                translated_subtitles, output_path_obj, encoding
            )

            total_time = time.time() - start_time

            results: WorkflowResults = {
                "input_path": str(input_path_obj),
                "output_path": str(output_path_obj),
                "src_lang": src_lang,
                "target_lang": target_lang,
                "original_segments": len(subtitles),
                "translated_segments": len(translated_subtitles),
                "total_time": total_time,
                "status": "completed",
                "transcription_time": None,
                "translation_time": total_time,
                "postprocessing_time": None,
                "postprocessing_success": None,
                "steps_completed": ["translation"],
                "file_size_mb": None,
            }

            logger.info("Translation completed in %.2f seconds", total_time)
            return results

        except Exception as e:
            raise WorkflowError(f"Subtitle translation workflow failed: {e}") from e

    def batch_process(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        postprocess_operations: Optional[List[str]] = None,
        **workflow_options: Union[str, int, float, bool],
    ) -> Dict[str, WorkflowResults]:
        """Process multiple files with the same workflow.

        Args:
            input_paths: List of input file paths
            output_dir: Output directory for processed files
            **workflow_options: Options to pass to workflow

        Returns:
            Dictionary mapping input paths to workflow results
        """
        output_dir_obj = Path(output_dir)
        ensure_directory(output_dir_obj)

        results: Dict[str, WorkflowResults] = {}
        successful = 0

        for i, input_path in enumerate(input_paths, 1):
            logger.info("Processing file %d/%d: %s", i, len(input_paths), input_path)

            try:
                input_path_obj = Path(input_path)
                output_path = output_dir_obj / input_path_obj.with_suffix(".srt").name

                # Determine workflow type based on input
                if is_video_file(input_path) or is_audio_file(input_path):
                    # Full transcription + translation workflow
                    result = self.transcribe_and_translate(
                        input_path,
                        output_path,
                        postprocess_operations=postprocess_operations,
                        **cast(Any, workflow_options),
                    )
                else:
                    # Translation-only workflow
                    result = self.translate_existing_subtitles(
                        input_path, output_path, **cast(Any, workflow_options)
                    )

                results[str(input_path)] = result
                successful += 1

            except (WorkflowError, TranscriptionError, TranslationError) as e:
                logger.error("Failed to process %s: %s", input_path, e)
                error_result: WorkflowResults = {
                    "status": "failed",
                    "input_path": str(input_path),
                    "output_path": "",
                    "total_time": 0.0,
                    "transcription_time": None,
                    "translation_time": None,
                    "postprocessing_time": None,
                    "postprocessing_success": None,
                    "translated_segments": None,
                    "steps_completed": [],
                    "error": str(e),
                }
                results[str(input_path)] = error_result
            except Exception as e:
                logger.error("Unexpected error processing %s: %s", input_path, e)
                unexpected_error_result: WorkflowResults = {
                    "status": "failed",
                    "input_path": str(input_path),
                    "output_path": "",
                    "total_time": 0.0,
                    "transcription_time": None,
                    "translation_time": None,
                    "postprocessing_time": None,
                    "postprocessing_success": None,
                    "translated_segments": None,
                    "steps_completed": [],
                    "error": str(e),
                }
                results[str(input_path)] = unexpected_error_result

        logger.info(
            "Batch processing completed: %d/%d successful", successful, len(input_paths)
        )
        return results

    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the current workflow configuration.

        Returns:
            Dictionary with workflow information
        """
        try:
            env_check = validate_postprocess_environment()
            postprocess_available = env_check.get("postprocess_available", False)
        except Exception:
            postprocess_available = False

        return {
            "transcriber": self.transcriber.get_model_info(),
            "translator": self.translator.get_service_info(),
            "postprocess_available": postprocess_available,
        }
