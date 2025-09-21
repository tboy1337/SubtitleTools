"""Transcription module for SubtitleTools.

This module provides audio/video to subtitle transcription functionality using
OpenAI's Whisper model, adapted from the original subwhisper implementation.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import whisper

from ..config.settings import DEFAULT_WHISPER_MODEL, WHISPER_MODELS, get_temp_dir
from ..utils.audio import cleanup_temp_dir, extract_audio, validate_audio_file
from ..utils.common import (
    format_timestamp,
    get_file_size_mb,
    is_audio_file,
    is_video_file,
    validate_file_exists,
)

logger = logging.getLogger(__name__)


class WhisperSegment(TypedDict):
    """Type definition for Whisper transcription segment."""

    start: float
    end: float
    text: str


class WhisperResult(TypedDict):
    """Type definition for Whisper transcription result."""

    segments: List[WhisperSegment]
    language: str


class TranscriptionError(Exception):
    """Exception raised for transcription errors."""


class SubWhisperTranscriber:
    """Main transcription class using OpenAI Whisper."""

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        language: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the transcriber.

        Args:
            model_name: Whisper model name to use
            language: Language code for transcription (None for auto-detect)
            device: Device to use for processing (None for auto-detect)

        Raises:
            TranscriptionError: If model is invalid or cannot be loaded
        """
        if model_name not in WHISPER_MODELS:
            raise TranscriptionError(
                f"Invalid model '{model_name}'. Supported models: {WHISPER_MODELS}"
            )

        self.model_name = model_name
        self.language = language
        self.device = device
        self._model: Optional[object] = None

        logger.info("Initialized transcriber with model: %s", model_name)

    @property
    def model(self) -> object:
        """Lazy-load the Whisper model."""
        if self._model is None:
            logger.info("Loading Whisper model: %s", self.model_name)
            try:
                self._model = cast(
                    Any, whisper.load_model(self.model_name, device=self.device)
                )
                logger.info("Successfully loaded Whisper model: %s", self.model_name)
            except Exception as e:
                raise TranscriptionError(
                    f"Failed to load Whisper model '{self.model_name}': {e}"
                ) from e
        return self._model

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        max_segment_length: Optional[int] = None,
        **whisper_options: Union[str, int, float, bool],
    ) -> WhisperResult:
        """Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            max_segment_length: Maximum segment length in characters
            **whisper_options: Additional options for Whisper

        Returns:
            Transcription result with segments and detected language

        Raises:
            TranscriptionError: If transcription fails
        """
        audio_path_obj = validate_file_exists(audio_path)
        logger.info("Starting audio transcription: %s", audio_path_obj)

        # Validate audio file
        if not validate_audio_file(audio_path_obj):
            raise TranscriptionError(
                f"Invalid or corrupted audio file: {audio_path_obj}"
            )

        file_size_mb = get_file_size_mb(audio_path_obj)
        logger.info("Audio file size: %.2f MB", file_size_mb)

        # Setup transcription options
        transcribe_options: Dict[str, Union[str, int, float, bool]] = {
            "verbose": False,
            "fp16": False,  # Disable fp16 which can cause issues on CPU
        }

        if self.language:
            transcribe_options["language"] = str(self.language)
            logger.info("Using specified language: %s", self.language)
        else:
            logger.info("Using automatic language detection")

        # Add any additional options
        transcribe_options.update(whisper_options)
        logger.debug("Transcription options: %s", transcribe_options)

        # Start transcription
        logger.info("Beginning audio transcription...")
        start_time = time.time()

        try:
            result = self._perform_transcription(audio_path_obj, transcribe_options)

            elapsed_time = time.time() - start_time
            logger.info("Transcription completed in %.2f seconds", elapsed_time)

            # Validate and log results
            if not result or "segments" not in result:
                raise TranscriptionError("Transcription returned invalid result")

            segments = result["segments"]
            detected_language = result.get("language", "unknown")

            logger.info(
                "Transcription successful: %d segments detected, language: %s",
                len(segments),
                detected_language,
            )

            if segments:
                total_duration = cast(float, cast(dict[str, Any], segments[-1])["end"])
                logger.info("Total transcribed duration: %.2f seconds", total_duration)

            # Apply segment length limit if specified
            if max_segment_length:
                segments = self._split_long_segments(segments, max_segment_length)  # type: ignore[assignment,arg-type]

            return WhisperResult(
                segments=cast(List[WhisperSegment], segments),
                language=cast(str, detected_language),
            )

        except Exception as e:
            raise TranscriptionError(f"Audio transcription failed: {e}") from e

    def _perform_transcription(
        self, audio_path: Path, options: Dict[str, Union[str, int, float, bool]]
    ) -> Dict[str, Union[str, List[Dict[str, Union[str, float]]]]]:
        """Perform the actual transcription with audio preprocessing."""
        try:
            # Try to load audio with scipy for preprocessing
            logger.debug("Loading audio with scipy for preprocessing...")
            sample_rate_raw, audio_data_raw = wav.read(str(audio_path))
            sample_rate = cast(int, sample_rate_raw)
            audio_data = cast(object, audio_data_raw)  # numpy array from scipy
            logger.debug(
                "Loaded audio: sample_rate=%s, shape=%s, dtype=%s",
                sample_rate,
                getattr(audio_data, "shape", "unknown"),
                getattr(audio_data, "dtype", "unknown"),
            )

            # Convert to float32 and normalize
            if str(getattr(audio_data, "dtype", None)) == "int16":
                audio_data = (
                    cast(Any, getattr(audio_data, "astype")(np.float32)) / 32768.0
                )
            elif str(getattr(audio_data, "dtype", None)) == "int32":
                audio_data = (
                    cast(Any, getattr(audio_data, "astype")(np.float32)) / 2147483648.0
                )
            else:
                logger.debug(
                    "Audio data already in format: %s",
                    getattr(audio_data, "dtype", "unknown"),
                )

            # Convert to mono if stereo
            if getattr(audio_data, "ndim", 1) > 1:
                logger.debug(
                    "Converting stereo to mono (shape: %s)",
                    getattr(audio_data, "shape", "unknown"),
                )
                audio_data = cast(object, np.mean(audio_data, axis=1))

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                logger.info("Resampling audio from %dHz to 16kHz", sample_rate)
                audio_data = cast(
                    object,
                    scipy.signal.resample(
                        audio_data, int(cast(Any, len(audio_data)) * 16000 / sample_rate)  # type: ignore[arg-type]
                    ),
                )

            logger.debug(
                "Final audio shape: %s", getattr(audio_data, "shape", "unknown")
            )

            # Perform transcription with preprocessed audio
            model_obj = cast(Any, self.model)
            result_raw = cast(
                Any, getattr(model_obj, "transcribe")(audio_data, **options)
            )
            return cast(
                Dict[str, Union[str, List[Dict[str, Union[str, float]]]]], result_raw
            )

        except (OSError, IOError, RuntimeError, ValueError) as audio_error:
            logger.warning(
                "Error loading audio with scipy: %s. Falling back to whisper's audio loading",
                audio_error,
            )
            # Fall back to whisper's default loading
            model_obj = self.model
            result_raw = getattr(model_obj, "transcribe")(str(audio_path), **options)
            result = cast(
                Dict[str, Union[str, List[Dict[str, Union[str, float]]]]], result_raw
            )
            return cast(
                Dict[str, Union[str, List[Dict[str, Union[str, float]]]]], dict(result)
            )

    def _split_long_segments(
        self, segments: List[WhisperSegment], max_length: int
    ) -> List[WhisperSegment]:
        """Split segments that exceed the maximum length."""
        split_segments = []

        for segment in segments:
            text = segment["text"].strip()
            if len(text) <= max_length:
                split_segments.append(segment)
                continue

            # Split long segment
            words = text.split()
            current_text: List[str] = []
            current_length = 0
            segment_start = segment["start"]
            segment_duration = segment["end"] - segment["start"]

            for word in words:
                word_length = len(word) + 1  # +1 for space

                if current_length + word_length <= max_length or not current_text:
                    current_text.append(word)
                    current_length += word_length
                else:
                    # Create subsegment
                    subsegment_text = " ".join(current_text)
                    text_ratio = len(subsegment_text) / len(text)
                    subsegment_duration = segment_duration * text_ratio
                    subsegment_end = segment_start + subsegment_duration

                    split_segments.append(
                        WhisperSegment(
                            start=segment_start,
                            end=subsegment_end,
                            text=subsegment_text,
                        )
                    )

                    # Start new subsegment
                    segment_start = subsegment_end
                    current_text = [word]
                    current_length = word_length

            # Add final subsegment if there's remaining text
            if current_text:
                subsegment_text = " ".join(current_text)
                split_segments.append(
                    WhisperSegment(
                        start=segment_start, end=segment["end"], text=subsegment_text
                    )
                )

        logger.info(
            "Split %d segments into %d segments", len(segments), len(split_segments)
        )
        return split_segments

    def transcribe_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_segment_length: Optional[int] = None,
        **whisper_options: Union[str, int, float, bool],
    ) -> str:
        """Transcribe video file by extracting audio and then transcribing.

        Args:
            video_path: Path to video file
            output_path: Optional path for output subtitle file
            max_segment_length: Maximum segment length in characters
            **whisper_options: Additional options for Whisper

        Returns:
            Path to generated subtitle file

        Raises:
            TranscriptionError: If transcription fails
        """
        video_path_obj = validate_file_exists(video_path)

        if output_path is None:
            output_path = video_path_obj.with_suffix(".srt")
        else:
            output_path = Path(output_path)

        logger.info("Transcribing video: %s -> %s", video_path_obj, output_path)

        try:
            # Extract audio from video
            logger.info("Extracting audio from video...")
            temp_dir = get_temp_dir()
            audio_path = extract_audio(video_path_obj, temp_dir=str(temp_dir))

            # Transcribe the extracted audio
            result = self.transcribe_audio(
                audio_path, max_segment_length=max_segment_length, **whisper_options
            )

            # Generate SRT file
            self.generate_srt(result["segments"], output_path)

            logger.info("Successfully transcribed video: %s", output_path)
            return str(output_path)

        except Exception as e:
            raise TranscriptionError(f"Video transcription failed: {e}") from e
        finally:
            # Clean up temporary files
            cleanup_temp_dir()

    def generate_srt(
        self, segments: List[WhisperSegment], output_file: Union[str, Path]
    ) -> None:
        """Generate SRT subtitle file from Whisper segments.

        Args:
            segments: List of transcription segments
            output_file: Path to output SRT file

        Raises:
            TranscriptionError: If SRT generation fails
        """
        if not segments:
            logger.warning("No segments provided for SRT generation")
            segments = []

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating SRT file with %d segments: %s", len(segments), output_path
        )

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    text = segment["text"].strip()

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

            logger.info("Successfully generated SRT file: %s", output_path)

            # Log file size for verification
            file_size = output_path.stat().st_size
            logger.debug("Generated SRT file size: %d bytes", file_size)

        except Exception as e:
            raise TranscriptionError(f"Failed to generate SRT file: {e}") from e

    def batch_transcribe(
        self,
        input_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        max_segment_length: Optional[int] = None,
        **whisper_options: Union[str, int, float, bool],
    ) -> Dict[str, Dict[str, Optional[str]]]:
        """Transcribe multiple video/audio files.

        Args:
            input_paths: List of input file paths
            output_dir: Output directory for subtitle files
            max_segment_length: Maximum segment length in characters
            **whisper_options: Additional options for Whisper

        Returns:
            Dictionary mapping input paths to results
        """
        results = {}

        for i, input_path in enumerate(input_paths, 1):
            logger.info("Processing file %d/%d: %s", i, len(input_paths), input_path)

            try:
                input_path_obj = Path(input_path)

                if output_dir:
                    output_path = (
                        Path(output_dir) / input_path_obj.with_suffix(".srt").name
                    )
                else:
                    output_path = input_path_obj.with_suffix(".srt")

                # Transcribe based on file type
                if is_video_file(input_path):
                    result_path = self.transcribe_video(
                        input_path, output_path, max_segment_length, **whisper_options
                    )
                elif is_audio_file(input_path):
                    transcription_result = self.transcribe_audio(
                        input_path, max_segment_length, **whisper_options
                    )
                    self.generate_srt(transcription_result["segments"], output_path)
                    result_path = str(output_path)
                else:
                    raise TranscriptionError(f"Unsupported file type: {input_path}")

                results[str(input_path)] = {
                    "status": "success",
                    "output": result_path,
                    "error": None,
                }

                logger.info("Successfully processed %s", input_path)

            except (TranscriptionError, OSError, IOError) as e:
                logger.error("Failed to process %s: %s", input_path, e)
                results[str(input_path)] = {
                    "status": "failed",
                    "output": None,
                    "error": str(e),
                }

        # Summary
        successful = sum(1 for r in results.values() if r["status"] == "success")
        logger.info(
            "Batch transcription complete: %d/%d successful",
            successful,
            len(input_paths),
        )

        return results

    def get_model_info(self) -> Dict[str, Union[str, Optional[str], bool]]:
        """Get information about the current model.

        Returns:
            Dictionary with model information
        """
        info: Dict[str, Union[str, Optional[str], bool]] = {
            "model_name": self.model_name,
            "language": self.language,
            "device": self.device,
            "loaded": self._model is not None,
        }

        if self._model is not None:
            try:
                # Get additional model info if available
                dims = cast(Dict[str, Any], getattr(self._model, "dims", {}))
                if hasattr(dims, "get"):
                    info["model_size"] = str(dims.get("n_mels", "unknown"))
                else:
                    info["model_size"] = "unknown"
            except (AttributeError, TypeError, ValueError):
                info["model_size"] = "unknown"

        return info
