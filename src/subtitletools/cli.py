"""Command-line interface for SubtitleTools.

This module provides a unified CLI that combines transcription, translation,
and post-processing functionality in a single tool.
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union, cast

from .config.settings import (
    DEFAULT_ENCODING,
    DEFAULT_SRC_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    SUPPORTED_ENCODINGS,
    SUPPORTED_SUBTITLE_FORMATS,
    SUPPORTED_TRANSLATION_SERVICES,
    WHISPER_MODELS,
)
from .core.subtitle import SubtitleError, SubtitleProcessor
from .core.transcription import SubWhisperTranscriber, TranscriptionError
from .core.translation import SubtitleTranslator, TranslationError, is_space_language
from .core.workflow import SubtitleWorkflow, WorkflowError
from .utils.common import (
    get_system_info,
    is_audio_file,
    is_subtitle_file,
    is_video_file,
    setup_logging,
)
from .utils.encoding import (
    convert_to_multiple_encodings,
    get_recommended_encodings,
)
from .utils.postprocess import get_supported_output_formats

if TYPE_CHECKING:
    from typing import Any
    from argparse import _SubParsersAction

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="SubtitleTools - Complete subtitle workflow tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe video to subtitles
  subtitletools transcribe video.mp4 --model medium --language en

  # Translate subtitles
  subtitletools translate input.srt output.srt --src-lang en --target-lang es

  # Convert subtitle encoding
  subtitletools encode subtitle.srt --to-encoding utf-8

  # Complete workflow: transcribe + translate
  subtitletools workflow video.mp4 --target-lang fr --fix-common-errors

For more information, visit: https://github.com/tboy1337/SubtitleTools
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="SubtitleTools 1.0.0",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--log-file",
        help="Log file path (in addition to console output)",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Transcription command
    _add_transcribe_parser(subparsers)

    # Translation command
    _add_translate_parser(subparsers)

    # Encoding command
    _add_encode_parser(subparsers)

    # Workflow command
    _add_workflow_parser(subparsers)

    return parser


def _add_transcribe_parser(subparsers: "_SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add transcription command parser."""
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Generate subtitles from video/audio files",
        description="Transcribe audio/video files to subtitle files using OpenAI Whisper",
    )

    # Input/Output
    transcribe_parser.add_argument(
        "input",
        help="Input video/audio file or directory (with --batch)",
    )
    transcribe_parser.add_argument(
        "--output",
        "-o",
        help="Output subtitle file or directory (default: same as input with .srt extension)",
    )

    # Model options
    transcribe_parser.add_argument(
        "--model",
        "-m",
        choices=WHISPER_MODELS,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_WHISPER_MODEL})",
    )
    transcribe_parser.add_argument(
        "--language",
        "-l",
        help="Language code for transcription (auto-detect if not specified)",
    )

    # Processing options
    transcribe_parser.add_argument(
        "--max-segment-length",
        type=int,
        help="Maximum character length for subtitle segments",
    )
    transcribe_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in the input directory",
    )
    transcribe_parser.add_argument(
        "--extensions",
        default="mp4,mkv,avi,mov,webm",
        help="File extensions to process in batch mode (comma-separated)",
    )


def _add_translate_parser(subparsers: "_SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add translation command parser."""
    translate_parser = subparsers.add_parser(
        "translate",
        help="Translate subtitle files between languages",
        description="Translate existing subtitle files from one language to another",
    )

    # Input/Output
    translate_parser.add_argument(
        "input",
        help="Input subtitle file or directory (with --batch)",
    )
    translate_parser.add_argument(
        "output",
        help="Output subtitle file or directory",
    )

    # Language options
    translate_parser.add_argument(
        "--src-lang",
        "-s",
        default=DEFAULT_SRC_LANGUAGE,
        help=f"Source language code (default: {DEFAULT_SRC_LANGUAGE})",
    )
    translate_parser.add_argument(
        "--target-lang",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"Target language code (default: {DEFAULT_TARGET_LANGUAGE})",
    )

    # Translation options
    translate_parser.add_argument(
        "--service",
        choices=SUPPORTED_TRANSLATION_SERVICES,
        default="google",
        help="Translation service to use (default: google)",
    )
    translate_parser.add_argument(
        "--api-key",
        help="API key for translation service",
    )
    translate_parser.add_argument(
        "--both",
        action="store_true",
        default=True,
        help="Include both original and translated text (default: True)",
    )
    translate_parser.add_argument(
        "--only-translation",
        dest="both",
        action="store_false",
        help="Include only translated text",
    )

    # File options
    translate_parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=f"Input file encoding (default: {DEFAULT_ENCODING})",
    )
    translate_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in the input directory",
    )
    translate_parser.add_argument(
        "--pattern",
        default="*.srt",
        help="File pattern for batch processing (default: *.srt)",
    )


def _add_encode_parser(subparsers: "_SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add encoding command parser."""
    encode_parser = subparsers.add_parser(
        "encode",
        help="Convert subtitle file encodings",
        description="Convert subtitle files between different character encodings",
    )

    # Input/Output
    encode_parser.add_argument(
        "input",
        nargs="?",
        help="Input subtitle file or directory (with --batch)",
    )
    encode_parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory (default: same as input)",
    )

    # Encoding options
    encode_parser.add_argument(
        "--from-encoding",
        "-f",
        help="Source encoding (auto-detect if not specified)",
    )
    encode_parser.add_argument(
        "--to-encoding",
        "-t",
        help="Target encoding (can specify multiple with comma separation)",
    )
    encode_parser.add_argument(
        "--recommended",
        "-r",
        action="store_true",
        help="Use recommended encodings for specified language",
    )
    encode_parser.add_argument(
        "--language",
        default="en",
        help="Language code for recommended encodings (default: en)",
    )
    encode_parser.add_argument(
        "--list-encodings",
        action="store_true",
        help="List all supported encodings and exit",
    )

    # File options
    encode_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in the input directory",
    )
    encode_parser.add_argument(
        "--pattern",
        default="*.srt",
        help="File pattern for batch processing (default: *.srt)",
    )


def _add_workflow_parser(subparsers: "_SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add workflow command parser."""
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Run end-to-end subtitle workflows",
        description="Complete workflows: transcribe video → translate → post-process",
    )

    # Input/Output
    workflow_parser.add_argument(
        "input",
        help="Input video/audio file or directory (with --batch)",
    )
    workflow_parser.add_argument(
        "--output",
        "-o",
        help="Output subtitle file or directory",
    )

    # Transcription options
    workflow_parser.add_argument(
        "--model",
        "-m",
        choices=WHISPER_MODELS,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_WHISPER_MODEL})",
    )
    workflow_parser.add_argument(
        "--max-segment-length",
        type=int,
        help="Maximum character length for subtitle segments",
    )

    # Translation options
    workflow_parser.add_argument(
        "--src-lang",
        "-s",
        default="auto",
        help="Source language code (default: auto-detect)",
    )
    workflow_parser.add_argument(
        "--target-lang",
        "-t",
        default=DEFAULT_TARGET_LANGUAGE,
        help=f"Target language code (default: {DEFAULT_TARGET_LANGUAGE})",
    )
    workflow_parser.add_argument(
        "--service",
        choices=SUPPORTED_TRANSLATION_SERVICES,
        default="google",
        help="Translation service to use (default: google)",
    )
    workflow_parser.add_argument(
        "--api-key",
        help="API key for translation service",
    )
    workflow_parser.add_argument(
        "--both",
        action="store_true",
        default=True,
        help="Include both original and translated text (default: True)",
    )
    workflow_parser.add_argument(
        "--only-translation",
        dest="both",
        action="store_false",
        help="Include only translated text",
    )

    # Post-processing options
    workflow_parser.add_argument(
        "--fix-common-errors",
        action="store_true",
        help="Apply common subtitle error fixes",
    )
    workflow_parser.add_argument(
        "--remove-hi",
        action="store_true",
        help="Remove hearing impaired text",
    )
    workflow_parser.add_argument(
        "--auto-split-long-lines",
        action="store_true",
        help="Automatically split long subtitle lines",
    )
    workflow_parser.add_argument(
        "--fix-punctuation",
        action="store_true",
        help="Fix punctuation issues",
    )
    workflow_parser.add_argument(
        "--ocr-fix",
        action="store_true",
        help="Apply OCR error fixes",
    )
    workflow_parser.add_argument(
        "--convert-to",
        choices=get_supported_output_formats(),
        help="Convert subtitle to specified format",
    )

    # File options
    workflow_parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all files in the input directory",
    )
    workflow_parser.add_argument(
        "--extensions",
        default="mp4,mkv,avi,mov,webm",
        help="File extensions to process in batch mode (comma-separated)",
    )
    workflow_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint if available (default: True)",
    )
    workflow_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not resume from checkpoint",
    )


def handle_transcribe_command(args: argparse.Namespace) -> int:  # pylint: disable=too-many-return-statements
    """Handle the transcribe command."""
    try:
        transcriber = SubWhisperTranscriber(cast(Any, args).model, cast(Any, args).language)

        if cast(Any, args).batch or os.path.isdir(cast(Any, args).input):
            # Batch processing
            input_dir = Path(cast(Any, args).input)
            if not input_dir.is_dir():
                logger.error("Input directory not found: %s", input_dir)
                return 1

            # Find files
            extensions = [ext.strip() for ext in cast(Any, args).extensions.split(",")]
            files: List[Union[str, Path]] = []
            for ext in extensions:
                files.extend(input_dir.glob(f"**/*.{ext}"))
                files.extend(input_dir.glob(f"**/*.{ext.upper()}"))

            if not files:
                logger.error("No files found with extensions: %s", cast(Any, args).extensions)
                return 1

            output_dir = Path(cast(Any, args).output) if cast(Any, args).output else input_dir

            results = transcriber.batch_transcribe(
                files,
                output_dir,
                cast(Any, args).max_segment_length,
            )

            # Summary
            successful = sum(1 for r in results.values() if r["status"] == "success")
            print(f"\nTranscription completed: {successful}/{len(files)} successful")

            return 0 if successful > 0 else 1

        # Single file processing
        input_path = Path(cast(Any, args).input)
        output_path = Path(cast(Any, args).output) if cast(Any, args).output else input_path.with_suffix(".srt")

        if is_video_file(input_path):
            result_path = transcriber.transcribe_video(
                input_path,
                output_path,
                cast(Any, args).max_segment_length,
            )
        elif is_audio_file(input_path):
            transcription_result = transcriber.transcribe_audio(
                input_path,
                cast(Any, args).max_segment_length,
            )
            transcriber.generate_srt(transcription_result["segments"], output_path)
            result_path = str(output_path)
        else:
            logger.error("Unsupported file type: %s", input_path)
            return 1

        print(f"Transcription completed: {result_path}")
        return 0

    except TranscriptionError as e:
        logger.error("Transcription error: %s", e)
        return 1
    except (OSError, IOError) as e:
        logger.error("File system error: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return 1


def handle_translate_command(args: argparse.Namespace) -> int:  # pylint: disable=too-many-return-statements
    """Handle the translate command."""
    try:
        translator = SubtitleTranslator(cast(Any, args).service, cast(Any, args).api_key)
        subtitle_processor = SubtitleProcessor()

        if cast(Any, args).batch or os.path.isdir(cast(Any, args).input):
            # Batch processing
            input_dir = Path(cast(Any, args).input)
            output_dir = Path(cast(Any, args).output)

            if not input_dir.is_dir():
                logger.error("Input directory not found: %s", input_dir)
                return 1

            output_dir.mkdir(parents=True, exist_ok=True)

            # Find subtitle files
            files = list(input_dir.glob(cast(Any, args).pattern))
            if not files:
                logger.error("No files matching pattern '%s' found", cast(Any, args).pattern)
                return 1

            successful = 0
            for file_path in files:
                try:
                    output_path = output_dir / file_path.name

                    # Parse subtitles
                    subtitles = subtitle_processor.parse_file(file_path, cast(Any, args).encoding)

                    # Extract and translate text
                    text_lines = subtitle_processor.extract_text(subtitles)
                    translated_lines = translator.translate_lines(
                        text_lines,
                        cast(Any, args).src_lang,
                        cast(Any, args).target_lang,
                    )

                    # Reconstruct and save
                    space = is_space_language(cast(Any, args).target_lang)

                    translated_subtitles = subtitle_processor.reconstruct_subtitles(
                        subtitles,
                        translated_lines,
                        space=space,
                        both=cast(Any, args).both,
                    )

                    subtitle_processor.save_file(translated_subtitles, output_path, cast(Any, args).encoding)
                    successful += 1
                    print(f"Translated: {file_path} -> {output_path}")

                except (TranslationError, SubtitleError) as e:
                    logger.error("Failed to translate %s: %s", file_path, e)
                except (OSError, IOError) as e:
                    logger.error("File error processing %s: %s", file_path, e)

            print(f"\nTranslation completed: {successful}/{len(files)} successful")
            return 0 if successful > 0 else 1

        # Single file processing
        input_path = Path(cast(Any, args).input)
        output_path = Path(cast(Any, args).output)

        # Parse subtitles
        subtitles = subtitle_processor.parse_file(input_path, cast(Any, args).encoding)

        # Extract and translate text
        text_lines = subtitle_processor.extract_text(subtitles)
        translated_lines = translator.translate_lines(
            text_lines,
            cast(Any, args).src_lang,
            cast(Any, args).target_lang,
        )

        # Reconstruct and save
        space = is_space_language(cast(Any, args).target_lang)

        translated_subtitles = subtitle_processor.reconstruct_subtitles(
            subtitles,
            translated_lines,
            space=space,
            both=cast(Any, args).both,
        )

        subtitle_processor.save_file(translated_subtitles, output_path, cast(Any, args).encoding)
        print(f"Translation completed: {output_path}")
        return 0

    except (TranslationError, SubtitleError) as e:
        logger.error("Translation error: %s", e)
        return 1
    except (OSError, IOError) as e:
        logger.error("File system error: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Translation interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return 1


def handle_encode_command(args: argparse.Namespace) -> int:  # pylint: disable=too-many-return-statements
    """Handle the encode command."""
    try:
        # List encodings if requested
        if cast(Any, args).list_encodings:
            print("Supported encodings:")
            for encoding in sorted(SUPPORTED_ENCODINGS):
                print(f"  {encoding}")
            return 0

        if not cast(Any, args).input:
            print("Error: Input file/directory is required unless --list-encodings is used")
            return 1

        # Determine target encodings
        if cast(Any, args).to_encoding:
            target_encodings = [enc.strip() for enc in cast(Any, args).to_encoding.split(",")]
        elif cast(Any, args).recommended:
            target_encodings = get_recommended_encodings(cast(Any, args).language)
            print(f"Using recommended encodings for '{cast(Any, args).language}': {', '.join(target_encodings)}")
        else:
            target_encodings = ["utf-8", "utf-8-sig"]
            print(f"Using default encodings: {', '.join(target_encodings)}")

        if cast(Any, args).batch or os.path.isdir(cast(Any, args).input):
            # Batch processing
            input_dir = Path(cast(Any, args).input)
            output_dir = Path(cast(Any, args).output_dir) if cast(Any, args).output_dir else input_dir

            files = list(input_dir.glob(cast(Any, args).pattern))
            if not files:
                logger.error("No files matching pattern '%s' found", cast(Any, args).pattern)
                return 1

            successful = 0
            for file_path in files:
                try:
                    results = convert_to_multiple_encodings(
                        str(file_path),
                        str(output_dir),
                        target_encodings,
                    )

                    # Print results for this file
                    print(f"\n{file_path.name}:")
                    for encoding, success in results.items():
                        status = "✓" if success else "✗"
                        print(f"  {status} {encoding}")

                    if any(results.values()):
                        successful += 1

                except (OSError, IOError, UnicodeError) as e:
                    logger.error("Failed to process %s: %s", file_path, e)
                except Exception as e:
                    logger.error("Failed to process %s: %s", file_path, e)

            print(f"\nEncoding conversion completed: {successful}/{len(files)} files successful")
            return 0 if successful > 0 else 1

        # Single file processing
        input_path = Path(cast(Any, args).input)
        output_dir = Path(cast(Any, args).output_dir) if cast(Any, args).output_dir else input_path.parent

        results = convert_to_multiple_encodings(
            str(input_path),
            str(output_dir),
            target_encodings,
        )

        # Print results
        print(f"\nConversion results for {input_path.name}:")
        for encoding, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {encoding}")

        return 0 if any(results.values()) else 1

    except (OSError, IOError, UnicodeError) as e:
        logger.error("Encoding error: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Encoding conversion interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return 1


def handle_workflow_command(args: argparse.Namespace) -> int:  # pylint: disable=too-many-return-statements
    """Handle the workflow command."""
    try:
        # Check post-processing environment if needed
        postprocess_ops = []
        if cast(Any, args).fix_common_errors:
            postprocess_ops.append("/fixcommonerrors")
        if cast(Any, args).remove_hi:
            postprocess_ops.append("/removetextforhi")
        if cast(Any, args).auto_split_long_lines:
            postprocess_ops.append("/splitlonglines")
        if cast(Any, args).fix_punctuation:
            postprocess_ops.append("/fixpunctuation")
        if cast(Any, args).ocr_fix:
            postprocess_ops.append("/ocrfix")

        if postprocess_ops:
            # Post-processing is now handled natively, no environment validation needed
            logger.debug("Using native post-processing implementation")

        workflow = SubtitleWorkflow(
            cast(Any, args).model,
            cast(Any, args).service,
            cast(Any, args).api_key,
        )

        if cast(Any, args).batch or os.path.isdir(cast(Any, args).input):
            # Batch processing
            input_dir = Path(cast(Any, args).input)
            output_dir = Path(cast(Any, args).output) if cast(Any, args).output else input_dir

            # Find files
            extensions = [ext.strip() for ext in cast(Any, args).extensions.split(",")]
            files: List[Union[str, Path]] = []
            for ext in extensions:
                files.extend(input_dir.glob(f"**/*.{ext}"))
                files.extend(input_dir.glob(f"**/*.{ext.upper()}"))

            # Also include subtitle files for translation-only workflows
            for fmt in SUPPORTED_SUBTITLE_FORMATS:
                files.extend(input_dir.glob(f"**/*.{fmt}"))

            if not files:
                logger.error("No supported files found")
                return 1

            results = workflow.batch_process(
                files,
                output_dir,
                src_lang=cast(Any, args).src_lang,
                target_lang=cast(Any, args).target_lang,
                max_segment_length=cast(Any, args).max_segment_length,
                both=cast(Any, args).both,
                resume=cast(Any, args).resume,
                postprocess_operations=postprocess_ops,
            )

            # Summary
            successful = sum(1 for r in results.values() if r.get("status") == "completed")
            print(f"\nWorkflow completed: {successful}/{len(files)} successful")

            return 0 if successful > 0 else 1

        # Single file processing
        input_path = Path(cast(Any, args).input)
        output_path = Path(cast(Any, args).output) if cast(Any, args).output else input_path.with_suffix(".srt")

        def progress_callback(message: str, progress: float) -> None:
            percent = int(progress * 100)
            print(f"\r[{percent:3d}%] {message}", end="", flush=True)

        if is_video_file(input_path) or is_audio_file(input_path):
            # Full workflow
            result = workflow.transcribe_and_translate(
                input_path,
                output_path,
                src_lang=cast(Any, args).src_lang,
                target_lang=cast(Any, args).target_lang,
                max_segment_length=cast(Any, args).max_segment_length,
                both=cast(Any, args).both,
                resume=cast(Any, args).resume,
                progress_callback=progress_callback,
                postprocess_operations=postprocess_ops,
            )
        elif is_subtitle_file(input_path):
            # Translation-only workflow
            result = workflow.translate_existing_subtitles(
                input_path,
                output_path,
                src_lang=cast(Any, args).src_lang,
                target_lang=cast(Any, args).target_lang,
                both=cast(Any, args).both,
            )
        else:
            logger.error("Unsupported file type: %s", input_path)
            return 1

        print(f"\nWorkflow completed: {result['output_path']}")
        print(f"Total time: {result['total_time']:.2f} seconds")

        return 0

    except WorkflowError as e:
        logger.error("Workflow error: %s", e)
        return 1
    except (OSError, IOError) as e:
        logger.error("File system error: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.info("Workflow interrupted by user")
        return 130
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return 1


def main(args: Optional[List[str]] = None) -> int:  # pylint: disable=too-many-return-statements
    """Main entry point for the CLI."""
    if args is None:
        args = sys.argv[1:]

    parsed_args = None
    try:
        parser = create_parser()
        parsed_args = parser.parse_args(args)

        # Setup logging
        log_level = logging.DEBUG if cast(Any, args).verbose else logging.INFO
        setup_logging(log_level, cast(Any, args).log_file)

        # Log system information in debug mode
        if cast(Any, args).verbose:
            sys_info = get_system_info()
            logger.debug("System info: %s", sys_info)

        # Print banner
        if not cast(Any, args).verbose:
            print("SubtitleTools v1.0.0 - Complete Subtitle Workflow Tool")
            print("=" * 50)

        # Dispatch to command handlers
        if cast(Any, args).command == "transcribe":
            return handle_transcribe_command(parsed_args)
        if cast(Any, args).command == "translate":
            return handle_translate_command(parsed_args)
        if cast(Any, args).command == "encode":
            return handle_encode_command(parsed_args)
        if cast(Any, args).command == "workflow":
            return handle_workflow_command(parsed_args)

        parser.print_help()
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except argparse.ArgumentError:
        # Argument parsing errors
        return 2
    except (OSError, IOError) as e:
        logger.error("System error: %s", e)
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        # Print traceback in verbose mode if parsed_args is available
        if parsed_args and getattr(parsed_args, 'verbose', False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
