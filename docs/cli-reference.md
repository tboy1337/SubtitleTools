# CLI Reference

Complete reference for the `subtitletools` command-line interface.

## Global options

| Option | Description |
|--------|-------------|
| `--version` | Print version and exit |
| `--verbose`, `-v` | Debug logging |
| `--log-file PATH` | Also write logs to a file |
| `--strict` | Fail if optional tools (e.g. FFprobe) are missing |

### Environment variables

These override defaults when the matching CLI flag is not set:

| Variable | Effect |
|----------|--------|
| `SUBTITLETOOLS_GOOGLE_API_KEY` | Fallback for `--api-key` |
| `SUBTITLETOOLS_LOG_FILE` | Fallback for `--log-file` |
| `SUBTITLETOOLS_WHISPER_MODEL` | Fallback for `--model` when it is still the default (`small`) |

## `transcribe`

Generate subtitles from video or audio using Whisper.

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `input` | — | Input file or directory (with `--batch`) |
| `--output`, `-o` | `<input>.srt` | Output path |
| `--model`, `-m` | `small` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `--language` | auto | Source language code |
| `--max-segment-length` | none | Max characters per subtitle cue |
| `--batch` | off | Process all matching files in a directory |
| `--extensions` | `mp4,mkv,avi,mov,webm` | Video extensions for batch mode |

## `translate`

Translate an existing subtitle file.

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `input` | — | Input subtitle file |
| `output` | — | Output subtitle file |
| `--src-lang` | `en` | Source language |
| `--target-lang` | `zh-CN` | Target language |
| `--service` | `google` | `google` or `google_cloud` |
| `--api-key` | env / none | Google Cloud API key |
| `--both` / `--only-translation` | both | Include original text or translated only |
| `--encoding` | `UTF-8` | File encoding |
| `--batch` | off | Batch directory mode |
| `--pattern` | `*` | Glob for batch mode |

## `encode`

Convert subtitle file encoding.

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `input` | — | Input subtitle file |
| `--output`, `-o` | derived | Output path |
| `--from-encoding` | auto-detect | Source encoding |
| `--to-encoding` | `utf-8` / `utf-8-sig` | Target encoding |
| `--recommended` | off | Use language-based encoding recommendations |
| `--list-encodings` | off | List supported encodings and exit |
| `--batch` | off | Batch directory mode |
| `--pattern` | `*` | Glob for batch mode |

## `workflow`

End-to-end pipeline: transcribe (video/audio), translate, and optionally post-process.

| Argument / option | Default | Description |
|-------------------|---------|-------------|
| `input` | — | Video, audio, or subtitle file (or directory with `--batch`) |
| `--output`, `-o` | derived | Output path |
| `--model`, `-m` | `small` | Whisper model |
| `--src-lang` | `auto` | Source language (`auto` for Whisper detection) |
| `--target-lang` | `zh-CN` | Target language |
| `--service` | `google` | Translation service |
| `--api-key` | env / none | Google Cloud API key |
| `--both` / `--only-translation` | both | Dual-language or translated-only output |
| `--resume` / `--no-resume` | resume on | Checkpoint resume for **video/audio** workflows |
| `--batch` | off | Batch mode |
| `--extensions` | `mp4,mkv,avi,mov,webm` | Extensions for batch video discovery |
| `--max-segment-length` | none | Max cue length for transcription |
| `--fix-common-errors` | off | Post-process: common error fixes |
| `--remove-hi` | off | Remove hearing-impaired text |
| `--split-long-lines` | off | Split long lines |
| `--fix-punctuation` | off | Fix punctuation |
| `--ocr-fix` | off | OCR corrections |
| `--convert-to` | none | Output format: `srt`, `vtt`, `ass`, `ssa`, `sami` |

### Resume behavior

Checkpoints apply to the **video/audio** `workflow` path (transcribe + translate + post-process). They are stored under:

- Windows: `%APPDATA%\SubtitleTools\cache\checkpoints\`
- Linux/macOS: `~/.subtitletools/cache/checkpoints/`

The standalone `translate` command does not use checkpoints.

## Python API

Heavy classes are lazy-loaded from the package root:

```python
from subtitletools import (
    SubWhisperTranscriber,
    SubtitleTranslator,
    SubtitleProcessor,
    SubtitleWorkflow,
)

workflow = SubtitleWorkflow(whisper_model="small", translation_service="google")
```

See source modules under `src/subtitletools/core/` for method signatures.
