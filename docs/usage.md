# Usage

See [CLI reference](cli-reference.md) for the full flag list and defaults.

## Commands

| Command | Purpose |
|---------|---------|
| `transcribe` | Video/audio to SRT via Whisper |
| `translate` | Translate existing subtitle files |
| `encode` | Convert subtitle encodings |
| `workflow` | End-to-end transcribe, translate, and post-process |

## Examples

```bash
# Transcribe
subtitletools transcribe video.mp4 --model medium

# Translate
subtitletools translate input.srt output.srt --src-lang en --target-lang es

# Encoding
subtitletools encode subtitle.srt --to-encoding utf-8

# Full workflow with post-processing and VTT output
subtitletools workflow video.mp4 --target-lang fr --fix-common-errors --convert-to vtt

# Subtitle-only workflow with post-processing
subtitletools workflow input.srt --target-lang es --fix-common-errors

# Batch workflow on a directory
subtitletools workflow ./videos --batch --output ./subtitles
```

## Resume

Workflow checkpoints apply to **video/audio** `workflow` runs (not the standalone `translate` command). Checkpoints are stored under your application data directory. Resume is enabled by default:

```bash
subtitletools workflow video.mp4 --target-lang es
# interrupted...
subtitletools workflow video.mp4 --target-lang es --resume
```

Use `--no-resume` to start fresh. If a checkpoint is corrupt, delete the matching file under `cache/checkpoints/` (see [troubleshooting](troubleshooting.md)).

## Environment variables

Optional fallbacks when CLI flags are omitted:

- `SUBTITLETOOLS_GOOGLE_API_KEY` — API key for `google_cloud`
- `SUBTITLETOOLS_LOG_FILE` — log file path
- `SUBTITLETOOLS_WHISPER_MODEL` — Whisper model when `--model` is default

## Logging and strict mode

```bash
subtitletools --verbose workflow video.mp4
subtitletools --log-file subtitletools.log workflow video.mp4
subtitletools --strict transcribe video.mp4   # require FFprobe
```

## Python API

```python
from subtitletools import SubtitleWorkflow

workflow = SubtitleWorkflow()
result = workflow.transcribe_and_translate("video.mp4", "output.srt", target_lang="es")
```

Classes are lazy-loaded; see [CLI reference](cli-reference.md#python-api).
