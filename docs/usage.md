# Usage

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

# Batch workflow on a directory
subtitletools workflow ./videos --batch --output ./subtitles
```

## Resume

Workflow checkpoints are stored under your application data directory. Resume is enabled by default:

```bash
subtitletools workflow video.mp4 --target-lang es
# interrupted...
subtitletools workflow video.mp4 --target-lang es --resume
```

Use `--no-resume` to start fresh.

## Logging

```bash
subtitletools --verbose workflow video.mp4
subtitletools --log-file subtitletools.log workflow video.mp4
```
