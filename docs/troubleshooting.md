# Troubleshooting

## FFmpeg not found

**Symptom:** `FFmpeg is required but was not found`

**Fix:** Install FFmpeg and ensure `ffmpeg` is on PATH. See [installation.md](installation.md).

## Translation fails immediately with `google` service

**Symptom:** Errors mentioning `pyexecjs` or token generation

**Fix:** Install Node.js, or switch to `--service google_cloud --api-key KEY`.

## `google_cloud` requires API key

**Symptom:** `google_cloud service requires an API key`

**Fix:** Pass `--api-key` or use `--service google` with Node.js installed.

## Workflow batch processes directory twice

This was fixed in 1.0.2+. Update to the latest release.

## Resume does not continue

Checkpoints use a stable ID derived from input path, model, languages, and post-processing options. Changing `--target-lang`, `--model`, or post-process flags starts a new checkpoint. Use the same options to resume.

Resume applies to **video/audio** `workflow` runs only, not standalone `translate`.

### Corrupt checkpoint

**Symptom:** `Corrupt checkpoint: missing temp_srt_path`

**Fix:** Delete checkpoint files under:

- Windows: `%APPDATA%\SubtitleTools\cache\checkpoints\`
- Linux/macOS: `~/.subtitletools/cache/checkpoints/`

Then rerun with `--no-resume` or allow a fresh checkpoint.

## Misaligned translated subtitles

If subtitle cues do not match translations, ensure you are on the latest version (per-cue translation mapping). Re-run translation on the source SRT if upgrading from an older build.

## Out of memory during transcription

Whisper loads the full audio into memory. Use a smaller model (`--model tiny` or `base`) or shorter source files.

## PyPI publish / release did not run

Releases run automatically when `version` in `pyproject.toml` changes on a push to `main`. If build or publish jobs were skipped, confirm the version field actually changed in that commit. Pushes that only change other files run quality checks only.
