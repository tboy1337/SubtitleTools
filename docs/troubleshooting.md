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

Checkpoints use a stable ID derived from input path and options. Changing `--target-lang` or `--model` starts a new checkpoint. Use the same options to resume.

## Out of memory during transcription

Whisper loads the full audio into memory. Use a smaller model (`--model tiny` or `base`) or shorter source files.

## PyPI publish / version mismatch

Releases are tag-based. Bump `version` in `pyproject.toml`, commit, then:

```bash
git tag v1.0.3
git push origin v1.0.3
```

The tag must match `pyproject.toml` (without the `v` prefix).
