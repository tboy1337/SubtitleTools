"""Integration tests for SubtitleTools (optional external dependencies)."""

from __future__ import annotations

import shutil

import pytest

from subtitletools.utils.audio import find_ffmpeg, find_ffprobe


@pytest.mark.integration
def test_find_ffmpeg_does_not_raise() -> None:
    """Smoke test: FFmpeg discovery completes without error."""
    result = find_ffmpeg()
    assert result is None or isinstance(result, str)


@pytest.mark.integration
def test_find_ffprobe_does_not_raise() -> None:
    """Smoke test: FFprobe discovery completes without error."""
    result = find_ffprobe()
    assert result is None or isinstance(result, str)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_ffmpeg_on_path_when_available() -> None:
    """When ffmpeg is on PATH, discovery should find it."""
    assert find_ffmpeg() is not None
