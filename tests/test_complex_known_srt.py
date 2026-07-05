"""Tests for complex_known.srt — unit checks plus live Google integration.

Run live Google tests (requires Node.js for pyexecjs):

    pytest -m integration tests/test_complex_known_srt.py -v

The golden reference ``complex_known_google_es.srt`` was captured from a real
``google`` service translation after the per-cue translation fix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, cast

import pytest
import srt

from subtitletools.cli import main as cli_main
from subtitletools.core.subtitle import SubtitleProcessor
from subtitletools.core.workflow import SubtitleWorkflow
from subtitletools.utils.common import check_execjs_runtime

FIXTURE_DIR = Path(__file__).parent / "test_data"
COMPLEX_SRT = FIXTURE_DIR / "complex_known.srt"
TRANSLATIONS_JSON = FIXTURE_DIR / "complex_known_translations.json"
GOOGLE_GOLDEN_SRT = FIXTURE_DIR / "complex_known_google_es.srt"

pytestmark_integration = pytest.mark.integration


def _requires_google_web() -> None:
    if not check_execjs_runtime():
        pytest.skip(
            "Google web translation requires a JavaScript runtime (install Node.js)"
        )
    if not GOOGLE_GOLDEN_SRT.is_file():
        pytest.skip("Missing golden file tests/test_data/complex_known_google_es.srt")


def _load_translation_spec() -> dict[str, Any]:
    with TRANSLATIONS_JSON.open(encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_srt_text(content: str) -> str:
    return content.replace("\r\n", "\n").strip() + "\n"


def _timing_signature(cues: list[srt.Subtitle]) -> list[tuple[int, str, str]]:
    return [(c.index, str(c.start), str(c.end)) for c in cues]


def _assert_matches_google_golden(output_path: Path) -> None:
    processor = SubtitleProcessor()
    expected_cues = cast(list[srt.Subtitle], processor.parse_file(GOOGLE_GOLDEN_SRT))
    actual_cues = cast(list[srt.Subtitle], processor.parse_file(output_path))

    assert len(actual_cues) == len(expected_cues) == 28
    assert _timing_signature(actual_cues) == _timing_signature(expected_cues)
    assert [c.content for c in actual_cues] == [c.content for c in expected_cues]
    assert _normalize_srt_text(output_path.read_text(encoding="utf-8")) == (
        _normalize_srt_text(GOOGLE_GOLDEN_SRT.read_text(encoding="utf-8"))
    )


class TestComplexKnownSrtFixture:
    """Validate the fixture before any pipeline tests."""

    def test_fixture_exists_and_has_28_cues(self) -> None:
        assert COMPLEX_SRT.is_file()
        processor = SubtitleProcessor()
        assert len(processor.parse_file(COMPLEX_SRT)) == 28

    def test_fixture_includes_complex_patterns(self) -> None:
        processor = SubtitleProcessor()
        contents = processor.extract_text(processor.parse_file(COMPLEX_SRT))
        joined = "\n".join(contents)
        assert "multi-line" in joined
        assert "https://example.com" in joined
        assert "日本語" in joined
        assert "<i>Speaker A:</i>" in joined


class TestComplexKnownRoundTrip:
    """Parse/save and encode paths must preserve content exactly."""

    def test_parse_save_round_trip_preserves_all_cues(self, tmp_path: Path) -> None:
        processor = SubtitleProcessor()
        output_path = tmp_path / "roundtrip.srt"
        originals = processor.parse_file(COMPLEX_SRT)
        processor.save_file(originals, output_path)
        roundtripped = processor.parse_file(output_path)

        def cue_signature(cues: list[srt.Subtitle]) -> list[tuple[int, str, str, str]]:
            return [(c.index, str(c.start), str(c.end), c.content) for c in cues]

        assert cue_signature(cast(list[srt.Subtitle], originals)) == cue_signature(
            cast(list[srt.Subtitle], roundtripped)
        )

    def test_cli_encode_utf8_round_trip(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "encoded"
        result = cli_main(
            [
                "encode",
                str(COMPLEX_SRT),
                "--output-dir",
                str(output_dir),
                "--to-encoding",
                "utf-8",
            ]
        )
        assert result == 0
        encoded_file = output_dir / "complex_known-utf-8.srt"
        processor = SubtitleProcessor()
        original_cues = processor.parse_file(COMPLEX_SRT)
        encoded_cues = processor.parse_file(encoded_file)
        assert [c.content for c in encoded_cues] == [c.content for c in original_cues]


class TestComplexKnownReconstructionUnit:
    """Reconstruction algorithm check using static translation pairs (no network)."""

    def test_reconstruct_subtitles_one_to_one_with_static_pairs(
        self, translation_spec: dict[str, Any]
    ) -> None:
        processor = SubtitleProcessor()
        originals = processor.parse_file(COMPLEX_SRT)
        translations: List[str] = translation_spec["translations"]
        reconstructed = processor.reconstruct_subtitles(
            originals, translations, space=True, both=False
        )
        assert len(reconstructed) == 28
        assert [c.content for c in reconstructed] == translations


@pytest.fixture(name="translation_spec")
def fixture_translation_spec() -> dict[str, Any]:
    return _load_translation_spec()


@pytest.mark.integration
@pytest.mark.network
@pytest.mark.timeout(600)
class TestComplexKnownGoogleLive:
    """Live end-to-end tests using the real Google web translation service."""

    def setup_method(self) -> None:
        _requires_google_web()

    def test_cli_translate_matches_google_golden(self, tmp_path: Path) -> None:
        output_path = tmp_path / "cli_google_es.srt"
        result = cli_main(
            [
                "translate",
                str(COMPLEX_SRT),
                str(output_path),
                "--src-lang",
                "en",
                "--target-lang",
                "es",
                "--only-translation",
                "--service",
                "google",
            ]
        )
        assert result == 0
        _assert_matches_google_golden(output_path)

    def test_workflow_translate_matches_google_golden(self, tmp_path: Path) -> None:
        output_path = tmp_path / "workflow_google_es.srt"
        workflow = SubtitleWorkflow(translation_service="google")
        result = workflow.translate_existing_subtitles(
            COMPLEX_SRT,
            output_path,
            src_lang="en",
            target_lang="es",
            both=False,
        )
        assert result["status"] == "completed"
        assert result["translated_segments"] == 28
        _assert_matches_google_golden(output_path)

    def test_workflow_cli_translate_produce_identical_output(
        self, tmp_path: Path
    ) -> None:
        cli_path = tmp_path / "cli.srt"
        workflow_path = tmp_path / "workflow.srt"

        assert (
            cli_main(
                [
                    "translate",
                    str(COMPLEX_SRT),
                    str(cli_path),
                    "--src-lang",
                    "en",
                    "--target-lang",
                    "es",
                    "--only-translation",
                    "--service",
                    "google",
                ]
            )
            == 0
        )

        workflow = SubtitleWorkflow(translation_service="google")
        workflow.translate_existing_subtitles(
            COMPLEX_SRT,
            workflow_path,
            src_lang="en",
            target_lang="es",
            both=False,
        )

        assert _normalize_srt_text(cli_path.read_text(encoding="utf-8")) == (
            _normalize_srt_text(workflow_path.read_text(encoding="utf-8"))
        )

    def test_multiline_cue_translated_as_single_block(self, tmp_path: Path) -> None:
        output_path = tmp_path / "multiline_check.srt"
        workflow = SubtitleWorkflow(translation_service="google")
        workflow.translate_existing_subtitles(
            COMPLEX_SRT,
            output_path,
            src_lang="en",
            target_lang="es",
            both=False,
        )
        cues = cast(
            list[srt.Subtitle],
            SubtitleProcessor().parse_file(output_path),
        )
        cue_three = cues[2]
        assert cue_three.content.count("\n") == 2
        assert "Línea uno" in cue_three.content
        assert "línea tres" in cue_three.content.lower()

    def test_no_source_english_left_in_simple_cues(self, tmp_path: Path) -> None:
        """Cue 2 must be translated — catches newline batching misalignment."""
        output_path = tmp_path / "alignment_check.srt"
        workflow = SubtitleWorkflow(translation_service="google")
        workflow.translate_existing_subtitles(
            COMPLEX_SRT,
            output_path,
            src_lang="en",
            target_lang="es",
            both=False,
        )
        cues = cast(
            list[srt.Subtitle],
            SubtitleProcessor().parse_file(output_path),
        )
        assert "intentionally complex" not in cues[1].content.lower()
        assert "intencionalmente complejo" in cues[1].content.lower()
