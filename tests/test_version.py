"""Tests for package version resolution."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from subtitletools._version import _fallback_version, _pyproject_path, get_version


class TestVersion:
    """Tests for get_version and fallback parsing."""

    def test_get_version_prefers_pyproject_when_present(self) -> None:
        """Test pyproject.toml is preferred when developing from source."""
        version_value = get_version()
        assert version_value
        assert version_value == _fallback_version()
        assert version_value != "unknown"

    def test_pyproject_path_points_at_repo_root(self) -> None:
        """Test pyproject path resolves beside the repository root."""
        assert _pyproject_path().name == "pyproject.toml"
        assert _pyproject_path().is_file()

    def test_fallback_reads_pyproject(self, mocker: MockerFixture) -> None:
        """Test fallback parses pyproject.toml when metadata is missing."""
        mocker.patch(
            "subtitletools._version.version",
            side_effect=PackageNotFoundError("subtitletools"),
        )
        _fallback_version.cache_clear()
        version_value = get_version()
        assert version_value
        assert version_value != "unknown"

    def test_fallback_unknown_when_pyproject_missing(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test fallback returns unknown when pyproject.toml is absent."""
        missing = tmp_path / "missing.toml"
        mocker.patch("subtitletools._version._pyproject_path", return_value=missing)
        _fallback_version.cache_clear()
        assert _fallback_version() == "unknown"

    def test_fallback_unknown_without_version_key(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test fallback returns unknown when pyproject has no version field."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'x'\n", encoding="utf-8")
        mocker.patch("subtitletools._version._pyproject_path", return_value=pyproject)
        _fallback_version.cache_clear()
        assert _fallback_version() == "unknown"

    def test_pyproject_path_uses_meipass_when_frozen(
        self, mocker: MockerFixture, tmp_path: Path
    ) -> None:
        """Test frozen executables resolve version from bundled pyproject.toml."""
        bundled = tmp_path / "pyproject.toml"
        bundled.write_text('[project]\nversion = "9.9.9"\n', encoding="utf-8")
        mocker.patch("subtitletools._version.sys.frozen", True, create=True)
        mocker.patch("subtitletools._version.sys._MEIPASS", str(tmp_path), create=True)
        _fallback_version.cache_clear()
        assert _pyproject_path() == bundled
        assert get_version() == "9.9.9"
