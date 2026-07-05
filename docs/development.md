# Development

## Setup

```bash
pip install -e ".[dev]"
```

## Quality checks

```bash
python scripts/verify.py
python scripts/verify.py --fix   # apply whitespace/import fixes first
```

`verify.py` runs autopep8, isort, black, mypy, pylint, bandit, pip-audit, and pytest.

## Tests

```bash
pytest
pytest -m unit
pytest -n auto    # parallel (requires pytest-xdist)
```

Minimum coverage threshold is **90%**, configured in `.coveragerc`.

## Releasing

1. Update `version` in `pyproject.toml`
2. Run `python scripts/verify.py`
3. Commit and push to `main`

When the `version` in `pyproject.toml` changes on `main`, CI automatically builds wheels, the Windows executable, creates a GitHub Release tagged `v<version>`, and publishes to PyPI. Pull requests and pushes that do not change the version run quality checks only.

The Windows executable is built with PyInstaller using [`SubtitleTools.spec`](../SubtitleTools.spec) (large download due to PyTorch/Whisper).

## Project layout

- `src/subtitletools/` — package source
- `tests/` — pytest suite
- `scripts/` — verify and PyInstaller helpers
- `SubtitleTools.spec` — PyInstaller specification
