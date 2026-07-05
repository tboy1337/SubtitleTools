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
4. Create and push a matching tag:

```bash
git tag v1.0.3
git push origin v1.0.3
```

CI runs quality checks, builds wheels and the Windows executable, creates a GitHub Release, and publishes to PyPI.

## Project layout

- `src/subtitletools/` — package source
- `tests/` — pytest suite
- `scripts/` — verify and PyInstaller helpers
- `SubtitleTools.spec` — PyInstaller specification
