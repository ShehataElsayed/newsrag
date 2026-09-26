# Contributing

Open an issue before proposing a large change. For a small fix, include a test that fails before the fix and passes afterward. Please do not commit private reporting notes, unpublished sources, credentials or API tokens.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest ruff mypy 'pypdf>=5,<7'
ruff check src tests examples setup.py
mypy src/newsrag
pytest -q
```

Keep adapters optional and avoid hidden network calls. API additions should preserve source IDs and exact offsets. Include tests with Arabic and English input, missing dates and malformed provider output where relevant. Document backward-incompatible changes in CHANGELOG.md.
