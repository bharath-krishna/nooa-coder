# nooa-coder

## Quickstart

```bash
python -m main
# or
python main.py
```

## Commands

- `python -m main` — runs the hello world entrypoint
- `python -m pytest` — runs tests (if any exist)
- `ruff check .` — lint
- `ruff format .` — format

## Project structure

- `main.py` — top-level entrypoint
- `pyproject.toml` — project config (Poetry/pip-based, no dependencies yet)
- `README.md` — project readme (currently empty)
- `.venv/` — virtual environment (activate with `source .venv/bin/activate`)

## Notes

- No database, external services, or complex build steps required
- This is a minimal template project; add tests, linting, and dependencies as needed