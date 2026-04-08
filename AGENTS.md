# AGENTS.md

- ALWAYS USE PARALLEL TOOLS WHEN APPLICABLE.
- The default branch in this repo is `main`.
- Package manager is `uv`. Always use `uv sync` to install dependencies and `uv run` to execute modules.
- Python version is **3.11+**. Do not use features unavailable in 3.11.
- Prefer automation: execute requested actions without confirmation unless blocked by missing info or safety/irreversibility.
- To understand the full project structure, read files inside `.docs/`.
- Run examples via `uv run vexor.examples.<module_name>` (e.g. `uv run vexor.examples.ingest_tabular`).

## Commands

```bash
# Install all dependencies
uv sync

# Install with optional group
uv sync --extra dev

# Run an example
uv run vexor.examples.ingest_tabular
uv run vexor.examples.search_dense

# Lint
ruff check vexor/

# Type check
mypy vexor/
```

## Code Style

- **Line length**: 120 characters max.
- **Type hints**: Strict. Use `Optional[X]` explicitly.
- **Naming**: PascalCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants.
- All code must pass `ruff check` before commit.
- All code must pass `mypy vexor`.
- Docstrings required for modules, classes, and public functions.
