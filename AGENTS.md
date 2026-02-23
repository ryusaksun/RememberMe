# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `src/remember_me/`:
- `cli.py`: Click entrypoint for all user commands.
- `importers/`: parsers for text/JSON/WeChat/NetEase inputs.
- `analyzer/`: persona extraction from chat history.
- `memory/`: ChromaDB-based retrieval storage.
- `engine/`: chat generation and proactive topic logic.

Runtime artifacts are stored in `data/` (`history/`, `profiles/`, `chroma/`, `images/`) and are gitignored. Use `examples/sample_chat.txt` for local import experiments. Keep tests in `tests/` (currently minimal/empty, so new features should add coverage).

## Build, Test, and Development Commands
- `uv sync`: install dependencies and create/update the local environment.
- `uv run remember-me import-chat examples/sample_chat.txt --format text --target "小明"`: import a sample chat and build persona/memory data.
- `uv run remember-me chat 小明`: start an interactive chat session.
- `uv run remember-me list-personas`: list generated personas.
- `uv run remember-me import-netease`: import NetEase private-message history (requires cookie).
- `uv build`: build distributable packages from `pyproject.toml`.
- `uv run pytest -q`: run tests (add `pytest` as a dev dependency if not installed locally).

## Coding Style & Naming Conventions
Target Python is `>=3.11`. Follow existing style:
- 4-space indentation, UTF-8 files.
- `snake_case` for modules/functions/variables, `PascalCase` for classes/dataclasses.
- Keep type hints and use explicit, small functions.
- Preserve clear Chinese-facing CLI copy in user messages.

No enforced formatter/linter is configured in-repo; keep changes consistent with surrounding code and avoid unrelated refactors.

## Testing Guidelines
Use `pytest` with files named `tests/test_*.py`. Prioritize parser correctness, persona-analysis edge cases, memory retrieval behavior, and CLI command flows. Prefer deterministic fixtures (for example, based on `examples/sample_chat.txt`) and assert observable outputs rather than implementation details.

## Commit & Pull Request Guidelines
History follows Conventional Commit prefixes such as `feat:` and `docs:`. Continue using concise, scoped subjects (example: `feat: improve proactive topic cooldown`).

For each PR, include:
- what changed and why,
- how to verify (exact commands),
- config/env changes (`GEMINI_API_KEY`, `NETEASE_COOKIE`) if applicable,
- terminal screenshots or sample output when behavior changes in CLI UX.

## Security & Configuration Tips
Never commit `.env`, cookies, or raw chat exports. Treat `data/` as local/private user data. Validate any new importer against malformed input and avoid logging secrets in exceptions or debug output.
