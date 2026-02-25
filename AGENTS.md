# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `src/remember_me/`:
- `cli.py`: Click entrypoint for all user commands.
- `importers/`: parsers for text/JSON/WeChat/NetEase inputs.
- `analyzer/`: persona extraction + relationship extraction (`relationship_extractor.py`).
- `memory/`: retrieval storage + governance + relationship store.
- `engine/`: chat generation, emotion, proactive topic logic.
- `controller.py`: async orchestration shared by CLI/GUI/Telegram.
- `telegram_bot.py`: production Telegram runtime (single persona mode).

Runtime artifacts are stored in `data/` and are gitignored, including:
- `history/`, `profiles/`, `chroma/`, `images/`
- `memories/` (core/session governance records)
- `relationships/` (relationship facts with candidate/confirmed/rejected)
- `sessions/`, `knowledge/`, `pending_events/`

Use `examples/sample_chat.txt` for local import experiments. Keep tests in `tests/` and extend existing suites when adding features.

## Build, Test, and Development Commands
- `uv sync`: install dependencies and create/update the local environment.
- `uv run remember-me import-chat examples/sample_chat.txt --format text --target "小明"`: import a sample chat and build persona/memory data.
- `uv run remember-me chat 小明`: start an interactive chat session.
- `uv run remember-me list-personas`: list generated personas.
- `uv run remember-me import-netease`: import NetEase private-message history (requires cookie).
- `uv run remember-me-tg`: run Telegram bot locally.
- `uv build`: build distributable packages from `pyproject.toml`.
- `uv run pytest -q`: run tests (add `pytest` as a dev dependency if not installed locally).
- `uv run pytest -q tests/test_relationship_memory.py`: focused regression for relationship-memory pipeline.

## Coding Style & Naming Conventions
Target Python is `>=3.11`. Follow existing style:
- 4-space indentation, UTF-8 files.
- `snake_case` for modules/functions/variables, `PascalCase` for classes/dataclasses.
- Keep type hints and use explicit, small functions.
- Preserve clear Chinese-facing CLI copy in user messages.

No enforced formatter/linter is configured in-repo; keep changes consistent with surrounding code and avoid unrelated refactors.

## Relationship Memory Rules
- Imported chat history defines the persona core and must not be overwritten.
- Runtime memory can accumulate, but conflicts with imported core should be marked/rejected by governance.
- Relationship facts should prefer structured `meta` when available (for example boundary topic/cooldown, shared event slots, addressing contexts).
- Prompt priority order should remain stable: `core -> relationship -> RAG/history -> session -> conflict -> scratchpad/emotion`.

## Testing Guidelines
Use `pytest` with files named `tests/test_*.py`. Prioritize:
- parser correctness and persona-analysis edge cases,
- relationship extraction/store/governance flows,
- prompt order and runtime race regressions (controller/telegram),
- memory retrieval behavior and CLI command flows.

Prefer deterministic fixtures and assert observable behavior instead of private implementation details.

## Commit & Pull Request Guidelines
History follows Conventional Commit prefixes such as `feat:` and `docs:`. Continue using concise, scoped subjects (example: `feat: improve proactive topic cooldown`).

For each PR, include:
- what changed and why,
- how to verify (exact commands),
- config/env changes (`GEMINI_API_KEY`, `NETEASE_COOKIE`) if applicable,
- terminal screenshots or sample output when behavior changes in CLI UX.

## VPS Deployment (Vultr)
Current production host runs:
- project path: `/opt/remember-me`
- service: `remember-me-tg.service`
- service exec uses `/root/.local/bin/uv`

Recommended deploy sequence:
1. `git push origin main`
2. `ssh vultr-tokyo 'cd /opt/remember-me && git pull --ff-only origin main'`
3. `ssh vultr-tokyo 'cd /opt/remember-me && /root/.local/bin/uv sync'`
4. `ssh vultr-tokyo 'systemctl restart remember-me-tg.service'`
5. `ssh vultr-tokyo 'systemctl is-active remember-me-tg.service && journalctl -u remember-me-tg.service --since "2 min ago" --no-pager -n 50'`

## Security & Configuration Tips
Never commit `.env`, cookies, or raw chat exports. Treat `data/` as local/private user data. Validate any new importer against malformed input and avoid logging secrets in exceptions or debug output.
