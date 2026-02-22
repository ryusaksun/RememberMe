# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                    # Install dependencies
uv run remember-me import-chat <file> --format <text|json|wechat> --target "名字"
uv run remember-me import-netease          # Interactive netease cloud music import
uv run remember-me chat <name>             # Start conversation
uv run remember-me list-personas           # Show all personas
```

No tests exist yet. `tests/` directory is empty.

## Architecture

Data flows through four phases: **Import → Analyze → Index → Chat**.

### Import Phase

All importers live in `src/remember_me/importers/` and implement:
```python
def parse(file_path, target_name, user_name=None) -> ChatHistory
```

Four formats: `plain_text`, `json_parser`, `wechat`, `netease`. The netease importer is different — it fetches data online via `netease_api.py` (self-implemented weapi AES+RSA encryption protocol) rather than parsing a local file. To add a new importer, implement `parse()` returning `ChatHistory` and register it in `cli.py`'s `IMPORTERS` dict.

`ChatHistory` and `ChatMessage` (in `importers/base.py`) are the universal data model. `ChatHistory.as_burst_dialogues()` extracts multi-message reply segments used for few-shot examples.

### Analysis Phase

`analyzer/persona.py`'s `analyze()` takes `ChatHistory` → returns `Persona` dataclass with 20+ features: linguistic stats, catchphrases, emoji/tone habits, burst patterns (avg length, probability distribution), active hours, greeting/farewell patterns, topic keywords, and real burst dialogue examples sampled from history.

Key: `burst_examples` (real multi-message conversations) are prioritized over `example_dialogues` (single-turn pairs) in prompt construction.

### Memory/RAG Phase

`memory/store.py` uses ChromaDB with persistent local storage. Messages are chunked into sliding windows of 5, vectorized with all-MiniLM-L6-v2, and stored with cosine similarity. Collection names are MD5-hashed (ChromaDB requires ASCII names).

### Chat Phase

`engine/chat.py`'s `ChatEngine` ties everything together:

1. **`_build_system_prompt(persona)`** — Constructs identity + style description + burst format rules + real dialogue examples + behavioral rules. Written as natural character description, not a data report.
2. **`_build_system(user_input)`** — Appends RAG context (top-5 relevant history fragments) to system prompt per query.
3. **`send_multi(user_input) -> list[str]`** — Primary method. Calls Gemini, splits response on `|||` separator into multiple messages simulating burst replies.

The `|||` separator is the bridge between single LLM responses and multi-message chat simulation. CLI displays each message with random 0.4-1.2s delays.

Conversation history is maintained in `_history` (Gemini Content objects), auto-trimmed to 40 entries.

## Data Storage

All user data in `data/` (gitignored):
- `data/history/{name}.json` — Full ChatHistory (serialized, reloadable via `ChatHistory.load()`)
- `data/profiles/{name}.json` — Persona analysis results
- `data/chroma/{name}/` — ChromaDB vector store
- `data/images/{name}/` — Downloaded chat images (netease)

## Environment Variables

- `GEMINI_API_KEY` — Required for chat
- `NETEASE_COOKIE` — MUSIC_U cookie value for netease import

Both read from `.env` via python-dotenv.

## LLM Model

Currently uses `gemini-3.1-pro-preview` (hardcoded in `engine/chat.py`, two occurrences).
