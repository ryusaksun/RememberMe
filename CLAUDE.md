# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                    # Install dependencies
uv run remember-me import-chat <file> --format <text|json|wechat> --target "名字"
uv run remember-me import-netease          # Interactive netease cloud music import
uv run remember-me chat <name>             # Start conversation (CLI)
uv run remember-me-gui                     # Start NiceGUI web interface
uv run remember-me-tg                      # Start Telegram Bot (single persona)
uv run remember-me list-personas           # Show all personas
```

No tests exist yet. `tests/` directory is empty.

## Architecture

Data flows through four phases: **Import → Analyze → Index → Chat**, with three delivery frontends (CLI, GUI, Telegram).

### Import Phase

All importers live in `src/remember_me/importers/` and implement:
```python
def parse(file_path, target_name, user_name=None) -> ChatHistory
```

Four formats: `plain_text`, `json_parser`, `wechat`, `netease`. The netease importer fetches data online via `netease_api.py` (self-implemented weapi AES+RSA encryption protocol) rather than parsing a local file. To add a new importer, implement `parse()` returning `ChatHistory` and register it in `cli.py`'s `IMPORTERS` dict.

`ChatHistory` and `ChatMessage` (in `importers/base.py`) are the universal data model. `ChatHistory.as_burst_dialogues()` extracts multi-message reply segments used for few-shot examples.

### Analysis Phase

`analyzer/persona.py`'s `analyze()` takes `ChatHistory` → returns `Persona` dataclass with 20+ features: linguistic stats, catchphrases, emoji/tone habits, burst patterns (avg length, probability distribution), active hours, greeting/farewell patterns, topic keywords, and real burst dialogue examples sampled from history.

Key: `burst_examples` (real multi-message conversations) are prioritized over `example_dialogues` (single-turn pairs) in prompt construction.

### Memory System (Three Tiers)

1. **Long-term RAG** (`memory/store.py`): ChromaDB with persistent local storage. Messages chunked into sliding windows of 5, vectorized with all-MiniLM-L6-v2, cosine similarity. Collection names are MD5-hashed (ChromaDB requires ASCII).

2. **Mid-term Scratchpad** (`memory/scratchpad.py`): LLM-maintained conversation notes updated every ~6 turns via background thread. Tracks open topics, facts, emotional tone. Uses `MODEL_LIGHT` to extract structured JSON (topics, facts, open_threads, emotion valence/arousal).

3. **Memory Governance** (`memory/governance.py`): Enforces "imported history is the single source of truth" — core persona facts from import are read-only, runtime session memories can accumulate but never override core. Builds layered prompt blocks: core facts > RAG retrieval > session context.

### Emotion System

`engine/emotion.py` implements a valence/arousal 2D continuous model. Keyword rules do instant micro-adjustments; Scratchpad LLM outputs sync deeper shifts. Emotion drives: reply delay factor, proactive cooldown factor, burst message count range, sticker probability, temperature delta.

### Chat Phase

`engine/chat.py`'s `ChatEngine` ties everything together:

1. **`_build_system_prompt(persona)`** — Identity + style + burst format rules + real dialogue examples + behavioral rules.
2. **`_build_system(user_input)`** — Layers on: current time, session phase guide, governance core facts, knowledge items, RAG results, session context, scratchpad, emotion state, open thread priorities, burst hint.
3. **`send_multi(user_input) -> list[str]`** — Primary method. Calls Gemini, splits response on `|||` separator into multiple messages. Applies reasoning-leak sanitization, human noise (typos/hesitations), sticker attachment.

The `|||` separator bridges single LLM responses and multi-message chat simulation. Conversation history auto-trims to 40 entries.

### Controller Layer

`controller.py`'s `ChatController` is the async orchestration layer shared by GUI and Telegram:
- Session phase state machine: `warmup → normal → deep_talk → cooldown → ending`
- Proactive message loop with two silence strategies: post-reply check-in vs post-proactive follow-up
- Pending event tracker (`engine/pending_events.py`): extracts follow-up-worthy events (trips, exams, illness) from conversation, triggers contextual check-ins later
- Message coalescing in Telegram bot: debounces rapid user messages before sending to LLM

### Proactive Features

- **TopicStarter** (`engine/topic_starter.py`): Uses Brave Search to find trending content matching persona's `topic_interests`, generates contextual opening messages or follow-ups.
- **KnowledgeFetcher/KnowledgeStore** (`knowledge/`): Daily knowledge base update — fetches articles via Brave Search + trafilatura extraction, summarizes with `MODEL_LIGHT`, stores in ChromaDB for RAG during chat.
- **Daily Scheduler** (Telegram only): Plans 1-2 proactive message times per day based on persona's `active_hours`.

### Delivery Frontends

- **CLI** (`cli.py`): Threading-based, with input worker + proactive worker threads. Rich console output.
- **GUI** (`gui/`): NiceGUI web app with cyberpunk terminal theme. Routes: `/` (home), `/chat/{name}`, `/import`. Uses `ChatController` async.
- **Telegram** (`telegram_bot.py`): Single-persona mode. Message coalescing (debounce rapid messages), photo/image support, `/note` command for manual memory notes, daily proactive scheduler.

## LLM Models

Defined in `models.py` (single source of truth):
- `MODEL_MAIN` = `gemini-3.1-pro-preview` — primary chat generation
- `MODEL_LIGHT` = `gemini-3-flash-preview` — scratchpad updates, event extraction, knowledge summaries, trivial replies

## Data Storage

All user data in `data/` (gitignored):
- `data/history/{name}.json` — Full ChatHistory (serialized, reloadable via `ChatHistory.load()`)
- `data/profiles/{name}.json` — Persona analysis results
- `data/chroma/{name}/` — ChromaDB vector store
- `data/images/{name}/` — Downloaded chat images (netease)
- `data/sessions/{name}.json` — Conversation session state (history + scratchpad + emotion)
- `data/stickers/{name}.json` — Classified sticker library
- `data/knowledge/{name}/` — Daily knowledge base articles
- `data/governance/{name}/` — Memory governance records (core + session)

## Environment Variables

- `GEMINI_API_KEY` — Required for chat
- `NETEASE_COOKIE` — MUSIC_U cookie value for netease import
- `BRAVE_API_KEY` — For topic starter / knowledge fetcher (optional, degrades gracefully)
- `TELEGRAM_BOT_TOKEN` — Required for Telegram bot
- `TELEGRAM_ALLOWED_USERS` — Comma-separated user IDs (optional whitelist)
- `PERSONA_NAME` — Telegram bot persona name (default: `阴暗扭曲爬行_-_-`)
- `TZ` — Timezone (default: `Asia/Shanghai`)

All read from `.env` via python-dotenv.
