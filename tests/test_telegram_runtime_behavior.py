from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from remember_me.telegram_bot import TelegramBot


def test_compute_coalesce_delay_prefers_sentence_end() -> None:
    bot = TelegramBot("token")
    bot._coalesce_first_at = time.monotonic()

    first_delay = bot._compute_coalesce_delay("好的。", is_first=True)
    assert first_delay == bot.COALESCE_ENDING_DEBOUNCE

    normal_delay = bot._compute_coalesce_delay("还没说完", is_first=False)
    ending_delay = bot._compute_coalesce_delay("那就这样吧。", is_first=False)
    assert ending_delay <= normal_delay


def test_shutdown_background_tasks_cancels_pending() -> None:
    bot = TelegramBot("token")

    async def _run():
        t1 = bot._track_task(asyncio.create_task(asyncio.sleep(60)), "x")
        bot._coalesce_timer = bot._track_task(asyncio.create_task(asyncio.sleep(60)), "coalesce")
        await bot._shutdown_background_tasks()
        assert t1.cancelled()
        assert bot._coalesce_timer is None
        assert not bot._bg_tasks

    asyncio.run(_run())


def test_deliver_messages_uses_phase_delay_sampling() -> None:
    phase_calls: list[str] = []

    class _FakeEngine:
        reply_delay_factor = 1.0

        def sample_inter_message_delay(self, phase: str):
            phase_calls.append(str(phase))
            return 0.0

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

        async def send_message(self, chat_id: int, text: str):
            return None

        async def send_photo(self, chat_id: int, photo):
            return None

    bot = TelegramBot("token")
    bot._controller = SimpleNamespace(_engine=_FakeEngine())

    asyncio.run(
        bot._deliver_messages(
            _FakeBot(),
            chat_id=123,
            msgs=["第一条", "第二条"],
            first_delay_phase="followup",
        )
    )

    assert phase_calls[0] == "followup"
    assert "burst" in phase_calls


def test_flush_coalesce_skip_when_controller_closed() -> None:
    calls = {"typing": 0, "send": 0}

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            calls["typing"] += 1

        async def send_message(self, chat_id: int, text: str):
            calls["send"] += 1

    bot = TelegramBot("token")
    bot._app = SimpleNamespace(bot=_FakeBot())
    bot._controller = None
    bot._chat_id = None
    bot._coalesce_buffer = ["你好"]

    asyncio.run(bot._flush_coalesce(chat_id=123, delay=0))
    assert calls["typing"] == 0
    assert calls["send"] == 0


def test_handle_photo_when_controller_dropped_replies_session_ended() -> None:
    class _FakeFile:
        async def download_as_bytearray(self):
            return bytearray(b"img")

    class _FakeContextBot:
        async def get_file(self, file_id: str):
            return _FakeFile()

    class _FakeAppBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

    class _FakeMessage:
        def __init__(self):
            self.photo = [SimpleNamespace(file_id="f1")]
            self.caption = ""
            self.replies: list[str] = []

        async def reply_text(self, text: str):
            self.replies.append(text)

    bot = TelegramBot("token")
    bot._app = SimpleNamespace(bot=_FakeAppBot())
    bot._controller = None
    bot._chat_id = None

    async def _ensure_session(chat_id: int, no_greet: bool = False) -> bool:
        return True

    bot._ensure_session = _ensure_session  # type: ignore[assignment]

    message = _FakeMessage()
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=1),
        effective_chat=SimpleNamespace(id=123),
        message=message,
    )
    context = SimpleNamespace(bot=_FakeContextBot())

    asyncio.run(bot._handle_photo(update, context))
    assert message.replies
    assert "当前会话已结束" in message.replies[-1]
