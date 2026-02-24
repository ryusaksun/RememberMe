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


def test_note_rel_list_confirm_reject_flow() -> None:
    class _Fact:
        def __init__(self, fid: str, status: str, content: str, fact_type: str = "shared_event"):
            self.id = fid
            self.status = status
            self.content = content
            self.type = fact_type
            self.confidence = 0.86
            self.evidence = ["上次我们一起看电影"]

    class _Store:
        def __init__(self):
            self.rows = [
                _Fact("rel_1", "candidate", "经常引用共同经历（上次/那次/还记得）"),
                _Fact("rel_2", "confirmed", "常用称呼偏好：宝宝", fact_type="addressing"),
            ]
            self.confirmed: list[str] = []
            self.rejected: list[tuple[str, str]] = []

        def list_facts(self, **kwargs):
            statuses = kwargs.get("statuses")
            if statuses:
                return [x for x in self.rows if x.status in statuses]
            return list(self.rows)

        def confirm_fact(self, ref):
            self.confirmed.append(str(ref))
            return True

        def reject_fact(self, ref, reason: str = "manual_reject"):
            self.rejected.append((str(ref), reason))
            return True

    class _Msg:
        def __init__(self, text: str):
            self.text = text
            self.replies: list[str] = []

        async def reply_text(self, text: str):
            self.replies.append(text)

    store = _Store()
    bot = TelegramBot("token")
    bot._get_relationship_store = lambda: store  # type: ignore[assignment]

    async def _run():
        msg_list = _Msg("/note rel list")
        upd_list = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_list,
        )
        await bot._cmd_note(upd_list, SimpleNamespace())
        assert msg_list.replies and "共同经历" in msg_list.replies[-1]

        msg_confirm = _Msg("/note rel confirm 1")
        upd_confirm = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_confirm,
        )
        await bot._cmd_note(upd_confirm, SimpleNamespace())
        assert store.confirmed == ["rel_1"]
        assert "已确认" in msg_confirm.replies[-1]

        msg_reject = _Msg("/note rel reject 2 不符合当前关系")
        upd_reject = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_reject,
        )
        await bot._cmd_note(upd_reject, SimpleNamespace())
        assert store.rejected == [("rel_2", "manual:不符合当前关系")]
        assert "已拒绝" in msg_reject.replies[-1]

    asyncio.run(_run())


def test_daily_proactive_uses_rhythm_policy() -> None:
    class _FakeEngine:
        def __init__(self):
            self.policy_calls: list[tuple[str, str]] = []
            self.injected: list[str] | None = None

        def plan_rhythm_policy(self, *, kind: str, user_input: str = ""):
            self.policy_calls.append((kind, user_input))
            return SimpleNamespace(
                min_count=1,
                max_count=2,
                prefer_count=1,
                min_len=6,
                max_len=22,
                prefer_len=12,
                allow_single_short_ack=False,
            )

        def get_recent_context(self) -> str:
            return "最近在聊游戏和工作"

        def inject_proactive_message(self, msgs: list[str]):
            self.injected = list(msgs)

    class _FakeStarter:
        def __init__(self):
            self.last_policy = None

        def generate(self, recent_context: str = "", count_policy=None):
            self.last_policy = count_policy
            return ["在吗"]

    class _FakeBot:
        pass

    bot = TelegramBot("token")
    engine = _FakeEngine()
    starter = _FakeStarter()
    bot._controller = SimpleNamespace(
        _engine=engine,
        _topic_starter=starter,
        _event_tracker=None,
        _update_activity=lambda: None,
    )
    bot._chat_id = 123
    bot._last_user_activity = 0.0
    bot._app = SimpleNamespace(bot=_FakeBot())

    async def _ensure_session(chat_id: int, no_greet: bool = False) -> bool:
        return True

    delivered: dict[str, object] = {}

    async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
        delivered["chat_id"] = chat_id
        delivered["msgs"] = list(msgs)
        delivered["phase"] = first_delay_phase

    bot._ensure_session = _ensure_session  # type: ignore[assignment]
    bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

    asyncio.run(bot._try_send_daily_message())
    assert starter.last_policy is not None
    assert engine.policy_calls and engine.policy_calls[0][0] == "proactive"
    assert engine.injected == ["在吗"]
    assert delivered["msgs"] == ["在吗"]
