from __future__ import annotations

import asyncio
import io
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import remember_me.telegram_bot as tg_mod
from telegram.error import NetworkError
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


def test_run_with_typing_heartbeat_repeats_until_task_done() -> None:
    calls = {"typing": 0}

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            calls["typing"] += 1

    async def _slow_reply():
        await asyncio.sleep(0.03)
        return ["ok"]

    bot = TelegramBot("token")
    bot.TYPING_HEARTBEAT_INTERVAL = 0.01

    async def _run():
        result = await bot._run_with_typing_heartbeat(_FakeBot(), 123, _slow_reply())
        assert result == ["ok"]

    asyncio.run(_run())
    assert calls["typing"] >= 2


def test_run_with_typing_heartbeat_tolerates_typing_errors() -> None:
    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            raise RuntimeError("typing failed")

    async def _slow_reply():
        await asyncio.sleep(0.01)
        return ["ok"]

    bot = TelegramBot("token")
    bot.TYPING_HEARTBEAT_INTERVAL = 0.005

    async def _run():
        result = await bot._run_with_typing_heartbeat(_FakeBot(), 123, _slow_reply())
        assert result == ["ok"]

    asyncio.run(_run())


def test_send_text_retries_on_retry_after(monkeypatch) -> None:
    calls = {"send": 0}

    class _FakeRetryAfter(Exception):
        def __init__(self, seconds: int):
            self._retry_after = timedelta(seconds=seconds)

    monkeypatch.setattr(tg_mod, "RetryAfter", _FakeRetryAfter)

    class _FakeBot:
        async def send_message(self, chat_id: int, text: str):
            calls["send"] += 1
            if calls["send"] == 1:
                raise _FakeRetryAfter(0)
            return None

    bot = TelegramBot("token")

    asyncio.run(bot._send_text(_FakeBot(), 123, "hello"))
    assert calls["send"] == 2


def test_send_photo_retry_resets_stream_cursor(monkeypatch) -> None:
    calls: list[bytes] = []

    class _FakeRetryAfter(Exception):
        def __init__(self, seconds: int):
            self._retry_after = timedelta(seconds=seconds)

    monkeypatch.setattr(tg_mod, "RetryAfter", _FakeRetryAfter)

    class _FakeBot:
        async def send_photo(self, chat_id: int, photo):
            calls.append(photo.read())
            if len(calls) == 1:
                raise _FakeRetryAfter(0)
            return None

    bot = TelegramBot("token")
    photo = io.BytesIO(b"fake_image_bytes")

    asyncio.run(bot._send_photo(_FakeBot(), 123, photo))
    assert calls == [b"fake_image_bytes", b"fake_image_bytes"]


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


def test_flush_coalesce_merges_messages_without_mechanical_prefix() -> None:
    captured: dict[str, object] = {}

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

        async def send_message(self, chat_id: int, text: str):
            return None

    class _FakeController:
        async def send_message(self, text: str):
            captured["merged"] = text
            return ["收到"]

    bot = TelegramBot("token")
    bot._app = SimpleNamespace(bot=_FakeBot())
    bot._controller = _FakeController()
    bot._chat_id = 123
    bot._coalesce_buffer = ["第一条", "第二条"]

    async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
        captured["delivered"] = list(msgs)

    bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

    asyncio.run(bot._flush_coalesce(chat_id=123, delay=0))
    assert captured["merged"] == "第一条\n第二条"
    assert "连续发了" not in str(captured["merged"])


def test_flush_coalesce_failure_uses_humanized_error_reply() -> None:
    sent: list[str] = []

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

        async def send_message(self, chat_id: int, text: str):
            sent.append(text)
            return None

    class _FakeController:
        async def send_message(self, text: str):
            raise RuntimeError("llm timeout 504")

    bot = TelegramBot("token")
    bot._app = SimpleNamespace(bot=_FakeBot())
    bot._controller = _FakeController()
    bot._chat_id = 123
    bot._coalesce_buffer = ["你好"]

    async def _run_with_typing_heartbeat(bot_obj, chat_id: int, coro):
        return await coro

    bot._run_with_typing_heartbeat = _run_with_typing_heartbeat  # type: ignore[assignment]

    asyncio.run(bot._flush_coalesce(chat_id=123, delay=0))
    assert sent
    assert all("llm timeout 504" not in msg for msg in sent)
    assert all(not msg.startswith("出错了：") for msg in sent)


def test_humanized_error_reply_is_rate_limited() -> None:
    sent: list[str] = []

    class _FakeBot:
        async def send_message(self, chat_id: int, text: str):
            sent.append(text)
            return None

    bot = TelegramBot("token")
    bot.ERROR_REPLY_COOLDOWN = 60
    bot._last_error_reply_at = time.time()
    asyncio.run(bot._maybe_send_humanized_error(_FakeBot(), 123))
    assert sent == []


def test_humanized_error_text_uses_persona_style_token(monkeypatch) -> None:
    bot = TelegramBot("token")
    bot._controller = SimpleNamespace(
        _persona=SimpleNamespace(
            self_references=["老子"],
            catchphrases=["笑死"],
            tone_markers=["啊"],
        )
    )
    monkeypatch.setattr(tg_mod.random, "choice", lambda seq: seq[0])

    text = bot._humanized_error_text()
    assert text.startswith("老子，")


def test_on_app_error_handles_transient_network_error() -> None:
    bot = TelegramBot("token")
    ctx = SimpleNamespace(error=NetworkError("Bad Gateway"))
    asyncio.run(bot._on_app_error(None, ctx))


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


def test_note_rel_confirm_uses_last_list_index_cache() -> None:
    class _Fact:
        def __init__(self, fid: str, status: str, content: str):
            self.id = fid
            self.status = status
            self.content = content
            self.type = "shared_event"
            self.confidence = 0.8
            self.evidence = []

    class _Store:
        def __init__(self):
            self.rows = [
                _Fact("rel_candidate", "candidate", "候选记录"),
                _Fact("rel_rejected", "rejected", "被拒绝记录"),
            ]
            self.confirmed: list[str] = []

        def list_facts(self, **kwargs):
            statuses = kwargs.get("statuses")
            if statuses:
                return [x for x in self.rows if x.status in statuses]
            return list(self.rows)

        def confirm_fact(self, ref):
            self.confirmed.append(str(ref))
            return True

        def reject_fact(self, ref, reason: str = "manual_reject"):
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
        msg_list = _Msg("/note rel list rejected")
        upd_list = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_list,
        )
        await bot._cmd_note(upd_list, SimpleNamespace())
        assert msg_list.replies and "被拒绝记录" in msg_list.replies[-1]

        msg_confirm = _Msg("/note rel confirm 1")
        upd_confirm = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_confirm,
        )
        await bot._cmd_note(upd_confirm, SimpleNamespace())
        assert store.confirmed == ["rel_rejected"]

    asyncio.run(_run())


def test_note_rel_confirm_cache_uses_direct_id_lookup_not_limited_list() -> None:
    class _Fact:
        def __init__(self, fid: str, status: str, content: str):
            self.id = fid
            self.status = status
            self.content = content
            self.type = "shared_event"
            self.confidence = 0.8
            self.evidence = []

    class _Store:
        def __init__(self):
            self.rows = [_Fact(f"rel_{i}", "candidate", f"候选 {i}") for i in range(600)]
            self.rows.append(_Fact("rel_target", "rejected", "目标记录"))
            self.confirmed: list[str] = []

        def list_facts(self, **kwargs):
            statuses = kwargs.get("statuses")
            limit = int(kwargs.get("limit", 50))
            rows = list(self.rows)
            if statuses:
                rows = [x for x in rows if x.status in statuses]
            # 模拟分页截断：全量查询时 target 在 500 之后会被截断
            return rows[:limit]

        def get_fact_by_id(self, fact_id: str):
            for row in self.rows:
                if row.id == fact_id:
                    return row
            return None

        def confirm_fact(self, ref):
            self.confirmed.append(str(ref))
            return True

        def reject_fact(self, ref, reason: str = "manual_reject"):
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
        msg_list = _Msg("/note rel list rejected")
        upd_list = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_list,
        )
        await bot._cmd_note(upd_list, SimpleNamespace())
        assert msg_list.replies and "目标记录" in msg_list.replies[-1]

        msg_confirm = _Msg("/note rel confirm 1")
        upd_confirm = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_confirm,
        )
        await bot._cmd_note(upd_confirm, SimpleNamespace())
        assert store.confirmed == ["rel_target"]

    asyncio.run(_run())


def test_note_rel_confirm_after_empty_list_cache_should_not_fallback_to_default_rows() -> None:
    class _Fact:
        def __init__(self, fid: str, status: str, content: str):
            self.id = fid
            self.status = status
            self.content = content
            self.type = "shared_event"
            self.confidence = 0.8
            self.evidence = []

    class _Store:
        def __init__(self):
            self.rows = [_Fact("rel_candidate", "candidate", "候选记录")]
            self.confirmed: list[str] = []

        def list_facts(self, **kwargs):
            statuses = kwargs.get("statuses")
            rows = list(self.rows)
            if statuses:
                rows = [x for x in rows if x.status in statuses]
            return rows

        def get_fact_by_id(self, fact_id: str):
            for row in self.rows:
                if row.id == fact_id:
                    return row
            return None

        def confirm_fact(self, ref):
            self.confirmed.append(str(ref))
            return True

        def reject_fact(self, ref, reason: str = "manual_reject"):
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
        msg_list = _Msg("/note rel list rejected")
        upd_list = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_list,
        )
        await bot._cmd_note(upd_list, SimpleNamespace())
        assert msg_list.replies and "暂无关系记忆记录" in msg_list.replies[-1]

        msg_confirm = _Msg("/note rel confirm 1")
        upd_confirm = SimpleNamespace(
            effective_user=SimpleNamespace(id=1),
            message=msg_confirm,
        )
        await bot._cmd_note(upd_confirm, SimpleNamespace())
        assert store.confirmed == []
        assert msg_confirm.replies and "结果为空" in msg_confirm.replies[-1]

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

        def get_proactive_context(self, max_chars: int = 460) -> str:
            return "压缩上下文"

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
    assert engine.policy_calls and engine.policy_calls[0] == ("proactive", "压缩上下文")
    assert engine.injected == ["在吗"]
    assert delivered["msgs"] == ["在吗"]


def test_daily_proactive_skips_recent_duplicate() -> None:
    class _FakeEngine:
        def __init__(self):
            self.injected: list[str] | None = None

        def plan_rhythm_policy(self, *, kind: str, user_input: str = ""):
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
        def generate(self, recent_context: str = "", count_policy=None):
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
    bot._mark_proactive_sent(["在吗"])

    async def _ensure_session(chat_id: int, no_greet: bool = False) -> bool:
        return True

    delivered = {"called": False}

    async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
        delivered["called"] = True

    bot._ensure_session = _ensure_session  # type: ignore[assignment]
    bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

    asyncio.run(bot._try_send_daily_message())
    assert engine.injected is None
    assert delivered["called"] is False


def test_daily_event_followup_marks_done_after_delivery() -> None:
    class _FakeTracker:
        def __init__(self):
            self.done_ids: list[str] = []

        def get_due_events(self):
            return [SimpleNamespace(
                id="evt_1",
                event="今天去体检",
                context="下午体检",
                followup_hint="问结果",
            )]

        def mark_done(self, event_id: str):
            self.done_ids.append(event_id)

    class _FakeEngine:
        def plan_rhythm_policy(self, *, kind: str, user_input: str = ""):
            return SimpleNamespace(
                min_count=1,
                max_count=2,
                prefer_count=1,
                min_len=6,
                max_len=22,
                prefer_len=12,
                allow_single_short_ack=False,
            )

        def get_proactive_context(self, max_chars: int = 460) -> str:
            return "压缩上下文"

        def inject_proactive_message(self, msgs: list[str]):
            return None

    class _FakeStarter:
        def generate_event_followup(self, *_args, **_kwargs):
            return ["体检结束了吗？结果还好吗"]

        def generate(self, **_kwargs):
            return ["在吗"]

    class _FakeBot:
        pass

    tracker = _FakeTracker()
    bot = TelegramBot("token")
    bot._controller = SimpleNamespace(
        _engine=_FakeEngine(),
        _topic_starter=_FakeStarter(),
        _event_tracker=tracker,
        _update_activity=lambda: None,
    )
    bot._chat_id = 123
    bot._last_user_activity = 0.0
    bot._app = SimpleNamespace(bot=_FakeBot())

    async def _ensure_session(chat_id: int, no_greet: bool = False) -> bool:
        return True

    async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
        return None

    bot._ensure_session = _ensure_session  # type: ignore[assignment]
    bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

    ok = asyncio.run(bot._try_send_daily_message())
    assert ok is True
    assert tracker.done_ids == ["evt_1"]


def test_daily_event_followup_not_mark_done_when_delivery_failed() -> None:
    class _FakeTracker:
        def __init__(self):
            self.done_ids: list[str] = []

        def get_due_events(self):
            return [SimpleNamespace(
                id="evt_1",
                event="今天去体检",
                context="下午体检",
                followup_hint="问结果",
            )]

        def mark_done(self, event_id: str):
            self.done_ids.append(event_id)

    class _FakeEngine:
        def __init__(self):
            self.injected: list[str] | None = None

        def plan_rhythm_policy(self, *, kind: str, user_input: str = ""):
            return SimpleNamespace(
                min_count=1,
                max_count=2,
                prefer_count=1,
                min_len=6,
                max_len=22,
                prefer_len=12,
                allow_single_short_ack=False,
            )

        def get_proactive_context(self, max_chars: int = 460) -> str:
            return "压缩上下文"

        def inject_proactive_message(self, msgs: list[str]):
            self.injected = list(msgs)

    class _FakeStarter:
        def generate_event_followup(self, *_args, **_kwargs):
            return ["体检结束了吗？结果还好吗"]

        def generate(self, **_kwargs):
            return ["在吗"]

    class _FakeBot:
        pass

    tracker = _FakeTracker()
    bot = TelegramBot("token")
    bot._controller = SimpleNamespace(
        _engine=_FakeEngine(),
        _topic_starter=_FakeStarter(),
        _event_tracker=tracker,
        _update_activity=lambda: None,
    )
    bot._chat_id = 123
    bot._last_user_activity = 0.0
    bot._app = SimpleNamespace(bot=_FakeBot())

    async def _ensure_session(chat_id: int, no_greet: bool = False) -> bool:
        return True

    async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
        raise RuntimeError("network down")

    bot._ensure_session = _ensure_session  # type: ignore[assignment]
    bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

    ok = asyncio.run(bot._try_send_daily_message())
    assert ok is False
    assert tracker.done_ids == []
    assert bot._controller._engine.injected is None


def test_telegram_proactive_signature_dedup_supports_near_match() -> None:
    bot = TelegramBot("token")
    bot._mark_proactive_sent(["在吗，你忙完了没"])
    assert bot._is_duplicate_proactive(["在吗？你忙完了没"])
    assert not bot._is_duplicate_proactive(["明天你有空吗"])


def test_telegram_call_topic_starter_supports_new_and_old_signatures() -> None:
    captured = {"new": "", "old": ""}

    def _new_fn(*, system_instruction=None, count_policy=None):
        captured["new"] = str(system_instruction or "")
        return ["ok"]

    def _old_fn(*, count_policy=None):
        captured["old"] = "called"
        return ["ok"]

    out_new = TelegramBot._call_topic_starter(
        _new_fn,
        system_instruction="SYS_BLOCK",
        count_policy="p",
    )
    out_old = TelegramBot._call_topic_starter(
        _old_fn,
        system_instruction="SYS_BLOCK",
        count_policy="p",
    )
    assert out_new == ["ok"]
    assert out_old == ["ok"]
    assert captured["new"] == "SYS_BLOCK"
    assert captured["old"] == "called"


def test_ensure_session_on_message_skip_proactive_when_user_active() -> None:
    delivered = {"n": 0}

    class _FakeController:
        def __init__(self, persona_name: str):
            self.persona_name = persona_name
            self.session_loaded = False
            self.rollback_calls: list[tuple[list[str], str]] = []

        async def start(self, on_message, on_typing=None, no_greet: bool = False):
            on_message(["在吗"], "proactive")

        def rollback_proactive_delivery(self, msgs: list[str], *, reason: str = "delivery_failed"):
            self.rollback_calls.append((list(msgs), reason))
            return True

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

    async def _run():
        bot = TelegramBot("token")
        bot._app = SimpleNamespace(bot=_FakeBot())
        bot._last_user_activity = time.time()

        async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
            delivered["n"] += 1

        bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

        original = tg_mod.ChatController
        try:
            tg_mod.ChatController = _FakeController  # type: ignore[assignment]
            ok = await bot._ensure_session(chat_id=123)
            assert ok
            await asyncio.sleep(0.02)
            assert bot._controller.rollback_calls == [(["在吗"], "telegram_skip_recent_user_activity")]
        finally:
            tg_mod.ChatController = original  # type: ignore[assignment]

    asyncio.run(_run())
    assert delivered["n"] == 0


def test_ensure_session_on_message_rolls_back_proactive_on_delivery_failure() -> None:
    class _FakeController:
        def __init__(self, persona_name: str):
            self.persona_name = persona_name
            self.session_loaded = False
            self.rollback_calls: list[list[str]] = []

        async def start(self, on_message, on_typing=None, no_greet: bool = False):
            on_message(["在吗"], "proactive")

        def rollback_proactive_delivery(self, msgs: list[str], *, reason: str = "delivery_failed"):
            self.rollback_calls.append(list(msgs))
            return True

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

    async def _run():
        bot = TelegramBot("token")
        bot._app = SimpleNamespace(bot=_FakeBot())
        bot._last_user_activity = 0.0

        async def _deliver_messages(bot_obj, chat_id: int, msgs: list[str], first_delay_phase: str = "first"):
            raise RuntimeError("send failed")

        bot._deliver_messages = _deliver_messages  # type: ignore[assignment]

        original = tg_mod.ChatController
        try:
            tg_mod.ChatController = _FakeController  # type: ignore[assignment]
            ok = await bot._ensure_session(chat_id=123)
            assert ok
            await asyncio.sleep(0.05)
            assert bot._controller.rollback_calls == [["在吗"]]
        finally:
            tg_mod.ChatController = original  # type: ignore[assignment]

    asyncio.run(_run())


def test_ensure_session_singleflight_under_concurrent_calls() -> None:
    counters = {"init": 0, "start": 0}

    class _FakeController:
        def __init__(self, persona_name: str):
            counters["init"] += 1
            self.persona_name = persona_name
            self.session_loaded = False

        async def start(self, on_message, on_typing=None, no_greet: bool = False):
            counters["start"] += 1
            await asyncio.sleep(0.03)

        async def stop(self):
            return None

    class _FakeBot:
        async def send_chat_action(self, chat_id: int, action):
            return None

    async def _run():
        bot = TelegramBot("token")
        bot._app = SimpleNamespace(bot=_FakeBot())

        original = tg_mod.ChatController
        try:
            tg_mod.ChatController = _FakeController  # type: ignore[assignment]
            r1, r2 = await asyncio.gather(
                bot._ensure_session(chat_id=123),
                bot._ensure_session(chat_id=123),
            )
            assert r1 is True and r2 is True
            assert counters["init"] == 1
            assert counters["start"] == 1
            assert bot._controller is not None
            assert bot._chat_id == 123
        finally:
            tg_mod.ChatController = original  # type: ignore[assignment]

    asyncio.run(_run())


def test_split_telegram_text_respects_limit() -> None:
    text = "a" * 100
    chunks = TelegramBot._split_telegram_text(text, limit=30)
    assert len(chunks) == 4
    assert all(len(c) <= 30 for c in chunks)
    assert "".join(chunks) == text


def test_plan_daily_times_skips_expired_slots_instead_of_pushing_to_tomorrow(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 26, 6, 33, tzinfo=tz)

    monkeypatch.setattr(tg_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(tg_mod.random, "sample", lambda seq, k: list(seq)[:k])
    monkeypatch.setattr(tg_mod.random, "randint", lambda a, b: 30)

    bot = TelegramBot("token")
    bot._load_persona_meta = lambda: {  # type: ignore[assignment]
        "active_hours": [1],  # 明显是今天已过太久的时段
        "chase_ratio": 0.0,
        "avg_burst_length": 1.0,
    }

    times = bot._plan_daily_times()
    assert times == []


def test_plan_daily_times_keeps_today_future_slots(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 26, 6, 33, tzinfo=tz)

    monkeypatch.setattr(tg_mod, "datetime", _FixedDateTime)
    monkeypatch.setattr(tg_mod.random, "sample", lambda seq, k: list(seq)[:k])
    monkeypatch.setattr(tg_mod.random, "randint", lambda a, b: 30)

    bot = TelegramBot("token")
    bot._load_persona_meta = lambda: {  # type: ignore[assignment]
        "active_hours": [8],
        "chase_ratio": 0.0,
        "avg_burst_length": 1.0,
    }

    times = bot._plan_daily_times()
    assert len(times) == 1
    assert times[0].date() == _FixedDateTime.now(tg_mod.TIMEZONE).date()
    assert times[0].hour == 8


def test_plan_daily_times_filters_before_sampling_avoids_empty_plan(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 26, 6, 33, tzinfo=tz)

    monkeypatch.setattr(tg_mod, "datetime", _FixedDateTime)
    # 如果先采样再过滤，这里会选中 1 点并被丢弃；修复后会先过滤，只能选到 8 点。
    monkeypatch.setattr(tg_mod.random, "sample", lambda seq, k: list(seq)[:k])
    monkeypatch.setattr(tg_mod.random, "randint", lambda a, b: 30)

    bot = TelegramBot("token")
    bot._load_persona_meta = lambda: {  # type: ignore[assignment]
        "active_hours": [1, 8],
        "chase_ratio": 0.0,
        "avg_burst_length": 1.0,
    }

    times = bot._plan_daily_times()
    assert len(times) == 1
    assert times[0].hour == 8


def test_daily_scheduler_requeues_slot_when_send_failed(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 26, 10, 0, tzinfo=tz)

    monkeypatch.setattr(tg_mod, "datetime", _FixedDateTime)

    bot = TelegramBot("token")
    bot._daily_date = "2026-02-26"
    bot._daily_times = [_FixedDateTime(2026, 2, 26, 9, 50, tzinfo=tg_mod.TIMEZONE)]

    async def _try_send():
        return False

    sleep_calls = {"n": 0}

    async def _sleep(_seconds: float):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise asyncio.CancelledError()

    bot._try_send_daily_message = _try_send  # type: ignore[assignment]
    monkeypatch.setattr(tg_mod.asyncio, "sleep", _sleep)

    try:
        asyncio.run(bot._daily_scheduler())
    except asyncio.CancelledError:
        pass

    assert len(bot._daily_times) == 1
    assert bot._daily_times[0].hour == 10
    assert bot._daily_times[0].minute == 10


def test_daily_scheduler_consumes_slot_on_send_success(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 26, 10, 0, tzinfo=tz)

    monkeypatch.setattr(tg_mod, "datetime", _FixedDateTime)

    bot = TelegramBot("token")
    bot._daily_date = "2026-02-26"
    bot._daily_times = [_FixedDateTime(2026, 2, 26, 9, 50, tzinfo=tg_mod.TIMEZONE)]

    async def _try_send():
        return True

    sleep_calls = {"n": 0}

    async def _sleep(_seconds: float):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise asyncio.CancelledError()

    bot._try_send_daily_message = _try_send  # type: ignore[assignment]
    monkeypatch.setattr(tg_mod.asyncio, "sleep", _sleep)

    try:
        asyncio.run(bot._daily_scheduler())
    except asyncio.CancelledError:
        pass

    assert bot._daily_times == []


def test_daily_scheduler_retry_backoff_escalates_after_previous_failure(monkeypatch) -> None:
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 2, 26, 10, 0, tzinfo=tz)

    monkeypatch.setattr(tg_mod, "datetime", _FixedDateTime)

    bot = TelegramBot("token")
    bot._daily_date = "2026-02-26"
    original_slot = _FixedDateTime(2026, 2, 26, 9, 50, tzinfo=tg_mod.TIMEZONE)
    bot._daily_times = [original_slot]
    bot._daily_retry_attempts[bot._daily_slot_key(original_slot)] = 1

    async def _try_send():
        return False

    sleep_calls = {"n": 0}

    async def _sleep(_seconds: float):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise asyncio.CancelledError()

    bot._try_send_daily_message = _try_send  # type: ignore[assignment]
    monkeypatch.setattr(tg_mod.asyncio, "sleep", _sleep)

    try:
        asyncio.run(bot._daily_scheduler())
    except asyncio.CancelledError:
        pass

    assert len(bot._daily_times) == 1
    assert bot._daily_times[0].hour == 10
    assert bot._daily_times[0].minute == 20
    key = bot._daily_slot_key(bot._daily_times[0])
    assert bot._daily_retry_attempts[key] == 2
