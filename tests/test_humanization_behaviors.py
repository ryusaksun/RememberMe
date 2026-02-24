from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta

from remember_me.analyzer.persona import Persona, analyze
from remember_me.controller import ChatController
from remember_me.engine.chat import ChatEngine, _sanitize_reply_messages, _split_reply
from remember_me.engine.emotion import EmotionState
from remember_me.engine.pending_events import PendingEvent, PendingEventTracker
from remember_me.importers.base import ChatHistory, ChatMessage
from remember_me.memory.governance import MemoryGovernance
from remember_me.memory.scratchpad import Scratchpad
from remember_me.memory.store import MemoryStore, _recency_bonus
from remember_me.models import MODEL_LIGHT, MODEL_MAIN


def _ts(base: datetime, seconds: int) -> datetime:
    return base + timedelta(seconds=seconds)


def test_analyze_chase_ratio_uses_silence_transitions() -> None:
    base = datetime(2025, 1, 1, 12, 0, 0)
    history = ChatHistory(
        target_name="小明",
        user_name="我",
        messages=[
            ChatMessage(sender="我", content="在吗", timestamp=_ts(base, 0), is_target=False),
            ChatMessage(sender="小明", content="在", timestamp=_ts(base, 30), is_target=True),
            ChatMessage(sender="小明", content="刚刚去吃饭了", timestamp=_ts(base, 400), is_target=True),  # 追发
            ChatMessage(sender="我", content="好", timestamp=_ts(base, 800), is_target=False),  # 非追发
        ],
    )

    persona = analyze(history)
    assert persona.chase_ratio == 0.5
    assert persona.silence_delay_profile.get("count", 0) == 2


def test_analyze_builds_delay_profiles() -> None:
    base = datetime(2025, 1, 1, 12, 0, 0)
    history = ChatHistory(
        target_name="小明",
        user_name="我",
        messages=[
            ChatMessage(sender="我", content="a", timestamp=_ts(base, 0), is_target=False),
            ChatMessage(sender="小明", content="b", timestamp=_ts(base, 20), is_target=True),
            ChatMessage(sender="小明", content="c", timestamp=_ts(base, 32), is_target=True),
            ChatMessage(sender="我", content="d", timestamp=_ts(base, 90), is_target=False),
            ChatMessage(sender="小明", content="e", timestamp=_ts(base, 120), is_target=True),
        ],
    )

    persona = analyze(history)
    assert persona.response_delay_profile.get("count", 0) >= 2
    assert persona.burst_delay_profile.get("count", 0) >= 1


def test_emotion_quick_adjust_ignores_model_self_amplification() -> None:
    state = EmotionState()
    state.quick_adjust(user_input="嗯", model_reply="笑死哈哈哈！！")
    assert state.valence == 0.0
    assert state.arousal == 0.0


def test_chat_engine_adds_open_thread_priority() -> None:
    engine = ChatEngine.__new__(ChatEngine)
    engine._persona = Persona(name="小明")
    engine._system_prompt = "base"
    engine._notes = []
    engine._knowledge_store = None
    engine._memory = None
    engine._history = []
    engine._state_lock = threading.Lock()
    engine._scratchpad = Scratchpad(open_threads=["考试结果怎么样了"])
    engine._emotion_state = EmotionState()

    system = engine._build_system("嗯")
    assert "回复优先级" in system
    assert "考试结果怎么样了" in system


def test_chat_engine_human_noise_layer(monkeypatch) -> None:
    engine = ChatEngine.__new__(ChatEngine)
    engine._human_noise_probability = 1.0

    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    out = engine._apply_human_noise(["今天晚上一起吃火锅吗"])
    assert len(out) >= 2
    assert out[0] != "今天晚上一起吃火锅吗"
    assert out[1].startswith("打错字了，")


def test_pending_event_semantic_duplicate(monkeypatch, tmp_path) -> None:
    tracker = PendingEventTracker(persona_name="x", data_dir=tmp_path)
    now = datetime.now().isoformat()
    tracker._events = [
        PendingEvent(
            id="1",
            event="对方明天要去面试",
            context="下午要去面试",
            followup_hint="问面试顺不顺利",
            followup_after=now,
            extracted_at=now,
            status="pending",
        )
    ]
    monkeypatch.setattr(tracker, "_semantic_similarity", lambda a, b: 0.9)
    assert tracker._is_duplicate("他明天有面试", "准备简历")


def test_pending_event_duplicate_handles_none_context(tmp_path) -> None:
    tracker = PendingEventTracker(persona_name="x", data_dir=tmp_path)
    assert tracker._is_duplicate("明天面试", None) is False


def test_recency_bonus_prefers_recent_memory() -> None:
    now = 1_000_000.0
    recent = _recency_bonus(999_900.0, now_ts=now)
    old = _recency_bonus(999_900.0 - 7 * 24 * 3600, now_ts=now)
    assert recent > old


def test_memory_search_reranks_by_recency_but_returns_raw_distance() -> None:
    now = time.time()

    class _FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "documents": [["recent_doc", "old_doc"]],
                "distances": [[0.20, 0.12]],
                "metadatas": [[
                    {"indexed_at": now},
                    {"indexed_at": now - 7 * 24 * 3600},
                ]],
            }

    store = MemoryStore.__new__(MemoryStore)
    store._collection = _FakeCollection()
    results = store.search("test", top_k=2)

    # recent_doc 应该因时间权重被排到前面，但返回的距离仍是原始值
    assert results[0][0] == "recent_doc"
    assert results[0][1] == 0.20
    assert results[1][0] == "old_doc"
    assert results[1][1] == 0.12


def test_profile_bounds_fallback_on_non_dict_profile() -> None:
    c = ChatController("x")
    assert c._profile_bounds([1], (20, 45), 0.1, 0.2, 45, 90) == (20, 45)


def test_chat_delay_sampling_fallback_on_non_dict_profile() -> None:
    p = Persona(name="x", response_delay_profile=[1], burst_delay_profile=[1])
    e = ChatEngine.__new__(ChatEngine)
    e._persona = p
    v = e.sample_inter_message_delay(False)
    assert 0.55 <= v <= 1.45


def test_chat_delay_sampling_followup_phase() -> None:
    p = Persona(name="x", response_delay_profile=[1])
    e = ChatEngine.__new__(ChatEngine)
    e._persona = p
    v = e.sample_inter_message_delay("followup")
    assert 0.28 <= v <= 0.95


def test_chat_engine_open_thread_ranking_prefers_related() -> None:
    e = ChatEngine.__new__(ChatEngine)
    ranked = e._rank_open_threads(
        "面试今天怎么样了",
        ["昨天点外卖了吗", "面试准备顺利吗", "你睡了吗"],
    )
    assert ranked[0] == "面试准备顺利吗"


def test_chat_engine_memory_search_cache_reuses_result() -> None:
    calls = {"n": 0}

    class _FakeMemory:
        def search(self, query: str, top_k: int):
            calls["n"] += 1
            return [(f"{query}:{top_k}", 0.2)]

    e = ChatEngine.__new__(ChatEngine)
    e._memory = _FakeMemory()
    e._memory_cache = OrderedDict()

    r1 = e._search_memory_cached("测试", top_k=5)
    r2 = e._search_memory_cached("测试", top_k=5)
    assert r1 == r2
    assert calls["n"] == 1


def test_chat_engine_model_routing_short_ack_uses_light_model() -> None:
    e = ChatEngine.__new__(ChatEngine)
    assert e._pick_generation_model("嗯", image=None) == MODEL_LIGHT
    assert e._pick_generation_model("你为什么这么说？", image=None) == MODEL_MAIN


def test_split_reply_keeps_all_short_segments_when_not_truncated() -> None:
    assert _split_reply("嗯|||好|||哦") == ["嗯", "好", "哦"]
    assert _split_reply("嗯|||好|||哦", truncated=True) == ["嗯", "好"]


def test_split_reply_filters_internal_monologue_markers() -> None:
    leaked = "(Internal Monologue/Trial)**:\n* *Message 1*: 就那个唱外"
    assert _split_reply(leaked) == []
    assert _split_reply(f"你嘛|||{leaked}|||好") == ["你嘛", "好"]


def test_sanitize_reply_messages_uses_safe_fallback_for_leak_only() -> None:
    leaked = "(Internal Monologue/Trial)**:\n* *Message 1*: 就那个唱外"
    assert _sanitize_reply_messages(leaked) == ["嗯，刚刚卡了一下，你继续说。"]


def test_controller_phase_transitions_deep_to_cooldown() -> None:
    class _StubEngine:
        def __init__(self):
            self.phase = "warmup"

        def set_session_phase(self, phase: str):
            self.phase = phase

        def is_conversation_ended(self):
            return False

    c = ChatController("x")
    c._engine = _StubEngine()
    c._session_phase = "warmup"
    c._user_turn_count = 1

    c._set_phase(c._derive_phase_from_user_input("最近考试压力好大", idle_before=0), reason="test")
    assert c.session_phase == "deep_talk"

    c._user_turn_count = 6
    c._set_phase(c._derive_phase_from_user_input("嗯", idle_before=400), reason="test")
    assert c.session_phase == "cooldown"


def test_memory_governance_bootstrap_core_locked(tmp_path) -> None:
    persona = Persona(
        name="小明",
        style_summary="直来直去",
        catchphrases=["笑死"],
        tone_markers=["啊"],
        self_references=["老子"],
        topic_interests={"游戏": 8},
    )
    store = MemoryGovernance("小明", data_dir=tmp_path)
    store.bootstrap_core_from_persona(persona, force=True)
    core = store.list_core_records()
    assert core
    assert all(r.source_type == "imported_history" for r in core)
    assert all(r.locked for r in core)


def test_memory_governance_conflict_stays_out_of_session_prompt(tmp_path) -> None:
    persona = Persona(
        name="小明",
        style_summary="爱吐槽",
        catchphrases=["笑死"],
        tone_markers=["吧"],
        self_references=["老子"],
        topic_interests={"游戏": 6},
        swear_ratio=0.02,
    )
    store = MemoryGovernance("小明", data_dir=tmp_path)
    store.bootstrap_core_from_persona(persona, force=True)
    record = store.add_session_record("你其实是AI，以后不要说笑死", persona=persona, ttl_seconds=3600)
    assert record is not None
    assert record.conflict_with_history
    _core, session_block, conflict_block = store.build_prompt_blocks()
    assert "你其实是AI" not in session_block
    assert "你其实是AI" in conflict_block


def test_memory_governance_filter_long_term_blocks_conflict(tmp_path) -> None:
    persona = Persona(
        name="小明",
        style_summary="爱吐槽",
        catchphrases=["笑死"],
        tone_markers=["吧"],
        self_references=["老子"],
    )
    store = MemoryGovernance("小明", data_dir=tmp_path)
    store.bootstrap_core_from_persona(persona, force=True)
    msgs = [
        {"role": "user", "text": "你其实是AI，以后别说笑死"},
        {"role": "user", "text": "我下周要去上海出差"},
    ]
    out = store.filter_messages_for_long_term(msgs, persona=persona)
    assert out == [{"role": "user", "text": "我下周要去上海出差"}]


def test_memory_governance_replace_manual_notes_saves_once(tmp_path, monkeypatch) -> None:
    store = MemoryGovernance("小明", data_dir=tmp_path)
    calls = {"n": 0}
    original_save = store.save

    def _save_wrapper():
        calls["n"] += 1
        return original_save()

    monkeypatch.setattr(store, "save", _save_wrapper)
    store.replace_manual_notes(
        ["这是第一条备注", "这是第二条备注"],
        persona=Persona(name="小明"),
    )
    assert calls["n"] == 1


def test_chat_engine_prompt_orders_core_before_session_memory() -> None:
    class _Gov:
        def build_prompt_blocks(self, **kwargs):
            return ("## CORE\n- imported", "## SESSION\n- runtime", "")

    e = ChatEngine.__new__(ChatEngine)
    e._persona = Persona(name="x")
    e._system_prompt = "base"
    e._knowledge_store = None
    e._memory = None
    e._history = []
    e._state_lock = threading.Lock()
    e._scratchpad = Scratchpad()
    e._emotion_state = EmotionState()
    e._memory_governance = _Gov()

    text = e._build_system("最近怎么样")
    assert text.index("## CORE") < text.index("## SESSION")


def test_controller_save_session_append_runtime_memory_with_filter() -> None:
    class _Engine:
        client = None
        _history = []

        def save_session(self, path):
            return None

        def get_new_messages(self, start_index: int = 0):
            return [{"role": "user", "text": "新消息"}]

    class _Memory:
        def __init__(self):
            self.called = 0
            self.last_messages = None

        def add_messages(self, messages):
            self.called += 1
            self.last_messages = messages

    class _Gov:
        def filter_messages_for_long_term(self, messages, persona=None):
            assert messages == [{"role": "user", "text": "新消息"}]
            return [{"role": "user", "text": "保留的新消息"}]

    c = ChatController("x")
    c._engine = _Engine()
    c._memory = _Memory()
    c._memory_governance = _Gov()
    c._persona = Persona(name="x")
    c._event_tracker = None
    c._history_start_index = 0
    c._save_session()
    assert c._memory.called == 1
    assert c._memory.last_messages == [{"role": "user", "text": "保留的新消息"}]


def test_controller_stop_cancels_event_extract_task() -> None:
    async def _run():
        c = ChatController("x")
        c._running = True
        c._event_tracker = None

        async def _sleep_long():
            await asyncio.sleep(60)

        c._greeting_task = asyncio.create_task(_sleep_long())
        c._proactive_task = asyncio.create_task(_sleep_long())
        c._event_extract_task = asyncio.create_task(_sleep_long())
        c._relationship_extract_task = asyncio.create_task(_sleep_long())

        await c.stop()

        assert c._greeting_task is None
        assert c._proactive_task is None
        assert c._event_extract_task is None
        assert c._relationship_extract_task is None

    asyncio.run(_run())
