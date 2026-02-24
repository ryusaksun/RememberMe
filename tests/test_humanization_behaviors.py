from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta

from remember_me.analyzer.persona import Persona, analyze
from remember_me.controller import ChatController
from remember_me.engine.chat import ChatEngine
from remember_me.engine.emotion import EmotionState
from remember_me.engine.pending_events import PendingEvent, PendingEventTracker
from remember_me.importers.base import ChatHistory, ChatMessage
from remember_me.memory.scratchpad import Scratchpad
from remember_me.memory.store import MemoryStore, _recency_bonus


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
