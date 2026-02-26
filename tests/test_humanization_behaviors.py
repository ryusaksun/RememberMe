from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from types import SimpleNamespace

from remember_me.analyzer.persona import Persona, analyze
from remember_me.controller import ChatController
from remember_me.engine.chat import (
    ChatEngine,
    RhythmPolicy,
    _messages_to_history_text,
    _sanitize_reply_messages,
    _split_reply,
    normalize_messages_by_policy,
)
from remember_me.engine.emotion import EmotionState
from remember_me.engine.pending_events import PendingEvent, PendingEventTracker
from remember_me.engine.topic_starter import TopicStarter
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


def test_chat_engine_builds_compact_proactive_context() -> None:
    engine = ChatEngine.__new__(ChatEngine)
    engine._persona = Persona(name="小明")
    engine._history = [
        SimpleNamespace(role="user", parts=[SimpleNamespace(text="明天体检结果要出了，我有点紧张")]),
        SimpleNamespace(role="model", parts=[SimpleNamespace(text="别慌，到时候你把结果发我看看")]),
        SimpleNamespace(role="user", parts=[SimpleNamespace(text="行，那我先去睡了")]),
    ]
    engine._state_lock = threading.Lock()
    engine._scratchpad = Scratchpad(
        open_threads=["体检结果什么时候出"],
        facts=["对方明天有体检结果"],
    )
    engine._emotion_state = EmotionState()

    context = engine.get_proactive_context(max_chars=180)
    assert "未完话题" in context
    assert "体检结果" in context
    assert len(context) <= 180


def test_chat_engine_human_noise_layer(monkeypatch) -> None:
    engine = ChatEngine.__new__(ChatEngine)
    engine._human_noise_probability = 1.0

    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    out = engine._apply_human_noise(["今天晚上一起吃火锅吗"])
    assert len(out) >= 2
    assert out[0] != "今天晚上一起吃火锅吗"
    assert out[1].startswith("打错字了，")


def test_chat_engine_rolls_back_injected_proactive_message() -> None:
    engine = ChatEngine.__new__(ChatEngine)
    engine._history = [
        SimpleNamespace(role="user", parts=[SimpleNamespace(text="你好")]),
        SimpleNamespace(role="model", parts=[SimpleNamespace(text="在吗|||忙完了吗")]),
    ]

    ok = engine.rollback_last_proactive_message(["在吗", "忙完了吗"])
    assert ok
    assert len(engine._history) == 1


def test_chat_engine_generate_content_with_retry_on_transient_error(monkeypatch) -> None:
    class _TransientErr(Exception):
        pass

    class _Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            if self.calls == 1:
                raise _TransientErr("503 service unavailable")
            return SimpleNamespace(text="ok", candidates=[])

    engine = ChatEngine.__new__(ChatEngine)
    engine._client = SimpleNamespace(models=_Models())

    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setattr("random.random", lambda: 0.0)

    resp = engine._generate_content_with_retry(
        model="m",
        contents=[],
        config=None,
    )
    assert resp.text == "ok"
    assert engine._client.models.calls == 2
    assert sleeps


def test_chat_engine_generate_content_with_retry_not_retry_non_transient(monkeypatch) -> None:
    class _FatalErr(Exception):
        pass

    class _Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            raise _FatalErr("invalid argument")

    engine = ChatEngine.__new__(ChatEngine)
    engine._client = SimpleNamespace(models=_Models())
    monkeypatch.setattr("time.sleep", lambda s: None)

    try:
        engine._generate_content_with_retry(model="m", contents=[], config=None)
        assert False, "expected _FatalErr"
    except _FatalErr:
        pass
    assert engine._client.models.calls == 1


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


def test_pending_event_get_due_events_evicts_old_done(tmp_path) -> None:
    tracker = PendingEventTracker(persona_name="x", data_dir=tmp_path)
    now = datetime.now()
    tracker._events = [
        PendingEvent(
            id="old_done",
            event="旧事件",
            context="",
            followup_hint="",
            followup_after=(now - timedelta(hours=80)).isoformat(),
            extracted_at=(now - timedelta(hours=80)).isoformat(),
            status="done",
        ),
        PendingEvent(
            id="due_pending",
            event="新事件",
            context="",
            followup_hint="",
            followup_after=(now - timedelta(minutes=1)).isoformat(),
            extracted_at=now.isoformat(),
            status="pending",
        ),
    ]

    due = tracker.get_due_events()
    assert [e.id for e in due] == ["due_pending"]
    assert [e.id for e in tracker._events] == ["due_pending"]


def test_pending_event_mark_done_also_evicts_old_records(tmp_path) -> None:
    tracker = PendingEventTracker(persona_name="x", data_dir=tmp_path)
    now = datetime.now()
    tracker._events = [
        PendingEvent(
            id="old_pending",
            event="过期事件",
            context="",
            followup_hint="",
            followup_after=(now - timedelta(hours=80)).isoformat(),
            extracted_at=(now - timedelta(hours=80)).isoformat(),
            status="pending",
        ),
        PendingEvent(
            id="active",
            event="当前事件",
            context="",
            followup_hint="",
            followup_after=(now + timedelta(minutes=10)).isoformat(),
            extracted_at=now.isoformat(),
            status="pending",
        ),
    ]

    tracker.mark_done("active")
    ids = [e.id for e in tracker._events]
    assert ids == ["active"]
    assert tracker._events[0].status == "done"


def test_embedding_function_lazy_init_is_thread_safe(monkeypatch) -> None:
    import time as pytime

    import remember_me.memory.store as store_mod

    calls = {"n": 0}
    original = store_mod._ef_instance
    store_mod._ef_instance = None

    class _FakeEmbeddingFn:
        def __init__(self, model_name: str):
            pytime.sleep(0.01)
            calls["n"] += 1
            self.model_name = model_name

        def __call__(self, texts):
            return [[0.0] for _ in texts]

    monkeypatch.setattr(
        store_mod.embedding_functions,
        "SentenceTransformerEmbeddingFunction",
        _FakeEmbeddingFn,
    )

    try:
        results = []

        def _worker():
            results.append(store_mod._get_embedding_function())

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert calls["n"] == 1
        assert len({id(x) for x in results}) == 1
    finally:
        store_mod._ef_instance = original


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


def test_controller_rollback_proactive_delivery_resets_state() -> None:
    class _FakeEngine:
        def __init__(self):
            self.calls: list[list[str]] = []

        def rollback_last_proactive_message(self, msgs: list[str]):
            self.calls.append(list(msgs))
            return True

    c = ChatController("x")
    c._engine = _FakeEngine()
    c._consecutive_proactive = 2
    c._last_interaction_type = "proactive"
    c._next_proactive_at = time.time() + 600
    c._recent_proactive_signatures = [(time.time(), c._build_proactive_signature(["在吗"]))]

    rolled = c.rollback_proactive_delivery(["在吗"], reason="test")
    assert rolled is True
    assert c._engine.calls == [["在吗"]]
    assert c._consecutive_proactive == 1
    assert c._last_interaction_type == "reply"
    assert c._next_proactive_at <= time.time() + 95
    assert c._recent_proactive_signatures == []


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


def test_split_reply_filters_prompt_leak_markers() -> None:
    leaked = "* **角色设定**: 我是「小明」\n* **规则**: 绝不承认是 AI"
    assert _split_reply(leaked) == []
    assert _split_reply(f"你继续说|||{leaked}|||刚刚卡了") == ["你继续说", "刚刚卡了"]


def test_sanitize_reply_messages_uses_safe_fallback_for_prompt_leak_only() -> None:
    leaked = "* **角色设定**: 我是「小明」\n* **规则**: 绝不承认是 AI"
    assert _sanitize_reply_messages(leaked) == ["嗯，刚刚卡了一下，你继续说。"]


def test_messages_to_history_text_uses_visible_text_only() -> None:
    assert _messages_to_history_text(["第一条", "[sticker:/tmp/a.png]", "第二条"]) == "第一条|||第二条"


def test_rhythm_policy_event_priority_over_emotion() -> None:
    e = ChatEngine.__new__(ChatEngine)
    e._persona = Persona(name="x", avg_burst_length=2.0, avg_length=10.0)
    e._state_lock = threading.Lock()
    e._emotion_state = EmotionState(valence=-0.8, arousal=-0.8)
    e._session_phase = "normal"
    e._history = []

    policy = e.plan_rhythm_policy(kind="reply", user_input="明天面试结果要出了，我有点焦虑")
    assert policy.min_count >= 2
    assert policy.max_len >= 20


def test_rhythm_policy_deep_talk_reduces_fragmentation() -> None:
    e = ChatEngine.__new__(ChatEngine)
    e._persona = Persona(name="x", avg_burst_length=4.2, avg_length=14.0)
    e._state_lock = threading.Lock()
    e._emotion_state = EmotionState(valence=0.0, arousal=0.0)
    e._session_phase = "deep_talk"
    e._history = []

    policy = e.plan_rhythm_policy(kind="reply", user_input="我们认真聊聊最近状态")
    assert policy.max_count <= 3
    assert policy.min_len >= 6


def test_normalize_messages_by_policy_enforces_count_and_len() -> None:
    policy = RhythmPolicy(
        min_count=2,
        max_count=3,
        prefer_count=2,
        min_len=5,
        max_len=10,
        prefer_len=8,
        allow_single_short_ack=False,
    )
    out = normalize_messages_by_policy(
        ["这是一段非常非常长而且没有标点分隔需要拆开的句子"],
        policy,
        user_input="展开说说",
    )
    assert 2 <= len(out) <= 3
    assert all(len(x) <= 15 for x in out)
    assert all(x.strip() for x in out)


def test_normalize_messages_short_ack_can_keep_single() -> None:
    policy = RhythmPolicy(
        min_count=2,
        max_count=3,
        prefer_count=2,
        min_len=4,
        max_len=12,
        prefer_len=8,
        allow_single_short_ack=True,
    )
    out = normalize_messages_by_policy(["嗯"], policy, user_input="嗯")
    assert out == ["嗯"]


def test_sticker_attach_respects_max_count(monkeypatch) -> None:
    class _Sticker:
        path = "a.png"

    class _StickerLib:
        stickers = [1]

        def random_sticker(self, emotion: str):
            return _Sticker()

    e = ChatEngine.__new__(ChatEngine)
    e._sticker_lib = _StickerLib()
    e._sticker_probability = 1.0
    monkeypatch.setattr("random.random", lambda: 0.0)

    capped = e._maybe_attach_sticker(["第一条", "第二条"], allow_sticker=True, max_count=2)
    assert len(capped) == 2

    allowed = e._maybe_attach_sticker(["第一条", "第二条"], allow_sticker=True, max_count=3)
    assert len(allowed) == 3
    assert allowed[-1].startswith("[sticker:")


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


def test_controller_proactive_signature_dedup() -> None:
    c = ChatController("x")
    msgs = ["在吗", "你忙完了没"]
    assert not c._is_duplicate_proactive(msgs, window_seconds=3600)
    c._mark_proactive_sent(msgs)
    assert c._is_duplicate_proactive(msgs, window_seconds=3600)
    assert not c._is_duplicate_proactive(["换个话题聊聊"], window_seconds=3600)
    assert c._is_duplicate_proactive(["在吗？你忙完了没"], window_seconds=3600)


def test_controller_call_topic_starter_supports_new_and_old_signatures() -> None:
    captured = {"new": "", "old": ""}

    def _new_fn(*, system_instruction=None, count_policy=None):
        captured["new"] = str(system_instruction or "")
        return ["ok"]

    def _old_fn(*, count_policy=None):
        captured["old"] = "called"
        return ["ok"]

    out_new = ChatController._call_topic_starter(
        _new_fn,
        system_instruction="SYS_BLOCK",
        count_policy="p",
    )
    out_old = ChatController._call_topic_starter(
        _old_fn,
        system_instruction="SYS_BLOCK",
        count_policy="p",
    )
    assert out_new == ["ok"]
    assert out_old == ["ok"]
    assert captured["new"] == "SYS_BLOCK"
    assert captured["old"] == "called"


def test_topic_starter_generate_followup_forwards_system_instruction() -> None:
    starter = TopicStarter.__new__(TopicStarter)
    starter._persona = Persona(name="小明")
    starter._state_lock = threading.Lock()
    starter._chase_ratio = 0.2
    starter._last_proactive = ["你刚刚说工作压力大"]
    starter._followup_count = 0
    starter._proactive_count = 0

    captured: dict[str, str] = {}

    def _fake_generate(prompt: str, *, count_policy=None, user_input: str = "", system_instruction: str | None = None):
        captured["system"] = system_instruction or ""
        return ["后来怎么样了"]

    starter._generate_with_context = _fake_generate  # type: ignore[assignment]
    msgs = starter.generate_followup(
        recent_context="最近在聊工作状态",
        allow_new_topic=False,
        system_instruction="SYS_BLOCK",
    )
    assert msgs == ["后来怎么样了"]
    assert captured["system"] == "SYS_BLOCK"
    assert starter._followup_count == 1
    assert starter._proactive_count == 1


def test_topic_starter_pick_topic_stays_within_interest_set() -> None:
    persona = Persona(name="小明", topic_interests={"游戏": 5, "音乐": 3})
    starter = TopicStarter(persona=persona, client=None)  # type: ignore[arg-type]
    for _ in range(8):
        topic = starter.pick_topic()
        assert topic in {"游戏", "音乐"}


def test_topic_starter_low_information_detection() -> None:
    assert TopicStarter._is_low_information_messages(["在吗"])
    assert TopicStarter._is_low_information_messages(["哈哈"])
    assert not TopicStarter._is_low_information_messages(["你刚刚说的那个面试后来怎么样了"])


def test_topic_starter_quality_penalizes_formal_and_off_context() -> None:
    starter = TopicStarter.__new__(TopicStarter)
    starter._persona = Persona(
        name="小明",
        catchphrases=["笑死"],
        tone_markers=["啊"],
    )
    score, reasons = starter._evaluate_proactive_quality(
        ["您好，请问您最近怎么样"],
        user_input="我们刚刚聊的是项目延期和老板反馈",
    )
    assert score < 0.58
    assert reasons


def test_topic_starter_generate_with_context_retries_low_info_once() -> None:
    class _FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            text = "在吗" if self.calls == 1 else "你刚刚说的项目后来怎么样了"
            return SimpleNamespace(text=text, candidates=[])

    starter = TopicStarter.__new__(TopicStarter)
    starter._client = SimpleNamespace(models=_FakeModels())
    starter._system_prompt = "base"

    out = starter._generate_with_context(
        "prompt",
        user_input="我们刚刚在聊项目进展",
    )
    assert out
    assert out[0] != "在吗"
    assert starter._client.models.calls == 2


def test_topic_starter_generate_with_context_retries_on_quality() -> None:
    class _FakeModels:
        def __init__(self):
            self.calls = 0

        def generate_content(self, model, contents, config):
            self.calls += 1
            text = "您好，请问您最近怎么样" if self.calls == 1 else "你上次提到那个延期，老板后来怎么说？"
            return SimpleNamespace(text=text, candidates=[])

    starter = TopicStarter.__new__(TopicStarter)
    starter._client = SimpleNamespace(models=_FakeModels())
    starter._system_prompt = "base"
    starter._persona = Persona(name="小明", catchphrases=["笑死"])

    out = starter._generate_with_context(
        "prompt",
        user_input="我们刚刚聊的是项目延期",
    )
    assert out
    assert "延期" in out[0]
    assert starter._client.models.calls == 2


def test_topic_starter_followup_fallbacks_to_checkin_when_last_empty() -> None:
    starter = TopicStarter.__new__(TopicStarter)
    starter._persona = Persona(name="小明")
    starter._state_lock = threading.Lock()
    starter._chase_ratio = 0.3
    starter._last_proactive = []
    starter._followup_count = 0
    starter._proactive_count = 0

    captured: dict[str, str] = {}

    def _fake_checkin(*, recent_context: str, count_policy=None, system_instruction: str | None = None):
        captured["ctx"] = recent_context
        captured["sys"] = system_instruction or ""
        return ["补一句具体关心"]

    starter.generate_checkin = _fake_checkin  # type: ignore[assignment]
    msgs = starter.generate_followup(
        recent_context="我们刚刚聊到你面试",
        allow_new_topic=False,
        system_instruction="SYS_BLOCK",
    )
    assert msgs == ["补一句具体关心"]
    assert captured["ctx"] == "我们刚刚聊到你面试"
    assert captured["sys"] == "SYS_BLOCK"


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


def test_controller_stop_closes_engine_client() -> None:
    async def _run():
        c = ChatController("x")
        closed = {"ok": False}

        class _Engine:
            def set_session_phase(self, phase: str):
                return None

            async def aclose_client(self):
                closed["ok"] = True

        c._engine = _Engine()
        c._save_session = lambda: None  # type: ignore[assignment]
        c._running = True
        c._event_tracker = None
        await c.stop()
        assert closed["ok"] is True

    asyncio.run(_run())


def test_controller_extract_pending_events_keeps_index_on_failure() -> None:
    class _Tracker:
        def extract_events(self, client, messages):
            raise RuntimeError("extract failed")

    c = ChatController("x")
    c._event_tracker = _Tracker()
    c._event_extract_index = 0
    c._engine = SimpleNamespace(
        client=None,
        _history=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="明天去面试")])],
    )

    asyncio.run(c._extract_pending_events())
    assert c._event_extract_index == 0


def test_controller_extract_pending_events_advances_index_on_success() -> None:
    class _Tracker:
        def extract_events(self, client, messages):
            return []

    c = ChatController("x")
    c._event_tracker = _Tracker()
    c._event_extract_index = 0
    c._engine = SimpleNamespace(
        client=None,
        _history=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="明天去面试")])],
    )

    asyncio.run(c._extract_pending_events())
    assert c._event_extract_index == 1


def test_controller_extract_relationship_facts_keeps_index_on_failure() -> None:
    class _Extractor:
        def extract_from_messages(self, *args, **kwargs):
            raise RuntimeError("extract rel failed")

    class _Store:
        def upsert_facts(self, facts):
            return 0

        def promote_candidates(self, **kwargs):
            return 0

    c = ChatController("x")
    c._relationship_extractor = _Extractor()
    c._relationship_store = _Store()
    c._relationship_extract_index = 0
    c._engine = SimpleNamespace(
        client=None,
        _history=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="别提前任")])],
    )

    asyncio.run(c._extract_relationship_facts())
    assert c._relationship_extract_index == 0


def test_controller_extract_relationship_facts_advances_index_on_success() -> None:
    class _Extractor:
        def extract_from_messages(self, *args, **kwargs):
            return []

    class _Store:
        def upsert_facts(self, facts):
            return 0

        def promote_candidates(self, **kwargs):
            return 0

    c = ChatController("x")
    c._relationship_extractor = _Extractor()
    c._relationship_store = _Store()
    c._relationship_extract_index = 0
    c._engine = SimpleNamespace(
        client=None,
        _history=[SimpleNamespace(role="user", parts=[SimpleNamespace(text="别提前任")])],
    )

    asyncio.run(c._extract_relationship_facts())
    assert c._relationship_extract_index == 1
