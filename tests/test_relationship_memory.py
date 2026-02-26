from __future__ import annotations

import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from types import SimpleNamespace

from remember_me.analyzer.relationship_extractor import RelationshipExtractor
from remember_me.engine.chat import ChatEngine
from remember_me.engine.emotion import EmotionState
from remember_me.importers.base import ChatHistory, ChatMessage
from remember_me.memory.governance import MemoryGovernance
from remember_me.memory.relationship import RelationshipFact, RelationshipMemoryStore
from remember_me.memory.scratchpad import Scratchpad


def _ts() -> datetime:
    return datetime(2025, 1, 1, 12, 0, 0)


def test_relationship_store_promote_candidates(tmp_path) -> None:
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    fact = RelationshipFact(
        id="rel_1",
        type="boundary",
        subject="user",
        content="存在明确的交流边界（别提/别问/不聊）",
        evidence=["别再提这个了", "不要再问这个了"],
        confidence=0.82,
        source="imported_history",
        status="candidate",
    )
    changed = store.upsert_facts([fact])
    assert changed == 1
    assert len(store.list_confirmed()) == 0
    promoted = store.promote_candidates(min_confidence=0.78, min_evidence=2)
    assert promoted == 1
    confirmed = store.list_confirmed()
    assert len(confirmed) == 1
    assert confirmed[0].status == "confirmed"


def test_relationship_store_build_prompt_block(tmp_path) -> None:
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    store.upsert_facts([
        RelationshipFact(
            id="rel_2",
            type="boundary",
            subject="user",
            content="存在明确的交流边界（别提/别问/不聊）",
            evidence=["别再提这个了", "不要再问这个了"],
            confidence=0.90,
            source="imported_history",
            status="confirmed",
        )
    ])
    block = store.build_prompt_block(limit=5)
    assert "你们关系记忆" in block
    assert "[互动边界]" in block


def test_relationship_store_upsert_merges_meta_and_boundary_cooldown(tmp_path) -> None:
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    store.upsert_facts([
        RelationshipFact(
            id="rel_meta_1",
            type="addressing",
            subject="persona",
            content="常用称呼偏好：宝宝",
            evidence=["宝宝早", "宝宝晚安"],
            confidence=0.72,
            status="candidate",
            meta={"term": "宝宝"},
        )
    ])
    changed = store.upsert_facts([
        RelationshipFact(
            id="rel_meta_2",
            type="addressing",
            subject="persona",
            content="常用称呼偏好：宝宝",
            evidence=["宝宝早点睡"],
            confidence=0.88,
            status="confirmed",
            meta={"preferred_contexts": ["daily_care"]},
        )
    ])
    assert changed == 1
    confirmed = store.list_confirmed(limit=5)
    assert confirmed and confirmed[0].meta.get("term") == "宝宝"
    assert confirmed[0].meta.get("preferred_contexts") == ["daily_care"]

    now = datetime.now()
    store.upsert_facts([
        RelationshipFact(
            id="rel_bound_cool",
            type="boundary",
            subject="user",
            content="边界话题：前任",
            evidence=["别提前任", "不要问前任"],
            confidence=0.9,
            status="confirmed",
            meta={
                "topic": "前任",
                "strength": "strict",
                "cooldown_seconds": 3600,
                "last_hit_at": (now - timedelta(minutes=10)).isoformat(),
            },
        )
    ])
    active = store.list_active_boundaries(now_ts=now)
    assert len(active) == 1
    assert store.mark_boundary_hit("你别提前任了", hit_at=now.isoformat())
    block = store.build_active_boundary_block(limit=3)
    assert "边界冷却" in block
    assert "前任" in block


def test_relationship_store_manual_confirm_and_reject(tmp_path) -> None:
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    store.upsert_facts([
        RelationshipFact(
            id="rel_a",
            type="shared_event",
            subject="both",
            content="经常引用共同经历（上次/那次/还记得）",
            evidence=["上次我们看电影", "那次我们淋雨"],
            confidence=0.75,
            status="candidate",
        ),
        RelationshipFact(
            id="rel_b",
            type="addressing",
            subject="persona",
            content="常用称呼偏好：宝宝",
            evidence=["宝宝早", "宝宝晚安"],
            confidence=0.84,
            status="confirmed",
        ),
    ])
    assert store.confirm_fact("rel_a")
    confirmed = [x.id for x in store.list_confirmed(limit=10)]
    assert "rel_a" in confirmed and "rel_b" in confirmed

    assert store.reject_fact("rel_b", reason="manual:not fit")
    rejected = [x.id for x in store.list_rejected(limit=10)]
    assert "rel_b" in rejected


def test_relationship_store_manual_confirm_reject_supports_large_fact_set(tmp_path) -> None:
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    facts = []
    for idx in range(520):
        facts.append(
            RelationshipFact(
                id=f"rel_many_{idx}",
                type="shared_event",
                subject="both",
                content=f"共同经历片段 {idx}",
                evidence=[f"证据 {idx}"],
                confidence=0.65,
                status="candidate",
            )
        )
    store.upsert_facts(facts)

    target = "rel_many_519"
    assert store.confirm_fact(target)
    assert any(f.id == target and f.status == "confirmed" for f in store.list_facts(limit=600))

    assert store.reject_fact(target, reason="manual:rollback")
    assert any(f.id == target and f.status == "rejected" for f in store.list_facts(limit=600))


def test_relationship_store_get_fact_by_id(tmp_path) -> None:
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    store.upsert_facts([
        RelationshipFact(
            id="rel_x",
            type="shared_event",
            subject="both",
            content="共同经历 X",
            evidence=["证据 X"],
            confidence=0.7,
            status="candidate",
        )
    ])
    fact = store.get_fact_by_id("rel_x")
    assert fact is not None
    assert fact.id == "rel_x"
    assert store.get_fact_by_id("not_exists") is None


def test_relationship_extractor_rules_and_conflict_filter() -> None:
    history = ChatHistory(
        target_name="小明",
        user_name="我",
        messages=[
            ChatMessage(sender="我", content="别再提这个了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="好", timestamp=_ts(), is_target=True),
            ChatMessage(sender="我", content="不要再问这个了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="上次我们一起看电影还记得吗", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="那次我们淋雨你还记得", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="宝宝你今天咋样", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="宝宝快去吃饭", timestamp=_ts(), is_target=True),
        ],
    )
    extractor = RelationshipExtractor()

    def _validator(text: str):
        if "交流边界" in text:
            return True, "冲突测试"
        return False, ""

    facts = extractor.extract_from_history(history, conflict_validator=_validator)
    assert any(f.type == "shared_event" for f in facts)
    assert any(f.type == "addressing" and "宝宝" in f.content for f in facts)
    boundary = next(f for f in facts if f.type == "boundary")
    assert boundary.conflict_with_core
    assert boundary.status == "rejected"


def test_relationship_extractor_structured_meta() -> None:
    history = ChatHistory(
        target_name="小明",
        user_name="我",
        messages=[
            ChatMessage(sender="我", content="别再提前任了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="好", timestamp=_ts(), is_target=True),
            ChatMessage(sender="我", content="不要再问前任了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="上次我们在南京看演唱会还记得吗", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="那次我们在地铁站迷路，真的尴尬", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="宝宝早安", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="宝宝早点睡", timestamp=_ts(), is_target=True),
        ],
    )
    extractor = RelationshipExtractor()
    facts = extractor.extract_from_history(history)

    boundary = next(f for f in facts if f.type == "boundary")
    assert boundary.meta.get("topic") == "前任"
    assert boundary.meta.get("cooldown_seconds", 0) > 0
    assert boundary.meta.get("strength") in {"strict", "normal", "soft"}

    shared = next(f for f in facts if f.type == "shared_event")
    assert "event" in shared.meta
    assert shared.meta.get("time_hint") in {"上次", "那次"}

    addressing = next(f for f in facts if f.type == "addressing" and "宝宝" in f.content)
    assert addressing.meta.get("term") == "宝宝"
    assert addressing.meta.get("preferred_contexts")


def test_relationship_extractor_history_windowed_dedup() -> None:
    history = ChatHistory(
        target_name="小明",
        user_name="我",
        messages=[
            ChatMessage(sender="我", content="别再提这个了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="好", timestamp=_ts(), is_target=True),
            ChatMessage(sender="我", content="不要再问这个了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="收到", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="上次我们看电影还记得吗", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="那次我们淋雨你还记得", timestamp=_ts(), is_target=True),
            ChatMessage(sender="我", content="别再提这个了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="行", timestamp=_ts(), is_target=True),
            ChatMessage(sender="我", content="不要再问这个了", timestamp=_ts(), is_target=False),
            ChatMessage(sender="小明", content="好好", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="宝宝你到家了吗", timestamp=_ts(), is_target=True),
            ChatMessage(sender="小明", content="宝宝早点休息", timestamp=_ts(), is_target=True),
        ],
    )
    extractor = RelationshipExtractor()
    facts = extractor.extract_from_history_in_windows(
        history,
        window_size=8,
        stride=4,
    )
    boundary = [f for f in facts if f.type == "boundary"]
    assert len(boundary) == 1
    assert any(f.type == "shared_event" for f in facts)


def test_governance_build_relationship_block(tmp_path) -> None:
    governance = MemoryGovernance("小明", data_dir=tmp_path)
    store = RelationshipMemoryStore("小明", data_dir=tmp_path)
    store.upsert_facts([
        RelationshipFact(
            id="rel_3",
            type="shared_event",
            subject="both",
            content="经常引用共同经历（上次/那次/还记得）",
            evidence=["上次我们一起看电影还记得吗", "那次我们淋雨你还记得"],
            confidence=0.88,
            source="imported_history",
            status="confirmed",
        )
    ])
    governance.set_relationship_store(store)
    block = governance.build_relationship_block(limit=5)
    assert "你们关系记忆" in block
    assert "共同经历" in block


def test_governance_validate_relationship_fact(tmp_path) -> None:
    governance = MemoryGovernance("小明", data_dir=tmp_path)
    persona = type("P", (), {
        "name": "小明",
        "style_summary": "自然口语",
        "catchphrases": ["笑死"],
        "tone_markers": ["啊"],
        "self_references": ["我"],
        "topic_interests": {"游戏": 3},
        "swear_ratio": 0.0,
        "avg_length": 12.0,
    })()
    governance.bootstrap_core_from_persona(persona, force=True)

    bad = RelationshipFact(
        id="rel_bad",
        type="addressing",
        subject="persona",
        content="常用称呼偏好：机器人",
        evidence=["你是机器人"],
        confidence=0.9,
        meta={"term": "AI客服"},
    )
    verdict = governance.validate_relationship_fact(bad, persona=persona)
    assert verdict.conflict

    bad_boundary = RelationshipFact(
        id="rel_bad_boundary",
        type="boundary",
        subject="user",
        content="边界话题：前任",
        evidence=["以后都不许聊前任"],
        confidence=0.9,
        meta={"topic": "以后都不许聊前任"},
    )
    verdict_bad_boundary = governance.validate_relationship_fact(bad_boundary, persona=persona)
    assert verdict_bad_boundary.conflict

    good = RelationshipFact(
        id="rel_ok",
        type="shared_event",
        subject="both",
        content="经常引用共同经历（上次/那次/还记得）",
        evidence=["上次我们一起看电影", "那次我们淋雨"],
        confidence=0.88,
    )
    verdict_ok = governance.validate_relationship_fact(good, persona=persona)
    assert not verdict_ok.conflict


def test_emotion_relationship_trigger_medium_weight() -> None:
    state = EmotionState()
    facts = [
        RelationshipFact(
            id="rel_evt",
            type="shared_event",
            subject="both",
            content="经常引用共同经历（上次/那次/还记得）",
            status="confirmed",
        ),
    ]
    state.apply_relationship_trigger(facts, "上次我们看的电影你还记得吗")
    assert state.valence > 0.0
    assert state.arousal > 0.0

    state2 = EmotionState(valence=0.2, arousal=0.2)
    boundary = [
        RelationshipFact(
            id="rel_bound",
            type="boundary",
            subject="user",
            content="存在明确的交流边界（别提/别问/不聊）",
            status="confirmed",
        )
    ]
    state2.apply_relationship_trigger(boundary, "这个不聊了，别问了")
    assert state2.arousal < 0.2


def test_chat_engine_prompt_orders_core_relation_session() -> None:
    class _Gov:
        def build_prompt_blocks(self, **kwargs):
            return ("## CORE\n- imported", "## SESSION\n- runtime", "")

        def build_relationship_block(self, limit: int = 10):
            return "## REL\n- relation"

    e = ChatEngine.__new__(ChatEngine)
    e._persona = type("P", (), {"name": "x"})()
    e._system_prompt = "base"
    e._knowledge_store = None
    e._memory = None
    e._history = []
    e._state_lock = threading.Lock()
    e._scratchpad = Scratchpad()
    e._emotion_state = EmotionState()
    e._memory_governance = _Gov()

    text = e._build_system("最近怎么样")
    assert text.index("## CORE") < text.index("## REL")
    assert text.index("## REL") < text.index("## SESSION")


def test_chat_engine_prompt_order_puts_knowledge_after_conflict() -> None:
    class _Gov:
        def build_prompt_blocks(self, **kwargs):
            return ("## CORE\n- imported", "## SESSION\n- runtime", "## CONFLICT\n- x")

        def build_relationship_block(self, limit: int = 10):
            return "## REL\n- relation"

    class _Memory:
        def search(self, query: str, top_k: int):
            return [("user: 上次我们一起看电影", 0.2)]

    class _KB:
        def search(self, query: str, top_k: int):
            return [SimpleNamespace(summary="今天有个新热点")]

    e = ChatEngine.__new__(ChatEngine)
    e._persona = type("P", (), {"name": "x"})()
    e._system_prompt = "base"
    e._knowledge_store = _KB()
    e._memory = _Memory()
    e._memory_cache = OrderedDict()
    e._history = []
    e._state_lock = threading.Lock()
    e._scratchpad = Scratchpad()
    e._emotion_state = EmotionState()
    e._memory_governance = _Gov()

    text = e._build_system("最近怎么样")
    assert text.index("## CORE") < text.index("## REL")
    assert text.index("## REL") < text.index("## 你们过去聊到类似话题时的真实对话")
    assert text.index("## 你们过去聊到类似话题时的真实对话") < text.index("## SESSION")
    assert text.index("## SESSION") < text.index("## CONFLICT")
    assert text.index("## CONFLICT") < text.index("## 你最近关注的新闻和动态")


def test_chat_engine_prompt_includes_active_boundary_block_after_relationship() -> None:
    class _Gov:
        def build_prompt_blocks(self, **kwargs):
            return ("## CORE\n- imported", "## SESSION\n- runtime", "## CONFLICT\n- x")

        def build_relationship_block(self, limit: int = 10):
            return "## REL\n- relation"

        def build_active_boundary_block(self, limit: int = 5):
            return "## BOUNDARY\n- 前任冷却中"

    e = ChatEngine.__new__(ChatEngine)
    e._persona = type("P", (), {"name": "x"})()
    e._system_prompt = "base"
    e._knowledge_store = None
    e._memory = None
    e._history = []
    e._state_lock = threading.Lock()
    e._scratchpad = Scratchpad()
    e._emotion_state = EmotionState()
    e._memory_governance = _Gov()

    text = e._build_system("最近怎么样")
    assert text.index("## CORE") < text.index("## REL")
    assert text.index("## REL") < text.index("## BOUNDARY")
    assert text.index("## BOUNDARY") < text.index("## SESSION")
