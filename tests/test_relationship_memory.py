from __future__ import annotations

import threading
from datetime import datetime

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
