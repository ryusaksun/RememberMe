"""关系记忆存储：记录你和对方之间的关系事实。"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_iso(ts: str) -> datetime:
    if not ts:
        return datetime.fromtimestamp(0)
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return datetime.fromtimestamp(0)


def _normalize_text(text: str) -> str:
    return "".join(ch for ch in (text or "").lower().strip() if not ch.isspace())


_STATUS_RANK = {
    "candidate": 1,
    "confirmed": 2,
    "rejected": 3,
}

_TYPE_LABEL = {
    "relation_stage": "关系阶段",
    "addressing": "称呼习惯",
    "boundary": "互动边界",
    "shared_event": "共同经历",
    "commitment": "承诺与跟进",
    "repair_pattern": "冲突修复",
    "preference": "偏好线索",
}
RELATION_TYPES = tuple(_TYPE_LABEL.keys())


@dataclass
class RelationshipFact:
    id: str
    type: str
    subject: str
    content: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "runtime_session"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    status: str = "candidate"
    conflict_with_core: bool = False
    conflict_reason: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "RelationshipFact":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in data.items() if k in valid}
        return cls(**payload)


class RelationshipMemoryStore:
    """关系事实持久化存储。"""

    def __init__(self, persona_name: str, data_dir: str | Path = Path("data")):
        self._name = persona_name
        self._data_dir = Path(data_dir)
        self._path = self._data_dir / "relationships" / f"{persona_name}.json"
        self._facts: list[RelationshipFact] = []
        self.load()

    def load(self):
        if not self._path.exists():
            self._facts = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._facts = [
                RelationshipFact.from_dict(item)
                for item in (raw.get("facts") or [])
                if isinstance(item, dict)
            ]
        except Exception:
            self._facts = []

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "persona_name": self._name,
            "updated_at": _now_iso(),
            "facts": [asdict(f) for f in self._facts],
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _fact_key(fact_type: str, content: str) -> str:
        return f"{fact_type}:{_normalize_text(content)}"

    @staticmethod
    def _sort_rows(rows: Iterable[RelationshipFact]) -> list[RelationshipFact]:
        sorted_rows = list(rows)
        sorted_rows.sort(
            key=lambda x: (x.confidence, _parse_iso(x.updated_at)),
            reverse=True,
        )
        return sorted_rows

    @staticmethod
    def _merge_evidence(old: list[str], new: list[str], limit: int = 3) -> list[str]:
        merged: list[str] = []
        for item in [*(old or []), *(new or [])]:
            text = str(item).strip()
            if not text:
                continue
            if text in merged:
                continue
            merged.append(text)
            if len(merged) >= limit:
                break
        return merged

    def upsert_facts(self, facts: list[RelationshipFact]) -> int:
        if not facts:
            return 0
        by_key = {
            self._fact_key(f.type, f.content): idx
            for idx, f in enumerate(self._facts)
        }
        changed = 0
        now = _now_iso()

        for incoming in facts:
            incoming.content = str(incoming.content or "").strip()
            if not incoming.content:
                continue
            incoming.type = str(incoming.type or "preference").strip() or "preference"
            incoming.subject = str(incoming.subject or "both").strip() or "both"
            incoming.id = incoming.id or f"rel_{uuid.uuid4().hex[:10]}"
            incoming.status = incoming.status if incoming.status in _STATUS_RANK else "candidate"
            incoming.confidence = max(0.0, min(1.0, float(incoming.confidence or 0.0)))
            incoming.updated_at = now
            incoming.created_at = incoming.created_at or now
            incoming.evidence = self._merge_evidence([], incoming.evidence)

            key = self._fact_key(incoming.type, incoming.content)
            idx = by_key.get(key)
            if idx is None:
                self._facts.append(incoming)
                by_key[key] = len(self._facts) - 1
                changed += 1
                continue

            current = self._facts[idx]
            merged_evidence = self._merge_evidence(current.evidence, incoming.evidence)
            better_status = _STATUS_RANK.get(incoming.status, 0) > _STATUS_RANK.get(current.status, 0)
            better_conf = incoming.confidence > current.confidence

            if better_status or better_conf or merged_evidence != current.evidence:
                current.subject = incoming.subject or current.subject
                current.evidence = merged_evidence
                current.confidence = max(current.confidence, incoming.confidence)
                if better_status:
                    current.status = incoming.status
                if incoming.conflict_with_core:
                    current.conflict_with_core = True
                    current.conflict_reason = incoming.conflict_reason or current.conflict_reason
                current.updated_at = now
                changed += 1

        if changed:
            self.save()
        return changed

    def list_facts(
        self,
        *,
        fact_type: str | None = None,
        statuses: set[str] | None = None,
        include_conflict: bool = True,
        limit: int = 20,
    ) -> list[RelationshipFact]:
        wanted_statuses = statuses or {"candidate", "confirmed", "rejected"}
        rows = [
            f for f in self._facts
            if f.status in wanted_statuses and (not fact_type or f.type == fact_type)
        ]
        if not include_conflict:
            rows = [f for f in rows if not f.conflict_with_core]
        return self._sort_rows(rows)[: max(0, limit)]

    def list_confirmed(self, limit: int = 20) -> list[RelationshipFact]:
        return self.list_facts(
            statuses={"confirmed"}, include_conflict=False, limit=limit,
        )

    def list_candidates(self, limit: int = 20) -> list[RelationshipFact]:
        return self.list_facts(
            statuses={"candidate"}, include_conflict=False, limit=limit,
        )

    def list_rejected(self, limit: int = 20) -> list[RelationshipFact]:
        return self.list_facts(
            statuses={"rejected"},
            include_conflict=True,
            limit=limit,
        )

    def promote_candidates(self, min_confidence: float = 0.78, min_evidence: int = 2) -> int:
        changed = 0
        now = _now_iso()
        for fact in self._facts:
            if fact.status != "candidate":
                continue
            if fact.conflict_with_core:
                continue
            if fact.confidence < min_confidence:
                continue
            if len(fact.evidence or []) < min_evidence:
                continue
            fact.status = "confirmed"
            fact.updated_at = now
            changed += 1
        if changed:
            self.save()
        return changed

    def build_prompt_block(self, limit: int = 10) -> str:
        rows = self.list_confirmed(limit=limit)
        if not rows:
            return ""
        lines = ["## 你们关系记忆（基于真实历史，优先于短期上下文）"]
        for fact in rows:
            label = _TYPE_LABEL.get(fact.type, fact.type)
            line = f"- [{label}] {fact.content}"
            if fact.evidence:
                line += f"（证据：{fact.evidence[0][:30]}）"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _resolve_ref(ref: str | int, rows: list[RelationshipFact]) -> RelationshipFact | None:
        if isinstance(ref, int):
            idx = ref - 1
            if 0 <= idx < len(rows):
                return rows[idx]
            return None
        target = str(ref or "").strip()
        if not target:
            return None
        for fact in rows:
            if fact.id == target:
                return fact
        return None

    def confirm_fact(self, ref: str | int) -> bool:
        rows = self.list_facts(limit=500)
        target = self._resolve_ref(ref, rows)
        if not target:
            return False
        if target.conflict_with_core:
            return False
        if target.status == "confirmed":
            return True
        target.status = "confirmed"
        target.updated_at = _now_iso()
        self.save()
        return True

    def reject_fact(self, ref: str | int, reason: str = "manual_reject") -> bool:
        rows = self.list_facts(limit=500)
        target = self._resolve_ref(ref, rows)
        if not target:
            return False
        target.status = "rejected"
        if reason:
            target.conflict_reason = str(reason)
        target.updated_at = _now_iso()
        self.save()
        return True
