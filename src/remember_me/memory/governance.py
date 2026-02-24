"""记忆治理层：导入历史为唯一核心真源，运行时信息可沉淀但不得覆盖核心。"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from remember_me.analyzer.persona import Persona
from remember_me.memory.relationship import RelationshipMemoryStore

logger = logging.getLogger(__name__)

_TRIVIAL_SESSION_RE = re.compile(
    r"^(嗯|好|哦|啊|哈+|呵呵|ok|行|对|是|收到|知道了|好的)$",
    re.I,
)
_AI_IDENTITY_RE = re.compile(r"(你是ai|你是人工智能|你是机器人|你只是程序|你不是真人)")
_OVERRIDE_CUE_RE = re.compile(r"(以后|从现在开始|别再|不要再|改成|改为|你其实|你不是|你应该)")
_NO_SWEAR_RE = re.compile(r"(别骂人|不要骂人|不许骂人|别说脏话|不要说脏话)")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _normalize_tokens(items: Iterable[str]) -> set[str]:
    tokens: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        tokens.add(text)
        for seg in re.split(r"[\s,，。！？!?:：;；、/]+", text):
            seg = seg.strip()
            if len(seg) >= 2:
                tokens.add(seg)
    return tokens


@dataclass
class ConflictResult:
    conflict: bool
    reason: str = ""


@dataclass
class MemoryRecord:
    id: str
    type: str  # core | session
    text: str
    source_type: str  # imported_history | runtime_session
    locked: bool
    conflict_with_history: bool = False
    conflict_reason: str = ""
    confidence: float = 1.0
    tags: list[str] = field(default_factory=list)
    ttl_seconds: int | None = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    expires_at: str = ""

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        expires_at = _parse_iso(self.expires_at)
        if not expires_at:
            return False
        return datetime.now() >= expires_at

    @classmethod
    def from_dict(cls, data: dict) -> MemoryRecord:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        payload = {k: v for k, v in data.items() if k in valid}
        return cls(**payload)


class MemoryGovernance:
    """核心记忆只读；运行时记忆可累积，但不允许覆盖核心。"""

    def __init__(self, persona_name: str, data_dir: str | Path = Path("data")):
        self._name = persona_name
        self._data_dir = Path(data_dir)
        self._path = self._data_dir / "memories" / f"{persona_name}.json"
        self._records: list[MemoryRecord] = []
        self._core_profile_snapshot: dict = {}
        self._relationship_store: RelationshipMemoryStore | None = None
        self.load()

    def load(self):
        if not self._path.exists():
            self._records = []
            self._core_profile_snapshot = {}
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._core_profile_snapshot = raw.get("core_profile_snapshot", {}) or {}
            self._records = [
                MemoryRecord.from_dict(item)
                for item in (raw.get("records") or [])
                if isinstance(item, dict)
            ]
        except Exception as e:
            logger.warning("加载记忆治理数据失败: %s", e)
            self._records = []
            self._core_profile_snapshot = {}

    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "persona_name": self._name,
            "updated_at": _now_iso(),
            "core_profile_snapshot": self._core_profile_snapshot,
            "records": [asdict(r) for r in self._records],
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _build_snapshot_from_persona(persona: Persona) -> dict:
        topics = sorted(
            (getattr(persona, "topic_interests", {}) or {}).items(),
            key=lambda x: -x[1],
        )[:8]
        return {
            "style_summary": persona.style_summary or "",
            "catchphrases": list((persona.catchphrases or [])[:12]),
            "tone_markers": list((persona.tone_markers or [])[:10]),
            "self_references": list((persona.self_references or [])[:8]),
            "topic_interests": {k: v for k, v in topics},
            "swear_ratio": float(getattr(persona, "swear_ratio", 0.0) or 0.0),
            "avg_length": float(getattr(persona, "avg_length", 0.0) or 0.0),
        }

    def _build_core_records(self, snapshot: dict) -> list[MemoryRecord]:
        now = _now_iso()
        rows: list[str] = []
        style = snapshot.get("style_summary", "").strip()
        if style:
            rows.append(f"说话风格基线：{style}")
        catch = snapshot.get("catchphrases", [])
        if catch:
            rows.append(f"高频口头禅：{'、'.join(catch[:8])}")
        tones = snapshot.get("tone_markers", [])
        if tones:
            rows.append(f"常见语气词：{'、'.join(tones[:6])}")
        self_refs = snapshot.get("self_references", [])
        if self_refs:
            rows.append(f"常见自称方式：{'、'.join(self_refs[:4])}")
        topics = list((snapshot.get("topic_interests", {}) or {}).keys())
        if topics:
            rows.append(f"长期话题偏好：{'、'.join(topics[:6])}")

        records: list[MemoryRecord] = []
        for text in rows:
            records.append(
                MemoryRecord(
                    id=f"core_{uuid.uuid4().hex[:10]}",
                    type="core",
                    text=text,
                    source_type="imported_history",
                    locked=True,
                    confidence=1.0,
                    tags=["core", "imported"],
                    ttl_seconds=None,
                    created_at=now,
                    updated_at=now,
                    expires_at="",
                )
            )
        return records

    def bootstrap_core_from_persona(self, persona: Persona, force: bool = False):
        snapshot = self._build_snapshot_from_persona(persona)
        existing_runtime = [
            r for r in self._records
            if r.source_type == "runtime_session" and not r.is_expired()
        ]
        existing_core = [r for r in self._records if r.source_type == "imported_history"]
        if existing_core and not force:
            if not self._core_profile_snapshot:
                self._core_profile_snapshot = snapshot
                self.save()
            return

        self._core_profile_snapshot = snapshot
        self._records = self._build_core_records(snapshot)
        if not force:
            self._records.extend(existing_runtime)
        self.save()

    def ensure_core_from_persona(self, persona: Persona):
        self.load()
        has_core = any(r.source_type == "imported_history" for r in self._records)
        if not has_core:
            self.bootstrap_core_from_persona(persona, force=False)
        elif not self._core_profile_snapshot:
            self._core_profile_snapshot = self._build_snapshot_from_persona(persona)
            self.save()

    def cleanup_expired(self):
        before = len(self._records)
        self._records = [
            r for r in self._records
            if not (r.source_type == "runtime_session" and r.is_expired())
        ]
        if len(self._records) != before:
            self.save()

    def _core_tokens(self) -> set[str]:
        snapshot = self._core_profile_snapshot or {}
        tokens: set[str] = set()
        tokens |= _normalize_tokens(snapshot.get("catchphrases", []))
        tokens |= _normalize_tokens(snapshot.get("tone_markers", []))
        tokens |= _normalize_tokens(snapshot.get("self_references", []))
        tokens |= _normalize_tokens((snapshot.get("topic_interests", {}) or {}).keys())
        style = snapshot.get("style_summary", "")
        if style:
            tokens |= _normalize_tokens([style])
        return {t for t in tokens if t}

    def validate_against_imported_history(
        self,
        text: str,
        persona: Persona | None = None,
    ) -> ConflictResult:
        msg = (text or "").strip()
        if not msg:
            return ConflictResult(False, "")

        if persona and not self._core_profile_snapshot:
            self._core_profile_snapshot = self._build_snapshot_from_persona(persona)

        if _AI_IDENTITY_RE.search(msg):
            return ConflictResult(True, "试图重写身份设定")

        if _NO_SWEAR_RE.search(msg):
            swear_ratio = float((self._core_profile_snapshot or {}).get("swear_ratio", 0.0) or 0.0)
            if swear_ratio > 0.005:
                return ConflictResult(True, "试图改写导入历史中的语气习惯")

        if _OVERRIDE_CUE_RE.search(msg):
            tokens = self._core_tokens()
            if any(tok and tok in msg for tok in tokens):
                return ConflictResult(True, "试图覆盖导入历史的核心表达")

        return ConflictResult(False, "")

    def list_core_records(self) -> list[MemoryRecord]:
        return [
            r for r in self._records
            if r.source_type == "imported_history" and r.type == "core"
        ]

    def list_session_records(
        self,
        tag: str | None = None,
        include_conflict: bool = True,
    ) -> list[MemoryRecord]:
        self.cleanup_expired()
        records = [
            r for r in self._records
            if r.source_type == "runtime_session" and r.type == "session"
        ]
        if tag:
            records = [r for r in records if tag in (r.tags or [])]
        if not include_conflict:
            records = [r for r in records if not r.conflict_with_history]
        records.sort(key=lambda x: x.updated_at, reverse=True)
        return records

    def add_session_record(
        self,
        text: str,
        *,
        persona: Persona | None = None,
        ttl_seconds: int | None = None,
        tags: list[str] | None = None,
        confidence: float = 0.55,
        persist: bool = True,
    ) -> MemoryRecord | None:
        message = (text or "").strip()
        if not message or len(message) < 4 or _TRIVIAL_SESSION_RE.match(message):
            return None

        # 去重：避免同一句短时间重复写入
        recent = self.list_session_records(include_conflict=True)[:20]
        if any(r.text == message for r in recent):
            return None

        verdict = self.validate_against_imported_history(message, persona=persona)
        now = datetime.now()
        expires_at = ""
        if ttl_seconds and ttl_seconds > 0:
            expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat()

        record = MemoryRecord(
            id=f"sess_{uuid.uuid4().hex[:10]}",
            type="session",
            text=message,
            source_type="runtime_session",
            locked=False,
            conflict_with_history=verdict.conflict,
            conflict_reason=verdict.reason,
            confidence=max(0.0, min(1.0, confidence)),
            tags=list(tags or ["runtime"]),
            ttl_seconds=ttl_seconds,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            expires_at=expires_at,
        )
        self._records.append(record)
        if persist:
            self.save()
        return record

    def filter_messages_for_long_term(
        self,
        messages: list[dict],
        persona: Persona | None = None,
    ) -> list[dict]:
        """筛选可进入长期向量记忆的消息，避免覆盖导入核心人格。"""
        filtered: list[dict] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip()
            text = str(msg.get("text", "")).strip()
            if not role or not text:
                continue
            if text.startswith("[sticker:"):
                continue
            if _TRIVIAL_SESSION_RE.match(text):
                continue

            if role == "user":
                verdict = self.validate_against_imported_history(text, persona=persona)
                if verdict.conflict:
                    continue
            else:
                # 模型回复里的自我身份越界信息不进入长期记忆，防止反向污染。
                if _AI_IDENTITY_RE.search(text):
                    continue
            filtered.append({"role": role, "text": text})
        return filtered

    def replace_manual_notes(self, notes: list[str], persona: Persona | None = None):
        self.cleanup_expired()
        keep = []
        for record in self._records:
            if record.source_type == "runtime_session" and "manual" in (record.tags or []):
                continue
            keep.append(record)
        self._records = keep

        for text in notes:
            self.add_session_record(
                text,
                persona=persona,
                ttl_seconds=None,
                tags=["manual", "session_note"],
                confidence=0.75,
                persist=False,
            )
        self.save()

    def list_manual_notes(self) -> list[MemoryRecord]:
        return self.list_session_records(tag="manual", include_conflict=True)

    def delete_manual_note_by_index(self, index: int) -> bool:
        notes = self.list_manual_notes()
        if index < 0 or index >= len(notes):
            return False
        target_id = notes[index].id
        before = len(self._records)
        self._records = [r for r in self._records if r.id != target_id]
        changed = len(self._records) != before
        if changed:
            self.save()
        return changed

    def build_prompt_blocks(
        self,
        *,
        core_limit: int = 6,
        session_limit: int = 5,
        conflict_limit: int = 2,
    ) -> tuple[str, str, str]:
        self.cleanup_expired()
        core = self.list_core_records()[:core_limit]
        session = self.list_session_records(include_conflict=False)[:session_limit]
        conflicts = [
            r for r in self.list_session_records(include_conflict=True)
            if r.conflict_with_history
        ][:conflict_limit]

        core_block = ""
        if core:
            lines = ["## 导入聊天记录的核心事实（最高优先级，不可违背）"]
            for record in core:
                lines.append(f"- {record.text}")
            core_block = "\n".join(lines)

        session_block = ""
        if session:
            lines = ["## 近期会话上下文（低优先级，可持续累积）"]
            for record in session:
                lines.append(f"- {record.text}")
            session_block = "\n".join(lines)

        conflict_block = ""
        if conflicts:
            lines = ["## 可能与导入历史冲突的新信息（仅参考，不得覆盖核心设定）"]
            for record in conflicts:
                reason = f"（{record.conflict_reason}）" if record.conflict_reason else ""
                lines.append(f"- {record.text}{reason}")
            conflict_block = "\n".join(lines)

        return core_block, session_block, conflict_block

    def set_relationship_store(self, store: RelationshipMemoryStore | None):
        self._relationship_store = store

    def build_relationship_block(self, limit: int = 10) -> str:
        if not self._relationship_store:
            return ""
        try:
            return self._relationship_store.build_prompt_block(limit=limit)
        except Exception as e:
            logger.warning("关系记忆块构建失败: %s", e)
            return ""
