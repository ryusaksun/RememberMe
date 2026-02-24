"""关系记忆抽取器：从聊天记录中提取关系事实。"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from typing import Callable

from google import genai
from google.genai import types

from remember_me.importers.base import ChatHistory
from remember_me.memory.relationship import RelationshipFact
from remember_me.models import MODEL_LIGHT

_BOUNDARY_RE = re.compile(r"(别再?|不要|不许|少)(?:.{0,8})(提|说|问|聊|叫|发)|这个不聊|不想聊|别问了|别提了")
_SHARED_EVENT_RE = re.compile(r"上次我们|那次我们|还记得|之前我们|那天我们")
_COMMITMENT_RE = re.compile(r"明天|待会|回头|之后|下次|我会|答应你")
_REPAIR_RE = re.compile(r"对不起|抱歉|不好意思|我错了|算我错|别生气|不吵了|和好")
_PREFERENCE_RE = re.compile(r"你喜欢|你不喜欢|你讨厌|我喜欢|我讨厌|你爱吃|你最爱")
_AFFECTION_RE = re.compile(r"爱你|想你|抱抱|亲亲|宝贝|宝宝|乖|老婆|老公")
_TENSION_RE = re.compile(r"滚|烦死|别烦|不想理|拉黑|绝交|懒得理")

_ADDRESS_TERMS = [
    "宝宝", "宝贝", "宝", "亲爱的", "哥们", "兄弟", "老师", "姐姐", "弟弟", "老婆", "老公",
]

_BASE_CONFIDENCE = {
    "relation_stage": 0.76,
    "addressing": 0.74,
    "boundary": 0.78,
    "shared_event": 0.72,
    "commitment": 0.73,
    "repair_pattern": 0.75,
    "preference": 0.70,
}

_VERIFY_PROMPT = """\
你是关系记忆提取器。给定候选关系事实与最近对话，请输出可信且简洁的结构化关系事实。

候选事实：
{candidates}

最近对话：
{messages}

仅输出 JSON 数组，每项格式：
{{
  "type": "relation_stage|addressing|boundary|shared_event|commitment|repair_pattern|preference",
  "subject": "user|persona|both",
  "content": "一句话关系事实",
  "confidence": 0.0,
  "evidence": ["证据句1", "证据句2"]
}}

规则：
- 宁缺毋滥，模糊内容不要输出
- content 不超过 28 字
- confidence 取值 0~1
- 最多输出 8 条
"""


class RelationshipExtractor:
    """规则+LLM混合关系事实抽取。"""

    def __init__(self):
        self._supported_types = set(_BASE_CONFIDENCE.keys())

    @staticmethod
    def _parse_json(raw: str) -> list[dict]:
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        try:
            data = json.loads(text)
        except Exception:
            return []
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []

    @staticmethod
    def _compact(text: str, limit: int = 36) -> str:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        return t[:limit]

    @staticmethod
    def _normalize_key(fact_type: str, content: str) -> str:
        return f"{fact_type}:{''.join(ch for ch in content.lower() if not ch.isspace())}"

    @staticmethod
    def _resolve_conflict(
        validator: Callable[[object], object] | None,
        fact: RelationshipFact,
    ) -> tuple[bool, str]:
        if not validator:
            return False, ""

        verdict = None
        called = False
        for payload in (fact, fact.content):
            try:
                verdict = validator(payload)
                called = True
                break
            except TypeError:
                continue
            except Exception:
                continue
        if not called:
            return False, ""

        if isinstance(verdict, tuple):
            if len(verdict) >= 2:
                return bool(verdict[0]), str(verdict[1] or "")
            if len(verdict) == 1:
                return bool(verdict[0]), ""
        if isinstance(verdict, bool):
            return verdict, ""
        conflict = bool(getattr(verdict, "conflict", False))
        reason = str(getattr(verdict, "reason", "") or "")
        return conflict, reason

    @staticmethod
    def _score(fact_type: str, evidence_count: int) -> float:
        base = _BASE_CONFIDENCE.get(fact_type, 0.7)
        confidence = base + max(0, evidence_count - 1) * 0.06
        if evidence_count >= 3:
            confidence += 0.03
        return max(0.0, min(0.95, confidence))

    @staticmethod
    def _to_role_texts_from_history(history: ChatHistory) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for m in history.messages:
            text = str(m.content or "").strip()
            if not text or text.startswith("["):
                continue
            rows.append(("persona" if m.is_target else "user", text))
        return rows

    @staticmethod
    def _to_role_texts_from_messages(messages: list[dict]) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for m in messages:
            role = str(m.get("role", "")).strip()
            text = str(m.get("text", "")).strip()
            if not role or not text or text.startswith("["):
                continue
            mapped = "persona" if role == "model" else "user"
            rows.append((mapped, text))
        return rows

    @staticmethod
    def _iter_windows(
        rows: list[tuple[str, str]],
        *,
        window_size: int,
        stride: int,
    ):
        if not rows:
            return
        if window_size <= 0:
            window_size = len(rows)
        if stride <= 0:
            stride = window_size

        n = len(rows)
        if n <= window_size:
            yield rows
            return

        start = 0
        while start < n:
            end = min(n, start + window_size)
            window = rows[start:end]
            if len(window) >= 8:
                yield window
            if end >= n:
                break
            start += stride

    def _extract_rule_candidates(self, rows: list[tuple[str, str]], source: str) -> list[RelationshipFact]:
        if not rows:
            return []

        candidate_map: dict[str, RelationshipFact] = {}

        def add(fact_type: str, subject: str, content: str, evidence_text: str):
            key = self._normalize_key(fact_type, content)
            fact = candidate_map.get(key)
            if fact is None:
                fact = RelationshipFact(
                    id=f"rel_{uuid.uuid4().hex[:10]}",
                    type=fact_type,
                    subject=subject,
                    content=content,
                    confidence=0.0,
                    source=source,
                    status="candidate",
                    evidence=[],
                )
                candidate_map[key] = fact
            snippet = self._compact(evidence_text)
            if snippet and snippet not in fact.evidence and len(fact.evidence) < 3:
                fact.evidence.append(snippet)

        # 规则 1: 关系阶段线索（亲近/摩擦）
        affection_hits = [text for _role, text in rows if _AFFECTION_RE.search(text)]
        tension_hits = [text for _role, text in rows if _TENSION_RE.search(text)]
        if len(affection_hits) >= 3:
            for item in affection_hits[:3]:
                add("relation_stage", "both", "关系语气偏亲近（亲昵称呼较多）", item)
        if len(tension_hits) >= 3:
            for item in tension_hits[:3]:
                add("relation_stage", "both", "关系语气存在摩擦（冲突表达较多）", item)

        # 规则 2: 称呼习惯（persona 对 user 的称呼）
        addressing_counter: Counter[str] = Counter()
        for role, text in rows:
            if role != "persona":
                continue
            lowered = text.lower()
            for term in _ADDRESS_TERMS:
                if term in lowered:
                    addressing_counter[term] += 1
        for term, cnt in addressing_counter.items():
            if cnt < 2:
                continue
            ev = [text for role, text in rows if role == "persona" and term in text.lower()][:3]
            for item in ev:
                add("addressing", "persona", f"常用称呼偏好：{term}", item)

        # 规则 3: 边界/共同经历/承诺/修复/偏好
        for role, text in rows:
            if _BOUNDARY_RE.search(text):
                add("boundary", role, "存在明确的交流边界（别提/别问/不聊）", text)
            if _SHARED_EVENT_RE.search(text):
                add("shared_event", "both", "经常引用共同经历（上次/那次/还记得）", text)
            if _COMMITMENT_RE.search(text):
                add("commitment", role, "存在未来约定与承诺表达（明天/回头/下次）", text)
            if _REPAIR_RE.search(text):
                add("repair_pattern", role, "冲突后倾向用道歉或安抚修复关系", text)
            if _PREFERENCE_RE.search(text):
                add("preference", role, "对彼此偏好有明确表达（喜欢/讨厌）", text)

        filtered: list[RelationshipFact] = []
        for fact in candidate_map.values():
            evidence_count = len(fact.evidence)
            if fact.type in {"boundary", "shared_event", "commitment", "repair_pattern", "preference"} and evidence_count < 2:
                continue
            if fact.type == "relation_stage" and evidence_count < 2:
                continue
            fact.confidence = self._score(fact.type, evidence_count)
            filtered.append(fact)

        return filtered

    def _verify_with_llm(
        self,
        client: genai.Client,
        rows: list[tuple[str, str]],
        candidates: list[RelationshipFact],
        source: str,
    ) -> list[RelationshipFact]:
        if not rows or not candidates:
            return []

        cand_json = json.dumps([
            {
                "type": c.type,
                "subject": c.subject,
                "content": c.content,
                "confidence": c.confidence,
                "evidence": c.evidence,
            }
            for c in candidates
        ], ensure_ascii=False, indent=2)
        messages_text = "\n".join(
            f"{'对方' if role == 'user' else '你'}: {text}"
            for role, text in rows[-60:]
        )
        prompt = _VERIFY_PROMPT.format(candidates=cand_json, messages=messages_text)

        try:
            response = client.models.generate_content(
                model=MODEL_LIGHT,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=900,
                ),
            )
        except Exception:
            return []

        items = self._parse_json(response.text or "")
        output: list[RelationshipFact] = []
        for item in items[:8]:
            fact_type = str(item.get("type", "")).strip()
            if fact_type not in self._supported_types:
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            subject = str(item.get("subject", "both")).strip() or "both"
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0) or 0.0)))
            evidence = [
                self._compact(str(e), limit=36)
                for e in (item.get("evidence") or [])
                if str(e).strip()
            ][:3]
            output.append(RelationshipFact(
                id=f"rel_{uuid.uuid4().hex[:10]}",
                type=fact_type,
                subject=subject,
                content=content,
                evidence=evidence,
                confidence=confidence,
                source=source,
                status="candidate",
            ))
        return output

    def _merge_rule_and_llm(
        self,
        rule_facts: list[RelationshipFact],
        llm_facts: list[RelationshipFact],
    ) -> list[RelationshipFact]:
        merged: dict[str, RelationshipFact] = {}
        for fact in rule_facts:
            merged[self._normalize_key(fact.type, fact.content)] = fact

        for fact in llm_facts:
            key = self._normalize_key(fact.type, fact.content)
            existing = merged.get(key)
            if existing is None:
                if fact.confidence >= 0.80 and fact.evidence:
                    merged[key] = fact
                continue
            existing.confidence = max(existing.confidence, fact.confidence)
            for ev in fact.evidence:
                if ev not in existing.evidence and len(existing.evidence) < 3:
                    existing.evidence.append(ev)

        return list(merged.values())

    def _extract_from_rows(
        self,
        rows: list[tuple[str, str]],
        *,
        source: str,
        client: genai.Client | None = None,
        conflict_validator: Callable[[object], object] | None = None,
    ) -> list[RelationshipFact]:
        if not rows:
            return []

        rule_facts = self._extract_rule_candidates(rows, source=source)
        llm_facts = self._verify_with_llm(client, rows, rule_facts, source=source) if client else []
        facts = self._merge_rule_and_llm(rule_facts, llm_facts)

        for fact in facts:
            conflict, reason = self._resolve_conflict(conflict_validator, fact)
            fact.conflict_with_core = conflict
            fact.conflict_reason = reason
            if conflict:
                fact.status = "rejected"
        return facts

    def extract_from_messages(
        self,
        messages: list[dict],
        *,
        client: genai.Client | None = None,
        source: str = "runtime_session",
        conflict_validator: Callable[[object], object] | None = None,
    ) -> list[RelationshipFact]:
        rows = self._to_role_texts_from_messages(messages)
        return self._extract_from_rows(
            rows, source=source, client=client, conflict_validator=conflict_validator,
        )

    def extract_from_history(
        self,
        history: ChatHistory,
        *,
        client: genai.Client | None = None,
        conflict_validator: Callable[[object], object] | None = None,
    ) -> list[RelationshipFact]:
        rows = self._to_role_texts_from_history(history)
        return self._extract_from_rows(
            rows,
            source="imported_history",
            client=client,
            conflict_validator=conflict_validator,
        )

    def extract_from_history_in_windows(
        self,
        history: ChatHistory,
        *,
        client: genai.Client | None = None,
        conflict_validator: Callable[[object], object] | None = None,
        window_size: int = 120,
        stride: int = 80,
    ) -> list[RelationshipFact]:
        rows = self._to_role_texts_from_history(history)
        if not rows:
            return []

        merged: dict[str, RelationshipFact] = {}
        for window_rows in self._iter_windows(rows, window_size=window_size, stride=stride):
            facts = self._extract_from_rows(
                window_rows,
                source="imported_history",
                client=client,
                conflict_validator=conflict_validator,
            )
            for fact in facts:
                key = self._normalize_key(fact.type, fact.content)
                existing = merged.get(key)
                if existing is None:
                    merged[key] = fact
                    continue
                existing.confidence = max(existing.confidence, fact.confidence)
                for ev in fact.evidence:
                    if ev not in existing.evidence and len(existing.evidence) < 3:
                        existing.evidence.append(ev)
                if fact.conflict_with_core:
                    existing.conflict_with_core = True
                    existing.status = "rejected"
                    existing.conflict_reason = fact.conflict_reason or existing.conflict_reason

        return list(merged.values())
