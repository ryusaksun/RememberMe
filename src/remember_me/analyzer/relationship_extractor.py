"""关系记忆抽取器：从聊天记录中提取关系事实。"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
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
_BOUNDARY_TOPIC_RE = re.compile(
    r"(?:别再?|不要|不许|少)(?:再)?(?:提|说|问|聊|叫|发)(?:我|这个|这件事|这事|一下|了)?(?P<topic>[^，。！？\n]{1,12})"
)
_EVENT_SLOT_RE = re.compile(
    r"(?:上次|那次|之前|那天)(?:我们)?(?P<event>[^，。！？\n]{2,18})"
)

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
  "evidence": ["证据句1", "证据句2"],
  "meta": {}
}}

规则：
- 宁缺毋滥，模糊内容不要输出
- content 不超过 28 字
- confidence 取值 0~1
- meta 仅保留有证据的结构化字段，不确定就留空对象
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
    def _clean_boundary_topic(topic: str) -> str:
        cleaned = re.sub(r"[，。！？!?\s]+", "", str(topic or "")).strip()
        cleaned = re.sub(r"(这个|这件事|这事|的话题|一下|了)$", "", cleaned)
        if cleaned in {"这个", "这事", "话题", "事情"}:
            return ""
        if len(cleaned) < 2:
            return ""
        return cleaned[:10]

    @staticmethod
    def _boundary_meta(text: str) -> dict:
        raw = str(text or "")
        topic = ""
        match = _BOUNDARY_TOPIC_RE.search(raw)
        if match:
            topic = RelationshipExtractor._clean_boundary_topic(match.group("topic") or "")

        strict = bool(re.search(r"(不许|必须|别再|不要再|永远不要)", raw))
        soft = bool(re.search(r"(先不聊|改天再聊|回头再说)", raw))
        strength = "strict" if strict else "soft" if soft else "normal"
        cooldown = 6 * 3600 if strict else 90 * 60 if soft else 3 * 3600
        meta: dict = {"strength": strength, "cooldown_seconds": cooldown}
        if topic:
            meta["topic"] = topic
        return meta

    @staticmethod
    def _guess_addressing_context(text: str) -> str:
        raw = str(text or "")
        if re.search(r"(早安|晚安|睡|起床|吃饭|到家|下班)", raw):
            return "daily_care"
        if re.search(r"(别生气|对不起|抱歉|和好|吵架)", raw):
            return "repair"
        if re.search(r"(爱你|想你|抱抱|亲亲)", raw):
            return "affection"
        if re.search(r"(考试|面试|工作|加班|开会)", raw):
            return "support"
        return "general"

    @staticmethod
    def _event_meta(text: str) -> dict:
        raw = str(text or "").strip()
        if not raw:
            return {}

        event = ""
        event_match = _EVENT_SLOT_RE.search(raw)
        if event_match:
            event = re.sub(r"[，。！？!?\s]+", " ", event_match.group("event") or "").strip()
        if not event:
            event = raw
        event = event[:22]

        time_hint = ""
        for token in ("上次", "那次", "那天", "之前", "昨天", "去年", "刚刚"):
            if token in raw:
                time_hint = token
                break

        place_hint = ""
        place_match = re.search(r"(在|去|到)([^，。！？\s]{1,8})", raw)
        if place_match:
            place_hint = f"{place_match.group(1)}{place_match.group(2)}"

        emotion_hint = ""
        for token in ("开心", "难过", "尴尬", "生气", "紧张", "激动"):
            if token in raw:
                emotion_hint = token
                break

        meta = {"event": event}
        if time_hint:
            meta["time_hint"] = time_hint
        if place_hint:
            meta["place_hint"] = place_hint
        if emotion_hint:
            meta["emotion_hint"] = emotion_hint
        return meta

    @staticmethod
    def _merge_meta(base: dict | None, incoming: dict | None) -> dict:
        out = dict(base or {})
        for key, val in (incoming or {}).items():
            if isinstance(out.get(key), dict) and isinstance(val, dict):
                child = dict(out[key])
                child.update(val)
                out[key] = child
                continue
            if isinstance(out.get(key), list) and isinstance(val, list):
                cur = list(out[key])
                for item in val:
                    if item in cur:
                        continue
                    cur.append(item)
                    if len(cur) >= 8:
                        break
                out[key] = cur
                continue
            out[key] = val
        return out

    @staticmethod
    def _sanitize_meta(fact_type: str, raw_meta: object) -> dict:
        if not isinstance(raw_meta, dict):
            return {}

        if fact_type == "boundary":
            out = {}
            topic = str(raw_meta.get("topic", "") or "").strip()
            if topic:
                out["topic"] = topic[:10]
            strength = str(raw_meta.get("strength", "") or "").strip().lower()
            if strength in {"strict", "normal", "soft"}:
                out["strength"] = strength
            try:
                cooldown = int(float(raw_meta.get("cooldown_seconds", 0) or 0))
            except (TypeError, ValueError):
                cooldown = 0
            if cooldown > 0:
                out["cooldown_seconds"] = min(cooldown, 24 * 3600)
            return out

        if fact_type == "shared_event":
            out = {}
            for key in ("event", "time_hint", "place_hint", "emotion_hint"):
                val = str(raw_meta.get(key, "") or "").strip()
                if val:
                    out[key] = val[:24]
            return out

        if fact_type == "addressing":
            out = {}
            term = str(raw_meta.get("term", "") or "").strip()
            if term:
                out["term"] = term[:8]
            contexts = raw_meta.get("preferred_contexts")
            if isinstance(contexts, list):
                cleaned = [str(x).strip() for x in contexts if str(x).strip()]
                if cleaned:
                    out["preferred_contexts"] = cleaned[:4]
            context_counts = raw_meta.get("context_counts")
            if isinstance(context_counts, dict):
                mapped = {}
                for key, val in context_counts.items():
                    k = str(key or "").strip()
                    if not k:
                        continue
                    try:
                        mapped[k] = int(val)
                    except (TypeError, ValueError):
                        continue
                if mapped:
                    out["context_counts"] = mapped
            return out

        return {}

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

        def add(
            fact_type: str,
            subject: str,
            content: str,
            evidence_text: str,
            meta: dict | None = None,
        ):
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
                    meta={},
                )
                candidate_map[key] = fact
            snippet = self._compact(evidence_text)
            if snippet and snippet not in fact.evidence and len(fact.evidence) < 3:
                fact.evidence.append(snippet)
            if meta:
                fact.meta = self._merge_meta(fact.meta, self._sanitize_meta(fact_type, meta))

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
        addressing_context_counter: dict[str, Counter[str]] = defaultdict(Counter)
        for role, text in rows:
            if role != "persona":
                continue
            lowered = text.lower()
            ctx = self._guess_addressing_context(text)
            for term in _ADDRESS_TERMS:
                if term in lowered:
                    addressing_counter[term] += 1
                    addressing_context_counter[term][ctx] += 1
        for term, cnt in addressing_counter.items():
            if cnt < 2:
                continue
            ev = [text for role, text in rows if role == "persona" and term in text.lower()][:3]
            context_counts = dict(addressing_context_counter.get(term, Counter()))
            preferred_contexts = [
                key for key, _val in addressing_context_counter.get(term, Counter()).most_common(3)
            ]
            for item in ev:
                add(
                    "addressing",
                    "persona",
                    f"常用称呼偏好：{term}",
                    item,
                    meta={
                        "term": term,
                        "context_counts": context_counts,
                        "preferred_contexts": preferred_contexts,
                    },
                )

        # 规则 3: 边界/共同经历/承诺/修复/偏好
        for role, text in rows:
            if _BOUNDARY_RE.search(text):
                boundary_meta = self._boundary_meta(text)
                topic = str(boundary_meta.get("topic", "") or "").strip()
                content = f"边界话题：{topic}" if topic else "存在明确的交流边界（别提/别问/不聊）"
                add("boundary", role, content, text, meta=boundary_meta)
            if _SHARED_EVENT_RE.search(text):
                add(
                    "shared_event",
                    "both",
                    "经常引用共同经历（上次/那次/还记得）",
                    text,
                    meta=self._event_meta(text),
                )
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
                "meta": c.meta,
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
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0) or 0.0)))
            except (TypeError, ValueError):
                confidence = 0.0
            evidence = [
                self._compact(str(e), limit=36)
                for e in (item.get("evidence") or [])
                if str(e).strip()
            ][:3]
            meta = self._sanitize_meta(fact_type, item.get("meta"))
            output.append(RelationshipFact(
                id=f"rel_{uuid.uuid4().hex[:10]}",
                type=fact_type,
                subject=subject,
                content=content,
                evidence=evidence,
                confidence=confidence,
                source=source,
                status="candidate",
                meta=meta,
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
            existing.meta = self._merge_meta(existing.meta, fact.meta)

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
                existing.meta = self._merge_meta(existing.meta, fact.meta)
                if fact.conflict_with_core:
                    existing.conflict_with_core = True
                    existing.status = "rejected"
                    existing.conflict_reason = fact.conflict_reason or existing.conflict_reason

        return list(merged.values())
