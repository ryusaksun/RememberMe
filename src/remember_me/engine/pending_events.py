"""待跟进事件追踪器 - 从对话中提取值得跟进的事件，定时触发追问。"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

from remember_me.models import MODEL_LIGHT

_MODEL = MODEL_LIGHT

_EXTRACT_PROMPT = """\
你是一个对话分析器。分析下面的对话片段，提取值得过一段时间后主动追问的事件。

只提取这类事件：
- 出行计划（"要去XX"、"坐车去XX"、"出发了"）→ 到了之后追问
- 待办/任务（"要去面试"、"明天考试"、"下午开会"）→ 结束后追问
- 身体状况（"感冒了"、"不舒服"、"受伤了"）→ 过一段时间关心
- 重要决定/变化（"要搬家"、"换工作"、"分手了"）→ 过几天关心进展
- 特定约定/事件（"周末聚餐"、"晚上打游戏"）→ 事后追问

不要提取：
- 闲聊、吐槽、玩笑、观点表达
- 已经完成的事（"今天吃了火锅"，除非有后续值得关心）
- 太模糊的信息（"最近有点忙"）
- 对方已经在这次对话中回答过结果的事

对话:
{messages}

如果有值得追问的事件，输出 JSON 数组:
[
  {{
    "event": "简短描述（如：对方要坐车去杭州）",
    "context": "用户原话摘录",
    "followup_hint": "追问方向（如：问到了没、顺不顺利）",
    "followup_after_minutes": 预估多少分钟后追问合适（整数，考虑事件性质）
  }}
]

时间估算参考：
- 短途出行（坐车1-3小时）→ 60-180 分钟
- 长途出行（飞机/火车）→ 180-480 分钟
- 考试/面试/开会 → 120-240 分钟
- 身体不适 → 360-720 分钟（半天到一天）
- 重要决定 → 1440-4320 分钟（1-3天）

如果没有值得追问的事件，输出空数组 []
只输出 JSON，不要其他文字。"""

# 每个事件最长存活时间（小时），超过后自动过期
_MAX_EVENT_AGE_HOURS = 48


@dataclass
class PendingEvent:
    id: str
    event: str
    context: str
    followup_hint: str
    followup_after: str  # ISO timestamp
    extracted_at: str  # ISO timestamp
    status: str = "pending"  # pending / done / expired


class PendingEventTracker:
    """从对话中提取待跟进事件，持久化存储，定时检查到期事件。"""

    def __init__(self, persona_name: str, data_dir: Path = Path("data")):
        self._path = data_dir / "pending_events" / f"{persona_name}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[PendingEvent] = self._load()
        self._embedding_fn = None

    def _load(self) -> list[PendingEvent]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [PendingEvent(**e) for e in data]
        except Exception as e:
            logger.warning("加载 pending_events 失败: %s", e)
            return []

    def _save(self):
        self._path.write_text(
            json.dumps([asdict(e) for e in self._events], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @property
    def pending_count(self) -> int:
        return sum(1 for e in self._events if e.status == "pending")

    def extract_events(
        self, client: genai.Client, recent_messages: list[dict]
    ) -> list[PendingEvent]:
        """用 LLM 从近期对话中提取待跟进事件。"""
        if not recent_messages:
            return []

        messages_text = "\n".join(
            f"{'对方' if m['role'] == 'user' else '你'}: {m['text']}"
            for m in recent_messages
        )

        prompt = _EXTRACT_PROMPT.format(messages=messages_text)

        try:
            response = client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            raw = (response.text or "").strip()
            # 处理 markdown 代码块
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                raw = raw.rsplit("```", 1)[0]
            items = json.loads(raw)
        except Exception as e:
            logger.debug("事件提取失败: %s", e)
            return []

        if not isinstance(items, list):
            return []

        now = datetime.now()
        new_events = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                minutes = int(item.get("followup_after_minutes", 60))
            except (TypeError, ValueError):
                minutes = 60
            followup_at = now + timedelta(minutes=max(minutes, 10))

            # 去重：如果已有类似事件（关键词重叠），跳过
            event_text = str(item.get("event", "") or "").strip()
            context_text = str(item.get("context", "") or "")
            if not event_text:
                continue
            if self._is_duplicate(event_text, context_text):
                continue

            ev = PendingEvent(
                id=uuid.uuid4().hex[:8],
                event=event_text,
                context=context_text,
                followup_hint=item.get("followup_hint", ""),
                followup_after=followup_at.isoformat(),
                extracted_at=now.isoformat(),
            )
            new_events.append(ev)
            self._events.append(ev)

        if new_events:
            self._evict_old()
            self._save()
            logger.info(
                "提取到 %d 个待跟进事件: %s",
                len(new_events),
                [e.event for e in new_events],
            )

        return new_events

    @staticmethod
    def _char_overlap_ratio(a: str, b: str) -> float:
        a_set = set(a)
        b_set = set(b)
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / max(len(a_set), len(b_set))

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        dot = sum(x * y for x, y in zip(v1, v2))
        n1 = math.sqrt(sum(x * x for x in v1))
        n2 = math.sqrt(sum(y * y for y in v2))
        if n1 == 0 or n2 == 0:
            return 0.0
        return dot / (n1 * n2)

    def _semantic_similarity(self, a: str, b: str) -> float:
        try:
            if self._embedding_fn is None:
                from remember_me.memory.store import _get_embedding_function
                self._embedding_fn = _get_embedding_function()
            emb = self._embedding_fn([a, b])
            if not emb or len(emb) != 2:
                return 0.0
            return self._cosine_similarity(emb[0], emb[1])
        except Exception:
            return 0.0

    def _is_duplicate(self, event_text: str, context: str | None = "") -> bool:
        """语义去重：文本重叠 + 向量相似度双重判定。"""
        event_text = str(event_text or "").strip()
        context = str(context or "")
        candidate = f"{event_text} {context[:80]}".strip()
        if len(candidate) < 2:
            return True

        for e in self._events:
            if e.status != "pending":
                continue

            existing_event = str(e.event or "").strip()
            existing_ctx = str(e.context or "")
            existing = f"{existing_event} {existing_ctx[:80]}".strip()
            if not existing:
                continue

            if event_text and existing_event and (event_text in existing_event or existing_event in event_text):
                return True

            if self._char_overlap_ratio(candidate, existing) >= 0.75:
                return True

            sim = self._semantic_similarity(candidate, existing)
            if sim >= 0.84:
                return True
        return False

    def get_due_events(self) -> list[PendingEvent]:
        """获取已到追问时间的 pending 事件（同时淘汰过期事件）。"""
        now = datetime.now()
        cutoff = now - timedelta(hours=_MAX_EVENT_AGE_HOURS)
        due = []
        evicted = False
        for e in self._events:
            if e.status != "pending":
                continue
            # 淘汰过期事件
            try:
                extracted = datetime.fromisoformat(e.extracted_at)
                if extracted < cutoff:
                    e.status = "expired"
                    evicted = True
                    continue
            except ValueError:
                pass
            try:
                followup_at = datetime.fromisoformat(e.followup_after)
                if followup_at <= now:
                    due.append(e)
            except ValueError:
                continue
        if evicted:
            self._save()
        return due

    def mark_done(self, event_id: str):
        """标记事件为已完成。"""
        for e in self._events:
            if e.id == event_id:
                e.status = "done"
                break
        self._save()

    def _evict_old(self):
        """清理过期事件。"""
        now = datetime.now()
        cutoff = now - timedelta(hours=_MAX_EVENT_AGE_HOURS)
        kept = []
        for e in self._events:
            try:
                extracted = datetime.fromisoformat(e.extracted_at)
                if extracted < cutoff:
                    continue  # 太旧了，丢弃
            except ValueError:
                pass
            kept.append(e)
        self._events = kept
