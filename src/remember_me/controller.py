"""ChatController — 异步对话控制层，供 GUI / CLI 共用。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
import zoneinfo
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from remember_me.analyzer.relationship_extractor import RelationshipExtractor
from remember_me.memory.governance import MemoryGovernance
from remember_me.memory.relationship import RelationshipMemoryStore

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
CHROMA_DIR = DATA_DIR / "chroma"
HISTORY_DIR = DATA_DIR / "history"
SESSIONS_DIR = DATA_DIR / "sessions"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

_TIMEZONE = zoneinfo.ZoneInfo(os.environ.get("TZ", "Asia/Shanghai"))
_SESSION_PHASES = {"warmup", "normal", "deep_talk", "cooldown", "ending"}
_ENDING_RE = re.compile(
    r"(再见|拜拜|bye|晚安|睡了|去忙|不聊了|回头聊|先这样|先撤|下次聊)", re.I,
)
_DEEP_TALK_RE = re.compile(
    r"(难过|压力|焦虑|崩溃|失眠|烦|生气|吵架|分手|考试|面试|工作|家庭|生病|抑郁|emo)", re.I,
)
_SHORT_REPLY_RE = re.compile(r"^(嗯|好|行|哦|ok|收到|知道了|好的|拜)$", re.I)


class ChatController:
    """异步聊天控制器，封装 ChatEngine + TopicStarter + MemoryStore。"""

    def __init__(self, persona_name: str, api_key: str | None = None):
        self._name = persona_name
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._engine = None
        self._persona = None
        self._topic_starter = None
        self._memory = None
        self._memory_governance: MemoryGovernance | None = None
        self._relationship_store: RelationshipMemoryStore | None = None
        self._relationship_extractor: RelationshipExtractor | None = None
        self._event_tracker = None
        self._on_message: Callable[[list[str], str], None] | None = None  # (msgs, msg_type)
        self._on_typing: Callable[[bool], None] | None = None
        self._running = False
        self._greeting_task: asyncio.Task | None = None
        self._proactive_task: asyncio.Task | None = None
        self._last_activity = 0.0
        self._history_start_index = 0
        self._event_extract_index = 0  # 上次事件提取时的历史位置
        self._event_extract_task: asyncio.Task | None = None
        self._relationship_extract_index = 0
        self._relationship_extract_task: asyncio.Task | None = None
        self._extract_lock = asyncio.Lock()
        self._last_relationship_followup_at = 0.0
        self._last_relationship_fact_id = ""
        self._session_loaded = False
        self._has_topics = False
        self._proactive_cooldown = 60
        self._reply_checkin_wait = 300
        self._next_proactive_at = 0.0
        self._consecutive_proactive = 0  # 连续主动消息计数（用户回复后归零）
        self._last_interaction_type: str = "none"  # "reply" | "proactive" | "none"
        self._fresh_session = True  # 新开 GUI 页面，跳过 is_conversation_ended 检查
        self._routine = None  # DailyRoutine（作息模板）
        self._session_phase = "warmup"
        self._user_turn_count = 0
        self._phase_updated_at = 0.0
        self._telemetry_seq = 0
        self._recent_proactive_signatures: list[tuple[float, str]] = []

    @property
    def persona_name(self) -> str:
        return self._name

    @property
    def session_phase(self) -> str:
        return self._session_phase

    def _emit_metric(self, event: str, **fields):
        self._telemetry_seq += 1
        payload = {
            "event": event,
            "persona": self._name,
            "phase": self._session_phase,
            "ts": round(time.time(), 3),
            "seq": self._telemetry_seq,
            **fields,
        }
        logger.info("telemetry %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _normalize_proactive_text(text: str) -> str:
        lowered = str(text or "").strip().lower()
        compact = re.sub(r"\s+", "", lowered)
        compact = re.sub(r"[，。！？!?~…,.:;；、\"'“”‘’（）()\[\]{}\-]", "", compact)
        return compact[:240]

    @staticmethod
    def _signature_tokens(signature: str) -> set[str]:
        text = str(signature or "").strip().lower()
        if not text:
            return set()
        words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
        tokens: set[str] = set()
        for w in words:
            if re.fullmatch(r"[\u4e00-\u9fff]+", w):
                if len(w) <= 1:
                    tokens.add(w)
                else:
                    for i in range(len(w) - 1):
                        tokens.add(w[i:i + 2])
            else:
                tokens.add(w)
        return tokens

    @classmethod
    def _signature_similarity(cls, left: str, right: str) -> float:
        lt = cls._signature_tokens(left)
        rt = cls._signature_tokens(right)
        if not lt or not rt:
            return 0.0
        return len(lt & rt) / max(1, len(lt | rt))

    def _build_proactive_signature(self, msgs: list[str]) -> str:
        rows = [self._normalize_proactive_text(m) for m in (msgs or []) if m and str(m).strip()]
        rows = [r for r in rows if r]
        if not rows:
            return ""
        return "|".join(rows)

    def _prune_proactive_signatures(self, window_seconds: int = 1800):
        if not self._recent_proactive_signatures:
            return
        cutoff = time.time() - max(60, int(window_seconds))
        self._recent_proactive_signatures = [
            (ts, sig) for ts, sig in self._recent_proactive_signatures if ts >= cutoff
        ]

    def _is_duplicate_proactive(self, msgs: list[str], window_seconds: int = 1800) -> bool:
        self._prune_proactive_signatures(window_seconds=window_seconds)
        sig = self._build_proactive_signature(msgs)
        if not sig:
            return False
        for _, old_sig in self._recent_proactive_signatures:
            if old_sig == sig:
                return True
            if self._signature_similarity(old_sig, sig) >= 0.82:
                return True
        return False

    def _mark_proactive_sent(self, msgs: list[str]):
        sig = self._build_proactive_signature(msgs)
        if not sig:
            return
        self._prune_proactive_signatures(window_seconds=1800)
        self._recent_proactive_signatures.append((time.time(), sig))

    def _unmark_proactive_sent(self, msgs: list[str]) -> bool:
        sig = self._build_proactive_signature(msgs)
        if not sig:
            return False
        for idx in range(len(self._recent_proactive_signatures) - 1, -1, -1):
            if self._recent_proactive_signatures[idx][1] == sig:
                del self._recent_proactive_signatures[idx]
                return True
        return False

    def rollback_proactive_delivery(self, msgs: list[str], *, reason: str = "delivery_failed") -> bool:
        """下游发送失败时，回滚主动消息注入与节奏状态。"""
        rolled_history = False
        if self._engine and hasattr(self._engine, "rollback_last_proactive_message"):
            try:
                rolled_history = bool(self._engine.rollback_last_proactive_message(msgs))
            except Exception:
                rolled_history = False
        self._unmark_proactive_sent(msgs)
        if self._consecutive_proactive > 0:
            self._consecutive_proactive -= 1
        if self._last_interaction_type == "proactive":
            self._last_interaction_type = "reply"
        now = time.time()
        if self._next_proactive_at <= 0 or self._next_proactive_at > now + 90:
            self._next_proactive_at = now + 90
        self._emit_metric(
            "proactive_delivery_rollback",
            rolled_history=bool(rolled_history),
            reason=reason,
        )
        return rolled_history

    def _build_outbound_system(self, user_input: str = "") -> str | None:
        engine = self._engine
        if not engine or not hasattr(engine, "build_system_for_generation"):
            return None
        try:
            return engine.build_system_for_generation(user_input or "")
        except Exception as e:
            logger.debug("构建主动消息 system 失败，回退默认 system: %s", e)
            return None

    @staticmethod
    def _get_engine_context(engine, max_chars: int = 480) -> str:
        if not engine:
            return ""
        get_proactive = getattr(engine, "get_proactive_context", None)
        if callable(get_proactive):
            try:
                return str(get_proactive(max_chars=max_chars) or "")
            except TypeError:
                return str(get_proactive() or "")
            except Exception as e:
                logger.debug("读取压缩上下文失败，回退 recent_context: %s", e)
        get_recent = getattr(engine, "get_recent_context", None)
        if callable(get_recent):
            try:
                return str(get_recent() or "")
            except Exception:
                return ""
        return ""

    @staticmethod
    def _call_topic_starter(
        fn,
        *args,
        system_instruction: str | None = None,
        **kwargs,
    ):
        if system_instruction:
            try:
                return fn(*args, system_instruction=system_instruction, **kwargs)
            except TypeError:
                # 兼容旧签名/测试桩
                pass
        return fn(*args, **kwargs)

    def _set_phase(self, phase: str, reason: str):
        if phase not in _SESSION_PHASES:
            phase = "normal"
        if phase == self._session_phase:
            return
        prev = self._session_phase
        self._session_phase = phase
        self._phase_updated_at = time.time()
        if self._engine:
            self._engine.set_session_phase(phase)
        self._emit_metric("phase_changed", reason=reason, prev=prev, new=phase)

    def _estimate_initial_phase(self) -> str:
        if not self._engine:
            return "warmup"
        if self._engine.is_conversation_ended():
            return "ending"
        history = list(getattr(self._engine, "_history", []))
        self._user_turn_count = sum(1 for h in history if getattr(h, "role", "") == "user")
        if self._user_turn_count <= 2:
            return "warmup"
        return "normal"

    def _derive_phase_from_user_input(self, text: str, idle_before: float) -> str:
        message = (text or "").strip()
        if not message:
            return self._session_phase
        if _ENDING_RE.search(message):
            return "ending"
        if self._engine and self._engine.is_conversation_ended():
            return "ending"
        if _DEEP_TALK_RE.search(message) or len(message) >= 42:
            return "deep_talk"
        if self._session_phase == "deep_talk" and (idle_before > 240 or _SHORT_REPLY_RE.match(message)):
            return "cooldown"
        if self._session_phase == "cooldown" and not _SHORT_REPLY_RE.match(message):
            return "normal"
        if self._user_turn_count <= 2:
            return "warmup"
        return "normal"

    def _derive_phase_from_idle(self, idle_seconds: float) -> str:
        if self._session_phase == "ending":
            return "ending"
        if idle_seconds > 900:
            return "cooldown"
        if self._session_phase == "warmup" and self._user_turn_count > 2:
            return "normal"
        return self._session_phase

    @staticmethod
    def _profile_bounds(
        profile: dict,
        fallback: tuple[int, int],
        lo_scale: float,
        hi_scale: float,
        lo_cap: int,
        hi_cap: int,
    ) -> tuple[int, int]:
        if not isinstance(profile, dict) or not profile:
            return fallback
        try:
            p25 = float(profile.get("p25", 0))
            p75 = float(profile.get("p75", 0))
        except (TypeError, ValueError):
            return fallback
        if p25 <= 0 or p75 <= 0:
            return fallback
        lo = max(fallback[0], min(int(p25 * lo_scale), lo_cap))
        hi = max(lo + 5, min(int(p75 * hi_scale), hi_cap))
        return lo, hi

    def _sample_initial_proactive_delay(self) -> int:
        profile = getattr(self._persona, "response_delay_profile", {}) if self._persona else {}
        lo, hi = self._profile_bounds(
            profile,
            fallback=(20, 45),
            lo_scale=0.08,
            hi_scale=0.15,
            lo_cap=45,
            hi_cap=90,
        )
        return random.randint(lo, hi)

    def _sample_reply_checkin_wait(self) -> int:
        profile = getattr(self._persona, "silence_delay_profile", {}) if self._persona else {}
        lo, hi = self._profile_bounds(
            profile,
            fallback=(300, 420),
            lo_scale=0.7,
            hi_scale=1.25,
            lo_cap=600,
            hi_cap=1200,
        )
        chase_ratio = float(getattr(self._persona, "chase_ratio", 0.0) or 0.0) if self._persona else 0.0
        if chase_ratio < 0.05:
            lo = int(lo * 1.15)
            hi = int(hi * 1.25)
        elif chase_ratio > 0.2:
            lo = int(lo * 0.8)
            hi = int(hi * 0.9)
        lo = max(120, lo)
        hi = max(lo + 10, min(1500, hi))
        return random.randint(lo, hi)

    def _sample_proactive_cooldown(self) -> int:
        profile = getattr(self._persona, "silence_delay_profile", {}) if self._persona else {}
        lo, hi = self._profile_bounds(
            profile,
            fallback=(60, 110),
            lo_scale=0.25,
            hi_scale=0.45,
            lo_cap=180,
            hi_cap=420,
        )
        chase_ratio = float(getattr(self._persona, "chase_ratio", 0.0) or 0.0) if self._persona else 0.0
        if chase_ratio < 0.05:
            lo = int(lo * 1.25)
            hi = int(hi * 1.35)
        elif chase_ratio > 0.2:
            lo = int(lo * 0.85)
            hi = int(hi * 0.9)
        lo = max(40, lo)
        hi = max(lo + 10, min(600, hi))
        return random.randint(lo, hi)

    def _load_notes(self) -> list[str]:
        """兼容旧接口：加载手动备注（实际来自运行时短期记忆 manual 标签）。"""
        try:
            store = MemoryGovernance(self._name, data_dir=DATA_DIR)
            return [r.text for r in store.list_manual_notes()]
        except Exception as e:
            logger.warning("加载备注失败: %s", e)
            return []

    def _relationship_validator(self, payload: object):
        if not self._memory_governance:
            return False, ""
        if hasattr(payload, "type") and hasattr(payload, "content"):
            return self._memory_governance.validate_relationship_fact(
                payload, persona=self._persona,
            )
        return self._memory_governance.validate_against_imported_history(
            str(payload), persona=self._persona,
        )

    def _pick_shared_event_fact(self):
        if not self._relationship_store:
            return None
        try:
            rows = self._relationship_store.list_facts(
                fact_type="shared_event",
                statuses={"confirmed"},
                include_conflict=False,
                limit=6,
            )
        except Exception:
            return None
        if not rows:
            return None
        now = time.time()
        for fact in rows:
            if fact.id != self._last_relationship_fact_id:
                return fact
        # 全是同一条时，至少间隔 2 小时再复用
        if now - self._last_relationship_followup_at >= 7200:
            return rows[0]
        return None

    @staticmethod
    def save_notes(persona_name: str, notes: list[str]):
        """兼容旧接口：保存手动备注到运行时短期记忆（不影响导入核心人格）。"""
        from remember_me.analyzer.persona import Persona

        profile_path = PROFILES_DIR / f"{persona_name}.json"
        persona = Persona.load(profile_path) if profile_path.exists() else Persona(name=persona_name)
        store = MemoryGovernance(persona_name, data_dir=DATA_DIR)
        store.ensure_core_from_persona(persona)
        store.replace_manual_notes(notes, persona=persona)

    @staticmethod
    def load_notes(persona_name: str) -> list[str]:
        """兼容旧接口：读取运行时短期记忆 manual 标签。"""
        try:
            store = MemoryGovernance(persona_name, data_dir=DATA_DIR)
            return [r.text for r in store.list_manual_notes()]
        except Exception:
            return []

    @property
    def session_loaded(self) -> bool:
        return self._session_loaded

    async def start(
        self,
        on_message: Callable[[list[str], str], None],
        on_typing: Callable[[bool], None] | None = None,
        no_greet: bool = False,
    ):
        """初始化引擎并启动主动消息循环。

        on_message(msgs, msg_type): msg_type 为 "reply" | "proactive" | "greet"
        on_typing(is_typing): 显示/隐藏输入状态
        """
        self._on_message = on_message
        self._on_typing = on_typing
        self._running = True

        # 延迟导入避免循环依赖
        from remember_me.analyzer.persona import Persona
        from remember_me.engine.chat import ChatEngine
        from remember_me.engine.topic_starter import TopicStarter
        from remember_me.memory.store import MemoryStore

        profile_path = PROFILES_DIR / f"{self._name}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"找不到 {self._name} 的人格档案")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY 未设置")

        persona = Persona.load(profile_path)
        self._persona = persona

        # 核心记忆治理：导入历史是唯一真源
        self._memory_governance = MemoryGovernance(self._name, data_dir=DATA_DIR)
        self._memory_governance.ensure_core_from_persona(persona)
        self._relationship_store = RelationshipMemoryStore(self._name, data_dir=DATA_DIR)
        self._relationship_extractor = RelationshipExtractor()
        self._memory_governance.set_relationship_store(self._relationship_store)

        # 加载记忆
        chroma_path = CHROMA_DIR / self._name
        if chroma_path.exists():
            self._memory = MemoryStore(chroma_path, persona_name=self._name)

        # 加载表情包库
        from remember_me.analyzer.sticker import StickerLibrary
        sticker_lib = None
        sticker_path = DATA_DIR / "stickers" / f"{self._name}.json"
        if sticker_path.exists():
            try:
                sticker_lib = StickerLibrary.load(sticker_path)
            except Exception as e:
                logger.debug("加载表情包库失败: %s", e)

        # 加载知识库
        from remember_me.knowledge.store import KnowledgeStore
        knowledge_store = None
        kb_dir = KNOWLEDGE_DIR / self._name
        chroma_path = CHROMA_DIR / self._name
        if kb_dir.exists():
            try:
                knowledge_store = KnowledgeStore(
                    chroma_dir=chroma_path, knowledge_dir=kb_dir,
                    persona_name=self._name,
                )
            except Exception as e:
                logger.debug("加载知识库失败: %s", e)

        self._engine = ChatEngine(
            persona=persona, memory=self._memory,
            api_key=self._api_key, sticker_lib=sticker_lib,
            notes=[], knowledge_store=knowledge_store,
            memory_governance=self._memory_governance,
        )

        # 加载上次对话
        session_path = SESSIONS_DIR / f"{self._name}.json"
        self._session_loaded = self._engine.load_session(session_path)
        self._history_start_index = len(self._engine._history)
        self._user_turn_count = sum(
            1 for h in self._engine._history if getattr(h, "role", "") == "user"
        )
        # 会话阶段：优先使用已加载会话，否则按历史估算
        loaded_phase = self._engine.session_phase if self._session_loaded else self._estimate_initial_phase()
        self._session_phase = loaded_phase if loaded_phase in _SESSION_PHASES else self._estimate_initial_phase()
        self._engine.set_session_phase(self._session_phase)
        self._phase_updated_at = time.time()

        # 加载作息模板 → 生成今日日程
        from remember_me.analyzer.routine import DailyRoutine
        routine_path = PROFILES_DIR / f"{self._name}_routine.json"
        if routine_path.exists():
            try:
                self._routine = DailyRoutine.load(routine_path)
                now_dt = datetime.now(_TIMEZONE)
                date_str = now_dt.strftime("%Y-%m-%d")
                if not self._engine._space_state.schedule or self._engine._space_state.schedule_date != date_str:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, lambda: self._engine.regenerate_daily_schedule(self._routine, now_dt.weekday(), date_str),
                    )
            except Exception as e:
                logger.warning("加载作息/生成日程失败: %s", e)

        self._topic_starter = TopicStarter(persona=persona, client=self._engine.client)
        self._has_topics = bool(getattr(persona, "topic_interests", None))

        # 待跟进事件追踪器
        from remember_me.engine.pending_events import PendingEventTracker
        self._event_tracker = PendingEventTracker(persona_name=self._name, data_dir=DATA_DIR)
        self._event_extract_index = len(self._engine._history)
        self._relationship_extract_index = len(self._engine._history)

        self._update_activity()
        self._reply_checkin_wait = self._sample_reply_checkin_wait()
        self._next_proactive_at = time.time() + self._sample_initial_proactive_delay()
        self._emit_metric(
            "session_started",
            session_loaded=self._session_loaded,
            has_topics=self._has_topics,
            user_turns=self._user_turn_count,
        )

        # 主动开场
        if not no_greet and self._has_topics:
            self._greeting_task = asyncio.create_task(self._send_greeting())

        # 启动后台主动消息循环
        self._proactive_task = asyncio.create_task(self._proactive_loop())

    async def _send_greeting(self):
        """发送主动开场消息。"""
        if self._on_typing:
            self._on_typing(True)
        try:
            ctx = self._get_engine_context(self._engine, max_chars=460) if self._session_loaded else ""
            loop = asyncio.get_event_loop()
            greet_policy = self._engine.plan_rhythm_policy(kind="greet", user_input=ctx)
            system_instruction = self._build_outbound_system(ctx)
            greet_msgs = await loop.run_in_executor(
                None,
                lambda: self._call_topic_starter(
                    self._topic_starter.generate,
                    recent_context=ctx,
                    count_policy=greet_policy,
                    system_instruction=system_instruction,
                ),
            )
            greet_msgs = [m for m in greet_msgs if m and m.strip()]
            if greet_msgs:
                if self._is_duplicate_proactive(greet_msgs, window_seconds=1800):
                    logger.info("开场消息跳过：与最近主动消息重复")
                    return
                self._engine.inject_proactive_message(greet_msgs)
                self._update_activity()
                self._last_interaction_type = "proactive"
                self._consecutive_proactive = 1
                self._next_proactive_at = time.time() + self._sample_proactive_cooldown()
                self._mark_proactive_sent(greet_msgs)
                self._set_phase("warmup", reason="greeting")
                self._emit_metric("greeting_sent", reply_count=len(greet_msgs))
                if self._on_message:
                    self._on_message(greet_msgs, "greet")
                # 后台预缓存
                loop.run_in_executor(None, self._topic_starter.prefetch)
            else:
                logger.warning("开场消息生成为空（可能 BRAVE_API_KEY 未设置，已使用降级方案）")
        except Exception as e:
            logger.warning("开场消息生成失败: %s", e)
        finally:
            if self._on_typing:
                self._on_typing(False)

    def _update_activity(self):
        self._last_activity = time.time()

    def _get_idle_seconds(self) -> float:
        return time.time() - self._last_activity

    async def send_message(self, text: str,
                           image: tuple[bytes, str] | None = None) -> list[str]:
        """发送用户消息，返回回复列表。

        image: 可选 (bytes, mime_type) 图片。
        """
        if not self._engine:
            raise RuntimeError("引擎未初始化，请先调用 start()")

        idle_before = self._get_idle_seconds()
        self._user_turn_count += 1
        next_phase = self._derive_phase_from_user_input(text, idle_before=idle_before)
        self._set_phase(next_phase, reason="user_input")
        if self._memory_governance:
            self._memory_governance.add_session_record(
                text,
                persona=self._persona,
                ttl_seconds=None,
                tags=["runtime", "user_turn"],
                confidence=0.45,
            )
        self._topic_starter.on_user_replied()
        self._update_activity()
        self._consecutive_proactive = 0
        self._last_interaction_type = "reply"
        self._fresh_session = False

        if self._on_typing:
            self._on_typing(True)
        try:
            loop = asyncio.get_event_loop()
            start_at = time.perf_counter()
            replies = await loop.run_in_executor(
                None, lambda: self._engine.send_multi(text, image=image)
            )
            self._update_activity()
            # 回复后按人格节奏等待一段时间，再考虑接话
            self._reply_checkin_wait = self._sample_reply_checkin_wait()
            self._next_proactive_at = time.time() + self._reply_checkin_wait
            elapsed_ms = int((time.perf_counter() - start_at) * 1000)
            self._emit_metric(
                "reply_generated",
                latency_ms=elapsed_ms,
                reply_count=len(replies),
                has_image=bool(image),
                input_len=len(text or ""),
            )
            if self._relationship_store:
                await loop.run_in_executor(
                    None, lambda: self._relationship_store.mark_boundary_hit(text)
                )

            # 异步提取待跟进事件（每 6 轮检查一次，与 scratchpad 同步）
            history_len = len(self._engine._history)
            async with self._extract_lock:
                if (
                    history_len - self._event_extract_index >= 6
                    and (not self._event_extract_task or self._event_extract_task.done())
                ):
                    self._event_extract_task = asyncio.create_task(self._extract_pending_events())
                if (
                    history_len - self._relationship_extract_index >= 8
                    and self._relationship_extractor
                    and self._relationship_store
                    and (not self._relationship_extract_task or self._relationship_extract_task.done())
                ):
                    self._relationship_extract_task = asyncio.create_task(
                        self._extract_relationship_facts()
                    )

            return replies
        finally:
            if self._on_typing:
                self._on_typing(False)

    async def _extract_pending_events(self):
        """异步从近期对话中提取待跟进事件。"""
        if not self._event_tracker or not self._engine:
            return
        snapshot_index = len(self._engine._history)
        advance_index = False
        try:
            messages = []
            for h in self._engine._history[self._event_extract_index:snapshot_index]:
                if h.parts and len(h.parts) > 0 and getattr(h.parts[0], "text", None):
                    messages.append({"role": h.role, "text": h.parts[0].text})

            if not messages:
                advance_index = True
                return

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self._event_tracker.extract_events(
                    self._engine.client, messages
                )
            )
            advance_index = True
        except Exception as e:
            logger.warning("事件提取失败: %s", e)
        finally:
            if advance_index:
                self._event_extract_index = snapshot_index

    async def _extract_relationship_facts(self):
        """异步提取关系记忆（低频增量，不阻塞主回复）。"""
        if not self._engine or not self._relationship_extractor or not self._relationship_store:
            return
        snapshot_index = len(self._engine._history)
        advance_index = False
        try:
            messages = []
            for h in self._engine._history[self._relationship_extract_index:snapshot_index]:
                if h.parts and len(h.parts) > 0 and getattr(h.parts[0], "text", None):
                    messages.append({"role": h.role, "text": h.parts[0].text})
            if not messages:
                advance_index = True
                return

            loop = asyncio.get_event_loop()

            facts = await loop.run_in_executor(
                None,
                lambda: self._relationship_extractor.extract_from_messages(
                    messages,
                    client=self._engine.client,
                    source="runtime_session",
                    conflict_validator=self._relationship_validator,
                ),
            )
            if facts:
                await loop.run_in_executor(None, lambda: self._relationship_store.upsert_facts(facts))
                await loop.run_in_executor(None, lambda: self._relationship_store.promote_candidates(
                    min_confidence=0.78, min_evidence=2,
                ))
            advance_index = True
        except Exception as e:
            logger.warning("关系记忆提取失败: %s", e)
        finally:
            if advance_index:
                self._relationship_extract_index = snapshot_index

    async def _proactive_loop(self):
        """后台主动消息循环。

        两种沉默场景，两种应对：
        - 回复后沉默（用户聊完不回了）→ 5 分钟后接话/关心，不引新话题
        - 主动消息后沉默（Bot 发了东西对方没理）→ 60 秒后追问一次
        新话题仅从每日调度触发，不在此循环中生成。
        """
        while self._running:
            await asyncio.sleep(2)
            if not self._running:
                continue
            now = time.time()
            idle = self._get_idle_seconds()
            self._set_phase(self._derive_phase_from_idle(idle), reason="idle")
            if idle <= 30 or now <= self._next_proactive_at:
                continue

            # 午夜日程刷新（run_in_executor 避免阻塞事件循环）
            if self._routine and self._engine:
                now_dt = datetime.now(_TIMEZONE)
                current_date = now_dt.strftime("%Y-%m-%d")
                if self._engine._space_state.schedule_date != current_date:
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, lambda: self._engine.regenerate_daily_schedule(
                                self._routine, now_dt.weekday(), current_date),
                        )
                    except Exception as e:
                        logger.warning("午夜日程刷新失败: %s", e)

            # 推进空间状态到当前时间 → 再检查是否允许主动消息
            if self._engine:
                self._engine.advance_space()
                space_mods = self._engine.space_modifiers
                if not space_mods.proactive_allowed:
                    continue
            if self._session_phase == "ending":
                continue
            # 新 GUI 会话跳过"对话已结束"检查（用户主动打开页面说明想聊）
            if not self._fresh_session and self._engine.is_conversation_ended():
                self._set_phase("ending", reason="conversation_ended")
                continue

            try:
                if self._on_typing:
                    self._on_typing(True)

                loop = asyncio.get_event_loop()
                msgs = None
                relation_fact_id = ""

                # 优先级 1：待跟进事件（上下文相关，始终可触发）
                if self._event_tracker and idle > 60:
                    due_events = self._event_tracker.get_due_events()
                    if due_events:
                        event = due_events[0]
                        event_input = f"{event.event}\n{event.context}\n{event.followup_hint}"
                        event_policy = self._engine.plan_rhythm_policy(
                            kind="event_followup",
                            user_input=event_input,
                        )
                        event_system = self._build_outbound_system(event_input)
                        logger.info("触发事件追问: %s", event.event)
                        msgs = await loop.run_in_executor(
                            None,
                            lambda: self._call_topic_starter(
                                self._topic_starter.generate_event_followup,
                                event.event,
                                event.context,
                                event.followup_hint,
                                count_policy=event_policy,
                                system_instruction=event_system,
                            ),
                        )
                        if msgs:
                            self._event_tracker.mark_done(event.id)

                # 优先级 2：关系记忆追问（shared_event 优先）
                if not msgs and idle > 90 and (now - self._last_relationship_followup_at) >= 1800:
                    fact = self._pick_shared_event_fact()
                    if fact:
                        ctx = self._get_engine_context(self._engine, max_chars=460)
                        relation_input = f"{ctx}\n{fact.content}"
                        rel_policy = self._engine.plan_rhythm_policy(
                            kind="relationship_followup",
                            user_input=relation_input,
                        )
                        relation_system = self._build_outbound_system(relation_input)
                        relation_fact_id = fact.id
                        msgs = await loop.run_in_executor(
                            None,
                            lambda: self._call_topic_starter(
                                self._topic_starter.generate_relationship_followup,
                                fact_type=fact.type,
                                fact_content=fact.content,
                                fact_meta=fact.meta,
                                recent_context=ctx,
                                count_policy=rel_policy,
                                system_instruction=relation_system,
                            ),
                        )

                # 优先级 3：根据上次交互类型决定
                if not msgs:
                    if self._last_interaction_type == "reply":
                        # 回复后沉默 → 接话/关心，不引新话题
                        if idle < self._reply_checkin_wait or self._consecutive_proactive >= 1:
                            continue
                        ctx = self._get_engine_context(self._engine, max_chars=460)
                        checkin_policy = self._engine.plan_rhythm_policy(
                            kind="followup",
                            user_input=ctx,
                        )
                        checkin_system = self._build_outbound_system(ctx)
                        msgs = await loop.run_in_executor(
                            None,
                            lambda: self._call_topic_starter(
                                self._topic_starter.generate_checkin,
                                ctx,
                                count_policy=checkin_policy,
                                system_instruction=checkin_system,
                            ),
                        )
                    else:
                        # 主动消息后沉默 → 追问（最多 1 次追问，加上原始消息共 2 条）
                        if self._consecutive_proactive >= 2:
                            continue
                        if self._topic_starter._last_proactive:
                            ctx = self._get_engine_context(self._engine, max_chars=460)
                            followup_policy = self._engine.plan_rhythm_policy(
                                kind="followup",
                                user_input=ctx,
                            )
                            followup_system = self._build_outbound_system(ctx)
                            msgs = await loop.run_in_executor(
                                None,
                                lambda: self._call_topic_starter(
                                    self._topic_starter.generate_followup,
                                    recent_context=ctx,
                                    allow_new_topic=False,
                                    count_policy=followup_policy,
                                    system_instruction=followup_system,
                                ),
                            )

                if msgs:
                    # 再次检查：发送期间用户可能回复了
                    if self._get_idle_seconds() < 15:
                        continue
                    if not self._fresh_session and self._engine.is_conversation_ended():
                        self._set_phase("ending", reason="conversation_ended")
                        continue
                    if self._is_duplicate_proactive(msgs, window_seconds=1800):
                        logger.info("主动消息跳过：命中重复去重")
                        continue
                    self._engine.inject_proactive_message(msgs)
                    self._update_activity()
                    self._consecutive_proactive += 1
                    self._last_interaction_type = "proactive"
                    self._fresh_session = False
                    self._mark_proactive_sent(msgs)
                    if relation_fact_id:
                        self._last_relationship_fact_id = relation_fact_id
                        self._last_relationship_followup_at = now
                    cooldown = self._sample_proactive_cooldown()
                    factor = getattr(self._engine, "proactive_cooldown_factor", 1.0) if self._engine else 1.0
                    if isinstance(factor, (int, float)) and factor > 0:
                        cooldown *= factor
                    # 叠加空间冷却因子
                    if self._engine:
                        s_factor = self._engine.space_modifiers.proactive_cooldown_factor
                        cooldown *= s_factor
                    self._next_proactive_at = now + cooldown
                    self._set_phase("cooldown", reason="proactive_sent")
                    self._emit_metric(
                        "proactive_sent",
                        reply_count=len(msgs),
                        idle_seconds=int(idle),
                        cooldown_seconds=int(cooldown),
                    )
                    if self._on_message:
                        self._on_message(msgs, "proactive")
            except Exception as e:
                logger.warning("主动消息生成失败: %s", e)
            finally:
                if self._on_typing:
                    self._on_typing(False)

    async def stop(self):
        """停止控制器，保存会话。"""
        self._running = False
        self._set_phase("ending", reason="controller_stop")
        # 先等待数据写入类任务自然结束（最多 5 秒），避免中断文件 I/O
        io_tasks = [t for t in (self._event_extract_task, self._relationship_extract_task)
                    if t and not t.done()]
        if io_tasks:
            done, pending = await asyncio.wait(io_tasks, timeout=5)
            if pending:
                logger.warning("停止会话时后台提取任务超时，强制取消: %d", len(pending))
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            if done:
                await asyncio.gather(*done, return_exceptions=True)
        # 其余任务直接取消
        for task in (self._greeting_task, self._proactive_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._greeting_task = None
        self._proactive_task = None
        self._event_extract_task = None
        self._relationship_extract_task = None
        self._save_session()
        if self._engine and hasattr(self._engine, "aclose_client"):
            try:
                await self._engine.aclose_client()
            except Exception as e:
                logger.debug("关闭聊天引擎客户端失败: %s", e)
        self._emit_metric("session_stopped")

    def _save_session(self):
        if not self._engine:
            return
        try:
            session_path = SESSIONS_DIR / f"{self._name}.json"
            self._engine.save_session(session_path)
            new_msgs = self._engine.get_new_messages(self._history_start_index)
            if new_msgs and self._memory:
                to_add = new_msgs
                if self._memory_governance:
                    to_add = self._memory_governance.filter_messages_for_long_term(
                        new_msgs, persona=self._persona,
                    )
                if to_add:
                    self._memory.add_messages(to_add)

            # 会话结束时补提取关系记忆（防止后台任务未触发或中断）
            if self._relationship_extractor and self._relationship_store:
                rel_msgs = []
                for h in self._engine._history[self._relationship_extract_index:]:
                    if h.parts and len(h.parts) > 0 and getattr(h.parts[0], "text", None):
                        rel_msgs.append({"role": h.role, "text": h.parts[0].text})
                if rel_msgs:
                    facts = self._relationship_extractor.extract_from_messages(
                        rel_msgs,
                        client=self._engine.client,
                        source="runtime_session",
                        conflict_validator=self._relationship_validator,
                    )
                    if facts:
                        self._relationship_store.upsert_facts(facts)
                        self._relationship_store.promote_candidates(
                            min_confidence=0.78, min_evidence=2,
                        )
                    self._relationship_extract_index = len(self._engine._history)

            # 会话结束时提取待跟进事件（确保不遗漏）
            if self._event_tracker:
                remaining = []
                for h in self._engine._history[self._event_extract_index:]:
                    if h.parts and len(h.parts) > 0 and getattr(h.parts[0], "text", None):
                        remaining.append({"role": h.role, "text": h.parts[0].text})
                if remaining:
                    try:
                        self._event_tracker.extract_events(self._engine.client, remaining)
                    except Exception as e:
                        logger.debug("会话结束事件提取失败: %s", e)
        except Exception as e:
            logger.warning("保存会话失败: %s", e)

    def get_total_messages(self) -> int:
        """获取历史消息总数。"""
        from remember_me.analyzer.persona import Persona
        profile_path = PROFILES_DIR / f"{self._name}.json"
        persona = Persona.load(profile_path)
        total = persona.total_messages
        history_path = HISTORY_DIR / f"{self._name}.json"
        if history_path.exists():
            try:
                data = json.loads(history_path.read_text(encoding="utf-8"))
                total = data.get("total_messages", total)
            except Exception:
                pass
        return total

    @staticmethod
    def list_personas() -> list[dict]:
        """列出所有人格档案。返回 [{name, total_messages}]。"""
        from remember_me.analyzer.persona import Persona
        if not PROFILES_DIR.exists():
            return []
        result = []
        for p in PROFILES_DIR.glob("*.json"):
            try:
                persona = Persona.load(p)
                result.append({
                    "name": persona.name,
                    "total_messages": persona.total_messages,
                    "style_summary": persona.style_summary,
                })
            except Exception:
                continue
        return result

    @staticmethod
    async def import_chat_file(
        file_path: str,
        fmt: str,
        target_name: str,
        user_name: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ):
        """导入聊天记录并生成人格档案 + 记忆索引。"""
        from remember_me.analyzer.persona import analyze
        from remember_me.importers import json_parser, plain_text, wechat
        from remember_me.memory.store import MemoryStore

        importers = {"text": plain_text, "json": json_parser, "wechat": wechat}
        importer = importers.get(fmt)
        if not importer:
            raise ValueError(f"不支持的格式: {fmt}")

        if on_progress:
            on_progress("正在解析聊天记录...")

        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(
            None, lambda: importer.parse(file_path, target_name=target_name, user_name=user_name)
        )

        if not history.messages:
            raise ValueError("未找到任何消息，请检查文件格式和目标名字。")

        target_count = len(history.target_messages)
        if on_progress:
            on_progress(f"导入 {len(history.messages)} 条消息（{target_name}：{target_count} 条）")

        # 保存聊天记录
        history_path = HISTORY_DIR / f"{target_name}.json"
        history.save(history_path)

        # 分析人格
        if on_progress:
            on_progress("正在分析人格特征...")
        persona = await loop.run_in_executor(None, lambda: analyze(history))
        profile_path = PROFILES_DIR / f"{target_name}.json"
        persona.save(profile_path)
        # 导入完成后重建核心记忆快照（唯一真源）
        governance = MemoryGovernance(target_name, data_dir=DATA_DIR)
        await loop.run_in_executor(None, lambda: governance.bootstrap_core_from_persona(persona, force=True))

        if on_progress:
            on_progress(f"人格分析完成 — {persona.style_summary[:60]}")

        # 提取作息模板
        if on_progress:
            on_progress("正在提取日常作息模式...")
        try:
            from remember_me.analyzer.routine import analyze_routine, DailyRoutine
            routine = await loop.run_in_executor(None, lambda: analyze_routine(history))
            routine_path = PROFILES_DIR / f"{target_name}_routine.json"
            routine.save(routine_path)
            slot_count = len(routine.weekday_slots) + len(routine.weekend_slots)
            if on_progress:
                on_progress(f"作息提取完成 — 识别到 {slot_count} 个日常时段")
        except Exception as e:
            logger.warning("作息提取失败: %s", e)
            if on_progress:
                on_progress(f"作息提取失败（不影响其他功能）: {e}")

        if on_progress:
            on_progress("正在提取关系记忆...")
        rel_store = RelationshipMemoryStore(target_name, data_dir=DATA_DIR)
        rel_extractor = RelationshipExtractor()

        def _validator(payload: object):
            if hasattr(payload, "type") and hasattr(payload, "content"):
                return governance.validate_relationship_fact(payload, persona=persona)
            return governance.validate_against_imported_history(
                str(payload), persona=persona,
            )

        rel_facts = await loop.run_in_executor(
            None,
            lambda: rel_extractor.extract_from_history_in_windows(
                history,
                conflict_validator=_validator,
                window_size=120,
                stride=80,
            ),
        )
        if rel_facts:
            await loop.run_in_executor(None, lambda: rel_store.upsert_facts(rel_facts))
            await loop.run_in_executor(
                None,
                lambda: rel_store.promote_candidates(min_confidence=0.78, min_evidence=2),
            )
        confirmed_rel = await loop.run_in_executor(None, lambda: len(rel_store.list_confirmed(limit=200)))
        if on_progress:
            on_progress(f"关系记忆提取完成 — 已确认 {confirmed_rel} 条")

        # 建立记忆索引
        if on_progress:
            on_progress("正在建立记忆索引...")
        store = MemoryStore(CHROMA_DIR / target_name, persona_name=target_name)
        await loop.run_in_executor(None, lambda: store.index_history(history))

        if on_progress:
            on_progress(f"[OK] {target_name} 导入完成！共 {len(history.messages)} 条消息。")

        return {
            "name": target_name,
            "total_messages": len(history.messages),
            "target_count": target_count,
            "style_summary": persona.style_summary,
        }
