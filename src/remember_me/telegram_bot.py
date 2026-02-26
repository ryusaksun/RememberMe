"""RememberMe Telegram Bot — 单 persona 模式 + 每日主动消息。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from remember_me.controller import ChatController
from remember_me.memory.relationship import RELATION_TYPES, RelationshipMemoryStore

logger = logging.getLogger(__name__)

PERSONA_NAME = os.environ.get("PERSONA_NAME", "阴暗扭曲爬行_-_-")
TIMEZONE = ZoneInfo(os.environ.get("TZ", "Asia/Shanghai"))
DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
_SENTENCE_END_RE = re.compile(r"[。！？!?~…]$")
_REL_TYPE_LABEL = {
    "relation_stage": "关系阶段",
    "addressing": "称呼习惯",
    "boundary": "互动边界",
    "shared_event": "共同经历",
    "commitment": "承诺与跟进",
    "repair_pattern": "冲突修复",
    "preference": "偏好线索",
}


class TelegramBot:
    """单 persona Telegram Bot，含每日主动消息调度。"""

    # ── 消息聚合参数 ──
    # 单条消息的短探测延迟（秒）：等这么久看看有没有后续消息
    COALESCE_PEEK = 0.8
    # 连发模式的 debounce（秒）：每条新消息重置此计时器
    COALESCE_DEBOUNCE = 2.0
    # 消息像“说完一句”时，尽快 flush（秒）
    COALESCE_ENDING_DEBOUNCE = 0.55
    # 从第一条消息算起的最大等待（秒）：超过此时间无论如何都处理
    COALESCE_MAX_WAIT = 8.0
    # 主动消息去重窗口（秒）：避免每日/主动消息连续重复
    PROACTIVE_DEDUP_WINDOW = 45 * 60
    # Telegram 单条文本上限 4096，这里留安全余量
    TELEGRAM_TEXT_LIMIT = 3800
    # typing 心跳频率（秒）：生成期间持续刷新，避免“卡住”观感
    TYPING_HEARTBEAT_INTERVAL = 4.5
    # Telegram API 发送层重试参数
    TELEGRAM_API_MAX_RETRIES = 4
    TELEGRAM_RETRY_BASE_DELAY = 0.45
    # 出错提示限频，避免连续异常时刷屏
    ERROR_REPLY_COOLDOWN = 20

    def __init__(self, token: str, allowed_users: set[int] | None = None):
        self._token = token
        self._allowed_users = allowed_users
        self._controller: ChatController | None = None
        self._chat_id: int | None = None
        self._send_lock = asyncio.Lock()
        self._app: Application | None = None
        self._last_user_activity: float = 0.0  # 用户最后交互时间
        # 消息聚合状态
        self._coalesce_buffer: list[str] = []
        self._coalesce_first_at: float = 0.0  # 缓冲区第一条消息的时间戳（monotonic）
        self._coalesce_timer: asyncio.Task | None = None
        self._coalesce_lock = asyncio.Lock()
        # 每日调度
        self._daily_times: list[datetime] = []
        self._daily_date: str = ""  # 当前计划对应的日期
        self._bg_tasks: set[asyncio.Task] = set()
        self._recent_proactive_signatures: list[tuple[float, str]] = []
        self._last_error_reply_at: float = 0.0

    def _is_allowed(self, user_id: int) -> bool:
        if self._allowed_users is None:
            return True
        return user_id in self._allowed_users

    def _track_task(self, task: asyncio.Task, name: str) -> asyncio.Task:
        """统一管理后台任务，避免 shutdown 时留下 pending task。"""
        self._bg_tasks.add(task)

        def _on_done(done: asyncio.Task):
            self._bg_tasks.discard(done)
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            if exc:
                logger.warning("后台任务失败 [%s]: %s", name, exc)

        task.add_done_callback(_on_done)
        return task

    def _cancel_coalesce_timer(self):
        if self._coalesce_timer and not self._coalesce_timer.done():
            self._coalesce_timer.cancel()
        self._coalesce_timer = None

    async def _shutdown_background_tasks(self):
        self._cancel_coalesce_timer()
        tasks = [t for t in self._bg_tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._bg_tasks.clear()

    @staticmethod
    def _is_sentence_finished(text: str) -> bool:
        return bool(_SENTENCE_END_RE.search((text or "").strip()))

    @staticmethod
    def _normalize_signature_text(text: str) -> str:
        """主动消息去重签名：忽略空白和常见标点，仅保留主体内容。"""
        lowered = str(text or "").strip().lower()
        compact = re.sub(r"\s+", "", lowered)
        compact = re.sub(r"[，。！？!?~…,.:;；、\"'“”‘’（）()\[\]{}\-]", "", compact)
        return compact[:240]

    def _build_proactive_signature(self, msgs: list[str]) -> str:
        rows = [self._normalize_signature_text(m) for m in (msgs or []) if m and str(m).strip()]
        rows = [r for r in rows if r]
        if not rows:
            return ""
        return "|".join(rows)

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

    def _prune_proactive_signatures(self):
        if not self._recent_proactive_signatures:
            return
        cutoff = time.time() - self.PROACTIVE_DEDUP_WINDOW
        self._recent_proactive_signatures = [
            (ts, sig) for ts, sig in self._recent_proactive_signatures if ts >= cutoff
        ]

    def _is_duplicate_proactive(self, msgs: list[str]) -> bool:
        self._prune_proactive_signatures()
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
        self._prune_proactive_signatures()
        self._recent_proactive_signatures.append((time.time(), sig))

    @staticmethod
    def _split_telegram_text(text: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
        raw = (text or "").strip()
        if not raw:
            return []
        if len(raw) <= limit:
            return [raw]

        chunks: list[str] = []
        remain = raw
        while len(remain) > limit:
            cut = max(
                remain.rfind("\n", 0, limit),
                remain.rfind("。", 0, limit),
                remain.rfind("！", 0, limit),
                remain.rfind("？", 0, limit),
                remain.rfind("，", 0, limit),
                remain.rfind(" ", 0, limit),
            )
            if cut < int(limit * 0.5):
                cut = limit
            chunk = remain[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remain = remain[cut:].strip()
        if remain:
            chunks.append(remain)
        return chunks

    @staticmethod
    def _typing_delay_scale(text: str) -> float:
        """根据消息长度微调打字延迟，让长句不至于“秒回”。"""
        length = len((text or "").strip())
        if length <= 0:
            return 1.0
        return min(1.7, 1.0 + (length / 180.0))

    async def _telegram_call_with_retry(self, op_name: str, op):
        """统一 Telegram API 发送重试：处理限流和短暂网络抖动。"""
        delay = self.TELEGRAM_RETRY_BASE_DELAY
        for attempt in range(1, self.TELEGRAM_API_MAX_RETRIES + 1):
            try:
                return await op()
            except RetryAfter as e:
                # PTB v22 的 retry_after 属性在默认配置下会触发弃用告警，优先读内部 timedelta 字段。
                retry_after = getattr(e, "_retry_after", None)
                if retry_after is None:
                    retry_after = getattr(e, "retry_after", 1)
                if isinstance(retry_after, timedelta):
                    wait = max(0.2, float(retry_after.total_seconds()))
                else:
                    try:
                        wait = max(0.2, float(retry_after))
                    except (TypeError, ValueError):
                        wait = 1.0
                if attempt >= self.TELEGRAM_API_MAX_RETRIES:
                    raise
                logger.warning(
                    "Telegram %s 遇到限流，第 %d/%d 次重试，%.2fs 后重试",
                    op_name, attempt, self.TELEGRAM_API_MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            except (TimedOut, NetworkError) as e:
                if attempt >= self.TELEGRAM_API_MAX_RETRIES:
                    raise
                logger.warning(
                    "Telegram %s 网络异常，第 %d/%d 次重试: %s",
                    op_name, attempt, self.TELEGRAM_API_MAX_RETRIES, e,
                )
                await asyncio.sleep(delay)
                delay = min(4.0, delay * 1.8)

    async def _send_typing(self, bot, chat_id: int):
        await self._telegram_call_with_retry(
            "send_chat_action",
            lambda: bot.send_chat_action(chat_id, ChatAction.TYPING),
        )

    async def _send_text(self, bot, chat_id: int, text: str):
        await self._telegram_call_with_retry(
            "send_message",
            lambda: bot.send_message(chat_id, text),
        )

    async def _send_photo(self, bot, chat_id: int, photo):
        async def _op():
            if hasattr(photo, "seek"):
                with contextlib.suppress(Exception):
                    photo.seek(0)
            return await bot.send_photo(chat_id, photo=photo)

        await self._telegram_call_with_retry(
            "send_photo",
            _op,
        )

    async def _run_with_typing_heartbeat(self, bot, chat_id: int, coro):
        """执行耗时协程期间持续发送 typing 心跳。"""
        task = asyncio.create_task(coro)
        try:
            while not task.done():
                try:
                    await self._send_typing(bot, chat_id)
                except Exception as e:
                    # typing 仅用于体验增强，不应影响真实回复生成。
                    logger.debug("typing 心跳发送失败，忽略并继续等待主任务: %s", e)
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(task),
                        timeout=self.TYPING_HEARTBEAT_INTERVAL,
                    )
                except asyncio.TimeoutError:
                    continue
            return await task
        except Exception:
            if not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task
            raise

    def _persona_style_token(self) -> str:
        controller = self._controller
        persona = getattr(controller, "_persona", None) if controller else None
        if not persona:
            return ""
        options: list[str] = []
        options.extend(list(getattr(persona, "self_references", []) or [])[:3])
        options.extend(list(getattr(persona, "catchphrases", []) or [])[:6])
        options.extend(list(getattr(persona, "tone_markers", []) or [])[:4])
        cleaned = []
        for item in options:
            token = str(item or "").strip()
            if not token:
                continue
            if " " in token or "\n" in token:
                continue
            if len(token) > 8:
                continue
            cleaned.append(token)
        if not cleaned:
            return ""
        return random.choice(cleaned)

    def _humanized_error_text(self) -> str:
        base = random.choice([
            "我这边刚卡了一下，你再发一条试试。",
            "等等，我这边网络抽了下，你再说一次。",
            "刚刚没接上，你再发一遍我马上回。",
        ])
        token = self._persona_style_token()
        if not token:
            return base
        if base.startswith(token):
            return base
        return f"{token}，{base}"

    async def _maybe_send_humanized_error(self, bot, chat_id: int):
        now = time.time()
        if now - self._last_error_reply_at < self.ERROR_REPLY_COOLDOWN:
            return
        self._last_error_reply_at = now
        await self._send_text(bot, chat_id, self._humanized_error_text())

    @staticmethod
    def _is_transient_app_error(exc: Exception) -> bool:
        if isinstance(exc, (NetworkError, TimedOut, RetryAfter)):
            return True
        text = f"{type(exc).__name__}: {exc}".lower()
        keys = ("bad gateway", "timeout", "temporar", "connection reset", "rate limit", "429")
        return any(k in text for k in keys)

    async def _on_app_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        err = getattr(context, "error", None)
        if not isinstance(err, Exception):
            logger.warning("Telegram runtime error handler 收到非异常对象: %r", err)
            return
        if self._is_transient_app_error(err):
            logger.warning("Telegram 运行时临时异常（已由全局 handler 捕获）: %s", err)
            return
        logger.exception("Telegram 运行时异常: %s", err)

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
                pass
        return fn(*args, **kwargs)

    @staticmethod
    def _get_engine_context(engine, max_chars: int = 460) -> str:
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

    def _compute_coalesce_delay(self, text: str, is_first: bool) -> float:
        if is_first:
            return self.COALESCE_ENDING_DEBOUNCE if self._is_sentence_finished(text) else self.COALESCE_PEEK
        elapsed = max(0.0, time.monotonic() - self._coalesce_first_at)
        remaining = max(0.05, self.COALESCE_MAX_WAIT - elapsed)
        debounce = (
            self.COALESCE_ENDING_DEBOUNCE
            if self._is_sentence_finished(text)
            else self.COALESCE_DEBOUNCE
        )
        return min(debounce, remaining)

    # ── Persona 数据加载 ──

    def _load_persona_meta(self) -> dict:
        """加载 persona 的 active_hours 和性格参数（不创建 ChatEngine）。"""
        from remember_me.analyzer.persona import Persona

        profile_path = PROFILES_DIR / f"{PERSONA_NAME}.json"
        if not profile_path.exists():
            logger.warning("persona 档案不存在: %s", profile_path)
            return {}
        persona = Persona.load(profile_path)
        return {
            "active_hours": getattr(persona, "active_hours", []),
            "chase_ratio": float(getattr(persona, "chase_ratio", 0.0) or 0.0),
            "avg_burst_length": float(getattr(persona, "avg_burst_length", 1.0) or 1.0),
        }

    # ── 每日调度 ──

    def _plan_daily_times(self) -> list[datetime]:
        """根据 persona 的 active_hours 生成今天的主动消息时刻。"""
        meta = self._load_persona_meta()
        active_hours = meta.get("active_hours", [])
        chase_ratio = meta.get("chase_ratio", 0.0)
        avg_burst = meta.get("avg_burst_length", 1.0)

        if not active_hours:
            logger.info("persona 无 active_hours，跳过每日调度")
            return []

        # 决定每天发几次
        if chase_ratio == 0 and avg_burst < 2:
            count = 1
        else:
            count = random.choice([1, 2])

        now = datetime.now(TIMEZONE)
        today = now.date()
        threshold = now - timedelta(hours=2)

        # 优先选 routine 中空闲的时段
        routine_path = PROFILES_DIR / f"{PERSONA_NAME}_routine.json"
        if routine_path.exists():
            try:
                from remember_me.analyzer.routine import DailyRoutine
                routine = DailyRoutine.load(routine_path)
                slots = routine.weekend_slots if now.weekday() >= 5 else routine.weekday_slots
                free_hours = [s.hour for s in slots if s.responsiveness >= 0.7]
                if free_hours:
                    preferred = [h for h in active_hours if h in free_hours]
                    if preferred:
                        active_hours = preferred
            except Exception:
                pass

        # 先过滤出“今天仍可触发”的小时，再抽样，避免抽到过期时段导致整天无计划。
        eligible_hours = []
        for hour in active_hours:
            latest_in_hour = datetime(today.year, today.month, today.day, hour, 59, tzinfo=TIMEZONE)
            if latest_in_hour >= threshold:
                eligible_hours.append(hour)
        if not eligible_hours:
            return []

        chosen_hours = random.sample(eligible_hours, min(count, len(eligible_hours)))

        times = []
        for hour in chosen_hours:
            min_minute = 0
            if threshold.date() == today and hour == threshold.hour:
                min_minute = max(0, min(59, threshold.minute))
            minute = random.randint(min_minute, 59)
            dt = datetime(today.year, today.month, today.day, hour, minute, tzinfo=TIMEZONE)
            # 每日计划只包含“今天”能触发的时刻。
            # 若该时段已过去太久（超过 2 小时），直接跳过，避免被推到明天后在午夜刷新时丢失。
            if dt < threshold:
                continue
            times.append(dt)

        times.sort()
        return times

    async def _daily_scheduler(self):
        """后台任务：每日主动消息调度，每 30 秒检查一次。"""
        await asyncio.sleep(5)  # 等 Bot 完全启动

        while True:
            try:
                now = datetime.now(TIMEZONE)
                today_str = now.strftime("%Y-%m-%d")

                # 新的一天或首次启动 → 生成计划 + 更新知识库
                if self._daily_date != today_str:
                    self._daily_times = self._plan_daily_times()
                    self._daily_date = today_str
                    if self._daily_times:
                        time_strs = [t.strftime("%H:%M") for t in self._daily_times]
                        logger.info("每日主动消息计划 [%s]: %s", today_str, ", ".join(time_strs))
                    # 触发知识库每日更新
                    self._track_task(asyncio.create_task(self._update_knowledge()), "daily_knowledge_update")
                    # 刷新空间日程（run_in_executor 避免阻塞事件循环）
                    if self._controller and self._controller._routine and self._controller._engine:
                        try:
                            ctrl = self._controller
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: ctrl._engine.regenerate_daily_schedule(
                                    ctrl._routine, now.weekday(), today_str,
                                ),
                            )
                        except Exception as e:
                            logger.warning("每日空间日程刷新失败: %s", e)

                # 检查是否到了计划时刻（每次最多触发 1 条，避免连发）
                remaining = []
                sent = False
                for planned_time in self._daily_times:
                    if not sent and now >= planned_time:
                        await self._try_send_daily_message()
                        sent = True
                    else:
                        remaining.append(planned_time)
                self._daily_times = remaining

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("每日调度异常: %s", e)

            await asyncio.sleep(30)

    async def _update_knowledge(self):
        """每日知识库更新（后台运行）。"""
        try:
            from remember_me.analyzer.persona import Persona
            from remember_me.knowledge.fetcher import KnowledgeFetcher
            from remember_me.knowledge.store import KnowledgeStore

            profile_path = PROFILES_DIR / f"{PERSONA_NAME}.json"
            if not profile_path.exists():
                return

            persona = Persona.load(profile_path)
            if not getattr(persona, "topic_interests", None):
                return

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return

            from google import genai
            client = genai.Client(api_key=api_key)

            kb_dir = DATA_DIR / "knowledge" / PERSONA_NAME
            chroma_dir = DATA_DIR / "chroma" / PERSONA_NAME
            images_dir = kb_dir / "images"

            store = KnowledgeStore(
                chroma_dir=chroma_dir, knowledge_dir=kb_dir,
                persona_name=PERSONA_NAME,
            )
            fetcher = KnowledgeFetcher(
                persona_name=PERSONA_NAME,
                topic_interests=persona.topic_interests,
                client=client,
                images_dir=images_dir,
            )

            loop = asyncio.get_event_loop()
            items = await loop.run_in_executor(None, fetcher.fetch_daily)
            if items:
                await loop.run_in_executor(None, store.add_items, items)
                logger.info("知识库更新完成: %d 条新知识", len(items))

            # 清理过期条目
            await loop.run_in_executor(None, store.evict)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("知识库更新失败: %s", e)

    async def _try_send_daily_message(self):
        """尝试发送每日主动消息。优先发送待跟进事件的追问。"""
        # 没有目标 chat_id（用户从未和 bot 交互过）
        async with self._send_lock:
            chat_id = self._chat_id
            if not chat_id and self._allowed_users:
                chat_id = next(iter(self._allowed_users))
        if not chat_id:
            logger.info("每日消息跳过：无目标 chat_id")
            return

        # 最近 30 分钟有互动 → 跳过（已经在聊了）
        if self._last_user_activity and time.time() - self._last_user_activity < 1800:
            logger.info("每日消息跳过：最近 30 分钟有互动")
            return

        logger.info("发送每日主动消息...")

        try:
            # 确保 session 存活
            ok = await self._ensure_session(chat_id, no_greet=True)
            if not ok:
                return

            controller = self._controller
            if not controller._engine or not controller._topic_starter:
                return

            loop = asyncio.get_event_loop()
            msgs = None

            # 优先级 1：检查待跟进事件
            if controller._event_tracker:
                due_events = controller._event_tracker.get_due_events()
                if due_events:
                    event = due_events[0]
                    event_input = f"{event.event}\n{event.context}\n{event.followup_hint}"
                    event_policy = controller._engine.plan_rhythm_policy(
                        kind="event_followup",
                        user_input=event_input,
                    )
                    event_system = None
                    if hasattr(controller._engine, "build_system_for_generation"):
                        try:
                            event_system = controller._engine.build_system_for_generation(event_input)
                        except Exception:
                            event_system = None
                    logger.info("每日消息触发事件追问: %s", event.event)
                    msgs = await loop.run_in_executor(
                        None,
                        lambda: self._call_topic_starter(
                            controller._topic_starter.generate_event_followup,
                            event.event,
                            event.context,
                            event.followup_hint,
                            count_policy=event_policy,
                            system_instruction=event_system,
                        ),
                    )
                    if msgs:
                        controller._event_tracker.mark_done(event.id)

            # 优先级 2：常规新话题
            if not msgs:
                if not controller._engine:
                    return
                ctx = self._get_engine_context(controller._engine, max_chars=460)
                proactive_policy = controller._engine.plan_rhythm_policy(
                    kind="proactive",
                    user_input=ctx,
                )
                proactive_system = None
                if hasattr(controller._engine, "build_system_for_generation"):
                    try:
                        proactive_system = controller._engine.build_system_for_generation(ctx)
                    except Exception:
                        proactive_system = None
                msgs = await loop.run_in_executor(
                    None,
                    lambda: self._call_topic_starter(
                        controller._topic_starter.generate,
                        recent_context=ctx,
                        count_policy=proactive_policy,
                        system_instruction=proactive_system,
                    ),
                )

            msgs = [m for m in (msgs or []) if m and m.strip()]
            if not msgs:
                logger.info("每日消息生成为空")
                return
            if self._is_duplicate_proactive(msgs):
                logger.info("每日消息跳过：与近期主动消息重复")
                return

            # 注入对话历史并发送（加锁防止与 send_message 并发修改 engine 状态）
            async with self._send_lock:
                controller._engine.inject_proactive_message(msgs)
                controller._update_activity()
                bot = self._app.bot
                await self._deliver_messages(bot, chat_id, msgs, first_delay_phase="followup")
                self._mark_proactive_sent(msgs)
            logger.info("每日主动消息已发送: %d 条", len(msgs))

        except Exception as e:
            logger.exception("每日主动消息发送失败: %s", e)

    # ── Session 管理 ──

    async def _ensure_session(self, chat_id: int, no_greet: bool = False) -> bool:
        """确保 controller 已启动。返回 True 表示就绪。"""
        if self._controller and self._chat_id == chat_id:
            return True

        # 已有其他用户在用
        if self._controller and self._chat_id != chat_id:
            return False

        controller = ChatController(PERSONA_NAME)
        bot = self._app.bot

        def on_message(msgs: list[str], msg_type: str):
            is_followup = msg_type in {"proactive", "greet"}
            async def _deliver_task():
                clean = [m for m in (msgs or []) if m and str(m).strip()]
                if not clean:
                    return
                async with self._send_lock:
                    if not self._controller or self._chat_id != chat_id:
                        return
                    if is_followup and time.time() - self._last_user_activity < 8:
                        logger.info("主动消息跳过：用户刚有输入活动")
                        return
                    if is_followup and self._is_duplicate_proactive(clean):
                        logger.info("主动消息跳过：命中 Telegram 去重")
                        return
                    await self._deliver_messages(
                        bot, chat_id, clean,
                        first_delay_phase="followup" if is_followup else "first",
                    )
                    if is_followup:
                        self._mark_proactive_sent(clean)

            self._track_task(asyncio.create_task(_deliver_task()), f"deliver_{msg_type}")

        def on_typing(is_typing: bool):
            if is_typing:
                self._track_task(
                    asyncio.create_task(self._send_typing(bot, chat_id)),
                    "typing_indicator",
                )

        try:
            await controller.start(
                on_message=on_message, on_typing=on_typing, no_greet=no_greet,
            )
        except Exception:
            # start() 失败时不写入状态，下次调用会重新初始化
            logger.exception("session 初始化失败")
            return False

        self._chat_id = chat_id
        self._controller = controller
        return True

    # ── 命令处理 ──

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("你没有权限使用此 Bot。")
            return

        chat_id = update.effective_chat.id
        self._last_user_activity = time.time()
        ok = await self._ensure_session(chat_id)
        if not ok:
            await update.message.reply_text("Bot 正在被其他用户使用。")
            return

        status = "（续接上次对话）" if self._controller.session_loaded else ""
        total = self._controller.get_total_messages()
        await update.message.reply_text(
            f"已连接 {PERSONA_NAME}{status}\n"
            f"共 {total} 条历史消息\n\n"
            f"直接发消息开始对话，/stop 结束。"
        )

    def _sync_notes_to_engine(self, notes: list[str]):
        """将备注同步到运行中的记忆治理层（仅短期上下文，不覆盖导入核心）。"""
        if self._controller and self._controller._memory_governance and self._controller._persona:
            self._controller._memory_governance.replace_manual_notes(notes, persona=self._controller._persona)

    def _get_relationship_store(self) -> RelationshipMemoryStore:
        if self._controller and self._controller._relationship_store:
            return self._controller._relationship_store
        return RelationshipMemoryStore(PERSONA_NAME, data_dir=DATA_DIR)

    @staticmethod
    def _format_relationship_item(idx: int, fact) -> str:
        status_label = {
            "confirmed": "已确认",
            "candidate": "待确认",
            "rejected": "已拒绝",
        }.get(getattr(fact, "status", "candidate"), "未知")
        fact_type = str(getattr(fact, "type", "") or "")
        type_label = _REL_TYPE_LABEL.get(fact_type, fact_type)
        content = str(getattr(fact, "content", "") or "").strip()
        conf = float(getattr(fact, "confidence", 0.0) or 0.0)
        evidence = list(getattr(fact, "evidence", []) or [])
        suffix = f"（置信 {conf:.2f}）"
        if evidence:
            suffix += f" 证据:{evidence[0][:20]}"
        return f"{idx}. [{status_label}][{type_label}] {content} {suffix}"

    async def _cmd_note_rel(self, update: Update, raw: str, parts: list[str]):
        if len(parts) < 3:
            await update.message.reply_text(
                "用法：\n"
                "/note rel list [type|all|rejected]\n"
                "/note rel confirm <序号>\n"
                "/note rel reject <序号> [原因]\n\n"
                f"type 可选：{', '.join(RELATION_TYPES)}"
            )
            return

        store = self._get_relationship_store()
        subcmd = parts[2].lower()

        if subcmd == "list":
            fact_type = None
            statuses = {"candidate", "confirmed"}
            include_conflict = False
            if len(parts) >= 4:
                arg = parts[3].strip().lower()
                if arg == "all":
                    statuses = {"candidate", "confirmed", "rejected"}
                    include_conflict = True
                elif arg == "rejected":
                    statuses = {"rejected"}
                    include_conflict = True
                else:
                    fact_type = arg
                    if fact_type not in RELATION_TYPES:
                        await update.message.reply_text(
                            f"type 无效，可选：{', '.join(RELATION_TYPES)}"
                        )
                        return
            rows = store.list_facts(
                fact_type=fact_type,
                statuses=statuses,
                include_conflict=include_conflict,
                limit=50,
            )
            if not rows:
                await update.message.reply_text("暂无关系记忆记录。")
                return
            lines = [self._format_relationship_item(i + 1, fact) for i, fact in enumerate(rows)]
            await update.message.reply_text("\n".join(lines))
            return

        if subcmd == "confirm":
            if len(parts) < 4:
                await update.message.reply_text("用法：/note rel confirm <序号>")
                return
            try:
                idx = int(parts[3])
            except ValueError:
                await update.message.reply_text("序号必须是数字。")
                return
            rows = store.list_facts(
                statuses={"candidate", "confirmed"},
                include_conflict=True,
                limit=200,
            )
            if idx < 1 or idx > len(rows):
                await update.message.reply_text(f"序号超出范围（1-{len(rows)}）。")
                return
            fact = rows[idx - 1]
            ok = store.confirm_fact(fact.id)
            if not ok:
                await update.message.reply_text("该记录无法确认（可能与核心人格冲突）。")
                return
            await update.message.reply_text(f"已确认：{fact.content}")
            return

        if subcmd == "reject":
            if len(parts) < 4:
                await update.message.reply_text("用法：/note rel reject <序号> [原因]")
                return
            try:
                idx = int(parts[3])
            except ValueError:
                await update.message.reply_text("序号必须是数字。")
                return
            rows = store.list_facts(
                statuses={"candidate", "confirmed"},
                include_conflict=True,
                limit=200,
            )
            if idx < 1 or idx > len(rows):
                await update.message.reply_text(f"序号超出范围（1-{len(rows)}）。")
                return
            reason = ""
            if len(parts) >= 5:
                reason = raw.split(None, 4)[4].strip()
            if not reason:
                reason = "manual_reject"
            fact = rows[idx - 1]
            store.reject_fact(fact.id, reason=f"manual:{reason}")
            await update.message.reply_text(f"已拒绝：{fact.content}")
            return

        await update.message.reply_text(
            "未知子命令。用法：/note rel list|confirm|reject"
        )

    async def _cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        raw = (update.message.text or "").strip()
        parts = raw.split()
        if len(parts) < 2:
            await update.message.reply_text(
                "用法：\n"
                "/note add <内容> — 添加备注\n"
                "/note list — 查看所有备注\n"
                "/note del <序号> — 删除备注\n"
                "/note rel list [type|all|rejected] — 查看关系记忆\n"
                "/note rel confirm <序号> — 确认关系记忆\n"
                "/note rel reject <序号> [原因] — 拒绝关系记忆\n\n"
                "备注是短期上下文，不会覆盖导入聊天记录的人设核心。\n"
                "修改即时生效，无需重启对话。"
            )
            return

        subcmd = parts[1].lower()
        if subcmd == "rel":
            await self._cmd_note_rel(update, raw, parts)
            return

        notes = ChatController.load_notes(PERSONA_NAME)

        if subcmd == "list":
            if not notes:
                await update.message.reply_text("暂无备注。")
            else:
                lines = [f"{i+1}. {n}" for i, n in enumerate(notes)]
                await update.message.reply_text("\n".join(lines))

        elif subcmd == "add":
            if len(parts) < 3:
                await update.message.reply_text("用法：/note add <内容>")
                return
            content = raw.split(None, 2)[2].strip()
            if not content:
                await update.message.reply_text("用法：/note add <内容>")
                return
            notes.append(content)
            ChatController.save_notes(PERSONA_NAME, notes)
            self._sync_notes_to_engine(notes)
            await update.message.reply_text(f"已添加第 {len(notes)} 条备注：{content}")

        elif subcmd == "del":
            if len(parts) < 3:
                await update.message.reply_text("用法：/note del <序号>")
                return
            try:
                idx = int(parts[2]) - 1
            except ValueError:
                await update.message.reply_text("序号必须是数字。")
                return
            if idx < 0 or idx >= len(notes):
                await update.message.reply_text(f"序号超出范围（1-{len(notes)}）。")
                return
            removed = notes.pop(idx)
            ChatController.save_notes(PERSONA_NAME, notes)
            self._sync_notes_to_engine(notes)
            await update.message.reply_text(f"已删除：{removed}")

        else:
            await update.message.reply_text("未知子命令。用 /note 查看用法。")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        if not self._controller or self._chat_id != update.effective_chat.id:
            await update.message.reply_text("当前没有进行中的对话。")
            return

        async with self._coalesce_lock:
            self._cancel_coalesce_timer()
            self._coalesce_buffer.clear()
            self._coalesce_first_at = 0.0
        async with self._send_lock:
            controller = self._controller
            if not controller or self._chat_id != update.effective_chat.id:
                await update.message.reply_text("当前没有进行中的对话。")
                return
            await controller.stop()
            self._controller = None
            self._chat_id = None
        await update.message.reply_text(
            f"已结束与 {PERSONA_NAME} 的对话。会话已保存。"
        )

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id
        self._last_user_activity = time.time()

        ok = await self._ensure_session(chat_id)
        if not ok:
            await update.message.reply_text("Bot 正在被其他用户使用。")
            return

        text = update.message.text.strip()
        if not text:
            return

        async with self._coalesce_lock:
            is_first = len(self._coalesce_buffer) == 0
            self._coalesce_buffer.append(text)

            # 取消旧的定时器
            self._cancel_coalesce_timer()

            if is_first:
                # 第一条消息：记录时间戳，设置短探测延迟
                self._coalesce_first_at = time.monotonic()
            delay = self._compute_coalesce_delay(text, is_first=is_first)

            self._coalesce_timer = self._track_task(
                asyncio.create_task(self._flush_coalesce(chat_id, delay)),
                "coalesce_flush",
            )

    async def _flush_coalesce(self, chat_id: int, delay: float):
        """等待指定延迟后，将缓冲区中的消息合并发送给 LLM。"""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return

        async with self._coalesce_lock:
            if not self._coalesce_buffer:
                return
            messages = list(self._coalesce_buffer)
            self._coalesce_buffer.clear()
            self._coalesce_first_at = 0.0

        # 单条消息直接发，多条消息加聚合提示
        if len(messages) == 1:
            merged = messages[0]
        else:
            # 保留用户原始连发文本，不注入机械提示，降低 persona 漂移风险。
            merged = "\n".join(messages)

        async with self._send_lock:
            bot = self._app.bot
            controller = self._controller
            if not controller or self._chat_id != chat_id:
                logger.info("聚合消息发送前会话已结束，丢弃本次缓冲内容")
                return

            try:
                replies = await self._run_with_typing_heartbeat(
                    bot, chat_id, controller.send_message(merged),
                )
                await self._deliver_messages(bot, chat_id, replies, first_delay_phase="first")
            except Exception as e:
                logger.exception("发送消息失败")
                with contextlib.suppress(Exception):
                    await self._maybe_send_humanized_error(bot, chat_id)

    async def _handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id
        self._last_user_activity = time.time()

        ok = await self._ensure_session(chat_id)
        if not ok:
            await update.message.reply_text("Bot 正在被其他用户使用。")
            return

        # 图片消息到达时，先把缓冲区里的文字消息冲掉（合并到图片的 caption 里）
        async with self._coalesce_lock:
            self._cancel_coalesce_timer()
            buffered = list(self._coalesce_buffer)
            self._coalesce_buffer.clear()
            self._coalesce_first_at = 0.0

        # 下载最大分辨率的图片
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = await file.download_as_bytearray()

        caption = (update.message.caption or "").strip() or "[图片]"
        # 如果之前缓冲区有文字消息，合并到 caption 前面
        if buffered:
            caption = "\n".join(buffered) + "\n" + caption

        async with self._send_lock:
            bot = self._app.bot
            controller = self._controller
            if not controller or self._chat_id != chat_id:
                await update.message.reply_text("当前会话已结束，请先 /start。")
                return

            try:
                replies = await self._run_with_typing_heartbeat(
                    bot,
                    chat_id,
                    controller.send_message(
                        caption, image=(bytes(image_bytes), "image/jpeg"),
                    ),
                )
                await self._deliver_messages(bot, chat_id, replies, first_delay_phase="first")
            except Exception as e:
                logger.exception("发送图片消息失败")
                with contextlib.suppress(Exception):
                    await self._maybe_send_humanized_error(bot, chat_id)

    async def _deliver_messages(
        self,
        bot,
        chat_id: int,
        msgs: list[str],
        first_delay_phase: str = "first",
    ):
        """逐条发送消息（带打字间隔）。"""
        msgs = [m for m in msgs if m and m.strip()]
        if not msgs:
            return

        # 获取情绪+空间驱动的延迟系数（已内化到属性中）
        delay_factor = 1.0
        engine = None
        if self._controller and self._controller._engine:
            engine = self._controller._engine
            delay_factor = engine.reply_delay_factor

        for i, msg in enumerate(msgs):
            # 第一条也加延迟，区分正常回复 / 主动接话
            if i == 0:
                if engine:
                    delay = engine.sample_inter_message_delay(first_delay_phase) * delay_factor
                else:
                    delay = (0.45 + random.random() * 0.9) * delay_factor
                delay *= self._typing_delay_scale(msg)
                await asyncio.sleep(max(0.05, delay))
                await self._send_typing(bot, chat_id)

            if msg.startswith("[sticker:"):
                sticker_path = Path(msg[9:].rstrip("]"))
                try:
                    if sticker_path.exists():
                        with open(sticker_path, "rb") as f:
                            await self._send_photo(bot, chat_id, f)
                    else:
                        logger.warning("表情包文件不存在: %s", sticker_path)
                except Exception as e:
                    logger.warning("发送表情包失败: %s", e)
            else:
                for chunk in self._split_telegram_text(msg):
                    await self._send_text(bot, chat_id, chunk)

            if i < len(msgs) - 1:
                if engine:
                    delay = engine.sample_inter_message_delay("burst") * delay_factor
                else:
                    delay = (0.4 + random.random() * 0.8) * delay_factor
                delay *= self._typing_delay_scale(msgs[i + 1])
                await asyncio.sleep(delay)
                await self._send_typing(bot, chat_id)

    def run(self):
        """启动 Bot（阻塞）。"""
        self._app = Application.builder().token(self._token).build()

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("note", self._cmd_note))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO, self._handle_photo)
        )
        self._app.add_error_handler(self._on_app_error)

        async def post_init(app: Application):
            await app.bot.set_my_commands([
                BotCommand("start", "开始对话"),
                BotCommand("stop", "结束对话"),
                BotCommand("note", "管理备注（手动补充信息）"),
            ])
            # 启动每日调度
            self._track_task(asyncio.create_task(self._daily_scheduler()), "daily_scheduler")

        async def post_shutdown(app: Application):
            await self._shutdown_background_tasks()
            if self._controller:
                with contextlib.suppress(Exception):
                    await self._controller.stop()
            self._controller = None
            self._chat_id = None

        self._app.post_init = post_init
        self._app.post_shutdown = post_shutdown

        logger.info("Telegram Bot 启动中... persona=%s", PERSONA_NAME)
        self._app.run_polling(drop_pending_updates=True)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("请设置 TELEGRAM_BOT_TOKEN 环境变量")
        raise SystemExit(1)

    allowed_env = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    allowed_users = None
    if allowed_env:
        try:
            allowed_users = {int(uid.strip()) for uid in allowed_env.split(",") if uid.strip()}
        except ValueError:
            print("TELEGRAM_ALLOWED_USERS 格式错误，应为逗号分隔的用户 ID")
            raise SystemExit(1)

    bot = TelegramBot(token=token, allowed_users=allowed_users)
    bot.run()


if __name__ == "__main__":
    main()
