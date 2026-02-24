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
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from remember_me.controller import ChatController

logger = logging.getLogger(__name__)

PERSONA_NAME = os.environ.get("PERSONA_NAME", "阴暗扭曲爬行_-_-")
TIMEZONE = ZoneInfo(os.environ.get("TZ", "Asia/Shanghai"))
DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
_SENTENCE_END_RE = re.compile(r"[。！？!?~…]$")


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
            "chase_ratio": getattr(persona, "chase_ratio", 0.0),
            "avg_burst_length": getattr(persona, "avg_burst_length", 1.0),
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

        # 从 active_hours 中随机选 count 个不同的小时
        chosen_hours = random.sample(active_hours, min(count, len(active_hours)))

        times = []
        for hour in chosen_hours:
            minute = random.randint(0, 59)
            dt = datetime(today.year, today.month, today.day, hour, minute, tzinfo=TIMEZONE)
            # 如果这个时间跨午夜（比如 hour=0 或 1），可能是"今天的凌晨"或"明天的凌晨"
            # 如果时间已过且距现在超过 2 小时，推到明天
            if dt < now - timedelta(hours=2):
                dt += timedelta(days=1)
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
        if not self._chat_id and not self._allowed_users:
            logger.info("每日消息跳过：无目标 chat_id")
            return
        # 如果有白名单但没有 chat_id，用白名单第一个用户
        chat_id = self._chat_id
        if not chat_id and self._allowed_users:
            chat_id = next(iter(self._allowed_users))

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
                    logger.info("每日消息触发事件追问: %s", event.event)
                    msgs = await loop.run_in_executor(
                        None, lambda: controller._topic_starter.generate_event_followup(
                            event.event, event.context, event.followup_hint
                        )
                    )
                    if msgs:
                        controller._event_tracker.mark_done(event.id)

            # 优先级 2：常规新话题
            if not msgs:
                ctx = controller._engine.get_recent_context() if controller._engine else ""
                msgs = await loop.run_in_executor(
                    None, lambda: controller._topic_starter.generate(recent_context=ctx)
                )

            msgs = [m for m in (msgs or []) if m and m.strip()]
            if not msgs:
                logger.info("每日消息生成为空")
                return

            # 注入对话历史
            controller._engine.inject_proactive_message(msgs)
            controller._update_activity()

            # 发送到 Telegram
            bot = self._app.bot
            await self._deliver_messages(bot, chat_id, msgs, first_delay_phase="followup")
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
            self._track_task(
                asyncio.create_task(
                    self._deliver_messages(
                        bot, chat_id, msgs,
                        first_delay_phase="followup" if is_followup else "first",
                    )
                ),
                f"deliver_{msg_type}",
            )

        def on_typing(is_typing: bool):
            if is_typing:
                self._track_task(
                    asyncio.create_task(bot.send_chat_action(chat_id, ChatAction.TYPING)),
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

    async def _cmd_note(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        args = update.message.text.split(maxsplit=2)  # /note [subcmd] [content]
        if len(args) < 2:
            await update.message.reply_text(
                "用法：\n"
                "/note add <内容> — 添加备注\n"
                "/note list — 查看所有备注\n"
                "/note del <序号> — 删除备注\n\n"
                "备注是短期上下文，不会覆盖导入聊天记录的人设核心。\n"
                "修改即时生效，无需重启对话。"
            )
            return

        subcmd = args[1].lower()
        notes = ChatController.load_notes(PERSONA_NAME)

        if subcmd == "list":
            if not notes:
                await update.message.reply_text("暂无备注。")
            else:
                lines = [f"{i+1}. {n}" for i, n in enumerate(notes)]
                await update.message.reply_text("\n".join(lines))

        elif subcmd == "add":
            if len(args) < 3 or not args[2].strip():
                await update.message.reply_text("用法：/note add <内容>")
                return
            content = args[2].strip()
            notes.append(content)
            ChatController.save_notes(PERSONA_NAME, notes)
            self._sync_notes_to_engine(notes)
            await update.message.reply_text(f"已添加第 {len(notes)} 条备注：{content}")

        elif subcmd == "del":
            if len(args) < 3:
                await update.message.reply_text("用法：/note del <序号>")
                return
            try:
                idx = int(args[2]) - 1
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
            merged = f"（对方连续发了 {len(messages)} 条消息）\n" + "\n".join(messages)

        async with self._send_lock:
            bot = self._app.bot
            controller = self._controller
            if not controller or self._chat_id != chat_id:
                logger.info("聚合消息发送前会话已结束，丢弃本次缓冲内容")
                return
            await bot.send_chat_action(chat_id, ChatAction.TYPING)

            try:
                replies = await controller.send_message(merged)
                await self._deliver_messages(bot, chat_id, replies, first_delay_phase="first")
            except Exception as e:
                logger.exception("发送消息失败")
                await bot.send_message(chat_id, f"出错了：{e}")

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
            await bot.send_chat_action(chat_id, ChatAction.TYPING)

            try:
                replies = await controller.send_message(
                    caption, image=(bytes(image_bytes), "image/jpeg"),
                )
                await self._deliver_messages(bot, chat_id, replies, first_delay_phase="first")
            except Exception as e:
                logger.exception("发送图片消息失败")
                await update.message.reply_text(f"出错了：{e}")

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

        # 获取情绪驱动的延迟系数
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
                await asyncio.sleep(max(0.05, delay))
                await bot.send_chat_action(chat_id, ChatAction.TYPING)

            if msg.startswith("[sticker:"):
                sticker_path = Path(msg[9:].rstrip("]"))
                try:
                    if sticker_path.exists():
                        with open(sticker_path, "rb") as f:
                            await bot.send_photo(chat_id, photo=f)
                    else:
                        logger.warning("表情包文件不存在: %s", sticker_path)
                except Exception as e:
                    logger.warning("发送表情包失败: %s", e)
            else:
                await bot.send_message(chat_id, msg)

            if i < len(msgs) - 1:
                if engine:
                    delay = engine.sample_inter_message_delay("burst") * delay_factor
                else:
                    delay = (0.4 + random.random() * 0.8) * delay_factor
                await asyncio.sleep(delay)
                await bot.send_chat_action(chat_id, ChatAction.TYPING)

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
