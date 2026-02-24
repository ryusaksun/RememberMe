"""RememberMe Telegram Bot — 单 persona 模式。"""

from __future__ import annotations

import asyncio
import logging
import os
import random

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


class TelegramBot:
    """单 persona Telegram Bot。"""

    def __init__(self, token: str, allowed_users: set[int] | None = None):
        self._token = token
        self._allowed_users = allowed_users
        self._controller: ChatController | None = None
        self._chat_id: int | None = None  # 当前绑定的 chat
        self._send_lock = asyncio.Lock()
        self._app: Application | None = None

    def _is_allowed(self, user_id: int) -> bool:
        if self._allowed_users is None:
            return True
        return user_id in self._allowed_users

    async def _ensure_session(self, chat_id: int) -> bool:
        """确保 controller 已启动。返回 True 表示就绪。"""
        if self._controller and self._chat_id == chat_id:
            return True

        # 已有其他用户在用
        if self._controller and self._chat_id != chat_id:
            return False

        self._chat_id = chat_id
        self._controller = ChatController(PERSONA_NAME)
        bot = self._app.bot

        def on_message(msgs: list[str], msg_type: str):
            asyncio.create_task(self._deliver_messages(bot, chat_id, msgs))

        def on_typing(is_typing: bool):
            if is_typing:
                asyncio.create_task(
                    bot.send_chat_action(chat_id, ChatAction.TYPING)
                )

        await self._controller.start(on_message=on_message, on_typing=on_typing)
        return True

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("你没有权限使用此 Bot。")
            return

        chat_id = update.effective_chat.id
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

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        if not self._controller or self._chat_id != update.effective_chat.id:
            await update.message.reply_text("当前没有进行中的对话。")
            return

        await self._controller.stop()
        self._controller = None
        self._chat_id = None
        await update.message.reply_text(
            f"已结束与 {PERSONA_NAME} 的对话。会话已保存。"
        )

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_allowed(update.effective_user.id):
            return

        chat_id = update.effective_chat.id

        # 自动启动 session
        ok = await self._ensure_session(chat_id)
        if not ok:
            await update.message.reply_text("Bot 正在被其他用户使用。")
            return

        text = update.message.text.strip()
        if not text:
            return

        async with self._send_lock:
            bot = self._app.bot
            await bot.send_chat_action(chat_id, ChatAction.TYPING)

            try:
                replies = await self._controller.send_message(text)
                await self._deliver_messages(bot, chat_id, replies)
            except Exception as e:
                logger.exception("发送消息失败")
                await update.message.reply_text(f"出错了：{e}")

    async def _deliver_messages(self, bot, chat_id: int, msgs: list[str]):
        """逐条发送消息（带打字间隔）。"""
        msgs = [m for m in msgs if m and m.strip()]
        if not msgs:
            return

        for i, msg in enumerate(msgs):
            if msg.startswith("[sticker:"):
                await bot.send_message(chat_id, "[表情包]")
            else:
                await bot.send_message(chat_id, msg)

            if i < len(msgs) - 1:
                delay = 0.4 + random.random() * 0.8
                await asyncio.sleep(delay)
                await bot.send_chat_action(chat_id, ChatAction.TYPING)

    def run(self):
        """启动 Bot（阻塞）。"""
        self._app = Application.builder().token(self._token).build()

        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        async def post_init(app: Application):
            await app.bot.set_my_commands([
                BotCommand("start", "开始对话"),
                BotCommand("stop", "结束对话"),
            ])

        self._app.post_init = post_init

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
