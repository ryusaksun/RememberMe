"""聊天页面 — 终端风格对话界面。"""

from __future__ import annotations

import asyncio
import queue
import random
from datetime import datetime

from nicegui import ui

from remember_me.controller import ChatController
from remember_me.gui.components.message_bubble import render_message, render_system_message
from remember_me.gui.components.typing_indicator import TypingIndicator
from remember_me.gui.theme import (
    BG_INPUT, BG_PRIMARY, BG_SECONDARY, NEON_CYAN, NEON_GREEN, NEON_MAGENTA, TEXT_DIM,
)


def create_chat_page(persona_name: str):
    """创建聊天页面。"""
    controller = ChatController(persona_name)
    typing_indicator = TypingIndicator(persona_name)
    messages_container: ui.element | None = None
    input_ref: ui.input | None = None
    sending = False

    # 消息队列：后台 task → UI 线程（避免 NiceGUI slot 上下文问题）
    _msg_queue: queue.Queue = queue.Queue()
    _typing_queue: queue.Queue = queue.Queue()

    ui.query("body").style(f"background: {BG_PRIMARY}; margin: 0; overflow: hidden;")

    # 外层容器：占满视口，flex 纵向排列
    with ui.column().style(
        "width: 100%; max-width: 800px; height: 100vh; margin: 0 auto; "
        "display: flex; flex-direction: column; padding: 0; gap: 0;"
    ):
        # ── 顶部标题栏 ──
        with ui.row().style(
            f"width: 100%; min-height: 48px; padding: 10px 16px; "
            f"border-bottom: 1px solid rgba(0,255,213,0.2); "
            f"background: {BG_SECONDARY}; flex-shrink: 0; "
            f"display: flex; align-items: center; justify-content: space-between; "
            f"gap: 12px; flex-wrap: nowrap;"
        ):
            # 左侧：返回 + 标题
            with ui.row().style(
                "display: flex; align-items: center; gap: 10px; flex-wrap: nowrap; "
                "overflow: hidden; min-width: 0;"
            ):
                ui.icon("arrow_back", size="20px").style(
                    f"color: {TEXT_DIM}; cursor: pointer; flex-shrink: 0;"
                ).on("click", lambda: _handle_back(controller))

                ui.label(f"═══ LINK: {persona_name} ═══").style(
                    f"color: {NEON_CYAN}; font-size: 13px; letter-spacing: 2px; "
                    f"white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
                )

            # 右侧：状态
            ui.label("[LIVE]").style(
                f"color: {NEON_GREEN}; font-size: 11px; flex-shrink: 0; "
                f"text-shadow: 0 0 8px rgba(57,255,20,0.5);"
            )

        # ── 消息区域（flex: 1 填满中间） ──
        with ui.scroll_area().style(
            f"flex: 1; width: 100%; background: {BG_PRIMARY}; min-height: 0;"
        ) as scroll_area:
            messages_container = ui.column().style(
                "width: 100%; padding: 16px 20px; gap: 2px;"
            )

        # typing indicator 放在消息区和输入栏之间
        typing_container = ui.row().style(
            f"width: 100%; padding: 2px 20px; flex-shrink: 0; min-height: 0; "
            f"background: {BG_PRIMARY};"
        )
        typing_indicator.create(typing_container)

        # ── 输入栏 ──
        with ui.row().style(
            f"width: 100%; padding: 8px 16px; "
            f"border-top: 1px solid rgba(0,255,213,0.2); "
            f"background: {BG_SECONDARY}; flex-shrink: 0; "
            f"display: flex; align-items: center; gap: 8px; flex-wrap: nowrap;"
        ):
            ui.label("$").style(
                f"color: {NEON_GREEN}; font-size: 14px; flex-shrink: 0;"
            )

            input_ref = ui.input(placeholder="输入消息...").props(
                "borderless dense"
            ).style(
                f"flex: 1; color: {NEON_GREEN}; font-size: 14px; caret-color: {NEON_GREEN};"
            )

            ui.icon("send", size="22px").style(
                f"color: {NEON_CYAN}; cursor: pointer; flex-shrink: 0;"
            ).on("click", lambda: _handle_send())

    # ── 事件处理 ──
    async def _handle_send():
        nonlocal sending
        if sending or not input_ref:
            return
        text = input_ref.value.strip() if input_ref.value else ""
        if not text:
            return

        sending = True
        input_ref.value = ""

        # 显示用户消息
        with messages_container:
            render_message(text, "you", is_target=False)
        await _scroll_bottom()

        # 发送并逐条显示回复
        try:
            replies = await controller.send_message(text)
            replies = [m for m in replies if m and m.strip()]
            for i, msg in enumerate(replies):
                await asyncio.sleep(0.4 + random.random() * 0.8)
                with messages_container:
                    render_message(
                        msg, persona_name, is_target=True,
                        is_burst_continuation=(i > 0),
                    )
                await _scroll_bottom()
        except Exception as e:
            with messages_container:
                render_system_message(f"出错了: {e}")
            await _scroll_bottom()
        finally:
            sending = False

    # 回车发送
    input_ref.on("keydown.enter", lambda: _handle_send())

    def _on_message(msgs: list[str], msg_type: str):
        """回调：收到主动消息或开场消息（可能从后台 task 调用，通过队列传递）。"""
        _msg_queue.put((list(msgs), msg_type))

    def _on_typing(is_typing: bool):
        """回调：显示/隐藏输入状态（可能从后台 task 调用，通过队列传递）。"""
        _typing_queue.put(is_typing)

    async def _poll_queues():
        """轮询消息队列，在 NiceGUI UI 上下文中安全地更新界面。"""
        # 处理 typing 状态
        while not _typing_queue.empty():
            try:
                is_typing = _typing_queue.get_nowait()
                if is_typing:
                    typing_indicator.show()
                else:
                    typing_indicator.hide()
            except queue.Empty:
                break

        # 处理消息
        if not _msg_queue.empty():
            try:
                msgs, msg_type = _msg_queue.get_nowait()
                msgs_clean = [m for m in msgs if m and m.strip()]
                for i, msg in enumerate(msgs_clean):
                    if i > 0:
                        await asyncio.sleep(0.4 + random.random() * 0.8)
                    with messages_container:
                        render_message(
                            msg, persona_name, is_target=True,
                            is_burst_continuation=(i > 0),
                        )
                    await _scroll_bottom()
            except queue.Empty:
                pass

    # 每 0.3 秒轮询队列（在 NiceGUI 上下文中执行，安全操作 UI）
    ui.timer(0.3, _poll_queues)

    async def _scroll_bottom():
        await asyncio.sleep(0.05)
        scroll_area.scroll_to(percent=1.0)

    # ── 初始化 ──
    async def _init():
        total = controller.get_total_messages()

        with messages_container:
            render_system_message(f"正在连接 {persona_name} 的记忆...")
        await _scroll_bottom()

        try:
            await controller.start(
                on_message=_on_message,
                on_typing=_on_typing,
            )
            session_hint = "（续接上次对话）" if controller.session_loaded else ""
            with messages_container:
                render_system_message(
                    f"连接成功 — 共 {total} 条历史消息 {session_hint}"
                )
                render_system_message("输入消息开始对话")
            await _scroll_bottom()
        except Exception as e:
            with messages_container:
                render_system_message(f"初始化失败: {e}")
            await _scroll_bottom()

    ui.timer(0.1, _init, once=True)


async def _handle_back(controller: ChatController):
    """返回首页前保存会话。"""
    await controller.stop()
    ui.navigate.to("/")
