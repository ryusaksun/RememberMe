"""导入流程页面 — 终端风格步骤式导入。"""

from __future__ import annotations

import asyncio

from nicegui import events, ui

from remember_me.controller import ChatController
from remember_me.gui.components.message_bubble import render_system_message
from remember_me.gui.theme import (
    BG_INPUT, BG_PRIMARY, BG_SECONDARY, NEON_CYAN, NEON_GREEN, NEON_MAGENTA, TEXT_DIM, TEXT_PRIMARY,
)


def create_import_page():
    """创建导入流程页面。"""
    ui.query("body").style(f"background: {BG_PRIMARY};")

    log_container: ui.element | None = None
    uploaded_path: str | None = None

    with ui.column().classes("w-full items-center").style(
        "min-height: 100vh; max-width: 700px; margin: 0 auto; padding: 20px;"
    ):
        # 标题
        with ui.row().classes("w-full items-center gap-3"):
            ui.icon("arrow_back", size="20px").style(
                f"color: {TEXT_DIM}; cursor: pointer;"
            ).on("click", lambda: ui.navigate.to("/"))

            ui.html(
                f'<span style="color: {NEON_MAGENTA}; font-size: 14px; letter-spacing: 2px;">'
                f'═══ IMPORT NEW TARGET ═══</span>'
            )

        ui.separator().style(
            f"margin: 20px 0; border-color: rgba(255,0,170,0.2); width: 100%;"
        )

        # 表单
        with ui.column().classes("w-full gap-4"):
            # 目标名字
            ui.html(f'<span style="color: {NEON_CYAN}; font-size: 12px;">[STEP 1/3] 目标信息</span>')

            target_input = ui.input(
                label="目标人物名字",
                placeholder="聊天记录中的显示名",
            ).props("borderless dense").classes("w-full").style(
                f"border: 1px solid rgba(0,255,213,0.2); border-radius: 4px; padding: 4px 8px;"
            )

            user_input = ui.input(
                label="你的名字（可选）",
                placeholder="留空则自动检测",
            ).props("borderless dense").classes("w-full").style(
                f"border: 1px solid rgba(0,255,213,0.2); border-radius: 4px; padding: 4px 8px;"
            )

            fmt_select = ui.select(
                options={"text": "纯文本 (text)", "json": "JSON", "wechat": "微信导出"},
                value="text",
                label="文件格式",
            ).props("borderless dense").classes("w-full").style(
                f"border: 1px solid rgba(0,255,213,0.2); border-radius: 4px; padding: 4px 8px;"
            )

            # 文件上传
            ui.html(f'<span style="color: {NEON_CYAN}; font-size: 12px;">[STEP 2/3] 上传文件</span>')

            upload_label = ui.html(
                f'<span style="color: {TEXT_DIM}; font-size: 12px;">'
                f'尚未选择文件</span>'
            )

            async def on_upload(e: events.UploadEventArguments):
                nonlocal uploaded_path
                # NiceGUI 3.x: e.file 是 FileUpload 对象，.read() 是 async
                import tempfile
                data = await e.file.read()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
                tmp.write(data)
                tmp.close()
                uploaded_path = tmp.name
                upload_label.content = (
                    f'<span style="color: {NEON_GREEN}; font-size: 12px;">'
                    f'[OK] 文件已上传: {_escape(e.file.name)}</span>'
                )

            ui.upload(
                label="选择聊天记录文件",
                on_upload=on_upload,
                auto_upload=True,
            ).props("flat bordered").classes("w-full").style(
                f"border: 1px solid rgba(0,255,213,0.2); "
                f"color: {TEXT_PRIMARY};"
            )

            # 执行导入
            ui.html(f'<span style="color: {NEON_CYAN}; font-size: 12px;">[STEP 3/3] 执行导入</span>')

            # 进度日志
            with ui.scroll_area().classes("w-full").style(
                f"height: 140px; background: {BG_SECONDARY}; border: 1px solid rgba(0,255,213,0.1); "
                f"border-radius: 4px; padding: 10px;"
            ) as log_scroll:
                log_container = ui.column().classes("w-full gap-1")

            import_btn = ui.button(
                "INITIATE IMPORT",
                on_click=lambda: _run_import(),
            ).props("flat no-caps").style(
                f"color: {NEON_GREEN}; border: 1px solid rgba(57,255,20,0.3); "
                f"font-size: 13px; letter-spacing: 1px; width: 100%;"
            )

        async def _run_import():
            target = target_input.value.strip() if target_input.value else ""
            user = user_input.value.strip() if user_input.value else None
            fmt = fmt_select.value

            if not target:
                with log_container:
                    _log_msg("[ERROR] 请输入目标人物名字", "error")
                return
            if not uploaded_path:
                with log_container:
                    _log_msg("[ERROR] 请上传聊天记录文件", "error")
                return

            import_btn.disable()

            def on_progress(msg: str):
                with log_container:
                    _log_msg(msg, "info")
                log_scroll.scroll_to(percent=1.0)

            try:
                result = await ChatController.import_chat_file(
                    file_path=uploaded_path,
                    fmt=fmt,
                    target_name=target,
                    user_name=user,
                    on_progress=on_progress,
                )
                with log_container:
                    _log_msg(f"[OK] Target loaded. Initiating link...", "success")

                await asyncio.sleep(1)
                ui.navigate.to(f"/chat/{result['name']}")
            except Exception as e:
                with log_container:
                    _log_msg(f"[FAIL] {e}", "error")
                import_btn.enable()


def _log_msg(text: str, level: str = "info"):
    """在日志区域添加一条消息。"""
    color_map = {
        "info": NEON_CYAN,
        "success": NEON_GREEN,
        "error": NEON_MAGENTA,
    }
    color = color_map.get(level, TEXT_DIM)
    ui.html(
        f'<span style="color: {color}; font-size: 12px; line-height: 1.5;">'
        f'&gt; {_escape(text)}</span>'
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
