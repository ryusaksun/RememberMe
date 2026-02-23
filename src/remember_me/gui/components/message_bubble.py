"""终端风格消息气泡组件。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from nicegui import ui

# 消息行统一样式
_LINE_STYLE = "white-space: pre-wrap; word-break: break-word; line-height: 1.7; font-size: 13px;"


def render_message(
    text: str,
    sender: str,
    is_target: bool,
    *,
    timestamp: datetime | None = None,
    is_burst_continuation: bool = False,
) -> ui.element:
    """渲染一条终端风格消息。

    - 对方消息: [HH:MM] <name> 消息内容  (cyan)
    - 用户消息: [HH:MM] you > 消息内容    (green)
    - burst 续行: 无时间戳前缀，缩进对齐
    """
    ts = timestamp or datetime.now()
    time_str = ts.strftime("%H:%M")

    # 表情包
    if text.startswith("[sticker:"):
        path = text[9:].rstrip("]")
        with ui.row().classes("msg-appear items-end gap-2") as row:
            if not is_burst_continuation:
                cls = "msg-target" if is_target else "msg-user"
                prefix = f"&lt;{_escape(sender)}&gt;" if is_target else "you &gt;"
                ui.html(
                    f'<span class="msg-time">[{time_str}]</span> '
                    f'<span class="{cls}">{prefix}</span>'
                ).style("white-space: pre;")
            if Path(path).exists():
                ui.image(path).classes("msg-sticker")
            else:
                cls = "msg-target" if is_target else "msg-user"
                ui.html(
                    f'<span class="{cls}" style="opacity:0.6;">[表情包: {Path(path).name}]</span>'
                )
        return row

    if is_target:
        if is_burst_continuation:
            html = (
                f'<span class="msg-time" style="visibility:hidden;">[{time_str}]</span> '
                f'<span class="msg-target" style="visibility:hidden;">'
                f'&lt;{_escape(sender)}&gt;</span> '
                f'<span class="msg-target">{_escape(text)}</span>'
            )
        else:
            html = (
                f'<span class="msg-time">[{time_str}]</span> '
                f'<span class="msg-target">'
                f'&lt;{_escape(sender)}&gt; {_escape(text)}</span>'
            )
    else:
        if is_burst_continuation:
            html = (
                f'<span class="msg-time" style="visibility:hidden;">[{time_str}]</span> '
                f'<span class="msg-user" style="visibility:hidden;">you &gt;</span> '
                f'<span class="msg-user">{_escape(text)}</span>'
            )
        else:
            html = (
                f'<span class="msg-time">[{time_str}]</span> '
                f'<span class="msg-user">you &gt; {_escape(text)}</span>'
            )

    el = ui.html(html).classes("msg-appear").style(_LINE_STYLE)
    return el


def render_system_message(text: str) -> ui.element:
    """渲染系统消息。"""
    return ui.html(
        f'<span class="msg-system">[SYSTEM] {_escape(text)}</span>'
    ).classes("msg-appear").style(_LINE_STYLE)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
