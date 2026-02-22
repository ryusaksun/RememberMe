"""微信聊天记录导出解析器。

支持微信导出的常见文本格式：
    昵称 2024-01-01 12:00:00
    消息内容

或（无时间戳）：
    昵称:
    消息内容
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .base import ChatHistory, ChatMessage

# 微信导出常见格式：昵称 YYYY-MM-DD HH:MM:SS
_HEADER_WITH_TS = re.compile(
    r"^(.+?)\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*$"
)
# 备选格式：昵称(时间)
_HEADER_WITH_TS_PAREN = re.compile(
    r"^(.+?)\((\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\)\s*$"
)


def _try_parse_ts(ts_str: str) -> datetime | None:
    ts_str = ts_str.replace("/", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def parse(file_path: str | Path, target_name: str, user_name: str | None = None) -> ChatHistory:
    path = Path(file_path)
    lines = path.read_text(encoding="utf-8").splitlines()

    messages: list[ChatMessage] = []
    current_sender: str | None = None
    current_ts: datetime | None = None
    content_lines: list[str] = []

    def _flush():
        nonlocal current_sender, current_ts, content_lines
        if current_sender and content_lines:
            content = "\n".join(content_lines).strip()
            if content:
                messages.append(
                    ChatMessage(
                        sender=current_sender,
                        content=content,
                        timestamp=current_ts,
                        is_target=(current_sender == target_name),
                    )
                )
        current_sender = None
        current_ts = None
        content_lines = []

    for line in lines:
        # 尝试匹配消息头
        m = _HEADER_WITH_TS.match(line) or _HEADER_WITH_TS_PAREN.match(line)
        if m:
            _flush()
            current_sender = m.group(1).strip()
            current_ts = _try_parse_ts(m.group(2))
            continue

        # 如果当前有发送者，则为消息内容
        if current_sender is not None:
            content_lines.append(line)
        # 忽略没有发送者上下文的行

    _flush()

    if not user_name:
        senders = {m.sender for m in messages if not m.is_target}
        user_name = senders.pop() if len(senders) == 1 else "我"

    return ChatHistory(target_name=target_name, user_name=user_name, messages=messages)
