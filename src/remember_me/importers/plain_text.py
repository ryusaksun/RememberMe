"""纯文本聊天记录解析器。

支持格式：
    名字: 消息内容
    名字: 消息内容
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import ChatHistory, ChatMessage


def parse(file_path: str | Path, target_name: str, user_name: str | None = None) -> ChatHistory:
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")

    messages: list[ChatMessage] = []
    pattern = re.compile(r"^(.+?):\s*(.+)$", re.MULTILINE)

    for match in pattern.finditer(text):
        sender = match.group(1).strip()
        content = match.group(2).strip()
        if not content:
            continue
        messages.append(
            ChatMessage(
                sender=sender,
                content=content,
                is_target=(sender == target_name),
            )
        )

    if not user_name:
        senders = {m.sender for m in messages if not m.is_target}
        user_name = senders.pop() if len(senders) == 1 else "我"

    return ChatHistory(target_name=target_name, user_name=user_name, messages=messages)
