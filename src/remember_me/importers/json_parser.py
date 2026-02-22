"""JSON 聊天记录解析器。

支持格式：
[
    {"sender": "小明", "content": "你好", "timestamp": "2024-01-01 12:00:00"},
    {"sender": "我", "content": "你好呀", "timestamp": "2024-01-01 12:01:00"}
]
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .base import ChatHistory, ChatMessage


def parse(file_path: str | Path, target_name: str, user_name: str | None = None) -> ChatHistory:
    path = Path(file_path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    messages: list[ChatMessage] = []
    for item in raw:
        sender = item.get("sender", "").strip()
        content = item.get("content", "").strip()
        if not sender or not content:
            continue

        ts = None
        if ts_str := item.get("timestamp"):
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                pass

        messages.append(
            ChatMessage(
                sender=sender,
                content=content,
                timestamp=ts,
                is_target=(sender == target_name),
            )
        )

    if not user_name:
        senders = {m.sender for m in messages if not m.is_target}
        user_name = senders.pop() if len(senders) == 1 else "我"

    return ChatHistory(target_name=target_name, user_name=user_name, messages=messages)
