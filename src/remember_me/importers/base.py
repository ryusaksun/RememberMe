from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ChatMessage:
    sender: str
    content: str
    timestamp: datetime | None = None
    is_target: bool = False


@dataclass
class ChatHistory:
    target_name: str
    user_name: str
    messages: list[ChatMessage] = field(default_factory=list)

    @property
    def target_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.is_target]

    @property
    def user_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if not m.is_target]

    def as_dialogue_pairs(self) -> list[tuple[ChatMessage, ChatMessage]]:
        """提取连续的 用户-目标 对话对，用于 few-shot 示例。"""
        pairs = []
        for i in range(len(self.messages) - 1):
            curr = self.messages[i]
            nxt = self.messages[i + 1]
            if not curr.is_target and nxt.is_target:
                pairs.append((curr, nxt))
        return pairs

    def as_burst_dialogues(self) -> list[tuple[str, list[str]]]:
        """提取用户说一句、目标连发多条的完整对话段。
        返回: [(用户消息, [目标回复1, 回复2, ...])]
        """
        segments: list[tuple[str, list[str]]] = []
        i = 0
        while i < len(self.messages):
            m = self.messages[i]
            if not m.is_target and not m.content.startswith("["):
                user_msg = m.content
                replies: list[str] = []
                j = i + 1
                while j < len(self.messages) and self.messages[j].is_target:
                    c = self.messages[j].content
                    if not c.startswith("["):
                        replies.append(c)
                    j += 1
                if replies and len(user_msg) > 1:
                    segments.append((user_msg, replies))
                i = j
            else:
                i += 1
        return segments

    def save(self, path: str | Path):
        """保存完整聊天记录为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "target_name": self.target_name,
            "user_name": self.user_name,
            "total_messages": len(self.messages),
            "messages": [
                {
                    "sender": m.sender,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "is_target": m.is_target,
                }
                for m in self.messages
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> ChatHistory:
        """从 JSON 文件加载聊天记录。"""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        messages = [
            ChatMessage(
                sender=m["sender"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]) if m.get("timestamp") else None,
                is_target=m["is_target"],
            )
            for m in raw["messages"]
        ]
        return cls(
            target_name=raw["target_name"],
            user_name=raw["user_name"],
            messages=messages,
        )
