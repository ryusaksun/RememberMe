"""中期记忆 - AI 自动维护的对话笔记（Scratchpad）。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime

from google import genai
from google.genai import types

_SCRATCHPAD_MODEL = "gemini-3-flash-preview"

_UPDATE_PROMPT = """\
你是一个对话记录员。分析下面的对话片段，更新对话笔记。

你正在记录的人物是「{persona_name}」。

当前笔记状态:
{current}

最近的对话（从上次更新后开始）:
{messages}

请输出更新后的笔记，JSON 格式:
{{
  "topics": [
    {{"name": "话题名", "status": "ongoing 或 concluded", "summary": "一句话概括"}}
  ],
  "facts": ["对方/你提到的重要事实，每条一句话"],
  "emotional_tone": "当前对话的情绪氛围",
  "open_threads": ["还没结束的话题/问题"],
  "emotion": {{
    "valence": 0.0,
    "arousal": 0.0,
    "trigger": "导致当前情绪的原因"
  }}
}}

emotion 字段说明（基于「{persona_name}」的性格来判断 TA 的情绪）:
- valence: -1.0(极不爽) ~ 0(平静) ~ 1.0(极开心)。聊到喜欢的话题/收到好消息→正值；被冒犯/无聊→负值
- arousal: -1.0(低沉/困) ~ 0(正常) ~ 1.0(极亢奋)。话题激烈/争论/游戏→高值；闲聊/困了→低值
- 如果对话平淡，valence 和 arousal 都接近 0

规则:
- 保留之前笔记中仍然相关的内容
- 如果话题已经聊完了，把 status 改为 "concluded"
- facts 只记录重要信息（地点、计划、情绪变化），不记录闲聊内容
- 最多保留 8 个 topics、10 个 facts、5 个 open_threads
- 只输出 JSON，不要其他文字
"""


@dataclass
class Scratchpad:
    """AI 维护的中期记忆，追踪当前对话的话题、事实和氛围。"""

    topics: list[dict] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    emotional_tone: str = ""
    open_threads: list[str] = field(default_factory=list)
    emotion_raw: dict = field(default_factory=dict)  # LLM 输出的结构化情绪
    last_update_turn: int = 0
    last_update_time: str = ""

    def to_prompt_block(self) -> str:
        """渲染为自然语言，插入 system prompt。"""
        if not self.topics and not self.facts:
            return ""

        lines = ["## 本次对话记录（你对这次聊天的记忆）"]

        if self.emotional_tone:
            lines.append(f"当前氛围: {self.emotional_tone}")
            lines.append("")

        if self.topics:
            lines.append("聊过的话题:")
            for t in self.topics:
                status = "还在聊" if t.get("status") == "ongoing" else "聊完了"
                lines.append(f"- {t.get('name', '')}（{status}）: {t.get('summary', '')}")
            lines.append("")

        if self.facts:
            lines.append("这次聊天中提到的重要信息:")
            for f in self.facts:
                lines.append(f"- {f}")
            lines.append("")

        if self.open_threads:
            lines.append("还没聊完的:")
            for t in self.open_threads:
                lines.append(f"- {t}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Scratchpad:
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})


def _parse_json(raw: str) -> dict | None:
    """从 LLM 响应中提取 JSON（处理 markdown 代码块）。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def update_scratchpad(
    client: genai.Client,
    scratchpad: Scratchpad,
    recent_history: list[dict],
    persona_name: str = "",
) -> Scratchpad:
    """调用 LLM 更新 scratchpad。recent_history: [{"role": "user"|"model", "text": str}]"""
    current_json = json.dumps(scratchpad.to_dict(), ensure_ascii=False, indent=2)
    messages_text = "\n".join(
        f"{'对方' if m['role'] == 'user' else '你'}: {m['text']}"
        for m in recent_history
    )

    prompt = _UPDATE_PROMPT.format(
        current=current_json,
        messages=messages_text,
        persona_name=persona_name or "对方",
    )

    response = client.models.generate_content(
        model=_SCRATCHPAD_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    data = _parse_json(response.text or "")
    if data is None:
        return scratchpad

    new_pad = Scratchpad.from_dict(data)
    # 强制限制列表长度，防止 LLM 忽略 prompt 中的上限
    new_pad.topics = new_pad.topics[:8]
    new_pad.facts = new_pad.facts[:10]
    new_pad.open_threads = new_pad.open_threads[:5]
    new_pad.emotion_raw = data.get("emotion", {})
    new_pad.last_update_time = datetime.now().isoformat()
    return new_pad
