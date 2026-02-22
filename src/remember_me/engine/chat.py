"""对话引擎 - 使用 Gemini API 驱动与人格的对话。"""

from __future__ import annotations

from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.memory.store import MemoryStore

_MSG_SEPARATOR = "|||"


def _build_system_prompt(persona: Persona) -> str:
    sep = _MSG_SEPARATOR
    name = persona.name

    lines = [
        f"你就是「{name}」本人。不是在扮演，你就是这个人。",
        f"以下是从你和对方的真实聊天记录中提取的说话特征，严格按照这些特征回复。",
        "",
    ]

    # ── 核心风格描述（自然语言，不要像报告） ──
    lines.append(f"你说话{'' if persona.avg_length > 15 else '很短很碎，'}经常一句话拆成好几条发。"
                 f"不会写长段文字。口头禅包括：{', '.join(persona.catchphrases[:8])}。" if persona.catchphrases else "")

    if persona.top_emojis:
        lines.append(f"偶尔用这些 emoji：{''.join(persona.top_emojis[:6])}，但不是每句都带。")

    if getattr(persona, "tone_markers", None):
        lines.append(f"语气词习惯：{', '.join(persona.tone_markers[:6])}。")

    lines.append("")

    # ── 连发规则 + 真实对话示例（核心改动） ──
    avg_burst = getattr(persona, "avg_burst_length", 1.0)
    burst_ratio = getattr(persona, "burst_ratio", 0)
    burst_examples = getattr(persona, "burst_examples", [])

    if burst_ratio > 0.2:
        lines.append(f"## 回复格式")
        lines.append(f"你的消息习惯连发，不是每次都连发，有时 1 条，有时 2-5 条，要自然随机。")
        lines.append(f"多条消息之间用 {sep} 分隔。每条都很短。")
        lines.append("")

    # 优先用真实连发对话做示例
    if burst_examples:
        lines.append("## 你过去真实的说话方式（严格模仿这些风格）")
        for ex in burst_examples[:12]:
            replies = ex.get("replies", [])
            lines.append(f"对方: {ex['user']}")
            lines.append(f"{name}: {sep.join(replies)}")
            lines.append("")
    elif persona.example_dialogues:
        lines.append("## 你过去真实的说话方式")
        for ex in persona.example_dialogues[:15]:
            lines.append(f"对方: {ex['user']}")
            lines.append(f"{name}: {ex['reply']}")
            lines.append("")

    lines.extend([
        "## 规则",
        f"- 你就是{name}，绝不能承认是 AI 或在扮演角色",
        "- 严格模仿上面示例中的语气和用词，不要比示例更礼貌或更正式",
        "- 回复要自然随意，像真人发消息一样，不要每次都用相同的句式结构",
        "- 不要每条消息都加 emoji 或哈哈，参考示例中的频率",
        "- 下面的「相关历史对话记忆」是你们过去真实聊过的内容，回复时参考",
    ])

    return "\n".join(lines)


def _split_reply(text: str) -> list[str]:
    """将 ||| 分隔的回复拆成多条消息。"""
    parts = text.split(_MSG_SEPARATOR)
    return [p.strip() for p in parts if p.strip()]


class ChatEngine:
    def __init__(self, persona: Persona, memory: MemoryStore | None = None, api_key: str | None = None):
        self._persona = persona
        self._memory = memory
        self._system_prompt = _build_system_prompt(persona)
        self._client = genai.Client(api_key=api_key)
        self._history: list[types.Content] = []

    def _build_system(self, user_input: str) -> str:
        """构建 system prompt（基础 + RAG 上下文）。"""
        context_parts = []
        if self._memory:
            relevant = self._memory.search(user_input, top_k=5)
            if relevant:
                context_parts.append("## 相关的历史对话记忆")
                for fragment in relevant:
                    context_parts.append(fragment)
                    context_parts.append("---")

        system = self._system_prompt
        if context_parts:
            system = system + "\n\n" + "\n".join(context_parts)
        return system

    def _trim_history(self):
        max_turns = 40
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]

    def send_multi(self, user_input: str) -> list[str]:
        """发送消息并获取多条回复（模拟连发）。"""
        system = self._build_system(user_input)

        self._history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        response = self._client.models.generate_content(
            model="gemini-3.1-pro-preview",
            contents=self._history,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.8,
                max_output_tokens=512,
            ),
        )

        raw_reply = response.text or ""

        self._history.append(types.Content(role="model", parts=[types.Part(text=raw_reply)]))
        self._trim_history()

        messages = _split_reply(raw_reply)
        return messages if messages else [raw_reply]

    def send(self, user_input: str) -> str:
        """发送消息并获取单条回复。"""
        msgs = self.send_multi(user_input)
        return "\n".join(msgs)

    def send_stream(self, user_input: str):
        """流式发送消息，yield 每个文本片段。"""
        system = self._build_system(user_input)

        self._history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))

        full_reply = []
        for chunk in self._client.models.generate_content_stream(
            model="gemini-3.1-pro-preview",
            contents=self._history,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.8,
                max_output_tokens=512,
            ),
        ):
            text = chunk.text or ""
            full_reply.append(text)
            yield text

        self._history.append(
            types.Content(role="model", parts=[types.Part(text="".join(full_reply))])
        )
        self._trim_history()
