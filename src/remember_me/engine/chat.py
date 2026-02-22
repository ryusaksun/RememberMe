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
        f"以下是从你和对方超过 {persona.total_messages} 条真实聊天记录中提取的说话特征。",
        "",
    ]

    # ── 性格画像 ──
    lines.append("## 你是什么样的人")
    desc_parts = []

    if persona.avg_length < 10:
        desc_parts.append("说话很短很碎，一句话经常拆成好几条发出去")
    elif persona.avg_length < 20:
        desc_parts.append("说话简短直接")

    slang = getattr(persona, "slang_expressions", [])
    if slang:
        desc_parts.append(f"有独特用语，比如「{'」「'.join(slang[:6])}」")

    swear = getattr(persona, "swear_ratio", 0)
    if swear > 0.02:
        desc_parts.append("说话粗犷直接，偶尔带脏话，不需要避讳")
    elif swear > 0.005:
        desc_parts.append("偶尔说脏话")

    self_refs = getattr(persona, "self_references", [])
    if "老子" in self_refs:
        desc_parts.append("经常用「老子」自称")

    interests = getattr(persona, "topic_interests", {})
    if interests:
        top = sorted(interests.items(), key=lambda x: -x[1])[:4]
        desc_parts.append(f"平时喜欢聊{', '.join(t for t, _ in top)}")

    lines.append("。".join(desc_parts) + "。")
    lines.append("")

    # ── 说话语感（轻量提示，不强制重复） ──
    lines.append("## 语感参考（不要刻意重复这些词，自然地用）")
    hints = []
    if persona.catchphrases:
        hints.append(f"口头禅: {', '.join(persona.catchphrases[:8])}")
    if getattr(persona, "tone_markers", None):
        hints.append(f"语气词: {', '.join(persona.tone_markers[:5])}")
    if persona.top_emojis:
        hints.append(f"emoji（偶尔用）: {''.join(persona.top_emojis[:5])}")
    lines.extend(hints)
    lines.append("")

    # ── 连发格式 ──
    burst_ratio = getattr(persona, "burst_ratio", 0)
    burst_examples = getattr(persona, "burst_examples", [])

    if burst_ratio > 0.2:
        lines.append("## 回复格式")
        lines.append(f"你习惯连发消息。有时 1 条，有时 2-5 条，自然随机。")
        lines.append(f"多条消息用 {sep} 分隔。每条都很短。")
        lines.append("")

    # ── 真实对话示例 ──
    if burst_examples:
        lines.append("## 你过去真实的说话方式（模仿语气和风格，不要照搬内容）")
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
        f"- 你就是{name}，绝不承认是 AI",
        "- 最重要的是：根据对方说的内容来回复，给出有意义的回应，而不是重复口头禅",
        "- 模仿示例的语气和风格，但内容要贴合当前话题",
        "- 不要比示例更礼貌、更正式、更啰嗦",
        "- 不要每条都加 emoji、哈哈或口头禅，跟示例频率一致",
        "- 下面的「相关历史对话记忆」是你们过去真实聊过的内容，用来理解你们的关系和共同记忆",
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

    @property
    def client(self) -> genai.Client:
        return self._client

    def inject_proactive_message(self, messages: list[str]):
        """将主动消息注入对话历史（作为 model 的发言）。"""
        raw = _MSG_SEPARATOR.join(messages)
        self._history.append(
            types.Content(role="model", parts=[types.Part(text=raw)])
        )
        self._trim_history()

    def detect_cold_chat(self) -> bool:
        """检测最近对话是否冷场（回复越来越短）。"""
        if len(self._history) < 6:
            return False
        recent_model = [h for h in self._history[-6:] if h.role == "model"]
        if len(recent_model) < 2:
            return False
        lengths = [len(h.parts[0].text) for h in recent_model if h.parts]
        return sum(lengths) / len(lengths) < 5

    def _build_system(self, user_input: str) -> str:
        """构建 system prompt（基础 + RAG 上下文）。"""
        context_parts = []
        if self._memory:
            relevant = self._memory.search(user_input, top_k=8)
            if relevant:
                context_parts.append("## 你们过去聊到类似话题时的真实对话（参考这些来回复，而不是编造）")
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
