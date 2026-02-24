"""对话引擎 - 使用 Gemini API 驱动与人格的对话。"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.memory.scratchpad import Scratchpad, update_scratchpad
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


def _split_reply(text: str, truncated: bool = False) -> list[str]:
    """将 ||| 分隔的回复拆成多条消息。自动检测并丢弃截断的末尾消息。"""
    parts = text.split(_MSG_SEPARATOR)
    result = [p.strip() for p in parts if p.strip()]
    if len(result) > 1:
        # 显式截断 或 最后一条异常短（≤2字且远短于前面平均长度），视为截断碎片
        avg_len = sum(len(m) for m in result[:-1]) / len(result[:-1])
        if truncated or (len(result[-1]) <= 2 and avg_len > 4):
            result = result[:-1]
    return result


class ChatEngine:
    def __init__(self, persona: Persona, memory: MemoryStore | None = None,
                 api_key: str | None = None, sticker_lib=None,
                 notes: list[str] | None = None,
                 knowledge_store=None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY 未提供，无法初始化对话引擎")
        self._persona = persona
        self._memory = memory
        self._notes = notes or []
        self._knowledge_store = knowledge_store
        self._system_prompt = _build_system_prompt(persona)
        self._client = genai.Client(api_key=api_key)
        self._history: list[types.Content] = []
        self._sticker_lib = sticker_lib
        self._sticker_probability = 0.14  # 约 14% 概率发表情包（基于真人数据）
        self._scratchpad = Scratchpad()
        self._scratchpad_updating = False

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

    def get_recent_context(self) -> str:
        """获取最近几轮对话的文本，供外部判断当前话题。"""
        recent = self._history[-6:] if len(self._history) >= 6 else self._history
        lines = []
        for h in recent:
            if h.parts and h.parts[0].text:
                role = "对方" if h.role == "user" else "你"
                lines.append(f"{role}: {h.parts[0].text[:100]}")
        return "\n".join(lines)

    def is_conversation_ended(self) -> bool:
        """检测最近对话是否已自然结束（说了再见/去忙了）。"""
        if not self._history:
            return False
        # 检查最后几条消息
        for h in reversed(self._history[-4:]):
            if h.parts and h.parts[0].text:
                text = h.parts[0].text.lower()
                if re.search(r"(再见|拜拜|bye|晚安|睡了|去了|走了|滚|不聊|去忙|"
                             r"不说了|激情王者|大开杀戒|去打游戏|上号去)", text):
                    return True
        return False

    _TRIVIAL_RE = re.compile(
        r"^(嗯|好|哈+|呵呵|ok|行|对|是|啊|哦|嗯嗯|好的|可以|没|没有|知道了|了解|真的|可以吧|好吧|嘻嘻)$", re.I,
    )

    def _expand_query(self, user_input: str) -> str:
        """对短/无意义消息用最近对话上下文扩展查询。"""
        if len(user_input) <= 4 or self._TRIVIAL_RE.match(user_input.strip()):
            recent = [
                h.parts[0].text[:80]
                for h in self._history[-6:]
                if h.parts and h.parts[0].text
            ]
            if recent:
                return " ".join(recent[-3:])
        return user_input

    def _build_system(self, user_input: str) -> str:
        """构建 system prompt（基础 + 备注 + 中期记忆 + RAG 上下文）。"""
        # 注入当前时间，让 persona 有时间感知
        now = datetime.now()
        time_block = (
            f"\n\n## 当前时间\n"
            f"现在是 {now.strftime('%Y年%m月%d日 %H:%M')}，"
            f"{'凌晨' if now.hour < 6 else '早上' if now.hour < 9 else '上午' if now.hour < 12 else '中午' if now.hour < 13 else '下午' if now.hour < 18 else '晚上' if now.hour < 23 else '深夜'}。"
            f"请根据当前时间自然地回复，不要在白天叫对方去睡觉，也不要在深夜像白天一样精力充沛。"
        )
        system = self._system_prompt + time_block

        # 手动备注（动态读取，修改后即时生效）
        if self._notes:
            lines = ["## 你知道的关于对方和你们关系的事"]
            for note in self._notes:
                lines.append(f"- {note}")
            system = system + "\n\n" + "\n".join(lines)

        # 中期记忆（scratchpad）
        scratchpad_block = self._scratchpad.to_prompt_block()
        if scratchpad_block:
            system = system + "\n\n" + scratchpad_block

        # 每日知识库（persona 最近关注的动态）
        if self._knowledge_store:
            try:
                query = self._expand_query(user_input)
                kb_items = self._knowledge_store.search(query, top_k=3)
                if kb_items:
                    kb_lines = ["## 你最近关注的新闻和动态（自然地提到，不要像背课文）"]
                    for item in kb_items:
                        kb_lines.append(f"- {item.summary}")
                    system = system + "\n\n" + "\n".join(kb_lines)
            except Exception as e:
                logger.debug("知识库检索失败: %s", e)

        # 长期记忆（RAG）
        if self._memory:
            query = self._expand_query(user_input)
            raw_results = self._memory.search(query, top_k=5)

            # 过滤低相关性结果 + 按行重叠率去重（overlap 窗口可能返回相似内容）
            seen_lines: set[str] = set()
            filtered: list[str] = []
            best_dist = raw_results[0][1] if raw_results else 0.0
            # 相对阈值：距离不超过最优结果的 2 倍，且绝对值不超过 1.2
            max_dist = min(best_dist * 2.0, 1.2)
            for doc, dist in raw_results:
                if dist > max_dist:
                    continue
                lines = set(doc.strip().split("\n"))
                overlap = len(lines & seen_lines) / len(lines) if lines else 1.0
                if overlap < 0.5:
                    filtered.append(doc)
                    seen_lines.update(lines)

            if filtered:
                context_parts = ["## 你们过去聊到类似话题时的真实对话（参考这些来回复，而不是编造）"]
                for fragment in filtered[:5]:
                    context_parts.append(fragment)
                    context_parts.append("---")
                system = system + "\n\n" + "\n".join(context_parts)

        return system

    def _trim_history(self):
        max_turns = 40
        if len(self._history) > max_turns:
            self._history = self._history[-max_turns:]

    def save_session(self, path: str | Path):
        """将对话历史保存到文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self._persona.name,
            "updated_at": datetime.now().isoformat(),
            "history": [
                {"role": h.role, "text": next((p.text for p in h.parts if p.text), "") if h.parts else ""}
                for h in self._history
            ],
            "scratchpad": self._scratchpad.to_dict(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_session(self, path: str | Path) -> bool:
        """从文件恢复对话历史。返回是否成功加载。"""
        path = Path(path)
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._history = [
                types.Content(role=h["role"], parts=[types.Part(text=h["text"])])
                for h in data.get("history", [])
            ]
            self._trim_history()
            if data.get("scratchpad"):
                self._scratchpad = Scratchpad.from_dict(data["scratchpad"])
            return bool(self._history)
        except Exception as e:
            logger.warning("加载会话失败: %s", e)
            return False

    def get_new_messages(self, start_index: int = 0) -> list[dict]:
        """获取从 start_index 开始的新消息（用于写入向量库）。"""
        result = []
        for h in self._history[start_index:]:
            if h.parts and h.parts[0].text:
                result.append({"role": h.role, "text": h.parts[0].text})
        return result

    # ── 中期记忆（Scratchpad）更新 ──

    def _should_update_scratchpad(self) -> bool:
        if self._scratchpad_updating:
            return False
        turns_since = len(self._history) - self._scratchpad.last_update_turn
        if self._scratchpad.last_update_turn == 0 and turns_since >= 4:
            return True
        return turns_since >= 6

    def _get_messages_since_last_update(self) -> list[dict]:
        result = []
        for h in self._history[self._scratchpad.last_update_turn:]:
            if h.parts and h.parts[0].text:
                result.append({"role": h.role, "text": h.parts[0].text})
        return result

    def _trigger_scratchpad_update(self):
        if self._scratchpad_updating:
            return
        recent = self._get_messages_since_last_update()
        if not recent:
            return
        self._scratchpad_updating = True
        current_turn = len(self._history)

        def _do_update():
            try:
                new_pad = update_scratchpad(self._client, self._scratchpad, recent)
                new_pad.last_update_turn = current_turn
                self._scratchpad = new_pad
            except Exception as e:
                logger.debug("Scratchpad 更新失败: %s", e)
            finally:
                self._scratchpad_updating = False

        threading.Thread(target=_do_update, daemon=True).start()

    def send_multi(self, user_input: str,
                   image: tuple[bytes, str] | None = None) -> list[str]:
        """发送消息并获取多条回复（模拟连发）。

        image: 可选 (bytes, mime_type) 图片数据，与文本一起发送给 LLM。
        """
        system = self._build_system(user_input)

        parts = [types.Part(text=user_input)]
        if image:
            img_bytes, mime_type = image
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        user_msg = types.Content(role="user", parts=parts)
        self._history.append(user_msg)

        try:
            response = self._client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.8,
                    max_output_tokens=2048,
                ),
            )
        except Exception:
            # API 失败时回滚用户消息，避免污染历史
            self._history.pop()
            raise

        raw_reply = response.text or ""
        truncated = (
            response.candidates
            and response.candidates[0].finish_reason
            and response.candidates[0].finish_reason.name == "MAX_TOKENS"
        )

        self._history.append(types.Content(role="model", parts=[types.Part(text=raw_reply)]))
        self._trim_history()

        # 异步更新中期记忆
        if self._should_update_scratchpad():
            self._trigger_scratchpad_update()

        messages = _split_reply(raw_reply, truncated=truncated)
        result = messages if messages else [raw_reply]

        # 按概率附加表情包
        result = self._maybe_attach_sticker(result)
        return result

    def _maybe_attach_sticker(self, replies: list[str]) -> list[str]:
        """按概率在回复后附加一张表情包。"""
        if not self._sticker_lib or not self._sticker_lib.stickers:
            return replies
        if random.random() > self._sticker_probability:
            return replies

        # 根据回复内容判断情感
        combined = " ".join(replies)
        emotion = self._detect_emotion(combined)
        sticker = self._sticker_lib.random_sticker(emotion)
        if sticker:
            replies.append(f"[sticker:{sticker.path}]")
        return replies

    @staticmethod
    def _detect_emotion(text: str) -> str:
        """简单情感检测。"""
        patterns = {
            "搞笑": r"(哈|笑|搞笑|笑死|好笑|离谱|绝了)",
            "喜爱": r"(爱|心|宝贝|想你|喜欢|可爱|好看)",
            "愤怒": r"(靠|妈的|吗的|操|气|烦|傻|几把|屎)",
            "难过": r"(呜|哭|难过|伤心|惨|寄|完蛋)",
            "震惊": r"(卧槽|我靠|天|啥|什么|真的假的)",
        }
        for emotion, pattern in patterns.items():
            if re.search(pattern, text):
                return emotion
        return "通用"

    def send(self, user_input: str) -> str:
        """发送消息并获取单条回复。"""
        msgs = self.send_multi(user_input)
        return "\n".join(msgs)

    def send_stream(self, user_input: str):
        """流式发送消息，yield 每个文本片段。"""
        system = self._build_system(user_input)

        user_msg = types.Content(role="user", parts=[types.Part(text=user_input)])
        self._history.append(user_msg)

        full_reply = []
        try:
            for chunk in self._client.models.generate_content_stream(
                model="gemini-3.1-pro-preview",
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.8,
                    max_output_tokens=2048,
                ),
            ):
                text = chunk.text or ""
                full_reply.append(text)
                yield text
        except Exception:
            # API 失败时回滚用户消息，避免污染历史
            self._history.pop()
            raise

        self._history.append(
            types.Content(role="model", parts=[types.Part(text="".join(full_reply))])
        )
        self._trim_history()

        # 异步更新中期记忆
        if self._should_update_scratchpad():
            self._trigger_scratchpad_update()
