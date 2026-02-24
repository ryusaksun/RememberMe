"""对话引擎 - 使用 Gemini API 驱动与人格的对话。"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_TIMEZONE = ZoneInfo(os.environ.get("TZ", "Asia/Shanghai"))

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.engine.emotion import EmotionState
from remember_me.memory.scratchpad import Scratchpad, update_scratchpad
from remember_me.memory.store import MemoryStore
from remember_me.models import MODEL_LIGHT, MODEL_MAIN

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
        desc_parts.append("说话很碎，一句话经常拆成好几条发出去，但每次都会把想说的说完")
    elif persona.avg_length < 20:
        desc_parts.append("说话简短直接，但不会话说到一半就停了")

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
    avg_burst_length = getattr(persona, "avg_burst_length", 1.0)
    burst_examples = getattr(persona, "burst_examples", [])

    if burst_ratio > 0.2:
        lines.append("## 回复格式（必须遵守）")
        lines.append(f"你习惯连发消息，平均每次发 {avg_burst_length:.0f} 条左右。多条消息用 {sep} 分隔。")
        lines.append("")
        lines.append("关键规则：每次回复必须至少包含一条【有实际内容】的消息。")
        lines.append(f"「笑死」「6」「牛逼」「哈哈哈」「不知道」这些不算有实际内容，它们后面必须跟一条真正回应话题的消息。")
        lines.append("")
        lines.append("示例（✗ 是错的，✓ 是对的）：")
        lines.append(f"✗ 笑死老子了")
        lines.append(f"✓ 笑死老子了{sep}南方要啥暖气啊你在做梦")
        lines.append(f"✗ 老子哪知道")
        lines.append(f"✓ 老子哪知道{sep}你自己不会搜啊")
        lines.append(f"✗ 哈哈哈哈哈{sep}你嘛又吃外卖啊")
        lines.append(f"✓ 哈哈哈哈哈{sep}你嘛又吃外卖啊{sep}你不会自己做饭吗")
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
        "- 心情再差也要把话说完，冷淡体现在语气上，不是不说话",
        "- 偶尔可以出现轻微口误并马上自我修正，但频率要很低，不能影响理解",
        "- 下面的「相关历史对话记忆」是你们过去真实聊过的内容，用来理解你们的关系和共同记忆",
    ])

    return "\n".join(lines)


_MAX_BURST = 8  # 单次回复最大消息条数安全上限（正常由 burst_range 引导）
_MEMORY_CACHE_MAX_SIZE = 64
_MEMORY_CACHE_TTL_SEC = 120.0

_SESSION_PHASE_GUIDE = {
    "warmup": "你们刚进入聊天，语气自然热身，不要突然沉重也别过度输出。",
    "normal": "按平时节奏聊，优先回应当下话题。",
    "deep_talk": "当前对话偏走心或严肃，先共情再表达观点，语气要稳。",
    "cooldown": "刚经历密集聊天，语气收一点，短句但要把意思说完整。",
    "ending": "对话接近收尾，不主动扩展新话题，语气自然结束。",
}


# 检测 LLM 推理泄漏：中文内容后跟随英文句子
# 匹配过渡符（引号、括号、标点、破折号等）+ 大写字母开头的英文长句
_REASONING_LEAK_RE = re.compile(
    r'[\s"\')}\]\.\-,;:。！？…—]+[A-Z][a-zA-Z\s,\'\"\-\*\(\)\.!?;:]{8,}$'
)


def _clean_reasoning_leak(msg: str) -> str:
    """剥离 LLM 偶尔泄漏的英文推理/元注释。"""
    m = _REASONING_LEAK_RE.search(msg)
    if m:
        cleaned = msg[:m.start()].strip()
        if cleaned:
            return cleaned
    return msg


def _is_reasoning_leak_msg(msg: str) -> bool:
    """检测整条消息是否为 LLM 推理泄漏（纯英文 / 中文推理片段）。"""
    stripped = msg.strip()
    if len(stripped) < 5:
        return False
    # 1) 纯英文消息（中文 persona 不应发纯英文，但短消息如 "OK" 不过滤）
    non_ascii = sum(1 for c in stripped if ord(c) > 127)
    if len(stripped) > 20 and non_ascii / len(stripped) < 0.1:
        return True
    # 2) 中文推理片段：消息前 6 字符内出现孤立右括号（无匹配左括号）
    #    如 "冬），直接改个字眼..." —— 这是 LLM 推理块被截断的尾部碎片
    for i, ch in enumerate(stripped[:6]):
        if ch in ')）':
            if '(' not in stripped[:i] and '（' not in stripped[:i]:
                return True
            break
    # 3) 以规划性语句结尾（"比如""例如" 不完整句，正在举例但没举完）
    if re.search(r'(?:比如|例如)\s*$', stripped):
        return True
    return False


def _split_reply(text: str, truncated: bool = False) -> list[str]:
    """将 ||| 分隔的回复拆成多条消息。自动检测并丢弃截断的末尾消息。"""
    parts = text.split(_MSG_SEPARATOR)
    result = [p.strip() for p in parts if p.strip()]
    # 清理 LLM 推理泄漏（末尾英文 + 整条纯英文）
    result = [_clean_reasoning_leak(m) for m in result]
    result = [m for m in result if m and not _is_reasoning_leak_msg(m)]
    if len(result) > 1:
        # 显式截断 或 最后一条异常短（≤2字且远短于前面平均长度），视为截断碎片
        # 计算 avg 时排除 ≤2 字的消息，避免被极短反应词拉低
        meaningful = [len(m) for m in result[:-1] if len(m) > 2]
        avg_len = sum(meaningful) / len(meaningful) if meaningful else 10
        if truncated or (len(result[-1]) <= 2 and avg_len > 4):
            result = result[:-1]
    # 硬上限：防止 LLM 输出过多条消息
    if len(result) > _MAX_BURST:
        result = result[:_MAX_BURST]
    return result


def _introduce_minor_typo(text: str) -> str:
    """制造一个轻微错字：重复一个字，模拟手滑。"""
    if len(text) < 6:
        return text
    candidates = [
        i for i, ch in enumerate(text)
        if ("\u4e00" <= ch <= "\u9fff" or ch.isalpha()) and 1 <= i < len(text) - 1
    ]
    if not candidates:
        return text
    idx = random.choice(candidates)
    return text[:idx] + text[idx] + text[idx:]


def _tokenize_for_similarity(text: str) -> set[str]:
    """将文本切成可用于简单相似度判断的 token（中英混合）。"""
    text = (text or "").strip().lower()
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    tokens: set[str] = set()
    for w in words:
        if not w:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", w):
            if len(w) == 1:
                tokens.add(w)
                continue
            for i in range(len(w) - 1):
                tokens.add(w[i : i + 2])
        else:
            tokens.add(w)
    return tokens


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
        self._session_phase = "warmup"
        self._human_noise_probability = max(
            0.01,
            min(0.06, 0.02 + getattr(persona, "short_msg_ratio", 0.0) * 0.03),
        )
        self._scratchpad = Scratchpad()
        self._scratchpad_updating = False
        self._emotion_state = EmotionState()
        self._state_lock = threading.Lock()  # 保护 scratchpad/emotion 的跨线程访问
        self._memory_cache: OrderedDict[str, tuple[float, list[tuple[str, float]]]] = OrderedDict()

    @property
    def client(self) -> genai.Client:
        return self._client

    @property
    def reply_delay_factor(self) -> float:
        """情绪驱动的回复延迟系数，供外部（telegram_bot/gui）使用。"""
        return self._emotion_state.get_modifiers(self._persona).reply_delay_factor

    @property
    def proactive_cooldown_factor(self) -> float:
        """情绪驱动的主动消息冷却系数。"""
        return self._emotion_state.get_modifiers(self._persona).proactive_cooldown_factor

    @property
    def session_phase(self) -> str:
        return self._session_phase

    def set_session_phase(self, phase: str):
        if phase not in _SESSION_PHASE_GUIDE:
            phase = "normal"
        self._session_phase = phase

    @staticmethod
    def _thread_similarity(query: str, thread: str) -> float:
        q_tokens = _tokenize_for_similarity(query)
        t_tokens = _tokenize_for_similarity(thread)
        if not q_tokens or not t_tokens:
            return 0.0
        union = q_tokens | t_tokens
        overlap = q_tokens & t_tokens
        score = len(overlap) / max(1, len(union))
        q = (query or "").strip().lower()
        t = (thread or "").strip().lower()
        if q and t and (q in t or t in q):
            score += 0.25
        return min(1.0, score)

    def _rank_open_threads(self, user_input: str, threads: list[str], top_k: int = 3) -> list[str]:
        cleaned = [t.strip() for t in threads if t and t.strip()]
        if not cleaned:
            return []
        query = (user_input or "").strip()
        if not query:
            return cleaned[:top_k]

        scored = [(t, self._thread_similarity(query, t)) for t in cleaned]
        strong = sorted((item for item in scored if item[1] >= 0.30), key=lambda x: x[1], reverse=True)
        medium = sorted((item for item in scored if 0.14 <= item[1] < 0.30), key=lambda x: x[1], reverse=True)
        weak = sorted((item for item in scored if item[1] < 0.14), key=lambda x: x[1], reverse=True)
        ordered = strong + medium + weak
        return [t for t, _score in ordered[:top_k]]

    def _build_phase_prompt_block(self) -> str:
        phase = getattr(self, "_session_phase", "normal")
        guide = _SESSION_PHASE_GUIDE.get(phase)
        if not guide:
            return ""
        return "\n".join([
            "## 当前对话阶段",
            f"阶段：{phase}",
            f"- {guide}",
        ])

    def _cache_memory_result(self, key: str, value: list[tuple[str, float]]):
        if not hasattr(self, "_memory_cache"):
            self._memory_cache = OrderedDict()
        self._memory_cache[key] = (time.time(), value)
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > _MEMORY_CACHE_MAX_SIZE:
            self._memory_cache.popitem(last=False)

    def _search_memory_cached(self, query: str, top_k: int) -> list[tuple[str, float]]:
        key = f"{top_k}:{query.strip()}"
        now_ts = time.time()
        cache = getattr(self, "_memory_cache", None)
        if cache is None:
            self._memory_cache = OrderedDict()
            cache = self._memory_cache
        cached = cache.get(key)
        if cached:
            ts, value = cached
            if now_ts - ts <= _MEMORY_CACHE_TTL_SEC:
                cache.move_to_end(key)
                return value
            cache.pop(key, None)
        if not self._memory:
            return []
        results = self._memory.search(query, top_k=top_k)
        self._cache_memory_result(key, results)
        return results

    def _pick_generation_model(self, user_input: str, image: tuple[bytes, str] | None) -> str:
        if image:
            return MODEL_MAIN
        text = (user_input or "").strip()
        if not text:
            return MODEL_MAIN
        if len(text) <= 8 and self._TRIVIAL_RE.match(text) and not re.search(r"[?？!！]", text):
            return MODEL_LIGHT
        return MODEL_MAIN

    @staticmethod
    def _sample_delay_from_profile(
        profile: dict,
        fallback: tuple[float, float],
        lo_scale: float,
        hi_scale: float,
        lo_cap: float,
        hi_cap: float,
    ) -> float:
        if not isinstance(profile, dict) or not profile:
            return random.uniform(*fallback)
        try:
            p25 = float(profile.get("p25", 0))
            p75 = float(profile.get("p75", 0))
        except (TypeError, ValueError):
            return random.uniform(*fallback)
        if p25 <= 0 or p75 <= 0:
            return random.uniform(*fallback)
        lo = max(fallback[0], min(p25 * lo_scale, lo_cap))
        hi = max(lo + 0.05, min(p75 * hi_scale, hi_cap))
        return random.uniform(lo, hi)

    def sample_inter_message_delay(self, phase: str | bool = "first") -> float:
        """采样展示延迟（用于 GUI/CLI/Telegram 打字节奏），单位秒。"""
        if isinstance(phase, bool):
            phase = "burst" if phase else "first"

        if phase == "burst":
            return self._sample_delay_from_profile(
                getattr(self._persona, "burst_delay_profile", {}),
                fallback=(0.40, 1.20),
                lo_scale=1 / 8.0,
                hi_scale=1 / 5.0,
                lo_cap=2.0,
                hi_cap=4.0,
            )

        if phase == "followup":
            return self._sample_delay_from_profile(
                getattr(self._persona, "response_delay_profile", {}),
                fallback=(0.28, 0.95),
                lo_scale=1 / 18.0,
                hi_scale=1 / 12.0,
                lo_cap=1.5,
                hi_cap=3.0,
            )

        return self._sample_delay_from_profile(
            getattr(self._persona, "response_delay_profile", {}),
            fallback=(0.55, 1.45),
            lo_scale=1 / 14.0,
            hi_scale=1 / 8.0,
            lo_cap=2.5,
            hi_cap=5.0,
        )

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
        return sum(lengths) / len(lengths) < 10

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
        # 注入当前时间（使用用户时区，非服务器时区）
        now = datetime.now(_TIMEZONE)
        time_block = (
            f"\n\n## 当前时间\n"
            f"现在是 {now.strftime('%Y年%m月%d日 %H:%M')}，"
            f"{'凌晨' if now.hour < 6 else '早上' if now.hour < 9 else '上午' if now.hour < 12 else '中午' if now.hour < 13 else '下午' if now.hour < 18 else '晚上' if now.hour < 23 else '深夜'}。"
            f"请根据当前时间自然地回复，不要在白天叫对方去睡觉，也不要在深夜像白天一样精力充沛。"
        )
        system = self._system_prompt + time_block
        phase_block = self._build_phase_prompt_block()
        if phase_block:
            system = system + "\n\n" + phase_block
        expanded_query = self._expand_query(user_input)

        # 手动备注（动态读取，修改后即时生效）
        if self._notes:
            lines = ["## 你知道的关于对方和你们关系的事"]
            for note in self._notes:
                lines.append(f"- {note}")
            system = system + "\n\n" + "\n".join(lines)

        # 中期记忆（scratchpad）+ 情绪引导（锁保护，防止后台线程写入时读到不一致状态）
        with self._state_lock:
            scratchpad_block = self._scratchpad.to_prompt_block()
            emotion_block = self._emotion_state.to_prompt_block(self._persona)
            burst_hint = self._emotion_state.burst_hint(self._persona)
            open_threads = self._rank_open_threads(
                user_input, list(self._scratchpad.open_threads),
            )
        if scratchpad_block:
            system = system + "\n\n" + scratchpad_block
        if emotion_block:
            system = system + "\n\n" + emotion_block
        if open_threads:
            lines = [
                "## 回复优先级",
                "- 如果对方这条消息和未完话题相关，优先把未完的话题聊完。",
                "- 只有在对方明显切换到新话题时，才放下未完话题。",
                f"- 当前最相关的未完话题：{open_threads[0]}",
            ]
            if len(open_threads) > 1:
                lines.append("其他未完话题：")
                for thread in open_threads[1:]:
                    lines.append(f"- {thread}")
            system = system + "\n\n" + "\n".join(lines)

        # 每日知识库（persona 最近关注的动态）
        if self._knowledge_store:
            try:
                kb_items = self._knowledge_store.search(expanded_query, top_k=3)
                if kb_items:
                    kb_lines = ["## 你最近关注的新闻和动态（自然地提到，不要像背课文）"]
                    for item in kb_items:
                        kb_lines.append(f"- {item.summary}")
                    system = system + "\n\n" + "\n".join(kb_lines)
            except Exception as e:
                logger.warning("知识库检索失败: %s", e)

        # 长期记忆（RAG）
        if self._memory:
            raw_results = self._search_memory_cached(expanded_query, top_k=5)

            # 过滤低相关性结果 + 按行重叠率去重（overlap 窗口可能返回相似内容）
            seen_lines: set[str] = set()
            filtered: list[str] = []
            best_dist = raw_results[0][1] if raw_results else 0.0
            # 相对阈值：距离不超过最优结果的 2 倍，且绝对值不超过 1.2
            max_dist = min(best_dist * 2.0, 1.2)
            for doc, dist in raw_results:
                if dist > max_dist:
                    continue
                lines = set(doc.strip().split("\n")) - {""}
                if not lines:
                    continue
                overlap = len(lines & seen_lines) / len(lines)
                if overlap < 0.5:
                    filtered.append(doc)
                    seen_lines.update(lines)

            if filtered:
                context_parts = ["## 你们过去聊到类似话题时的真实对话（参考这些来回复，而不是编造）"]
                for fragment in filtered[:5]:
                    context_parts.append(fragment)
                    context_parts.append("---")
                system = system + "\n\n" + "\n".join(context_parts)

        # burst_hint 放在最末尾，确保 LLM 注意力最高
        if burst_hint:
            system = system + f"\n\n⚠️ {burst_hint}用 ||| 分隔多条消息。记住：反应词之后必须跟实际回应。"

        return system

    def _trim_history(self):
        max_turns = 40
        if len(self._history) > max_turns:
            trimmed = len(self._history) - max_turns
            self._history = self._history[-max_turns:]
            # 同步 scratchpad 索引，防止越界
            self._scratchpad.last_update_turn = max(0, self._scratchpad.last_update_turn - trimmed)

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
            "session_phase": self._session_phase,
            "scratchpad": self._scratchpad.to_dict(),
            "emotion_state": self._emotion_state.to_dict(),
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
            # 确保索引不超过实际历史长度
            self._scratchpad.last_update_turn = min(
                self._scratchpad.last_update_turn, len(self._history)
            )
            if data.get("emotion_state"):
                self._emotion_state = EmotionState.from_dict(data["emotion_state"])
            self.set_session_phase(str(data.get("session_phase", "normal")))
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
                new_pad = update_scratchpad(
                    self._client, self._scratchpad, recent,
                    persona_name=self._persona.name,
                )
                new_pad.last_update_turn = current_turn
                with self._state_lock:
                    self._scratchpad = new_pad
                    # 从 Scratchpad LLM 输出同步情绪（覆盖规则引擎微调）
                    if new_pad.emotion_raw:
                        self._emotion_state.sync_from_scratchpad(new_pad.emotion_raw)
            except Exception as e:
                logger.warning("Scratchpad 更新失败: %s", e)
            finally:
                self._scratchpad_updating = False

        threading.Thread(target=_do_update, daemon=True).start()

    def send_multi(self, user_input: str,
                   image: tuple[bytes, str] | None = None) -> list[str]:
        """发送消息并获取多条回复（模拟连发）。

        image: 可选 (bytes, mime_type) 图片数据，与文本一起发送给 LLM。
        """
        # 情绪衰减（距上次更新到现在的时间回归）
        with self._state_lock:
            self._emotion_state.decay(self._persona)
            mods = self._emotion_state.get_modifiers(self._persona)
            burst_range = self._emotion_state.compute_burst_range(self._persona)

        # 根据 burst 范围动态计算 token 预算（下限 512，保证多条短消息有足够空间）
        _low, high = burst_range
        tokens_per_msg = 120
        mods.max_output_tokens = min(1536, max(512, high * tokens_per_msg))

        system = self._build_system(user_input)

        parts = [types.Part(text=user_input)]
        if image:
            img_bytes, mime_type = image
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        user_msg = types.Content(role="user", parts=parts)
        self._history.append(user_msg)

        temperature = max(0.1, min(1.5, 0.8 + mods.temperature_delta))
        model = self._pick_generation_model(user_input, image)
        if model == MODEL_LIGHT:
            mods.max_output_tokens = min(mods.max_output_tokens, 768)

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    max_output_tokens=mods.max_output_tokens,
                ),
            )
        except Exception:
            # API 失败时安全回滚用户消息
            if self._history and self._history[-1] is user_msg:
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

        # 即时情绪微调（关键词规则）
        with self._state_lock:
            self._emotion_state.quick_adjust(user_input, raw_reply, self._persona)
        # 动态更新表情包概率
        self._sticker_probability = mods.sticker_probability

        # 异步更新中期记忆
        if self._should_update_scratchpad():
            self._trigger_scratchpad_update()

        messages = _split_reply(raw_reply, truncated=truncated)
        result = messages if messages else [raw_reply]

        # 低频人类噪声：偶尔打错字并自我修正，避免回复过于“工整”
        result, noise_applied = self._apply_human_noise_with_flag(result)

        # 按概率附加表情包
        result = self._maybe_attach_sticker(result, allow_sticker=not noise_applied)
        return result

    def _apply_human_noise(self, replies: list[str]) -> list[str]:
        out, _applied = self._apply_human_noise_with_flag(replies)
        return out

    def _apply_human_noise_with_flag(self, replies: list[str]) -> tuple[list[str], bool]:
        capped = min(max(self._human_noise_probability, 0.0), 0.08)
        if not replies or random.random() > capped:
            return replies, False

        if random.random() < 0.7:
            strategies = [self._apply_typo_noise, self._apply_hesitation_noise]
        else:
            strategies = [self._apply_hesitation_noise, self._apply_typo_noise]
        for strategy in strategies:
            mutated = strategy(replies)
            if mutated != replies:
                return mutated[:_MAX_BURST], True
        return replies, False

    def _apply_typo_noise(self, replies: list[str]) -> list[str]:
        if not replies:
            return replies

        candidates = [
            (i, m) for i, m in enumerate(replies)
            if 6 <= len(m) <= 40 and not m.startswith("[sticker:")
        ]
        if not candidates:
            return replies

        idx, msg = random.choice(candidates)
        typo = _introduce_minor_typo(msg)
        if typo == msg:
            return replies

        out = list(replies)
        out[idx] = typo
        if random.random() < 0.6 and len(out) < _MAX_BURST:
            out.insert(idx + 1, f"打错字了，{msg}")
        return out[:_MAX_BURST]

    def _apply_hesitation_noise(self, replies: list[str]) -> list[str]:
        candidates = [
            (i, m.strip()) for i, m in enumerate(replies)
            if 4 <= len(m.strip()) <= 35 and not m.startswith("[sticker:")
        ]
        if not candidates:
            return replies
        idx, msg = random.choice(candidates)
        if msg.startswith(("呃", "额", "嗯", "啊")):
            return replies
        prefix = random.choice(["呃，", "等下，", "啊对了，"])
        out = list(replies)
        out[idx] = f"{prefix}{msg}"
        return out

    def _maybe_attach_sticker(self, replies: list[str], allow_sticker: bool = True) -> list[str]:
        """按概率在回复后附加一张表情包。"""
        if not allow_sticker:
            return replies
        if not self._sticker_lib or not self._sticker_lib.stickers:
            return replies
        sticker_prob = max(0.01, min(0.35, self._sticker_probability))
        if random.random() > sticker_prob:
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
                model=MODEL_MAIN,
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
            if self._history and self._history[-1] is user_msg:
                self._history.pop()
            raise

        self._history.append(
            types.Content(role="model", parts=[types.Part(text="".join(full_reply))])
        )
        self._trim_history()

        # 异步更新中期记忆
        if self._should_update_scratchpad():
            self._trigger_scratchpad_update()
