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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_TIMEZONE = ZoneInfo(os.environ.get("TZ", "Asia/Shanghai"))

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.engine.emotion import EmotionState
from remember_me.memory.governance import MemoryGovernance
from remember_me.memory.scratchpad import Scratchpad, update_scratchpad
from remember_me.memory.store import MemoryStore
from remember_me.models import MODEL_LIGHT, MODEL_MAIN

_MSG_SEPARATOR = "|||"

_SHORT_ACK_RE = re.compile(r"^(嗯+|哦+|好+|行|ok|好的?|收到|知道了|明白|了解|是的?)$", re.I)
_EVENT_IMPORTANT_RE = re.compile(
    r"(面试|考试|体检|手术|住院|生病|离职|裁员|分手|吵架|deadline|ddl|汇报|开会|签约|搬家|相亲|见家长|复合|焦虑|崩溃|失眠|抑郁|emo)",
    re.I,
)


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
    avg_burst_length = float(getattr(persona, "avg_burst_length", 1.0) or 1.0)
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


@dataclass(frozen=True)
class RhythmInputs:
    kind: str
    user_input: str
    session_phase: str
    event_score: float
    emotion_valence: float
    emotion_arousal: float
    persona_avg_burst: float
    persona_avg_len: float
    context_density: float
    burst_low: int
    burst_high: int


@dataclass(frozen=True)
class RhythmPolicy:
    min_count: int
    max_count: int
    prefer_count: int
    min_len: int
    max_len: int
    prefer_len: int
    allow_single_short_ack: bool = False


# 检测 LLM 推理泄漏：中文内容后跟随英文句子
# 匹配过渡符（引号、括号、标点、破折号等）+ 大写字母开头的英文长句
_REASONING_LEAK_RE = re.compile(
    r'[\s"\')}\]\.\-,;:。！？…—]+[A-Z][a-zA-Z\s,\'\"\-\*\(\)\.!?;:]{8,}$'
)
_MONOLOGUE_LEAK_RE = re.compile(
    r"(?i)(internal\s+monologue|chain\s*of\s*thought|thought\s*process|"
    r"reasoning(?:\s*trace|\s*process)?|/trial|trial\)\*\*|思考过程|推理过程|内心独白)"
)
_MESSAGE_LINE_RE = re.compile(r"(?is).*?message\s*\d+\s*[:：]\s*")
_PROMPT_LEAK_STRUCT_RE = re.compile(r"(?m)(^\s*##\s*|^\s*[-*]\s+|\*\*[^*]{2,}\*\*)")
_PROMPT_LEAK_KEYWORDS = (
    "角色设定",
    "系统提示",
    "system prompt",
    "system instruction",
    "你就是「",
    "你就是\"",
    "以下是从你和对方",
    "真实聊天记录中提取",
    "你是什么样的人",
    "语感参考",
    "回复格式（必须遵守）",
    "关键规则",
    "你过去真实的说话方式",
    "最重要的是：根据对方说的内容来回复",
    "绝不承认是 ai",
)
_SAFE_REPLY_FALLBACK = "嗯，刚刚卡了一下，你继续说。"


def _clean_reasoning_leak(msg: str) -> str:
    """剥离 LLM 偶尔泄漏的英文推理/元注释。"""
    if _MONOLOGUE_LEAK_RE.search(msg):
        # 尝试从 "Message 1:" 之后挽救正文，否则整条丢弃
        candidate = _MESSAGE_LINE_RE.sub("", msg).strip(" \t\r\n*-`：:")
        if candidate and not _MONOLOGUE_LEAK_RE.search(candidate):
            return candidate
        return ""
    m = _REASONING_LEAK_RE.search(msg)
    if m:
        cleaned = msg[:m.start()].strip()
        if cleaned:
            return cleaned
    return msg


def _is_prompt_leak_msg(msg: str) -> bool:
    """检测系统提示泄漏（如「角色设定/规则」块被模型误输出）。"""
    stripped = (msg or "").strip()
    if len(stripped) < 8:
        return False
    lowered = stripped.lower()
    hit_count = sum(1 for kw in _PROMPT_LEAK_KEYWORDS if kw in lowered or kw in stripped)
    if hit_count >= 2:
        return True
    if hit_count == 1 and _PROMPT_LEAK_STRUCT_RE.search(stripped):
        return True
    return False


def _is_reasoning_leak_msg(msg: str) -> bool:
    """检测整条消息是否为 LLM 推理泄漏（纯英文 / 中文推理片段）。"""
    stripped = msg.strip()
    if len(stripped) < 5:
        return False
    if _MONOLOGUE_LEAK_RE.search(stripped):
        return True
    if _is_prompt_leak_msg(stripped):
        return True
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
        should_drop_short_tail = False
        if meaningful:
            avg_len = sum(meaningful) / len(meaningful)
            should_drop_short_tail = len(result[-1]) <= 2 and avg_len > 4
        if truncated or should_drop_short_tail:
            result = result[:-1]
    # 硬上限：防止 LLM 输出过多条消息
    if len(result) > _MAX_BURST:
        result = result[:_MAX_BURST]
    return result


def _messages_to_history_text(messages: list[str]) -> str:
    """将最终发送给用户的多条消息转成历史文本，避免污染后续上下文。"""
    rows = [str(m).strip() for m in (messages or []) if str(m).strip()]
    if not rows:
        return _SAFE_REPLY_FALLBACK
    texts = [m for m in rows if not m.startswith("[sticker:")]
    if texts:
        return _MSG_SEPARATOR.join(texts)
    return rows[0]


def _is_short_ack(text: str) -> bool:
    return bool(_SHORT_ACK_RE.match((text or "").strip()))


def _split_long_message_to_segments(
    text: str,
    target_parts: int,
    *,
    min_len: int = 4,
    max_len: int = 24,
) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    if target_parts <= 1:
        return [raw]

    parts = re.split(r"(?<=[，。！？!?；;、\n])", raw)
    punct_segments = [p.strip() for p in parts if p and p.strip()]
    punct_segments = [p for p in punct_segments if p]

    segments: list[str] = []
    if punct_segments and len(punct_segments) >= 2:
        segments = punct_segments
    else:
        chunk = max(min_len, min(max_len, max(4, len(raw) // target_parts)))
        cursor = 0
        while cursor < len(raw):
            end = min(len(raw), cursor + chunk)
            segments.append(raw[cursor:end].strip())
            cursor = end

    merged: list[str] = []
    for seg in segments:
        if not seg:
            continue
        if merged and len(merged[-1]) < min_len:
            candidate = f"{merged[-1]}{seg}"
            if len(candidate) <= max_len + min_len:
                merged[-1] = candidate
                continue
        merged.append(seg)

    return [s for s in merged if s]


def _merge_overflow_segments(messages: list[str], target_count: int) -> list[str]:
    if target_count <= 0:
        return []
    out = [m.strip() for m in messages if m and m.strip()]
    if not out:
        return []
    while len(out) > target_count and len(out) >= 2:
        tail = out.pop()
        head = out.pop()
        glue = "" if head.endswith(("，", "。", "！", "？", "!", "?")) else "，"
        out.append(f"{head}{glue}{tail}")
    return out


def normalize_messages_by_policy(
    messages: list[str],
    policy: RhythmPolicy,
    *,
    user_input: str = "",
) -> list[str]:
    rows = [str(m).strip() for m in (messages or []) if str(m).strip()]
    if not rows:
        return []

    max_count = max(1, min(_MAX_BURST, int(policy.max_count)))
    min_count = max(1, min(max_count, int(policy.min_count)))
    min_len = max(2, int(policy.min_len))
    max_len = max(min_len + 2, int(policy.max_len))
    prefer_len = max(min_len, min(max_len, int(policy.prefer_len)))

    stickers = [m for m in rows if m.startswith("[sticker:")]
    texts = [m for m in rows if not m.startswith("[sticker:")]
    if not texts:
        return rows[:max_count]

    # 先按单条上限拆分过长文本
    expanded: list[str] = []
    for msg in texts:
        if len(msg) <= max_len:
            expanded.append(msg)
            continue
        est_parts = max(2, round(len(msg) / max(1, prefer_len)))
        est_parts = min(max_count, max(2, est_parts))
        expanded.extend(
            _split_long_message_to_segments(
                msg,
                est_parts,
                min_len=min_len,
                max_len=max_len,
            )
        )
    texts = [m for m in expanded if m]

    # 条数超上限时先合并文本，再丢弃贴纸，最后硬裁
    while len(texts) + len(stickers) > max_count and len(texts) > 1:
        texts = _merge_overflow_segments(texts, len(texts) - 1)
    while len(texts) + len(stickers) > max_count and stickers:
        stickers.pop()
    if len(texts) + len(stickers) > max_count:
        keep_text = max(1, max_count - len(stickers))
        texts = _merge_overflow_segments(texts, keep_text)[:keep_text]

    should_relax_min = policy.allow_single_short_ack and _is_short_ack(user_input)
    if not should_relax_min and len(texts) + len(stickers) < min_count and texts:
        need = min_count - (len(texts) + len(stickers))
        for _ in range(need):
            idx = max(range(len(texts)), key=lambda i: len(texts[i]), default=-1)
            if idx < 0:
                break
            msg = texts[idx]
            if len(msg) < max(min_len * 2, 8):
                break
            parts = _split_long_message_to_segments(msg, 2, min_len=min_len, max_len=max_len)
            if len(parts) < 2:
                break
            texts[idx:idx + 1] = parts
            if len(texts) + len(stickers) >= min_count:
                break

    # 消除过短碎片：和后句合并
    i = 0
    while i < len(texts) - 1:
        cur = texts[i]
        nxt = texts[i + 1]
        if len(cur) >= min_len:
            i += 1
            continue
        glue = "" if cur.endswith(("，", "。", "！", "？", "!", "?")) else "，"
        merged = f"{cur}{glue}{nxt}"
        if len(merged) <= max_len + min_len:
            texts[i] = merged
            del texts[i + 1]
            continue
        i += 1

    # 最终安全限流
    while len(texts) + len(stickers) > max_count and len(texts) > 1:
        texts = _merge_overflow_segments(texts, len(texts) - 1)
    while len(texts) + len(stickers) > max_count and stickers:
        stickers.pop()

    output = texts + stickers
    if not output:
        return rows[:1]
    return output[:max_count]


def _sanitize_reply_messages(raw_reply: str, truncated: bool = False) -> list[str]:
    """对模型原始输出做安全清洗，确保不会把推理文本直接发给用户。"""
    messages = _split_reply(raw_reply, truncated=truncated)
    if messages:
        return messages
    cleaned = _clean_reasoning_leak(raw_reply).strip()
    if cleaned and not _is_reasoning_leak_msg(cleaned):
        return [cleaned]
    return [_SAFE_REPLY_FALLBACK]


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
                 knowledge_store=None,
                 memory_governance: MemoryGovernance | None = None):
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
        self._memory_governance = memory_governance
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

    def _estimate_context_density(self, user_input: str) -> float:
        text = (user_input or "").strip()
        if not text:
            return 0.2
        recent = self.get_recent_context()
        recent_len = len(recent.replace("\n", ""))
        text_len = len(text)
        raw = min(1.0, (recent_len / 360.0) * 0.5 + (text_len / 120.0) * 0.5)
        return max(0.0, raw)

    @staticmethod
    def _estimate_event_score(kind: str, user_input: str) -> float:
        kind_key = str(kind or "").strip().lower()
        if kind_key == "event_followup":
            return 1.0
        if kind_key == "relationship_followup":
            return 0.85
        if kind_key in {"greet", "proactive"}:
            base = 0.35
        elif kind_key == "followup":
            base = 0.45
        else:
            base = 0.30

        text = (user_input or "").strip()
        if not text:
            return base
        if _EVENT_IMPORTANT_RE.search(text):
            return max(base, 0.82)
        if re.search(r"(上次|那次|还记得|之后|后来|结果|进展|怎么样了)", text):
            return max(base, 0.62)
        return base

    def build_rhythm_inputs(
        self,
        *,
        kind: str,
        user_input: str = "",
        event_score: float | None = None,
    ) -> RhythmInputs:
        with self._state_lock:
            valence = float(self._emotion_state.valence)
            arousal = float(self._emotion_state.arousal)
            burst_low, burst_high = self._emotion_state.compute_burst_range(self._persona)
        persona_avg_burst = float(getattr(self._persona, "avg_burst_length", 2.0) or 2.0)
        persona_avg_len = float(getattr(self._persona, "avg_length", 12.0) or 12.0)
        score = float(event_score) if event_score is not None else self._estimate_event_score(kind, user_input)
        score = max(0.0, min(1.0, score))
        return RhythmInputs(
            kind=str(kind or "reply"),
            user_input=str(user_input or ""),
            session_phase=str(getattr(self, "_session_phase", "normal") or "normal"),
            event_score=score,
            emotion_valence=max(-1.0, min(1.0, valence)),
            emotion_arousal=max(-1.0, min(1.0, arousal)),
            persona_avg_burst=persona_avg_burst,
            persona_avg_len=persona_avg_len,
            context_density=self._estimate_context_density(user_input),
            burst_low=max(1, min(_MAX_BURST, int(burst_low))),
            burst_high=max(1, min(_MAX_BURST, int(burst_high))),
        )

    def plan_rhythm_policy(
        self,
        *,
        kind: str,
        user_input: str = "",
        event_score: float | None = None,
    ) -> RhythmPolicy:
        inputs = self.build_rhythm_inputs(kind=kind, user_input=user_input, event_score=event_score)
        kind_key = inputs.kind.lower()

        if kind_key == "reply":
            min_count = max(1, min(_MAX_BURST, inputs.burst_low))
            max_count = max(min_count, min(_MAX_BURST, inputs.burst_high))
            prefer_count = max(min_count, min(max_count, round((min_count + max_count) / 2)))
            base_len = max(8, min(26, round(inputs.persona_avg_len)))
            min_len = max(4, min(16, base_len - 5))
            max_len = max(min_len + 4, min(38, base_len + 9))
            prefer_len = max(min_len, min(max_len, base_len))
            allow_single_short_ack = True
        else:
            min_count, max_count, prefer_count = 1, 2, 1
            min_len, max_len, prefer_len = 6, 22, 12
            allow_single_short_ack = False

        # 事件优先：关键事件优先保证信息完整
        if inputs.event_score >= 0.8:
            min_count = min(_MAX_BURST, min_count + 1)
            max_count = min(_MAX_BURST, max(max_count, min_count + 1))
            prefer_count = min(max_count, prefer_count + 1)
            min_len = min(28, min_len + 4)
            max_len = min(44, max_len + 8)
            prefer_len = min(max_len, prefer_len + 5)
        elif inputs.event_score >= 0.5:
            max_count = min(_MAX_BURST, max_count + 1)
            prefer_count = min(max_count, prefer_count + 1)
            min_len = min(26, min_len + 2)
            max_len = min(40, max_len + 4)
            prefer_len = min(max_len, prefer_len + 2)

        # 情绪微调：不覆盖事件优先，只做轻量偏移
        if inputs.emotion_arousal > 0.45:
            max_count = min(_MAX_BURST, max_count + 1)
            prefer_count = min(max_count, prefer_count + 1)
            min_len = max(4, min_len - 1)
            max_len = max(min_len + 4, max_len - 2)
        elif inputs.emotion_arousal < -0.35 or inputs.emotion_valence < -0.45:
            max_count = max(min_count, max_count - 1)
            prefer_count = max(min_count, prefer_count - 1)
            min_len = min(30, min_len + 2)
            max_len = min(46, max_len + 4)

        phase = inputs.session_phase
        if phase == "deep_talk":
            max_count = max(min_count, min(max_count, 3))
            min_len = min(30, min_len + 2)
            max_len = min(42, max_len + 2)
        elif phase == "cooldown":
            max_count = max(min_count, min(max_count, 2))
            prefer_count = min(prefer_count, 2)
            max_len = min(max_len, 22)
        elif phase == "ending":
            max_count = max(min_count, min(max_count, 2))
            prefer_count = 1
            max_len = min(max_len, 18)
        elif phase == "warmup":
            max_count = max(min_count, min(max_count, 3))

        max_count = max(1, min(_MAX_BURST, max_count))
        min_count = max(1, min(max_count, min_count))
        prefer_count = max(min_count, min(max_count, prefer_count))
        min_len = max(2, min(min_len, 40))
        max_len = max(min_len + 2, min(max_len, 48))
        prefer_len = max(min_len, min(max_len, prefer_len))

        return RhythmPolicy(
            min_count=min_count,
            max_count=max_count,
            prefer_count=prefer_count,
            min_len=min_len,
            max_len=max_len,
            prefer_len=prefer_len,
            allow_single_short_ack=allow_single_short_ack,
        )

    @staticmethod
    def format_rhythm_hint(policy: RhythmPolicy) -> str:
        if policy.min_count == policy.max_count:
            count_hint = f"{policy.max_count}条"
        else:
            count_hint = f"{policy.min_count}-{policy.max_count}条"
        if policy.min_len == policy.max_len:
            len_hint = f"{policy.max_len}字"
        else:
            len_hint = f"{policy.min_len}-{policy.max_len}字"
        return f"这次回复 {count_hint}，单条大约 {len_hint}。"

    def normalize_messages_by_rhythm(
        self,
        messages: list[str],
        policy: RhythmPolicy,
        *,
        user_input: str = "",
    ) -> list[str]:
        return normalize_messages_by_policy(messages, policy, user_input=user_input)

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
        lengths = [len(h.parts[0].text) for h in recent_model if h.parts and getattr(h.parts[0], "text", None)]
        return bool(lengths) and sum(lengths) / len(lengths) < 10

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
        """构建 system prompt（core > relationship > boundary > RAG > session > conflict > scratchpad/emotion）。"""
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
        burst_hint = ""
        rhythm_hint = ""
        gov_core_block = ""
        gov_relationship_block = ""
        gov_boundary_block = ""
        gov_session_block = ""
        gov_conflict_block = ""
        governance = getattr(self, "_memory_governance", None)

        # 1) 导入聊天记录核心事实（最高优先级，只读）
        if governance:
            try:
                gov_core_block, gov_session_block, gov_conflict_block = governance.build_prompt_blocks(
                    core_limit=6, session_limit=5, conflict_limit=2,
                )
                if hasattr(governance, "build_relationship_block"):
                    gov_relationship_block = governance.build_relationship_block(limit=10)
                if hasattr(governance, "build_active_boundary_block"):
                    gov_boundary_block = governance.build_active_boundary_block(limit=5)
            except Exception as e:
                logger.warning("核心记忆块构建失败: %s", e)
        if gov_core_block:
            system = system + "\n\n" + gov_core_block
        if gov_relationship_block:
            system = system + "\n\n" + gov_relationship_block
        if gov_boundary_block:
            system = system + "\n\n" + gov_boundary_block

        # 2) 导入历史检索（只依赖 import 建立的向量库，不写入运行时消息）
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

        # 3) 会话短期上下文（可过期，且不得覆盖核心）
        if gov_session_block:
            system = system + "\n\n" + gov_session_block
        if gov_conflict_block:
            system = system + "\n\n" + gov_conflict_block

        # 每日知识库（作为补充，不高于核心/关系/RAG/会话）
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

        # 中期记忆（scratchpad）+ 情绪引导（锁保护，防止后台线程写入时读到不一致状态）
        # 最小化锁持有时间：只在读取共享状态时持锁，排序/格式化在锁外进行
        with self._state_lock:
            scratchpad_block = self._scratchpad.to_prompt_block()
            emotion_block = self._emotion_state.to_prompt_block(self._persona)
            burst_hint = self._emotion_state.burst_hint(self._persona)
            _raw_threads = list(self._scratchpad.open_threads)
        open_threads = self._rank_open_threads(user_input, _raw_threads)
        rhythm_hint = self.format_rhythm_hint(
            self.plan_rhythm_policy(kind="reply", user_input=user_input)
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

        # burst_hint 放在最末尾，确保 LLM 注意力最高
        if burst_hint:
            system = system + (
                f"\n\n⚠️ {burst_hint}{rhythm_hint}用 ||| 分隔多条消息。"
                "记住：反应词之后必须跟实际回应。"
            )
        elif rhythm_hint:
            system = system + f"\n\n⚠️ {rhythm_hint}用 ||| 分隔多条消息。"

        return system

    def _apply_relationship_emotion_trigger(self, user_input: str):
        governance = getattr(self, "_memory_governance", None)
        if not governance:
            return
        store = getattr(governance, "_relationship_store", None)
        if not store:
            return
        try:
            facts = store.list_confirmed(limit=12)
        except Exception as e:
            logger.debug("读取关系记忆失败，跳过情绪触发: %s", e)
            return
        if not facts:
            return
        self._emotion_state.apply_relationship_trigger(facts, user_input)

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
        rhythm_policy = self.plan_rhythm_policy(kind="reply", user_input=user_input)

        # 根据节奏策略动态计算 token 预算（下限 512，保证多条短消息有足够空间）
        tokens_per_msg = max(80, min(180, rhythm_policy.prefer_len * 8))
        mods.max_output_tokens = min(1536, max(512, rhythm_policy.max_count * tokens_per_msg))

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

        result = _sanitize_reply_messages(raw_reply, truncated=truncated)

        # 低频人类噪声：偶尔打错字并自我修正，避免回复过于“工整”
        result, noise_applied = self._apply_human_noise_with_flag(result)
        result = self.normalize_messages_by_rhythm(
            result, rhythm_policy, user_input=user_input,
        )

        # 按概率附加表情包（计入条数上限）
        result = self._maybe_attach_sticker(
            result,
            allow_sticker=not noise_applied,
            max_count=rhythm_policy.max_count,
        )

        # 只把用户可见文本写入历史，防止原始泄漏文本污染后续上下文。
        history_text = _messages_to_history_text(result)
        self._history.append(types.Content(role="model", parts=[types.Part(text=history_text)]))
        self._trim_history()

        # 即时情绪微调（关键词规则）
        with self._state_lock:
            self._emotion_state.quick_adjust(user_input, history_text, self._persona)
            self._apply_relationship_emotion_trigger(user_input)
        # 动态更新表情包概率
        self._sticker_probability = mods.sticker_probability

        # 异步更新中期记忆
        if self._should_update_scratchpad():
            self._trigger_scratchpad_update()

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

    def _maybe_attach_sticker(
        self,
        replies: list[str],
        allow_sticker: bool = True,
        max_count: int = _MAX_BURST,
    ) -> list[str]:
        """按概率在回复后附加一张表情包。"""
        if not allow_sticker:
            return replies
        if not self._sticker_lib or not self._sticker_lib.stickers:
            return replies
        max_count = max(1, min(_MAX_BURST, int(max_count)))
        if len(replies) >= max_count:
            return replies[:max_count]
        sticker_prob = max(0.01, min(0.35, self._sticker_probability))
        if random.random() > sticker_prob:
            return replies[:max_count]

        # 根据回复内容判断情感
        combined = " ".join(replies)
        emotion = self._detect_emotion(combined)
        sticker = self._sticker_lib.random_sticker(emotion)
        if sticker:
            replies.append(f"[sticker:{sticker.path}]")
        return replies[:max_count]

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
