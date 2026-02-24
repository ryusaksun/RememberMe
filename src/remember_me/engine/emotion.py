"""情绪系统 - valence/arousal 二维连续模型，驱动行为参数动态调整。"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

# ── 情绪标签映射：(valence_min, valence_max, arousal_min, arousal_max) ──

_LABEL_REGIONS = [
    ("兴奋", 0.3, 1.0, 0.3, 1.0),
    ("开心", 0.3, 1.0, -0.3, 0.3),
    ("放松", 0.3, 1.0, -1.0, -0.3),
    ("愤怒", -1.0, -0.3, 0.3, 1.0),
    ("烦躁", -0.3, 0.0, 0.3, 1.0),
    ("低落", -1.0, -0.3, -1.0, -0.3),
    ("无聊", -0.3, 0.0, -1.0, -0.3),
    ("平静", -0.3, 0.3, -0.3, 0.3),
]

# ── 关键词检测模式 ──

_POSITIVE_RE = re.compile(r"(哈{2,}|笑死|绝了|牛逼|666|好家伙|太好了|开心|爽|赢了|嘻嘻)")
_NEGATIVE_RE = re.compile(r"(傻逼|妈的|操|滚|烦死|恶心|吐了|无语|服了|气死)")
_SAD_RE = re.compile(r"(难过|心累|呜|哭|寄了|完蛋|废了|裂开|emo|伤心)")
_EXCITED_RE = re.compile(r"(！{2,}|？{2,}|卧槽|我靠|天|啊{2,}|真的假的)")

# 半衰期（秒）
_HALF_LIFE = 900.0  # 15 分钟


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class EmotionModifiers:
    """情绪对各项行为参数的修正。"""

    temperature_delta: float = 0.0
    burst_count_bias: int = 0
    sticker_probability: float = 0.14
    reply_delay_factor: float = 1.0
    proactive_cooldown_factor: float = 1.0
    max_output_tokens: int = 1024


@dataclass
class EmotionState:
    """结构化情绪状态（valence-arousal 二维模型）。"""

    valence: float = 0.0   # 愉悦度 [-1, 1]
    arousal: float = 0.0   # 激动度 [-1, 1]
    label: str = "平静"
    intensity: float = 0.0  # 综合强度 [0, 1]
    trigger: str = ""
    updated_at: str = ""

    # ── 标签推导 ──

    def _compute_label(self) -> str:
        v, a = self.valence, self.arousal
        for name, v_lo, v_hi, a_lo, a_hi in _LABEL_REGIONS:
            if v_lo <= v <= v_hi and a_lo <= a <= a_hi:
                return name
        return "平静"

    def _update_derived(self):
        self.label = self._compute_label()
        self.intensity = _clamp((abs(self.valence) + abs(self.arousal)) / 2, 0, 1)

    # ── 时间衰减 ──

    def decay(self, persona=None):
        """情绪向基线自然回归。半衰期 15 分钟。"""
        if not self.updated_at:
            return

        try:
            elapsed = (datetime.now() - datetime.fromisoformat(self.updated_at)).total_seconds()
        except ValueError:
            return

        if elapsed <= 0:
            return

        decay_factor = 0.5 ** (elapsed / _HALF_LIFE)

        # 获取 persona 情绪基线（没有则用 0）
        baseline_v = 0.0
        baseline_a = 0.0
        if persona:
            ep = getattr(persona, "emotion_profile", None) or {}
            baseline_v = ep.get("default_valence", 0.0)
            baseline_a = ep.get("default_arousal", 0.0)

        self.valence = baseline_v + (self.valence - baseline_v) * decay_factor
        self.arousal = baseline_a + (self.arousal - baseline_a) * decay_factor
        self.updated_at = datetime.now().isoformat()
        self._update_derived()

    # ── 即时微调（关键词规则，无 LLM） ──

    def quick_adjust(self, user_input: str, model_reply: str, persona=None):
        """基于关键词的即时情绪微调。"""
        combined = user_input + " " + model_reply

        # 正向情绪
        if _POSITIVE_RE.search(combined):
            self.valence = _clamp(self.valence + 0.1, -1, 1)
            self.arousal = _clamp(self.arousal + 0.08, -1, 1)

        # 负向高激动（用户骂人/表达不满）
        if _NEGATIVE_RE.search(user_input):
            self.valence = _clamp(self.valence - 0.15, -1, 1)
            self.arousal = _clamp(self.arousal + 0.15, -1, 1)

        # 低落情绪
        if _SAD_RE.search(user_input):
            self.valence = _clamp(self.valence - 0.12, -1, 1)
            self.arousal = _clamp(self.arousal - 0.1, -1, 1)

        # 惊叹（高激动但方向不定）
        if _EXCITED_RE.search(combined):
            self.arousal = _clamp(self.arousal + 0.12, -1, 1)

        # 话题情绪（如果 persona 有 topic_valence）
        if persona:
            ep = getattr(persona, "emotion_profile", None) or {}
            topic_valence = ep.get("topic_valence", {})
            for topic, v_shift in topic_valence.items():
                if topic in combined:
                    self.valence = _clamp(self.valence + v_shift * 0.1, -1, 1)
                    break

        self.updated_at = datetime.now().isoformat()
        self._update_derived()

    # ── Scratchpad 更新同步 ──

    def sync_from_scratchpad(self, emotion_raw: dict):
        """从 Scratchpad LLM 输出同步情绪状态（覆盖规则引擎的微调）。"""
        if not emotion_raw:
            return
        try:
            v = emotion_raw.get("valence")
            a = emotion_raw.get("arousal")
            if v is not None:
                v_f = float(v)
                if not math.isfinite(v_f):
                    return
                self.valence = _clamp(v_f, -1, 1)
            if a is not None:
                a_f = float(a)
                if not math.isfinite(a_f):
                    return
                self.arousal = _clamp(a_f, -1, 1)
        except (ValueError, TypeError):
            return  # LLM 输出异常值，跳过本次同步
        trigger = emotion_raw.get("trigger", "")
        if trigger:
            self.trigger = str(trigger)
        self.updated_at = datetime.now().isoformat()
        self._update_derived()

    # ── 参数修正 ──

    def get_modifiers(self, persona=None) -> EmotionModifiers:
        """根据当前情绪生成行为参数修正。"""
        v, a = self.valence, self.arousal
        mods = EmotionModifiers()

        # temperature: 高 arousal → 升温，低 arousal → 降温
        mods.temperature_delta = _clamp(a * 0.15, -0.15, 0.15)

        # burst 偏移
        burst_bias = (v * 0.5 + a * 0.5) * 2
        mods.burst_count_bias = round(_clamp(burst_bias, -2, 2))

        # 表情包概率
        base = 0.14
        if v > 0.3:
            mods.sticker_probability = min(base + v * 0.12, 0.30)
        elif v < -0.3:
            mods.sticker_probability = max(base + v * 0.1, 0.03)
        else:
            mods.sticker_probability = base

        # 回复延迟
        if a > 0:
            mods.reply_delay_factor = 1.0 - a * 0.4  # 兴奋时更快 [0.6, 1.0]
        else:
            mods.reply_delay_factor = 1.0 - a * 0.5  # 低沉时更慢 [1.0, 1.5]

        # 主动消息冷却（限制范围 0.6-1.5，避免低落时完全沉默）
        if v > 0.3 and a > 0.3:
            mods.proactive_cooldown_factor = 0.6
        elif v < -0.3:
            mods.proactive_cooldown_factor = 1.5
        else:
            mods.proactive_cooldown_factor = 1.0

        # max_output_tokens（控制总输出长度，防止过多条消息）
        if v < -0.3 and a < 0:
            mods.max_output_tokens = 512
        elif v < -0.3 and a > 0.3:
            mods.max_output_tokens = 768
        else:
            mods.max_output_tokens = 1024

        return mods

    # ── 情绪 Prompt 生成 ──

    def to_prompt_block(self, persona=None) -> str:
        """渲染为 system prompt 中的情绪引导块。"""
        if self.intensity < 0.1:
            return ""

        lines = ["## 你当前的心情"]

        intensity_word = "有点" if self.intensity < 0.4 else "很" if self.intensity < 0.7 else "非常"
        lines.append(f"你现在{intensity_word}{self.label}。")

        if self.trigger:
            lines.append(f"原因：{self.trigger}")

        v, a = self.valence, self.arousal
        if v > 0.3 and a > 0.3:
            lines.append("你现在很来劲，消息会发得比平时更多更快，语气更活跃。")
        elif v < -0.3 and a > 0.3:
            lines.append("你现在有点烦/生气，说话会更直接更冲，消息可能更短。")
        elif v < -0.3 and a < -0.3:
            lines.append("你现在心情不太好，话会变少，可能只回一两个字。不要装没事，就是话少了。")
        elif v > 0.3 and a < 0:
            lines.append("你现在心情不错但比较放松，语气温和随意。")

        lines.append("注意：不要刻意提到自己的情绪，自然地通过说话方式表现出来。")
        return "\n".join(lines)

    def burst_hint(self) -> str:
        """根据情绪生成动态 burst 条数提示。"""
        mods = self.get_modifiers()
        bias = mods.burst_count_bias
        if bias >= 2:
            return "你现在话很多，倾向于发 3-5 条，但绝不超过 5 条。"
        elif bias == 1:
            return "你现在话比较多，倾向于发 2-4 条。"
        elif bias == -1:
            return "你现在话比较少，倾向于只发 1-2 条。"
        elif bias <= -2:
            return "你现在不太想说话，大多数时候只回 1 条，而且很短。"
        return "大多数时候 1-3 条，偶尔最多 5 条。"

    # ── 序列化 ──

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "label": self.label,
            "intensity": round(self.intensity, 3),
            "trigger": self.trigger,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmotionState:
        state = cls(
            valence=data.get("valence", 0.0),
            arousal=data.get("arousal", 0.0),
            trigger=data.get("trigger", ""),
            updated_at=data.get("updated_at", ""),
        )
        state._update_derived()
        return state
