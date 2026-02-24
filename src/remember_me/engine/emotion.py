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
        # 只以用户输入为主，避免模型自己的措辞反向放大情绪漂移。
        user_text = user_input or ""

        # 冷却：距上次调整不足 30 秒，幅度减半（避免连续相同刺激线性叠加）
        try:
            elapsed = (datetime.now() - datetime.fromisoformat(self.updated_at)).total_seconds()
        except (ValueError, TypeError):
            elapsed = 999
        scale = 0.5 if elapsed < 30 else 1.0

        # 正向情绪
        if _POSITIVE_RE.search(user_text):
            self.valence = _clamp(self.valence + 0.1 * scale, -1, 1)
            self.arousal = _clamp(self.arousal + 0.08 * scale, -1, 1)

        # 负向高激动（用户骂人/表达不满）
        if _NEGATIVE_RE.search(user_input):
            self.valence = _clamp(self.valence - 0.15 * scale, -1, 1)
            self.arousal = _clamp(self.arousal + 0.15 * scale, -1, 1)

        # 低落情绪
        if _SAD_RE.search(user_input):
            self.valence = _clamp(self.valence - 0.12 * scale, -1, 1)
            self.arousal = _clamp(self.arousal - 0.1 * scale, -1, 1)

        # 惊叹（高激动但方向不定）
        if _EXCITED_RE.search(user_text):
            self.arousal = _clamp(self.arousal + 0.12 * scale, -1, 1)

        # 话题情绪（如果 persona 有 topic_valence）
        if persona:
            ep = getattr(persona, "emotion_profile", None) or {}
            topic_valence = ep.get("topic_valence", {})
            for topic, v_shift in topic_valence.items():
                if topic in user_text:
                    self.valence = _clamp(self.valence + v_shift * 0.1, -1, 1)
                    break

        self.updated_at = datetime.now().isoformat()
        self._update_derived()

    def apply_relationship_trigger(self, facts, user_input: str):
        """根据关系记忆做中等强度情绪微调，增强长期关系一致性。"""
        text = (user_input or "").strip().lower()
        if not text:
            return

        delta_v = 0.0
        delta_a = 0.0
        triggers: list[str] = []

        for fact in list(facts or [])[:12]:
            fact_type = str(getattr(fact, "type", "") or "")
            content = str(getattr(fact, "content", "") or "").lower()
            if not content:
                continue

            hit = False
            if fact_type == "shared_event":
                if any(k in text for k in ("上次", "那次", "还记得", "之前", "那天")):
                    hit = True
                elif any(seg and len(seg) >= 2 and seg in text for seg in re.split(r"[，。！？、\s]+", content)):
                    hit = True
                if hit:
                    delta_v += 0.06
                    delta_a += 0.04
                    triggers.append("共同经历被提及")
            elif fact_type == "addressing":
                if any(seg and len(seg) >= 2 and seg in text for seg in re.split(r"[：，。！？、\s]+", content)):
                    hit = True
                if hit:
                    delta_v += 0.04
                    delta_a += 0.02
                    triggers.append("称呼习惯命中")
            elif fact_type == "boundary":
                if any(k in text for k in ("别提", "不聊", "别问", "不要再问", "不想聊")):
                    hit = True
                if hit:
                    delta_v -= 0.02
                    delta_a -= 0.03
                    triggers.append("边界线索命中")

        if delta_v == 0.0 and delta_a == 0.0:
            return

        # 中等权重：限制总偏移幅度，避免压过既有情绪机制
        delta_v = _clamp(delta_v, -0.10, 0.10)
        delta_a = _clamp(delta_a, -0.08, 0.08)
        self.valence = _clamp(self.valence + delta_v, -1, 1)
        self.arousal = _clamp(self.arousal + delta_a, -1, 1)
        self.updated_at = datetime.now().isoformat()
        if triggers:
            self.trigger = triggers[0]
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

        # max_output_tokens 不再硬编码，由 compute_burst_range() 驱动
        # 保留默认值 1024，send_multi() 会根据 burst 范围覆盖

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
            lines.append("你现在有点烦/生气，说话更冲更不耐烦。但怼人要怼完整，别怼一半就没了。")
        elif v < -0.3 and a < -0.3:
            lines.append("你现在心情不太好，语气冷淡敷衍，爱搭不理的感觉。但回复内容要完整，冷淡体现在语气上而不是把话吞一半。")
        elif v > 0.3 and a < 0:
            lines.append("你现在心情不错但比较放松，语气温和随意。")

        lines.append("注意：不要刻意提到自己的情绪，自然地通过说话方式表现出来。")
        return "\n".join(lines)

    def compute_burst_range(self, persona) -> tuple[int, int]:
        """返回 (min_burst, max_burst) 建议范围。仅基于 persona 习惯 + 情绪。

        具体发几条由 LLM 根据对话内容自行判断，这里只给合理范围。
        """
        # persona 基线
        base = getattr(persona, "avg_burst_length", 2.0)

        # 情绪系数：兴奋话多 ×1.5，低落话少 ×0.5
        mods = self.get_modifiers(persona)
        bias = mods.burst_count_bias  # -2 ~ +2
        emotion_factor = 1.0 + bias * 0.25  # 0.5 ~ 1.5

        center = base * emotion_factor
        low = max(2 if base >= 1.5 else 1, round(center * 0.7))
        high = max(low, min(8, round(center * 1.3)))

        return (low, high)

    def burst_hint(self, persona=None) -> str:
        """根据情绪+persona 生成动态 burst 条数提示。"""
        if persona:
            low, high = self.compute_burst_range(persona)
            if low == high:
                return f"这次回复 {low} 条左右。"
            return f"这次回复 {low}-{high} 条。"
        # 降级：没有 persona 时用旧逻辑
        mods = self.get_modifiers()
        bias = mods.burst_count_bias
        if bias >= 2:
            return "你现在话很多，倾向于发 3-5 条。"
        elif bias == 1:
            return "你现在话比较多，倾向于发 2-4 条。"
        elif bias == -1:
            return "你现在话比较少，倾向于只发 1-2 条。"
        elif bias <= -2:
            return "你现在不太想说话，条数少但每次要把话说完。"
        return "大多数时候 1-3 条。"

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
