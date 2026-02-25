"""空间状态 + 每日日程引擎 — 追踪 persona 当前位置和活动。"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from remember_me.models import MODEL_LIGHT

logger = logging.getLogger(__name__)

# ── 有效位置 ──
LOCATIONS = frozenset({
    "bedroom", "living_room", "kitchen", "bathroom",
    "school", "work", "commute",
    "outside_dining", "outside_shopping", "outside_other",
})

# ── 响应度等级 → 数值 ──
RESPONSIVENESS_LEVELS = {
    "free": 1.0,
    "relaxing": 0.85,
    "eating": 0.7,
    "commuting": 0.6,
    "busy": 0.4,
    "sleeping": 0.05,
}

# ── 位置的中文标签 ──
_LOCATION_LABEL = {
    "bedroom": "卧室", "living_room": "客厅", "kitchen": "厨房",
    "bathroom": "浴室", "school": "学校", "work": "公司",
    "commute": "路上（通勤）",
    "outside_dining": "外面（吃饭）", "outside_shopping": "外面（逛街）",
    "outside_other": "外面",
}


@dataclass
class ScheduleEntry:
    """今日日程中的单个时段。"""
    hour: int = 0
    minute: int = 0
    activity: str = "休息"
    location: str = "bedroom"
    responsiveness: str = "free"
    context_hint: str = ""


@dataclass
class SpaceModifiers:
    """空间驱动的行为修饰（与 EmotionModifiers 并行叠加）。"""
    reply_delay_factor: float = 1.0
    message_length_factor: float = 1.0
    proactive_allowed: bool = True
    proactive_cooldown_factor: float = 1.0
    burst_count_bias: int = 0


@dataclass
class SpaceState:
    """Persona 当前的空间/位置/活动状态。"""
    current_location: str = "bedroom"
    current_activity: str = "休息"
    context_hint: str = ""
    responsiveness: str = "free"
    schedule: list[dict] = field(default_factory=list)
    schedule_date: str = ""
    updated_at: str = ""

    def advance(self, now: datetime):
        """根据当前时间推进位置和活动状态。"""
        if not self.schedule:
            return
        now_minutes = now.hour * 60 + now.minute
        # 找到当前时间对应的最新时段（不超过当前时间的最晚条目）
        best: dict | None = None
        for entry in self.schedule:
            entry_minutes = int(entry.get("hour", 0)) * 60 + int(entry.get("minute", 0))
            if entry_minutes <= now_minutes:
                best = entry
        if not best:
            # 当前时间早于日程第一条，取最后一条（通常是前一天晚上的状态）
            best = self.schedule[-1]

        self.current_location = str(best.get("location", "bedroom"))
        self.current_activity = str(best.get("activity", "休息"))
        self.context_hint = str(best.get("context_hint", ""))
        self.responsiveness = str(best.get("responsiveness", "free"))
        self.updated_at = now.isoformat()

    def to_prompt_block(self) -> str:
        """生成 system prompt 注入块。"""
        loc_label = _LOCATION_LABEL.get(self.current_location, self.current_location)
        lines = [
            "## 你现在的状态",
            f"你现在在{loc_label}，正在{self.current_activity}。",
        ]
        if self.context_hint:
            lines.append(f"（{self.context_hint}）")
        lines.append("不要刻意提到自己在哪里或在做什么，但可以在自然的场景下体现出来。")

        # 忙碌/通勤时的额外提示
        if self.responsiveness in ("busy", "sleeping"):
            lines.append("你现在不太方便看手机，回复可能简短或慢。")
        elif self.responsiveness == "commuting":
            lines.append("你在路上，可能会断断续续地回消息。")

        return "\n".join(lines)

    def get_modifiers(self) -> SpaceModifiers:
        """根据当前响应度计算行为修饰。"""
        mods = SpaceModifiers()

        if self.responsiveness == "sleeping":
            mods.reply_delay_factor = 5.0
            mods.message_length_factor = 0.3
            mods.proactive_allowed = False
            mods.burst_count_bias = -3
        elif self.responsiveness == "busy":
            mods.reply_delay_factor = 2.5
            mods.message_length_factor = 0.6
            mods.proactive_allowed = False
            mods.proactive_cooldown_factor = 3.0
            mods.burst_count_bias = -1
        elif self.responsiveness == "commuting":
            mods.reply_delay_factor = 1.5
            mods.message_length_factor = 0.7
            mods.proactive_cooldown_factor = 1.5
        elif self.responsiveness == "eating":
            mods.reply_delay_factor = 1.8
            mods.message_length_factor = 0.75
            mods.proactive_cooldown_factor = 2.0
        elif self.responsiveness == "relaxing":
            mods.reply_delay_factor = 0.9
            mods.message_length_factor = 1.0
        # "free" 使用默认值 (全 1.0)

        return mods

    def to_dict(self) -> dict:
        return {
            "current_location": self.current_location,
            "current_activity": self.current_activity,
            "context_hint": self.context_hint,
            "responsiveness": self.responsiveness,
            "schedule": list(self.schedule),
            "schedule_date": self.schedule_date,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SpaceState:
        if not isinstance(data, dict):
            return cls()
        return cls(
            current_location=str(data.get("current_location", "bedroom")),
            current_activity=str(data.get("current_activity", "休息")),
            context_hint=str(data.get("context_hint", "")),
            responsiveness=str(data.get("responsiveness", "free")),
            schedule=list(data.get("schedule") or []),
            schedule_date=str(data.get("schedule_date", "")),
            updated_at=str(data.get("updated_at", "")),
        )


# ── 每日日程生成 ──

_DAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

_RESPONSIVENESS_MAP = {
    "睡觉": "sleeping", "起床": "free", "上课": "busy", "上班": "busy",
    "开会": "busy", "下班后": "relaxing", "通勤": "commuting",
    "洗澡": "busy", "做饭": "eating", "吃饭": "eating",
    "休闲": "relaxing", "外出吃饭": "eating", "逛街": "relaxing",
    "外出": "relaxing", "吃午饭": "eating", "吃晚饭": "eating",
    # 补充 LLM 可能生成的活动
    "自习": "busy", "写作业": "busy", "复习": "busy", "考试": "busy",
    "健身": "busy", "运动": "busy", "跑步": "busy",
    "约会": "relaxing", "聚会": "relaxing", "和朋友": "relaxing",
    "打游戏": "relaxing", "玩游戏": "relaxing", "看剧": "relaxing",
    "看电影": "relaxing", "刷手机": "free", "发呆": "free",
    "早餐": "eating", "吃早饭": "eating", "吃早餐": "eating",
    "午休": "sleeping", "午睡": "sleeping", "小憩": "sleeping",
    "购物": "relaxing", "买东西": "relaxing",
    "工作": "busy", "加班": "busy", "实习": "busy",
}


def generate_daily_schedule(
    client,
    routine,
    persona,
    day_of_week: int,
    date_str: str,
) -> list[ScheduleEntry]:
    """基于 DailyRoutine 模板 + LLM 生成今日日程。

    LLM 失败时 fallback 到模板 + 随机偏移。
    """
    from google import genai
    from google.genai import types

    is_weekend = day_of_week >= 5
    slots = routine.weekend_slots if is_weekend else routine.weekday_slots

    if not slots:
        return _default_schedule(routine)

    # 构建 LLM prompt
    template_text = "\n".join(
        f"  {s.hour:02d}:{s.minute:02d} - {s.activity}（{s.location}）"
        for s in slots
    )
    persona_name = getattr(persona, "name", "对方")
    style = getattr(persona, "style_summary", "")

    prompt = (
        f"你需要为「{persona_name}」生成今天（{date_str}，{_DAY_NAMES[day_of_week]}）的日程表。\n"
        f"人物风格：{style[:100]}\n\n"
        f"以下是从聊天记录中提取的作息模板：\n{template_text}\n\n"
        f"请基于模板生成今天的日程，可以：\n"
        f"- 每个时段随机偏移 ±15-30 分钟\n"
        f"- 根据星期几微调（周末可以晚起、多休闲）\n"
        f"- 为每个时段写一句自然的 context_hint（第一人称，如\"刚下课回到家\"\"在公交上刷手机\"）\n\n"
        f"返回 JSON 数组，每个元素：\n"
        f'{{"hour": 8, "minute": 30, "activity": "通勤", "location": "commute", "context_hint": "在公交上"}}\n\n'
        f"有效 location：bedroom, living_room, kitchen, bathroom, school, work, commute, "
        f"outside_dining, outside_shopping, outside_other\n\n"
        f"只输出 JSON 数组，不要其他内容。"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_LIGHT,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=1024,
            ),
        )
        raw = (response.text or "").strip()
        # 清理 markdown 代码块
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        entries_data = json.loads(raw)
        if not isinstance(entries_data, list):
            raise ValueError("LLM 返回非数组")

        entries = []
        for item in entries_data:
            try:
                if not isinstance(item, dict):
                    continue
                activity = str(item.get("activity", "休息"))
                location = str(item.get("location", "bedroom"))
                if location not in LOCATIONS:
                    location = "bedroom"
                responsiveness = _RESPONSIVENESS_MAP.get(activity, "free")
                entries.append(ScheduleEntry(
                    hour=int(item.get("hour", 0)) % 24,
                    minute=int(item.get("minute", 0)) % 60,
                    activity=activity,
                    location=location,
                    responsiveness=responsiveness,
                    context_hint=str(item.get("context_hint", "")),
                ))
            except (TypeError, ValueError, KeyError) as exc:
                logger.debug("跳过无效日程条目 %s: %s", item, exc)

        if entries:
            entries.sort(key=lambda e: (e.hour, e.minute))
            logger.info("LLM 生成日程成功：%d 个时段", len(entries))
            return entries

    except Exception as e:
        logger.warning("LLM 日程生成失败，使用 fallback: %s", e)

    return _fallback_schedule(slots, routine)


def _fallback_schedule(slots, routine) -> list[ScheduleEntry]:
    """LLM 失败时的 fallback：模板 + 随机偏移。"""
    entries = []
    for s in slots:
        jitter = random.randint(-15, 15)
        total_min = s.hour * 60 + s.minute + jitter
        total_min = max(0, min(23 * 60 + 59, total_min))
        hour = total_min // 60
        minute = total_min % 60
        location = s.location if s.location in LOCATIONS else "bedroom"
        responsiveness = _RESPONSIVENESS_MAP.get(s.activity, "free")
        entries.append(ScheduleEntry(
            hour=hour, minute=minute,
            activity=s.activity, location=location,
            responsiveness=responsiveness,
            context_hint="",
        ))
    entries.sort(key=lambda e: (e.hour, e.minute))
    return entries or _default_schedule(routine)


def _default_schedule(routine) -> list[ScheduleEntry]:
    """最小默认日程（按时间排序，覆盖全天）。"""
    sleep_start = getattr(routine, "sleep_start", 1)
    sleep_end = getattr(routine, "sleep_end", 8)
    entries = [
        ScheduleEntry(hour=0, minute=0, activity="睡觉",
                       location="bedroom", responsiveness="sleeping"),
        ScheduleEntry(hour=sleep_end, minute=30, activity="起床",
                       location="bedroom", responsiveness="free",
                       context_hint="刚醒来"),
        ScheduleEntry(hour=12, minute=0, activity="吃午饭",
                       location="kitchen", responsiveness="eating"),
        ScheduleEntry(hour=14, minute=0, activity="休闲",
                       location="living_room", responsiveness="relaxing"),
        ScheduleEntry(hour=19, minute=0, activity="吃晚饭",
                       location="kitchen", responsiveness="eating"),
        ScheduleEntry(hour=21, minute=0, activity="休闲",
                       location="living_room", responsiveness="relaxing",
                       context_hint="在沙发上刷手机"),
        ScheduleEntry(hour=sleep_start, minute=0, activity="睡觉",
                       location="bedroom", responsiveness="sleeping"),
    ]
    entries.sort(key=lambda e: e.hour * 60 + e.minute)
    return entries
