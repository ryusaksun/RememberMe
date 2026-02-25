"""从聊天记录中提取日常作息模式。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median

from remember_me.importers.base import ChatHistory

# ── 活动关键词 → (activity, location) ──

_ACTIVITY_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # 学校/工作
    (re.compile(r"(下课|放学|课[完了间]|到教室|上课)"), "上课", "school"),
    (re.compile(r"(到公司|上班|到工位|开始干活|打卡|在公司)"), "上班", "work"),
    (re.compile(r"(开会|会议中)"), "开会", "work"),
    (re.compile(r"(下班|收工|撤了|走了.*公司)"), "下班后", "living_room"),
    # 通勤
    (re.compile(r"(在路上|公交|地铁|到站|快到了|等车|打车|骑车|走路.*回)"), "通勤", "commute"),
    # 睡眠
    (re.compile(r"(睡了|睡觉|困了|晚安|要睡了|先睡|去睡|碎觉|躺平睡)"), "睡觉", "bedroom"),
    (re.compile(r"(起来了|起床|醒了|早安|早$|刚醒)"), "起床", "bedroom"),
    # 家务/日常
    (re.compile(r"(洗澡|冲凉|冲个澡)"), "洗澡", "bathroom"),
    (re.compile(r"(做饭|煮饭|热饭|下厨|炒菜|煮面)"), "做饭", "kitchen"),
    (re.compile(r"(吃饭|吃东西|干饭|恰饭|吃午饭|吃晚饭|吃早饭)"), "吃饭", "kitchen"),
    # 休闲
    (re.compile(r"(看剧|追剧|看电影|打游戏|刷手机|躺着|沙发|看视频|看番|打排位|开黑)"), "休闲", "living_room"),
    # 外出
    (re.compile(r"(出去吃|出来吃|餐厅|吃火锅|吃烧烤|点外卖)"), "外出吃饭", "outside_dining"),
    (re.compile(r"(逛街|商场|买东西|购物|超市)"), "逛街", "outside_shopping"),
    (re.compile(r"(出门|出去[了玩]|外面)"), "外出", "outside_other"),
]

# 30 分钟时间桶
_BUCKET_MINUTES = 30
_MIN_OCCURRENCES = 3  # 至少在 3 个不同日期出现才算稳定模式


@dataclass
class RoutineSlot:
    """单个作息时段。"""
    hour: int
    minute: int
    activity: str
    location: str
    confidence: float = 0.0
    responsiveness: float = 1.0


@dataclass
class DailyRoutine:
    """提取的每日作息模板。"""
    weekday_slots: list[RoutineSlot] = field(default_factory=list)
    weekend_slots: list[RoutineSlot] = field(default_factory=list)
    sleep_start: int = 1       # 入睡小时
    sleep_end: int = 8         # 起床小时

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> DailyRoutine:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        return {
            "weekday_slots": [asdict(s) for s in self.weekday_slots],
            "weekend_slots": [asdict(s) for s in self.weekend_slots],
            "sleep_start": self.sleep_start,
            "sleep_end": self.sleep_end,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DailyRoutine:
        def _parse_slots(raw: list) -> list[RoutineSlot]:
            result = []
            for item in (raw or []):
                if not isinstance(item, dict):
                    continue
                result.append(RoutineSlot(
                    hour=int(item.get("hour", 0)),
                    minute=int(item.get("minute", 0)),
                    activity=str(item.get("activity", "")),
                    location=str(item.get("location", "bedroom")),
                    confidence=float(item.get("confidence", 0.0)),
                    responsiveness=float(item.get("responsiveness", 1.0)),
                ))
            return result

        return cls(
            weekday_slots=_parse_slots(data.get("weekday_slots")),
            weekend_slots=_parse_slots(data.get("weekend_slots")),
            sleep_start=int(data.get("sleep_start", 1)),
            sleep_end=int(data.get("sleep_end", 8)),
        )


def _time_bucket(hour: int, minute: int) -> int:
    """将时间转为 30 分钟桶编号 (0-47)。"""
    return hour * 2 + (1 if minute >= 30 else 0)


def _bucket_to_time(bucket: int) -> tuple[int, int]:
    """桶编号 → (hour, minute_center)。"""
    hour = bucket // 2
    minute = 15 if bucket % 2 == 0 else 45
    return hour, minute


def analyze_routine(history: ChatHistory) -> DailyRoutine:
    """从聊天记录中提取日常作息模式。"""
    # 收集活动命中：{(bucket, is_weekend): [(activity, location, date_str)]}
    hits: dict[tuple[int, bool], list[tuple[str, str, str]]] = defaultdict(list)
    # 收集响应延迟：{bucket: [delay_seconds]}
    delays: dict[int, list[float]] = defaultdict(list)

    target_msgs = history.target_messages
    all_msgs = history.messages

    # 1) 提取活动关键词
    for msg in target_msgs:
        if not msg.timestamp:
            continue
        content = msg.content
        if content.startswith("["):  # 跳过 [图片] 等占位符
            continue
        for pattern, activity, location in _ACTIVITY_PATTERNS:
            if pattern.search(content):
                bucket = _time_bucket(msg.timestamp.hour, msg.timestamp.minute)
                is_weekend = msg.timestamp.weekday() >= 5
                date_str = msg.timestamp.strftime("%Y-%m-%d")
                hits[(bucket, is_weekend)].append((activity, location, date_str))
                break  # 每条消息只匹配第一个活动

    # 2) 计算各时段响应延迟（用户发消息到目标回复的间隔）
    for i in range(len(all_msgs) - 1):
        curr = all_msgs[i]
        nxt = all_msgs[i + 1]
        if not curr.is_target and nxt.is_target and curr.timestamp and nxt.timestamp:
            delay = (nxt.timestamp - curr.timestamp).total_seconds()
            if 0 < delay < 7200:  # 2 小时以内的合理回复
                bucket = _time_bucket(nxt.timestamp.hour, nxt.timestamp.minute)
                delays[bucket].append(delay)

    # 3) 聚合为 RoutineSlot
    def _aggregate(is_weekend: bool) -> list[RoutineSlot]:
        slots: list[RoutineSlot] = []
        for bucket in range(48):
            key = (bucket, is_weekend)
            entries = hits.get(key, [])
            if not entries:
                continue
            # 去重日期计数
            unique_dates = set(date_str for _, _, date_str in entries)
            if len(unique_dates) < _MIN_OCCURRENCES:
                continue
            # 投票选最频繁的活动
            activity_votes: dict[tuple[str, str], int] = defaultdict(int)
            for act, loc, _ in entries:
                activity_votes[(act, loc)] += 1
            (best_activity, best_location), _ = max(activity_votes.items(), key=lambda x: x[1])

            # 响应度：该时段中位延迟归一化（延迟越短响应越高）
            bucket_delays = delays.get(bucket, [])
            if bucket_delays:
                med = median(bucket_delays)
                # 30 秒以内 → 1.0，600 秒(10分钟) → 0.3，映射到 [0.1, 1.0]
                responsiveness = max(0.1, min(1.0, 1.0 - (med - 30) / 800))
            else:
                responsiveness = 0.7  # 默认中等

            hour, minute = _bucket_to_time(bucket)
            confidence = min(1.0, len(unique_dates) / 10.0)  # 10 天 → 1.0

            slots.append(RoutineSlot(
                hour=hour, minute=minute,
                activity=best_activity, location=best_location,
                confidence=confidence, responsiveness=responsiveness,
            ))
        # 按时间排序
        slots.sort(key=lambda s: (s.hour, s.minute))
        return slots

    weekday_slots = _aggregate(is_weekend=False)
    weekend_slots = _aggregate(is_weekend=True)

    # 4) 睡眠检测：从 farewell/greeting 时间推断
    sleep_hours: list[int] = []
    wake_hours: list[int] = []
    _SLEEP_RE = re.compile(r"(睡了|晚安|要睡了|先睡|碎觉|困了)")
    _WAKE_RE = re.compile(r"(起来了|起床|醒了|早安|早$|刚醒)")

    for msg in target_msgs:
        if not msg.timestamp or msg.content.startswith("["):
            continue
        if _SLEEP_RE.search(msg.content):
            sleep_hours.append(msg.timestamp.hour)
        elif _WAKE_RE.search(msg.content):
            wake_hours.append(msg.timestamp.hour)

    sleep_start = _most_common_hour(sleep_hours, default=1)
    sleep_end = _most_common_hour(wake_hours, default=8)

    routine = DailyRoutine(
        weekday_slots=weekday_slots,
        weekend_slots=weekend_slots,
        sleep_start=sleep_start,
        sleep_end=sleep_end,
    )

    # 如果提取结果过少，补充默认时段
    if len(weekday_slots) < 3:
        routine.weekday_slots = _fill_defaults(weekday_slots, sleep_start, sleep_end)
    if len(weekend_slots) < 3:
        routine.weekend_slots = _fill_defaults(weekend_slots, sleep_start, sleep_end)

    return routine


def _most_common_hour(hours: list[int], default: int) -> int:
    """返回出现最多的小时，无数据时返回默认值。"""
    if not hours:
        return default
    counts: dict[int, int] = defaultdict(int)
    for h in hours:
        counts[h] += 1
    return max(counts, key=counts.get)


def _fill_defaults(
    existing: list[RoutineSlot],
    sleep_start: int,
    sleep_end: int,
) -> list[RoutineSlot]:
    """当提取的时段过少时，补充默认作息。"""
    existing_hours = {s.hour for s in existing}
    defaults = [
        RoutineSlot(hour=sleep_start, minute=0, activity="睡觉", location="bedroom",
                    confidence=0.3, responsiveness=0.05),
        RoutineSlot(hour=sleep_end, minute=30, activity="起床", location="bedroom",
                    confidence=0.3, responsiveness=0.8),
        RoutineSlot(hour=12, minute=0, activity="吃午饭", location="kitchen",
                    confidence=0.3, responsiveness=0.7),
        RoutineSlot(hour=19, minute=0, activity="吃晚饭", location="kitchen",
                    confidence=0.3, responsiveness=0.7),
        RoutineSlot(hour=21, minute=0, activity="休闲", location="living_room",
                    confidence=0.3, responsiveness=0.9),
    ]
    merged = list(existing)
    for d in defaults:
        if d.hour not in existing_hours:
            merged.append(d)
    merged.sort(key=lambda s: (s.hour, s.minute))
    return merged
