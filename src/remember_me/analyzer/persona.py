"""人格分析器 - 从聊天记录中提取说话风格和人格特征。"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from remember_me.importers.base import ChatHistory

_EMOJI_RE = re.compile(
    r"[\U0001f600-\U0001f64f"
    r"\U0001f300-\U0001f5ff"
    r"\U0001f680-\U0001f6ff"
    r"\U0001f1e0-\U0001f1ff"
    r"\U00002702-\U000027b0"
    r"\U0000fe00-\U0000fe0f"
    r"\U0001f900-\U0001f9ff"
    r"\U0001fa00-\U0001fa6f"
    r"\U0001fa70-\U0001faff"
    r"\U00002600-\U000026ff"
    r"]+",
    re.UNICODE,
)

# 语气词/句尾词
_TONE_MARKERS = [
    "hh", "haha", "哈哈", "哈哈哈", "嘿嘿", "嘻嘻", "呜呜",
    "hhh", "哈哈哈哈", "233", "2333",
    "啊", "呀", "呢", "吧", "嘛", "噢", "哦", "嗯",
    "~", "～", "...", "。。。", "。。",
    "doge", "awsl", "yyds", "绝了", "无语", "救命",
    "好吧", "算了", "随便", "不知道", "可以", "好的",
    "emmm", "emm", "em",
]


@dataclass
class Persona:
    name: str
    total_messages: int = 0

    # 基础语言特征
    avg_length: float = 0.0
    short_msg_ratio: float = 0.0
    question_ratio: float = 0.0
    exclamation_ratio: float = 0.0

    # 词汇习惯
    catchphrases: list[str] = field(default_factory=list)
    top_emojis: list[str] = field(default_factory=list)
    tone_markers: list[str] = field(default_factory=list)
    sentence_endings: list[str] = field(default_factory=list)
    slang_expressions: list[str] = field(default_factory=list)   # 方言/网络用语/独特表达
    swear_ratio: float = 0.0                                      # 粗口比例

    # 行为模式
    active_hours: list[int] = field(default_factory=list)
    burst_ratio: float = 0.0
    avg_burst_length: float = 1.0
    burst_distribution: list[float] = field(default_factory=list)
    greeting_patterns: list[str] = field(default_factory=list)
    farewell_patterns: list[str] = field(default_factory=list)
    self_references: list[str] = field(default_factory=list)     # 自称方式
    chase_ratio: float = 0.0                                      # 用户沉默后追发的概率
    response_delay_profile: dict = field(default_factory=dict)    # 对方发言后你的响应时延（秒）
    burst_delay_profile: dict = field(default_factory=dict)       # 你连发内部时延（秒）
    silence_delay_profile: dict = field(default_factory=dict)     # 长沉默（>5分钟）后的时延（秒）

    # 话题偏好
    topic_keywords: list[str] = field(default_factory=list)
    topic_interests: dict[str, int] = field(default_factory=dict) # 话题→频次

    # 典型回复样例
    example_dialogues: list[dict[str, str]] = field(default_factory=list)
    burst_examples: list[dict] = field(default_factory=list)

    # 情绪画像（用于情绪系统基线）
    emotion_profile: dict = field(default_factory=dict)

    # 作息模板（用于空间/日程系统）
    daily_routine: dict = field(default_factory=dict)

    # 综合描述
    style_summary: str = ""

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Persona:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # 兼容旧版 JSON（缺少新字段时用默认值）
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


def _get_text_contents(target_msgs):
    """提取纯文本消息内容（排除 [图片] 等占位符）。"""
    contents = [m.content for m in target_msgs if not m.content.startswith("[")]
    return contents if contents else [m.content for m in target_msgs]


def _analyze_tone_markers(contents: list[str]) -> list[str]:
    """分析语气词使用习惯。"""
    counter: Counter[str] = Counter()
    for c in contents:
        c_lower = c.lower()
        for marker in _TONE_MARKERS:
            if marker in c_lower:
                counter[marker] += 1
    # 只返回出现次数 > 总消息 1% 的
    threshold = max(5, len(contents) * 0.01)
    return [m for m, count in counter.most_common(15) if count >= threshold]


def _analyze_sentence_endings(contents: list[str]) -> list[str]:
    """分析句尾习惯（最后1-2个字符）。"""
    ending_counter: Counter[str] = Counter()
    for c in contents:
        c = c.rstrip()
        if len(c) < 2:
            continue
        # 取最后一个字符
        last = c[-1]
        if last in "啊呀呢吧嘛哦噢吗呐诶哎耶喔咯":
            ending_counter[last] += 1
        # 取最后两个字符
        last2 = c[-2:]
        if last2 in ("hh", "HH", "哈哈", "嘿嘿", "嘻嘻", "呜呜", "哇塞"):
            ending_counter[last2] += 1
        # 省略号
        if c.endswith("...") or c.endswith("。。。") or c.endswith("。。"):
            ending_counter["..."] += 1
        if c.endswith("~") or c.endswith("～"):
            ending_counter["~"] += 1

    threshold = max(5, len(contents) * 0.01)
    return [e for e, count in ending_counter.most_common(10) if count >= threshold]


def _analyze_active_hours(target_msgs) -> list[int]:
    """分析活跃时段。"""
    hour_counter: Counter[int] = Counter()
    for m in target_msgs:
        if m.timestamp:
            hour_counter[m.timestamp.hour] += 1
    return [h for h, _ in hour_counter.most_common(5)]


def _analyze_burst_ratio(history) -> float:
    """分析连发消息比例（连续发 2 条以上算连发）。"""
    if len(history.messages) < 2:
        return 0.0
    burst_count = 0
    target_total = 0
    for i in range(1, len(history.messages)):
        curr = history.messages[i]
        prev = history.messages[i - 1]
        if curr.is_target:
            target_total += 1
            if prev.is_target:
                burst_count += 1
    return round(burst_count / max(target_total, 1), 3)


def _analyze_burst_pattern(history) -> tuple[float, list[float]]:
    """分析连发模式：平均连发长度、1-5条各自概率。"""
    bursts: list[int] = []
    current = 0
    for m in history.messages:
        if m.is_target:
            current += 1
        else:
            if current > 0:
                bursts.append(current)
            current = 0
    if current > 0:
        bursts.append(current)

    if not bursts:
        return 1.0, [1.0, 0.0, 0.0, 0.0, 0.0]

    avg = round(sum(bursts) / len(bursts), 1)

    # 计算 1-5+ 条的概率分布
    total = len(bursts)
    dist = []
    for n in range(1, 6):
        if n < 5:
            dist.append(round(sum(1 for b in bursts if b == n) / total, 3))
        else:
            dist.append(round(sum(1 for b in bursts if b >= 5) / total, 3))

    return avg, dist


def _percentile(sorted_values: list[float], p: float) -> float:
    """线性插值分位数。sorted_values 必须已升序。"""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _build_delay_profile(values: list[float]) -> dict:
    """将秒级样本压缩为分位统计，便于运行时采样。"""
    filtered = sorted(v for v in values if 0 < v <= 12 * 3600)
    if not filtered:
        return {}
    return {
        "count": len(filtered),
        "p25": round(_percentile(filtered, 0.25), 1),
        "p50": round(_percentile(filtered, 0.50), 1),
        "p75": round(_percentile(filtered, 0.75), 1),
        "mean": round(sum(filtered) / len(filtered), 1),
    }


def _analyze_greetings_farewells(contents: list[str]):
    """分析打招呼和告别方式。"""
    greeting_patterns = [
        "早", "早安", "早上好", "上午好", "中午好", "下午好", "晚上好",
        "嗨", "hi", "hello", "hey", "你好", "哈喽", "在吗", "在不",
        "醒了", "起来了", "起床了",
    ]
    farewell_patterns = [
        "晚安", "拜拜", "拜", "bye", "再见", "睡了", "睡觉",
        "去了", "先走了", "下次聊", "回聊", "good night", "gn",
        "我睡了", "困了",
    ]
    greet_counter: Counter[str] = Counter()
    farewell_counter: Counter[str] = Counter()
    for c in contents:
        c_lower = c.lower().strip()
        for g in greeting_patterns:
            if c_lower.startswith(g) or c_lower == g:
                greet_counter[g] += 1
                break
        for f in farewell_patterns:
            if f in c_lower:
                farewell_counter[f] += 1
                break
    greetings = [g for g, _ in greet_counter.most_common(5) if greet_counter[g] >= 2]
    farewells = [f for f, _ in farewell_counter.most_common(5) if farewell_counter[f] >= 2]
    return greetings, farewells


def _analyze_topic_keywords(contents: list[str]) -> list[str]:
    """提取高频话题关键词（排除常见停用词）。"""
    stopwords = set(
        "的了是我你他她它在有不这那就也都要会可以"
        "吗呢啊吧嘛哦好说去看来到想知道觉得没什么怎么"
        "一个人还很多少能但和跟被把被让对为因所以"
        "上下中大小点些里面外时候东西事情比较如果然后"
    )
    _noise_re = re.compile(
        r"^(https?|//|www|douyin|com|cn|mp4|html"
        r"|哈{3,}|嘿{3,}|嘻{3,}|呜{3,}|嗯{3,}|啊{3,})$"
    )

    word_counter: Counter[str] = Counter()
    for c in contents:
        # 跳过含 URL 的消息
        if "http" in c or "douyin" in c:
            continue
        segments = re.split(r"[，。！？,.\s!?~…、；;：:""''\"'()\\[\\]【】]+", c)
        for seg in segments:
            seg = seg.strip()
            if len(seg) < 2 or len(seg) > 8:
                continue
            if any(ch in stopwords for ch in seg) and len(seg) <= 2:
                continue
            if seg.startswith("[") or _noise_re.match(seg):
                continue
            word_counter[seg] += 1

    threshold = max(5, len(contents) * 0.001)
    return [w for w, count in word_counter.most_common(50) if count >= threshold][:20]


_POSITIVE_WORDS_RE = re.compile(r"(哈{2,}|笑死|绝了|牛逼|666|好家伙|太好了|开心|爽|赢了|嘻嘻|好耶)")
_NEGATIVE_WORDS_RE = re.compile(r"(傻逼|妈的|操|滚|烦死|恶心|吐了|无语|服了|气死)")
_SAD_WORDS_RE = re.compile(r"(难过|心累|呜|哭|寄了|完蛋|废了|裂开|emo|伤心)")
_EXCITED_WORDS_RE = re.compile(r"(！{2,}|？{2,}|卧槽|我靠|天|啊{2,}|真的假的)")


def _analyze_emotion_profile(history: ChatHistory, contents_no_url: list[str],
                             topic_interests: dict[str, int]) -> dict:
    """从历史数据中提取情绪画像：基线 valence/arousal + 话题情绪倾向。"""
    if not contents_no_url:
        return {}

    # 统计正/负/悲/兴奋关键词出现率，推算基线 valence/arousal
    total = len(contents_no_url)
    pos_count = sum(1 for c in contents_no_url if _POSITIVE_WORDS_RE.search(c))
    neg_count = sum(1 for c in contents_no_url if _NEGATIVE_WORDS_RE.search(c))
    sad_count = sum(1 for c in contents_no_url if _SAD_WORDS_RE.search(c))
    excited_count = sum(1 for c in contents_no_url if _EXCITED_WORDS_RE.search(c))

    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    sad_ratio = sad_count / total
    excited_ratio = excited_count / total

    # 基线 valence：正向词越多越偏正，负向词越多越偏负
    default_valence = round((pos_ratio - neg_ratio - sad_ratio) * 2, 2)
    default_valence = max(-0.5, min(0.5, default_valence))

    # 基线 arousal：兴奋词和负面高激动词越多越高，悲伤词拉低
    default_arousal = round((excited_ratio + neg_ratio * 0.5 - sad_ratio * 0.5) * 2, 2)
    default_arousal = max(-0.5, min(0.5, default_arousal))

    # 话题情绪倾向：每个话题中正/负关键词出现的比例差异
    topic_valence: dict[str, float] = {}
    topic_defs = {
        "游戏": r"(王者|游戏|上号|排位|吃鸡)",
        "音乐": r"(听歌|歌|音乐|演唱会|专辑)",
        "学习": r"(学习|上课|作业|考试|老师)",
        "美食": r"(好吃|外卖|奶茶|火锅|烧烤)",
        "影视": r"(电影|电视|追剧|动漫|B站)",
    }
    for topic, pattern in topic_defs.items():
        related = [c for c in contents_no_url if re.search(pattern, c, re.I)]
        if len(related) < 5:
            continue
        t_pos = sum(1 for c in related if _POSITIVE_WORDS_RE.search(c))
        t_neg = sum(1 for c in related if _NEGATIVE_WORDS_RE.search(c))
        t_ratio = (t_pos - t_neg) / len(related)
        if abs(t_ratio) > 0.05:
            topic_valence[topic] = round(t_ratio * 3, 2)

    return {
        "default_valence": default_valence,
        "default_arousal": default_arousal,
        "topic_valence": topic_valence,
    }


def analyze(history: ChatHistory, max_examples: int = 30) -> Persona:
    target_msgs = history.target_messages
    if not target_msgs:
        return Persona(name=history.target_name)

    contents = _get_text_contents(target_msgs)
    total = len(contents)
    if total == 0:
        return Persona(name=history.target_name)

    # ── 基础统计 ──
    lengths = [len(c) for c in contents]
    avg_length = sum(lengths) / total
    short_count = sum(1 for l in lengths if l < 10)
    short_msg_ratio = short_count / total
    question_count = sum(1 for c in contents if "?" in c or "？" in c)
    exclamation_count = sum(1 for c in contents if "!" in c or "！" in c)

    # ── emoji ──
    emoji_counter: Counter[str] = Counter()
    for c in contents:
        for match in _EMOJI_RE.finditer(c):
            emoji_counter[match.group()] += 1
    top_emojis = [e for e, _ in emoji_counter.most_common(10)]

    # ── 口头禅（降低阈值，提取更多） ──
    phrase_counter: Counter[str] = Counter()
    for c in contents:
        segments = re.split(r"[，。！？,.\s!?~…]+", c)
        for seg in segments:
            seg = seg.strip()
            if 2 <= len(seg) <= 8:
                phrase_counter[seg] += 1
    _noise_phrase_re = re.compile(
        r"^(哈{4,}|嘿{3,}|嘻{3,}|呜{3,}|嗯{3,}|啊{3,}"
        r"|https?|douyin|com|www|mp4|html|//v"
        r"|哈{3,}笑|笑死哈|哈{3,}过)$"
    )
    threshold = max(3, total * 0.0005)  # 0.05% 阈值，捕获更多特色表达
    catchphrases = [p for p, count in phrase_counter.most_common(100)
                    if count >= threshold
                    and not _noise_phrase_re.match(p)][:30]

    # ── 语气词 ──
    tone_markers = _analyze_tone_markers(contents)

    # ── 句尾习惯 ──
    sentence_endings = _analyze_sentence_endings(contents)

    # ── 活跃时段 ──
    active_hours = _analyze_active_hours(target_msgs)

    # ── 连发消息比例 ──
    burst_ratio = _analyze_burst_ratio(history)
    avg_burst_length, burst_distribution = _analyze_burst_pattern(history)

    # ── 节奏统计（用于运行时采样）+ 用户沉默后追发概率 ──
    response_gaps: list[float] = []        # 用户发言 -> 目标回复
    burst_gaps: list[float] = []           # 目标连续发言的内部间隔（<=5分钟）
    silence_gaps: list[float] = []         # 目标发言后出现长沉默（>5分钟）
    chase_count = 0                        # 长沉默后，目标继续主动发言
    no_chase_count = 0                     # 长沉默后，用户先回复

    msgs = history.messages
    for i in range(1, len(msgs)):
        prev = msgs[i - 1]
        curr = msgs[i]
        if not prev.timestamp or not curr.timestamp:
            continue
        gap = (curr.timestamp - prev.timestamp).total_seconds()
        if gap <= 0 or gap > 12 * 3600:
            continue

        if not prev.is_target and curr.is_target:
            response_gaps.append(gap)

        if prev.is_target and curr.is_target and gap <= 300:
            burst_gaps.append(gap)

        if prev.is_target and gap > 300:
            silence_gaps.append(gap)
            if curr.is_target:
                chase_count += 1
            else:
                no_chase_count += 1

    chase_total = chase_count + no_chase_count
    chase_ratio = round(chase_count / chase_total, 3) if chase_total else 0.0
    response_delay_profile = _build_delay_profile(response_gaps)
    burst_delay_profile = _build_delay_profile(burst_gaps)
    silence_delay_profile = _build_delay_profile(silence_gaps)

    # ── 打招呼 & 告别 ──
    greeting_patterns, farewell_patterns = _analyze_greetings_farewells(contents)

    # ── 话题关键词 ──
    topic_keywords = _analyze_topic_keywords(contents)

    # ── 方言/网络用语/独特表达 ──
    slang_map = {
        "你嘛": r"你嘛", "吗的": r"吗的", "要得": r"要得", "对头": r"对头",
        "噻": r"噻", "老子": r"老子", "笑西": r"笑西", "孽障": r"孽障",
        "书背": r"书背", "书币": r"书币", "hiehie": r"hiehie", "赫赫": r"赫赫",
        "干巴爹": r"干巴爹", "速速": r"速速", "笑死": r"笑死",
        "我靠": r"我靠", "他妈的": r"他妈的", "几把": r"几把",
        "懂的都懂": r"懂的都懂", "谁懂": r"谁懂", "是这个理": r"是这个理",
        "真羡慕你": r"真羡慕你", "嘤嘤嘤": r"嘤嘤嘤",
    }
    slang_expressions = []
    for label, pattern in slang_map.items():
        cnt = sum(1 for c in contents if re.search(pattern, c))
        if cnt >= 3:
            slang_expressions.append(label)

    # ── 粗口比例 ──
    swear_re = re.compile(r"他妈|妈的|吗的|卧槽|我靠|操|屎|几把|傻逼|牛逼")
    swear_count = sum(1 for c in contents if swear_re.search(c))
    swear_ratio = round(swear_count / total, 3)

    # ── 自称方式 ──
    self_ref_map = {"老子": r"老子", "爹": r"爹[^爹]|^爹$", "我": r"^我[^的]|[^是]我$"}
    self_references = [label for label, pattern in self_ref_map.items()
                       if sum(1 for c in contents if re.search(pattern, c)) >= 5]

    # ── 话题兴趣（精细化，区分子类别） ──
    # 先过滤掉含 URL 的消息（避免 https 中的 ps 等误匹配）
    contents_no_url = [c for c in contents if "http" not in c and "douyin" not in c]
    topic_defs = {
        "手游/王者荣耀": r"(王者|排位|上号|五排|双排|连跪|段位|钻石|铂金|星耀|吕布|韩信|甄姬|匹配|mvp|吃鸡|和平精英|荣耀)",
        "游戏(泛)": r"(游戏|打游戏)",
        "电竞赛事": r"(edg|dk|rng|比赛|世界赛|msi|战队)",
        "音乐": r"(听歌|歌[^词]|耳机|音乐|音乐节|专辑|演唱会)",
        "学习/校园": r"(学习|上课|作业|考试|ppt|老师|学年|学期|宿舍|体测)",
        "影视/追剧": r"(电影|电视|追剧|动漫|漫画|哈利波特|弥留|B站|b站|番剧)",
        "美食": r"(好吃|恰|麻辣|米线|外卖|烧烤|奶茶|火锅|肥牛)",
        "吐槽/搞笑": r"(无语|离谱|笑死|受不了|绝了|我靠|服了|搞笑)",
        "日常生活": r"(睡觉|起床|洗澡|出门|回家|快递|天气)",
        "热搜/社交": r"(热搜|微博|知乎|最右|b站|B站|抖音|小红书|豆瓣|吃瓜|八卦|热帖)",
    }
    topic_interests = {}
    topic_threshold = max(3, len(contents_no_url) * 0.005)
    for topic, pattern in topic_defs.items():
        cnt = sum(1 for c in contents_no_url if re.search(pattern, c, re.I))
        if cnt >= topic_threshold:
            topic_interests[topic] = cnt
    # 兜底：保证"热搜/社交"始终存在（低权重），让主动消息有更多话题来源
    if "热搜/社交" not in topic_interests:
        topic_interests["热搜/社交"] = max(1, topic_threshold)

    # ── 情绪画像 ──
    emotion_profile = _analyze_emotion_profile(history, contents_no_url, topic_interests)

    # ── 典型对话样例（均匀采样） ──
    pairs = history.as_dialogue_pairs()
    pairs = [(u, r) for u, r in pairs
             if not u.content.startswith("[") and not r.content.startswith("[")]
    if len(pairs) <= max_examples:
        sampled = pairs
    else:
        step = len(pairs) / max_examples
        sampled = [pairs[min(int(i * step), len(pairs) - 1)] for i in range(max_examples)]
    example_dialogues = [
        {"user": p[0].content, "reply": p[1].content}
        for p in sampled
    ]

    # ── 真实连发对话样例（均匀采样，包含不同长度的连发） ──
    burst_segs = history.as_burst_dialogues()
    burst_segs = [(u, rs) for u, rs in burst_segs if len(rs) >= 2]
    burst_examples: list[dict] = []
    if burst_segs:
        max_burst_ex = 15
        if len(burst_segs) <= max_burst_ex:
            burst_sample = burst_segs
        else:
            step = len(burst_segs) / max_burst_ex
            burst_sample = [burst_segs[min(int(i * step), len(burst_segs) - 1)] for i in range(max_burst_ex)]
        burst_examples = [
            {"user": u, "replies": rs}
            for u, rs in burst_sample
        ]

    # ── 综合风格描述 ──
    style_parts = []

    if avg_length < 10:
        style_parts.append("说话非常简短，偏好短句")
    elif avg_length < 20:
        style_parts.append("说话简短")
    elif avg_length > 50:
        style_parts.append("经常写长段文字")
    else:
        style_parts.append("说话长度适中")

    if short_msg_ratio > 0.7:
        style_parts.append("大量消息不超过10个字，习惯把一句话拆成多条发")
    elif short_msg_ratio > 0.5:
        style_parts.append("较多短消息")

    if burst_ratio > 0.4:
        style_parts.append("经常连续发多条消息（不等对方回复就接着说）")
    elif burst_ratio > 0.2:
        style_parts.append("有时会连发消息")

    if question_count / total > 0.15:
        style_parts.append("经常提问")

    if exclamation_count / total > 0.15:
        style_parts.append("语气比较激动，经常用感叹号")

    if tone_markers:
        style_parts.append(f"常用语气词: {', '.join(tone_markers[:6])}")

    if sentence_endings:
        style_parts.append(f"句尾习惯: 经常以「{'」「'.join(sentence_endings[:5])}」结尾")

    if top_emojis:
        style_parts.append(f"常用 emoji: {''.join(top_emojis[:6])}")

    if catchphrases:
        style_parts.append(f"口头禅: {', '.join(catchphrases[:8])}")

    if greeting_patterns:
        style_parts.append(f"打招呼方式: {', '.join(greeting_patterns[:3])}")

    if farewell_patterns:
        style_parts.append(f"告别方式: {', '.join(farewell_patterns[:3])}")

    if active_hours:
        hour_strs = [f"{h}点" for h in active_hours[:3]]
        style_parts.append(f"最活跃时段: {', '.join(hour_strs)}")

    if topic_keywords:
        style_parts.append(f"常聊话题: {', '.join(topic_keywords[:10])}")

    if slang_expressions:
        style_parts.append(f"方言/独特用语: {', '.join(slang_expressions[:10])}")

    if swear_ratio > 0.01:
        style_parts.append(f"说话比较粗犷，偶尔带脏话（约{swear_ratio*100:.0f}%的消息）")

    if self_references and "老子" in self_references:
        style_parts.append("经常用「老子」自称")

    if topic_interests:
        sorted_topics = sorted(topic_interests.items(), key=lambda x: -x[1])
        style_parts.append(f"最感兴趣: {', '.join(t for t, _ in sorted_topics[:5])}")

    style_summary = "。".join(style_parts) + "。"

    return Persona(
        name=history.target_name,
        total_messages=total,
        avg_length=round(avg_length, 1),
        short_msg_ratio=round(short_msg_ratio, 3),
        question_ratio=round(question_count / total, 3),
        exclamation_ratio=round(exclamation_count / total, 3),
        catchphrases=catchphrases,
        top_emojis=top_emojis,
        tone_markers=tone_markers,
        sentence_endings=sentence_endings,
        slang_expressions=slang_expressions,
        swear_ratio=swear_ratio,
        active_hours=active_hours,
        burst_ratio=burst_ratio,
        avg_burst_length=avg_burst_length,
        burst_distribution=burst_distribution,
        chase_ratio=chase_ratio,
        response_delay_profile=response_delay_profile,
        burst_delay_profile=burst_delay_profile,
        silence_delay_profile=silence_delay_profile,
        greeting_patterns=greeting_patterns,
        farewell_patterns=farewell_patterns,
        self_references=self_references,
        topic_keywords=topic_keywords,
        topic_interests=topic_interests,
        example_dialogues=example_dialogues,
        burst_examples=burst_examples,
        emotion_profile=emotion_profile,
        daily_routine={},  # 由 cli/controller 的 import 流程单独提取并保存
        style_summary=style_summary,
    )


