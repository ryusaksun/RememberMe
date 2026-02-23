"""主动话题发起器 - Brave Search 搜索热点 + Gemini 3.1 Pro 生成。"""

from __future__ import annotations

import logging
import os
import random
import re as _re
from datetime import date

logger = logging.getLogger(__name__)

import requests as _requests
from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.engine.chat import _MSG_SEPARATOR, _build_system_prompt, _split_reply

# 每个话题对应多个搜索词，随机选一个，避免每次搜到同样内容
_TOPIC_SEARCH_HINTS: dict[str, list[str]] = {
    "手游/王者荣耀": [
        "王者荣耀最新更新", "王者荣耀新赛季 新英雄", "王者荣耀攻略 热门打法",
        "手游新闻 热门手游", "王者荣耀 赛事 KPL",
    ],
    "游戏(泛)": [
        "手游新闻 热门游戏", "steam 热门游戏 新游", "游戏圈最新资讯",
        "二次元手游 新游预约", "独立游戏 推荐",
    ],
    "电竞赛事": [
        "电竞比赛 LPL赛程", "英雄联盟赛事 最新", "电竞选手 热搜",
        "无畏契约赛事", "电竞圈 八卦新闻",
    ],
    "音乐": [
        "最新音乐 新歌发布", "音乐节 演唱会 最新", "网易云音乐 热歌榜",
        "华语乐坛 新专辑", "QQ音乐 热门歌曲",
    ],
    "音乐/rap": [
        "姜云升 新歌 最新动态", "中文说唱 rapper 新闻", "说唱新世代 嘻哈",
        "中文rap 热门歌曲", "说唱歌手 巡演 音乐节",
    ],
    "滑板": [
        "滑板 skateboard 最新", "滑板技巧 教程", "滑板赛事 极限运动",
        "街头滑板 潮流文化", "滑板品牌 新品",
    ],
    "美食": [
        "美食热搜 网红美食", "新开的餐厅 探店", "外卖 奶茶 热门推荐",
        "火锅烧烤 美食推荐", "美食博主 测评",
    ],
    "影视/追剧": [
        "最新电影 热映", "热播电视剧 追剧推荐", "新番动漫 B站热门",
        "综艺节目 最新一期", "豆瓣电影 高分推荐",
    ],
    "学习/校园": [
        "大学生 校园热点", "大学生活 搞笑日常", "考研 考试 学生热搜",
        "校园恋爱 大学趣事", "毕业季 校招",
    ],
    "吐槽/搞笑": [
        "微博热搜 今日", "最右 热门 搞笑", "知乎 热榜 离谱",
        "B站 热门视频 搞笑", "今日热搜 沙雕新闻",
        "微博 吃瓜 八卦", "抖音 热门 离谱事件",
    ],
    "日常生活": [
        "微博热搜榜 今天", "知乎热榜 今日话题", "B站热门 日常",
        "最右 热门帖子", "生活热搜 今日热点",
        "小红书 热门话题", "豆瓣 热门讨论",
    ],
    # ── 社交平台热搜（兜底话题，任何人都能聊） ──
    "热搜/社交": [
        "微博热搜榜 实时", "知乎热榜 今天", "B站每周必看 热门",
        "最右 搞笑热帖", "抖音热搜 今天", "小红书 热门话题",
    ],
}

_MODEL = "gemini-3.1-pro-preview"


def _brave_search(query: str, count: int = 5) -> list[dict]:
    """调用 Brave Search API 搜索。返回 [{title, description, url}]。"""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        logger.warning("BRAVE_API_KEY 未设置，跳过搜索")
        return []

    try:
        resp = _requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": count, "search_lang": "zh-hans", "freshness": "pw"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "url": item.get("url", ""),
            })
        return results
    except _requests.Timeout:
        logger.warning("Brave Search 超时: %s", query)
        return []
    except _requests.HTTPError as e:
        logger.warning("Brave Search API 错误 %s: %s", e.response.status_code, query)
        return []
    except Exception as e:
        logger.warning("Brave Search 异常: %s", e)
        return []


class TopicStarter:
    """基于 persona 的兴趣话题，Brave 搜索热点并用 Gemini 生成主动消息。"""

    def __init__(self, persona: Persona, client: genai.Client):
        self._persona = persona
        self._client = client
        self._system_prompt = _build_system_prompt(persona)
        self._used_topics: set[str] = set()
        self._cached: list[str] | None = None
        self._last_proactive: list[str] = []
        self._followup_count = 0
        self._proactive_count = 0
        self._chase_ratio = getattr(persona, "chase_ratio", 0.0)
        avg_burst = getattr(persona, "avg_burst_length", 1.0)
        # 综合评分：chase_ratio 为主，avg_burst_length 为辅
        chase_score = self._chase_ratio * 0.7 + min((avg_burst - 1) * 0.1, 0.3)
        if chase_score < 0.05:
            self._max_proactive_per_silence = 1
        elif chase_score < 0.15:
            self._max_proactive_per_silence = 2
        else:
            self._max_proactive_per_silence = 3

    def prefetch(self):
        """后台预生成消息，不修改任何状态。"""
        # 保存并恢复状态，防止 generate() 的副作用
        saved_last = self._last_proactive
        saved_followup = self._followup_count
        saved_proactive = self._proactive_count
        self._cached = self.generate()
        self._last_proactive = saved_last
        self._followup_count = saved_followup
        self._proactive_count = saved_proactive

    def pop_cached(self) -> list[str] | None:
        cached = self._cached
        self._cached = None
        if cached:
            self._last_proactive = cached
            self._followup_count = 0
            self._proactive_count += 1
        return cached

    def pick_topic(self) -> str | None:
        interests = self._persona.topic_interests
        if not interests:
            return None
        available = {t: w for t, w in interests.items() if t not in self._used_topics}
        if not available:
            # 保留上一个话题，避免清空后立即重复
            last = self._used_topics.copy()
            self._used_topics.clear()
            available = {t: w for t, w in interests.items() if t not in last}
            if not available:
                available = dict(interests)
        topics = list(available.keys())
        weights = [available[t] for t in topics]
        chosen = random.choices(topics, weights=weights, k=1)[0]
        self._used_topics.add(chosen)
        return chosen

    def _generate_with_context(self, prompt: str) -> list[str]:
        """用 Gemini 3.1 Pro 生成消息。"""
        response = self._client.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.9,
                max_output_tokens=2048,
            ),
        )
        raw = response.text or ""
        truncated = (
            response.candidates
            and response.candidates[0].finish_reason
            and response.candidates[0].finish_reason.name == "MAX_TOKENS"
        )
        messages = _split_reply(raw, truncated=truncated)
        return [m for m in messages if m]

    def generate(self, topic: str | None = None, recent_context: str = "") -> list[str]:
        """搜索热点 + 生成主动消息。会参考当前对话内容决定如何切入。"""
        if topic is None:
            topic = self.pick_topic()
        if topic is None:
            return []

        name = self._persona.name
        today = date.today().strftime("%Y年%m月%d日")
        hints = _TOPIC_SEARCH_HINTS.get(topic, [f"{topic} 最新"])
        search_hint = random.choice(hints)

        # Brave Search
        results = _brave_search(f"{search_hint}", count=5)

        if not results:
            return []

        # 把搜索结果拼成上下文
        news_context = f"今天是{today}。以下是搜索到的最新「{topic}」相关新闻：\n\n"
        for i, r in enumerate(results, 1):
            news_context += f"{i}. {r['title']}\n   {r['description']}\n\n"

        # 如果有当前对话上下文，让模型判断是否适合插入
        chat_context = ""
        if recent_context:
            chat_context = (
                f"你们刚刚在聊的内容：\n{recent_context}\n\n"
                f"注意：如果你们正在聊比较重要或严肃的话题（比如身体健康、感情问题、考试压力等），"
                f"不要突然换话题，而是继续关心对方。"
                f"只有在对话已经聊得差不多了、或者话题比较轻松时，才自然地分享新闻。\n\n"
            )

        prompt = (
            f"{chat_context}"
            f"{news_context}"
            f"从上面的新闻中选一个最有意思的，用{name}的语气主动跟对方分享。\n"
            f"如果你们刚才的话题还没聊完或比较重要，就先不分享新闻，而是接着之前的话题继续聊。\n\n"
            f"要求：\n"
            f"- 像是随手在聊天窗口打字，不要像新闻播报\n"
            f"- 加上自己的评价/吐槽\n"
            f"- 用 {_MSG_SEPARATOR} 分隔多条消息\n"
            f"- 1-3 条短消息就够了\n"
        )

        msgs = self._generate_with_context(prompt)
        if msgs:
            self._last_proactive = msgs
            self._followup_count = 0
            self._proactive_count += 1
        return msgs

    def should_send_proactive(self) -> bool:
        return self._proactive_count < self._max_proactive_per_silence

    def generate_followup(self, recent_context: str = "") -> list[str]:
        """对方没回复时的行为，根据 chase_ratio 决定。"""

        if self._chase_ratio < 0.05:
            self._last_proactive = []
            self._followup_count = 0
            return self.generate(recent_context=recent_context)

        if self._followup_count >= max(1, round(self._chase_ratio * 10)):
            self._last_proactive = []
            self._followup_count = 0
            return self.generate(recent_context=recent_context)

        name = self._persona.name
        last_msg = _MSG_SEPARATOR.join(self._last_proactive)

        prompt = (
            f"你刚刚跟对方说了：「{last_msg}」\n"
            f"但对方没有回复。你想追问一下。\n\n"
            f"用{name}的语气，自然地追一句，1 条短消息就行。\n"
        )

        msgs = self._generate_with_context(prompt)
        if msgs:
            self._followup_count += 1
            self._proactive_count += 1
        return msgs

    def on_user_replied(self):
        self._followup_count = 0
        self._last_proactive = []
        self._proactive_count = 0
