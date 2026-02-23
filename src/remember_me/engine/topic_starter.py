"""主动话题发起器 - 联网搜索热点并以人格风格发起话题。"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.engine.chat import _MSG_SEPARATOR, _build_system_prompt, _split_reply

_TOPIC_SEARCH_HINTS = {
    "手游/王者荣耀": "王者荣耀最新更新 王者荣耀新赛季 手游新闻",
    "游戏(泛)": "手游新闻 王者荣耀 热门手游",
    "电竞赛事": "电竞比赛 LPL赛程 英雄联盟赛事",
    "音乐": "最新音乐 新歌发布 音乐节 演唱会",
    "美食": "美食热搜 网红美食 新开的餐厅",
    "影视/追剧": "最新电影 热播电视剧 新番动漫",
    "学习/校园": "大学生 校园热点 大学生活",
    "吐槽/搞笑": "今日热搜 搞笑新闻 离谱事件",
    "日常生活": "生活热搜 今日热点",
}


class TopicStarter:
    """基于 persona 的兴趣话题，联网搜索热点并生成主动消息。"""

    def __init__(self, persona: Persona, client: genai.Client):
        self._persona = persona
        self._client = client
        self._system_prompt = _build_system_prompt(persona)
        self._used_topics: list[str] = []
        self._cached: list[str] | None = None
        self._last_proactive: list[str] = []
        self._followup_count = 0
        self._proactive_count = 0  # 本轮沉默中已发的主动消息数
        # 从聊天记录分析：这个人沉默后追发的概率
        self._chase_ratio = getattr(persona, "chase_ratio", 0.0)
        # 根据追发概率决定最多主动发几次（沉默期间）
        if self._chase_ratio < 0.05:
            self._max_proactive_per_silence = 1  # 几乎不追发，最多1条
        elif self._chase_ratio < 0.2:
            self._max_proactive_per_silence = 2
        else:
            self._max_proactive_per_silence = 3

    def prefetch(self):
        """后台预生成一条消息，缓存起来。"""
        self._cached = self.generate()

    def pop_cached(self) -> list[str] | None:
        """取出缓存的消息（如果有），取后清空。"""
        cached = self._cached
        self._cached = None
        return cached

    def pick_topic(self) -> str | None:
        """从 topic_interests 中加权随机选择一个话题。"""
        interests = self._persona.topic_interests
        if not interests:
            return None

        available = {t: w for t, w in interests.items() if t not in self._used_topics}
        if not available:
            self._used_topics.clear()
            available = dict(interests)

        topics = list(available.keys())
        weights = [available[t] for t in topics]
        chosen = random.choices(topics, weights=weights, k=1)[0]
        self._used_topics.append(chosen)
        return chosen

    def _call_with_search(self, prompt: str) -> list[str]:
        """调用 Gemini + Google Search grounding 生成消息。"""
        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.9,
                max_output_tokens=512,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        raw = response.text or ""
        messages = _split_reply(raw)
        return [m for m in messages if m]

    def _call_without_search(self, prompt: str) -> list[str]:
        """不联网搜索，直接生成（用于追问/跟进）。"""
        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.9,
                max_output_tokens=256,
            ),
        )
        raw = response.text or ""
        messages = _split_reply(raw)
        return [m for m in messages if m]

    def generate(self, topic: str | None = None) -> list[str]:
        """生成一条主动消息（联网搜索热点）。"""
        if topic is None:
            topic = self.pick_topic()
        if topic is None:
            return []

        name = self._persona.name
        search_hint = _TOPIC_SEARCH_HINTS.get(topic, f"{topic} 最新热点")

        prompt = (
            f"你在网上看到了一个跟「{topic}」相关的有意思的事情，想主动跟对方分享。\n\n"
            f"请搜索「{search_hint}」找一个最近的真实热点，"
            f"然后用{name}的语气主动跟对方说。\n\n"
            f"要求：\n"
            f"- 必须基于搜索到的真实新闻/事件，不要编造\n"
            f"- 像是随手在聊天窗口打字分享，不要像新闻播报\n"
            f"- 可以加上自己的评价/吐槽\n"
            f"- 用 {_MSG_SEPARATOR} 分隔多条消息\n"
            f"- 1-3 条短消息就够了\n"
        )
        msgs = self._call_with_search(prompt)
        if msgs:
            self._last_proactive = msgs
            self._followup_count = 0
        return msgs

    def should_send_proactive(self) -> bool:
        """根据这个人的真实行为习惯，判断是否应该主动发消息。"""
        return self._proactive_count < self._max_proactive_per_silence

    def generate_followup(self) -> list[str]:
        """对方没回复时的行为。根据 chase_ratio 决定：
        - chase_ratio 低（<5%）：不追问，直接换新话题（或不发）
        - chase_ratio 中等：偶尔追一句
        - chase_ratio 高：多次追问
        """
        self._proactive_count += 1

        # 这个人几乎不追发 → 不追问，等很久后发个新话题
        if self._chase_ratio < 0.05:
            self._last_proactive = []
            self._followup_count = 0
            return self.generate()

        # 追问超限 → 换新话题
        if self._followup_count >= max(1, round(self._chase_ratio * 10)):
            self._last_proactive = []
            self._followup_count = 0
            return self.generate()

        name = self._persona.name
        last_msg = _MSG_SEPARATOR.join(self._last_proactive)

        prompt = (
            f"你刚刚跟对方说了：「{last_msg}」\n"
            f"但对方没有回复。你想追问一下。\n\n"
            f"用{name}的语气，自然地追一句，1 条短消息就行。\n"
        )

        msgs = self._call_without_search(prompt)
        if msgs:
            self._followup_count += 1
        return msgs

    def on_user_replied(self):
        """用户回复了，重置追问状态。"""
        self._followup_count = 0
        self._last_proactive = []
        self._proactive_count = 0
