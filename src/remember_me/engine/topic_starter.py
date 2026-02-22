"""主动话题发起器 - 联网搜索热点并以人格风格发起话题。"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

from remember_me.analyzer.persona import Persona
from remember_me.engine.chat import _MSG_SEPARATOR, _build_system_prompt, _split_reply

_TOPIC_SEARCH_HINTS = {
    "游戏": "最新游戏新闻 新游戏发布 游戏更新",
    "音乐": "最新音乐 新歌发布 音乐节 演唱会",
    "美食": "美食热搜 新开的餐厅 美食趋势",
    "影视": "最新电影 热播电视剧 新番动漫",
    "学习": "大学生 校园热点 考试",
    "吐槽": "今日热搜 搞笑新闻 离谱事件",
}


class TopicStarter:
    """基于 persona 的兴趣话题，联网搜索热点并生成主动消息。"""

    def __init__(self, persona: Persona, client: genai.Client):
        self._persona = persona
        self._client = client
        self._system_prompt = _build_system_prompt(persona)
        self._used_topics: list[str] = []

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
            model="gemini-3.1-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                temperature=0.9,
                max_output_tokens=256,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        raw = response.text or ""
        messages = _split_reply(raw)
        return messages if messages else [raw]

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
        return self._call_with_search(prompt)

    def generate_cold_topic(self) -> list[str]:
        """冷场时生成一条转换话题的消息。"""
        topic = self.pick_topic()
        if topic is None:
            return []

        name = self._persona.name
        search_hint = _TOPIC_SEARCH_HINTS.get(topic, f"{topic} 最新")

        prompt = (
            f"你们聊天聊到没什么好说的了，你想换个话题。\n"
            f"你刚好看到了一个跟「{topic}」相关的事情，想分享给对方。\n\n"
            f"请搜索「{search_hint}」找一个最近的热点，"
            f"然后自然地转换话题。\n\n"
            f"要求：\n"
            f"- 基于搜索到的真实内容\n"
            f"- 用{name}的语气，像是突然想到的\n"
            f"- 可以用「哦对了」「诶你看到xxx了吗」这种转折\n"
            f"- 用 {_MSG_SEPARATOR} 分隔多条消息\n"
            f"- 1-3 条短消息\n"
        )
        return self._call_with_search(prompt)
