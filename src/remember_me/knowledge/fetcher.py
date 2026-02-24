"""知识库抓取器 — 每日根据 persona 兴趣搜索文章和图片。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types

from remember_me.engine.brave import brave_image_search, brave_search
from remember_me.engine.topic_starter import _TOPIC_SEARCH_HINTS
from remember_me.knowledge.store import KnowledgeItem

from remember_me.models import MODEL_LIGHT

_SUMMARY_MODEL = MODEL_LIGHT
_TOPIC_CONCURRENCY = 3


def _extract_article(url: str) -> str | None:
    """用 trafilatura 提取文章全文，截断到 3000 字。"""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text:
            return text[:3000]
        return None
    except Exception as e:
        logger.debug("文章提取失败 %s: %s", url, e)
        return None


def _download_thumbnail(url: str, save_dir: Path, name_hint: str) -> str | None:
    """下载缩略图，返回本地路径。"""
    if not url:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    elif ".gif" in url.lower():
        ext = ".gif"
    filename = hashlib.md5(name_hint.encode()).hexdigest()[:12] + ext
    save_path = save_dir / filename
    if save_path.exists():
        return str(save_path)
    try:
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return str(save_path)
    except Exception as e:
        logger.debug("缩略图下载失败: %s", e)
        return None


def _summarize_articles(
    client: genai.Client,
    articles: list[dict],
    persona_name: str,
) -> list[str]:
    """批量摘要文章，返回口语化摘要列表。"""
    if not articles:
        return []

    parts = []
    for i, a in enumerate(articles, 1):
        text = a["text"][:1500]  # 每篇截断，避免 prompt 过长
        parts.append(f"【文章{i}】标题：{a['title']}\n话题：{a['topic']}\n内容：{text}")

    prompt = (
        f"你是 {persona_name} 的助手。把下面每篇文章各用一两句话概括。\n"
        f"要求：像 {persona_name} 会怎么跟朋友提起这件事那样，口语化、有态度。\n"
        f"每篇一行，用序号对应。不要加多余解释。\n\n"
        + "\n\n".join(parts)
    )

    try:
        response = client.models.generate_content(
            model=_SUMMARY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        raw = response.text or ""
        # 解析序号格式的摘要
        summaries = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # 去掉序号前缀（"1. " / "1、" / "【文章1】" 等）
            import re
            line = re.sub(r"^[\d]+[.、）\]]\s*", "", line)
            line = re.sub(r"^【文章\d+】\s*", "", line)
            if line:
                summaries.append(line)
        return summaries
    except Exception as e:
        logger.warning("摘要生成失败: %s", e)
        return []


class KnowledgeFetcher:
    """根据 persona 兴趣每日抓取知识。"""

    def __init__(self, persona_name: str, topic_interests: dict[str, int],
                 client: genai.Client, images_dir: Path):
        self._name = persona_name
        self._topics = topic_interests
        self._client = client
        self._images_dir = images_dir

    def _select_topics(self, count: int = 4) -> list[str]:
        """加权随机选择话题。"""
        if not self._topics:
            return []
        topics = list(self._topics.keys())
        weights = [self._topics[t] for t in topics]
        # 不重复选
        selected = []
        remaining = list(zip(topics, weights))
        for _ in range(min(count, len(remaining))):
            if not remaining:
                break
            t_list, w_list = zip(*remaining)
            chosen = random.choices(t_list, weights=w_list, k=1)[0]
            selected.append(chosen)
            remaining = [(t, w) for t, w in remaining if t != chosen]
        return selected

    def fetch_daily(self) -> list[KnowledgeItem]:
        """执行每日抓取，返回 KnowledgeItem 列表。"""
        selected = self._select_topics(count=4)
        if not selected:
            logger.info("无兴趣话题，跳过知识库更新")
            return []

        logger.info("知识库更新: 选中话题 %s", selected)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._fetch_daily_async(selected))
        # fetch_daily 通常在后台线程执行（无运行中事件循环）；若在 loop 内调用则降级串行。
        logger.warning("fetch_daily 在运行中的事件循环内被调用，降级为串行抓取")
        all_items: list[KnowledgeItem] = []
        for topic in selected:
            try:
                all_items.extend(self._fetch_topic(topic))
            except Exception as e:
                logger.warning("话题 %s 抓取失败: %s", topic, e)
        return all_items

    async def _fetch_daily_async(self, topics: list[str]) -> list[KnowledgeItem]:
        sem = asyncio.Semaphore(_TOPIC_CONCURRENCY)

        async def _run_topic(topic: str):
            async with sem:
                return await asyncio.to_thread(self._fetch_topic, topic)

        tasks = [_run_topic(topic) for topic in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_items: list[KnowledgeItem] = []
        for topic, result in zip(topics, results):
            if isinstance(result, Exception):
                logger.warning("话题 %s 抓取失败: %s", topic, result)
                continue
            all_items.extend(result)
        return all_items

    def _fetch_topic(self, topic: str) -> list[KnowledgeItem]:
        """抓取单个话题的文章和图片。"""
        hints = _TOPIC_SEARCH_HINTS.get(topic, [f"{topic} 最新"])
        search_hint = random.choice(hints)

        # Web 搜索
        results = brave_search(search_hint, count=5)
        items = []
        articles = []  # [{title, text, topic, url}]
        now = datetime.now().isoformat()

        # 提取 top-2 文章全文
        extracted_count = 0
        for r in results:
            if extracted_count >= 2:
                break
            url = r.get("url", "")
            if not url:
                continue

            full_text = _extract_article(url)
            time.sleep(1)  # 抓取间隔

            if not full_text or len(full_text) < 50:
                # 提取失败，用搜索描述作为 fallback
                full_text = r.get("description", "")
                if not full_text:
                    continue

            articles.append({
                "title": r["title"],
                "text": full_text,
                "topic": topic,
                "url": url,
            })
            extracted_count += 1

        # 批量摘要
        if articles and extracted_count > 0:
            to_summarize = articles[:extracted_count]
            summaries = _summarize_articles(self._client, to_summarize, self._name)

            for idx, article in enumerate(to_summarize):
                summary = summaries[idx] if idx < len(summaries) else article["text"][:120]
                items.append(KnowledgeItem(
                    topic=article["topic"],
                    title=article["title"],
                    url=article["url"],
                    summary=summary,
                    full_text=article["text"],
                    image_path=None,
                    fetched_at=now,
                ))

        # 图片搜索（1 张缩略图）
        try:
            img_results = brave_image_search(search_hint, count=3)
            if img_results:
                img = img_results[0]
                thumb_url = img.get("thumbnail_url", "")
                if thumb_url:
                    local_path = _download_thumbnail(
                        thumb_url, self._images_dir,
                        f"{topic}_{now}",
                    )
                    # 把图片路径附加到最后一个 item
                    if local_path and items:
                        items[-1].image_path = local_path
        except Exception as e:
            logger.debug("图片搜索失败: %s", e)

        return items
