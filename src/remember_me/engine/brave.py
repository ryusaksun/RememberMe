"""Brave Search API 共享模块 — Web 搜索 + Image 搜索。"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)


def brave_search(query: str, count: int = 5) -> list[dict]:
    """调用 Brave Web Search API。返回 [{title, description, url}]。"""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        logger.warning("BRAVE_API_KEY 未设置，跳过搜索")
        return []

    try:
        resp = requests.get(
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
    except requests.Timeout:
        logger.warning("Brave Search 超时: %s", query)
        return []
    except requests.HTTPError as e:
        logger.warning("Brave Search API 错误 %s: %s", e.response.status_code, query)
        return []
    except Exception as e:
        logger.warning("Brave Search 异常: %s", e)
        return []


def brave_image_search(query: str, count: int = 3) -> list[dict]:
    """调用 Brave Image Search API。返回 [{title, thumbnail_url, source_url}]。"""
    api_key = os.environ.get("BRAVE_API_KEY", "")
    if not api_key:
        return []

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/images/search",
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": count, "search_lang": "zh-hans"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
            thumbnail = item.get("thumbnail", {})
            results.append({
                "title": item.get("title", ""),
                "thumbnail_url": thumbnail.get("src", ""),
                "source_url": item.get("url", ""),
            })
        return results
    except Exception as e:
        logger.warning("Brave Image Search 异常: %s", e)
        return []
