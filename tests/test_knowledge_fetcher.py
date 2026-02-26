from __future__ import annotations

from types import SimpleNamespace

import remember_me.knowledge.fetcher as fetcher_mod
from remember_me.knowledge.fetcher import KnowledgeFetcher, _should_skip_full_extract


def test_should_skip_full_extract_for_dynamic_sites() -> None:
    assert _should_skip_full_extract("https://www.bilibili.com/video/BV1xx")
    assert _should_skip_full_extract("https://www.douyin.com/video/123")
    assert not _should_skip_full_extract("https://news.qq.com/a/20260226/000001.htm")


def test_fetch_topic_uses_search_description_for_dynamic_site(monkeypatch, tmp_path) -> None:
    called = {"extract": 0}

    def _fake_search(query: str, count: int = 5):
        return [{
            "title": "B站热点",
            "description": "这是动态站点降级摘要，长度足够用于知识条目构建。" * 2,
            "url": "https://www.bilibili.com/video/BV1xx",
        }]

    def _fake_extract(url: str):
        called["extract"] += 1
        return "不应命中"

    def _fake_summarize(client, articles, persona_name: str):
        return ["口语化摘要"]

    monkeypatch.setattr(fetcher_mod, "brave_search", _fake_search)
    monkeypatch.setattr(fetcher_mod, "_extract_article", _fake_extract)
    monkeypatch.setattr(fetcher_mod, "_summarize_articles", _fake_summarize)
    monkeypatch.setattr(fetcher_mod, "brave_image_search", lambda *_args, **_kwargs: [])

    fetcher = KnowledgeFetcher(
        persona_name="小明",
        topic_interests={"吐槽/搞笑": 5},
        client=SimpleNamespace(),
        images_dir=tmp_path / "images",
    )
    items = fetcher._fetch_topic("吐槽/搞笑")

    assert items
    assert called["extract"] == 0
    assert items[0].summary == "口语化摘要"

