"""知识库存储 — ChromaDB 向量检索 + JSON 索引。"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeItem:
    topic: str              # 来源话题（如 "音乐/rap"）
    title: str              # 标题
    url: str                # 原始 URL
    summary: str            # 口语化摘要（100-200 字）
    full_text: str          # 提取的全文（截断 3000 字）
    image_path: str | None  # 本地图片路径
    fetched_at: str         # ISO datetime

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> KnowledgeItem:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class KnowledgeStore:
    """管理 persona 的每日知识库：ChromaDB 向量搜索 + JSON 元数据。"""

    def __init__(self, chroma_dir: str | Path, knowledge_dir: str | Path,
                 persona_name: str):
        """
        chroma_dir: 已有的 ChromaDB 持久化目录（与 MemoryStore 共享）
        knowledge_dir: data/knowledge/{name}/ — 存放 index.json 和 images/
        """
        self._knowledge_dir = Path(knowledge_dir)
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._knowledge_dir / "index.json"
        self._images_dir = self._knowledge_dir / "images"

        # 共享已有的 ChromaDB PersistentClient
        import chromadb
        from remember_me.memory.store import _get_embedding_function

        chroma_dir = Path(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        safe_name = "kb_" + hashlib.md5(persona_name.encode()).hexdigest()[:12]
        self._ef = _get_embedding_function()
        try:
            self._collection = self._client.get_or_create_collection(
                name=safe_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._ef,
            )
        except ValueError:
            self._client.delete_collection(name=safe_name)
            self._collection = self._client.get_or_create_collection(
                name=safe_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._ef,
            )

    def _load_index(self) -> list[dict]:
        if not self._index_path.exists():
            return []
        try:
            return json.loads(self._index_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_index(self, items: list[dict]):
        self._index_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    def add_items(self, items: list[KnowledgeItem]):
        """添加知识条目到 ChromaDB + JSON 索引。"""
        if not items:
            return

        index = self._load_index()
        existing_urls = {entry["url"] for entry in index}

        documents, ids, metadatas = [], [], []
        new_entries = []

        for item in items:
            if item.url in existing_urls:
                continue
            doc_id = "kb_" + hashlib.md5(item.url.encode()).hexdigest()[:16]
            documents.append(item.summary)
            ids.append(doc_id)
            metadatas.append({"topic": item.topic, "title": item.title, "url": item.url})
            new_entries.append(item.to_dict())
            existing_urls.add(item.url)

        if documents:
            self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            index.extend(new_entries)
            self._save_index(index)
            logger.info("知识库新增 %d 条", len(documents))

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeItem]:
        """向量搜索知识库，返回最相关的条目。"""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []

        # 用 summary 和 metadata 重建 KnowledgeItem（轻量返回）
        items = []
        index = {entry["url"]: entry for entry in self._load_index()}
        for summary, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0],
        ):
            if dist > 1.2:
                continue
            full = index.get(meta.get("url", ""))
            if full:
                items.append(KnowledgeItem.from_dict(full))
            else:
                items.append(KnowledgeItem(
                    topic=meta.get("topic", ""),
                    title=meta.get("title", ""),
                    url=meta.get("url", ""),
                    summary=summary,
                    full_text="",
                    image_path=None,
                    fetched_at="",
                ))
        return items

    def evict(self, max_age_days: int = 7, max_size_bytes: int = 15 * 1024**3):
        """清理过期条目。先按 TTL 清理，再按大小强制清理。"""
        index = self._load_index()
        if not index:
            return

        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        keep, remove_urls = [], []

        for entry in index:
            if entry.get("fetched_at", "") < cutoff:
                remove_urls.append(entry["url"])
                # 删除关联图片
                img = entry.get("image_path")
                if img:
                    p = Path(img)
                    if p.exists():
                        p.unlink(missing_ok=True)
            else:
                keep.append(entry)

        # 从 ChromaDB 删除
        if remove_urls:
            remove_ids = [
                "kb_" + hashlib.md5(url.encode()).hexdigest()[:16]
                for url in remove_urls
            ]
            try:
                self._collection.delete(ids=remove_ids)
            except Exception as e:
                logger.warning("ChromaDB 删除失败: %s", e)
            logger.info("知识库淘汰 %d 条过期条目", len(remove_urls))

        self._save_index(keep)

        # 大小检查（极端情况）
        total = self.total_size_bytes()
        if total > max_size_bytes:
            logger.warning("知识库超过 %d GB，强制清理最旧条目", max_size_bytes // (1024**3))
            keep.sort(key=lambda x: x.get("fetched_at", ""))
            while keep and self.total_size_bytes() > max_size_bytes:
                oldest = keep.pop(0)
                img = oldest.get("image_path")
                if img:
                    Path(img).unlink(missing_ok=True)
                doc_id = "kb_" + hashlib.md5(oldest["url"].encode()).hexdigest()[:16]
                try:
                    self._collection.delete(ids=[doc_id])
                except Exception:
                    pass
            self._save_index(keep)

    def total_size_bytes(self) -> int:
        """计算知识库目录总大小。"""
        total = 0
        for f in self._knowledge_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
