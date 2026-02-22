"""记忆存储 - 基于 ChromaDB 的向量检索，用于 RAG。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from remember_me.importers.base import ChatHistory


class MemoryStore:
    def __init__(self, persist_dir: str | Path, persona_name: str):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        # ChromaDB collection name 只支持 ASCII，用哈希处理中文名
        safe_name = "mem_" + hashlib.md5(persona_name.encode()).hexdigest()[:12]
        self._collection = self._client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_history(self, history: ChatHistory):
        """将聊天记录索引到向量数据库。按对话窗口分组存储。"""
        if not history.messages:
            return

        # 按窗口（每5条消息）分组
        window_size = 5
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for i in range(0, len(history.messages), window_size):
            window = history.messages[i : i + window_size]
            text = "\n".join(f"{m.sender}: {m.content}" for m in window)
            doc_id = f"window_{i}"
            documents.append(text)
            ids.append(doc_id)
            metadatas.append({"start_idx": i, "end_idx": i + len(window)})

        # ChromaDB 会自动用内置 embedding 函数
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """检索与查询最相关的历史对话片段。"""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(query_texts=[query], n_results=min(top_k, self._collection.count()))
        return results["documents"][0] if results["documents"] else []
