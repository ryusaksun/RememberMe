"""记忆存储 - 基于 ChromaDB 的向量检索，用于 RAG。"""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path

# 屏蔽 sentence-transformers / HF 加载时的冗余输出
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import chromadb
from chromadb.utils import embedding_functions

_ef_instance: embedding_functions.SentenceTransformerEmbeddingFunction | None = None


def _get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """懒加载嵌入模型，屏蔽加载时的冗余输出。"""
    global _ef_instance
    if _ef_instance is not None:
        return _ef_instance
    import io
    import sys
    # 模型加载时 HF/safetensors 直接 print，临时重定向 stdout/stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        _ef_instance = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_BGE_MODEL,
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
    return _ef_instance

from remember_me.importers.base import ChatHistory

_BGE_MODEL = "BAAI/bge-small-zh-v1.5"


class MemoryStore:
    def __init__(self, persist_dir: str | Path, persona_name: str):
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        # ChromaDB collection name 只支持 ASCII，用哈希处理中文名
        self._safe_name = "mem_" + hashlib.md5(persona_name.encode()).hexdigest()[:12]
        self._ef = _get_embedding_function()
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._safe_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._ef,
            )
        except ValueError:
            # 嵌入模型变更（如从 default 迁移到 bge），删除旧 collection 重建
            self._client.delete_collection(name=self._safe_name)
            self._collection = self._client.get_or_create_collection(
                name=self._safe_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._ef,
            )

    def index_history(self, history: ChatHistory):
        """将聊天记录索引到向量数据库。按对话窗口分组存储。"""
        if not history.messages:
            return

        # 先准备数据，再清空+写入（减少 delete 和 insert 之间的窗口）
        window_size = 5
        step = 3  # window_size - overlap(2)
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for i in range(0, len(history.messages), step):
            window = history.messages[i : i + window_size]
            if not window:
                break
            text = "\n".join(f"{m.sender}: {m.content}" for m in window)
            doc_id = f"window_{i}"
            documents.append(text)
            ids.append(doc_id)
            metadatas.append({"start_idx": i, "end_idx": i + len(window)})

        if not documents:
            return

        # 数据已在内存中准备好，删除旧 collection 后立即写入
        # 若 upsert 失败，下次 import-chat 会重建
        self._client.delete_collection(name=self._safe_name)
        self._collection = self._client.get_or_create_collection(
            name=self._safe_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def add_messages(self, messages: list[dict]):
        """将新对话消息追加到向量库（运行时产生的对话）。"""
        if not messages:
            return

        import uuid
        window_size = 5
        step = 3  # window_size - overlap(2)

        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        uid = uuid.uuid4().hex[:12]
        for i in range(0, len(messages), step):
            window = messages[i : i + window_size]
            if not window:
                break
            role_map = {"user": "对方", "model": "你"}
            text = "\n".join(f"{role_map.get(m['role'], m['role'])}: {m['text']}" for m in window)
            doc_id = f"session_{uid}_{i}"
            documents.append(text)
            ids.append(doc_id)
            metadatas.append({"start_idx": i, "end_idx": i + len(window)})

        if documents:
            self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        """检索与查询最相关的历史对话片段。返回 [(文档, cosine距离)]，距离越小越相似。"""
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return list(zip(results["documents"][0], results["distances"][0]))
