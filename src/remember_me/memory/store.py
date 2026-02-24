"""记忆存储 - 基于 ChromaDB 的向量检索，用于 RAG。"""

from __future__ import annotations

import hashlib
import logging
import os
import time
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


def _recency_bonus(indexed_at: float, now_ts: float | None = None) -> float:
    """根据索引时间给距离一个轻量奖励，近期记忆优先。"""
    if indexed_at <= 0:
        return 0.0
    if now_ts is None:
        now_ts = time.time()
    age_hours = max(0.0, (now_ts - indexed_at) / 3600.0)
    # 0h 时约 0.18，24h 时约 0.09，越旧越接近 0
    return min(0.18, 0.18 / (1.0 + age_hours / 24.0))


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

    @staticmethod
    def _build_history_turns(history: ChatHistory) -> list[dict]:
        """按 turn 边界切分历史：说话人切换或同人间隔超过 5 分钟都算新 turn。"""
        turns: list[dict] = []
        current: dict | None = None

        for idx, m in enumerate(history.messages):
            text = m.content.strip()
            if not text:
                continue
            ts = m.timestamp
            role = m.sender

            start_new = False
            if current is None:
                start_new = True
            else:
                gap = None
                if current["last_ts"] and ts:
                    gap = (ts - current["last_ts"]).total_seconds()
                if role != current["role"] or (gap is not None and gap > 300):
                    start_new = True

            if start_new:
                if current:
                    turns.append(current)
                current = {
                    "role": role,
                    "texts": [text],
                    "start_idx": idx,
                    "end_idx": idx + 1,
                    "last_ts": ts,
                }
                continue

            current["texts"].append(text)
            current["end_idx"] = idx + 1
            if ts:
                current["last_ts"] = ts

        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _build_runtime_turns(messages: list[dict]) -> list[dict]:
        """按角色切分运行时 turn。"""
        role_map = {"user": "对方", "model": "你"}
        turns: list[dict] = []
        current: dict | None = None

        for idx, m in enumerate(messages):
            text = str(m.get("text", "")).strip()
            if not text:
                continue
            role = role_map.get(m.get("role"), str(m.get("role", "未知")))
            if current is None or role != current["role"]:
                if current:
                    turns.append(current)
                current = {
                    "role": role,
                    "texts": [text],
                    "start_idx": idx,
                    "end_idx": idx + 1,
                }
            else:
                current["texts"].append(text)
                current["end_idx"] = idx + 1

        if current:
            turns.append(current)
        return turns

    @staticmethod
    def _build_turn_windows(
        turns: list[dict],
        id_prefix: str,
        window_turns: int = 4,
        step_turns: int = 2,
    ) -> tuple[list[str], list[str], list[dict]]:
        """将 turn 序列打成可检索窗口，并写入时间元数据。"""
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        if not turns:
            return documents, ids, metadatas

        now_ts = time.time()
        total_turns = len(turns)
        for i in range(0, total_turns, step_turns):
            window = turns[i : i + window_turns]
            if not window:
                break
            lines = []
            for turn in window:
                lines.extend(f"{turn['role']}: {txt}" for txt in turn["texts"])
            text = "\n".join(lines).strip()
            if not text:
                continue

            last_ts = window[-1].get("last_ts")
            if last_ts is not None:
                indexed_at = last_ts.timestamp()
            else:
                # 没有原始时间戳时按顺序构造相对时间，保证“越近越优先”
                indexed_at = now_ts - max(0, (total_turns - (i + len(window))) * 90)

            doc_id = f"{id_prefix}_{i}"
            documents.append(text)
            ids.append(doc_id)
            metadatas.append({
                "start_idx": window[0]["start_idx"],
                "end_idx": window[-1]["end_idx"],
                "turn_start": i,
                "turn_end": i + len(window),
                "indexed_at": round(indexed_at, 3),
            })

        return documents, ids, metadatas

    def index_history(self, history: ChatHistory):
        """将聊天记录索引到向量数据库，按 turn 边界切片。"""
        turns = self._build_history_turns(history)
        documents, ids, metadatas = self._build_turn_windows(turns, id_prefix="turn")
        if not documents:
            return

        # 数据已在内存中准备好，删除旧 collection 后立即写入
        self._client.delete_collection(name=self._safe_name)
        self._collection = self._client.get_or_create_collection(
            name=self._safe_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def add_messages(self, messages: list[dict]):
        """将运行时新消息按 turn 追加到向量库。"""
        if not messages:
            return

        import uuid

        turns = self._build_runtime_turns(messages)
        uid = uuid.uuid4().hex[:12]
        documents, ids, metadatas = self._build_turn_windows(turns, id_prefix=f"session_{uid}")
        if documents:
            self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        """检索相关片段并按“语义距离 + 时间权重”重排。"""
        count = self._collection.count()
        if count == 0:
            return []

        candidate_k = min(max(top_k * 2, top_k), count)
        results = self._collection.query(
            query_texts=[query],
            n_results=candidate_k,
            include=["documents", "distances", "metadatas"],
        )
        docs = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        if not docs:
            return []

        now_ts = time.time()
        reranked: list[tuple[str, float, float]] = []
        for doc, dist, meta in zip(docs, dists, metas):
            indexed_at = 0.0
            if isinstance(meta, dict):
                try:
                    indexed_at = float(meta.get("indexed_at", 0.0))
                except (TypeError, ValueError):
                    indexed_at = 0.0
            adjusted = max(0.0, dist - _recency_bonus(indexed_at, now_ts))
            # 保持返回值中的距离语义仍为原始 cosine distance，
            # 仅用 adjusted 参与排序，避免上层阈值逻辑被破坏。
            reranked.append((doc, dist, adjusted))

        reranked.sort(key=lambda x: x[2])
        return [(doc, dist) for doc, dist, _ in reranked[:top_k]]
