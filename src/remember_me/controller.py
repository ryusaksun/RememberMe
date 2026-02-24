"""ChatController — 异步对话控制层，供 GUI / CLI 共用。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
CHROMA_DIR = DATA_DIR / "chroma"
HISTORY_DIR = DATA_DIR / "history"
SESSIONS_DIR = DATA_DIR / "sessions"
NOTES_DIR = DATA_DIR / "notes"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"


class ChatController:
    """异步聊天控制器，封装 ChatEngine + TopicStarter + MemoryStore。"""

    def __init__(self, persona_name: str, api_key: str | None = None):
        self._name = persona_name
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._engine = None
        self._topic_starter = None
        self._memory = None
        self._event_tracker = None
        self._on_message: Callable[[list[str], str], None] | None = None  # (msgs, msg_type)
        self._on_typing: Callable[[bool], None] | None = None
        self._running = False
        self._greeting_task: asyncio.Task | None = None
        self._proactive_task: asyncio.Task | None = None
        self._last_activity = 0.0
        self._history_start_index = 0
        self._event_extract_index = 0  # 上次事件提取时的历史位置
        self._session_loaded = False
        self._has_topics = False
        self._proactive_cooldown = 60
        self._next_proactive_at = 0.0
        self._fresh_session = True  # 新开 GUI 页面，跳过 is_conversation_ended 检查

    @property
    def persona_name(self) -> str:
        return self._name

    def _load_notes(self) -> list[str]:
        """加载手动备注。"""
        notes_path = NOTES_DIR / f"{self._name}.json"
        if not notes_path.exists():
            return []
        try:
            return json.loads(notes_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("加载备注失败: %s", e)
            return []

    @staticmethod
    def save_notes(persona_name: str, notes: list[str]):
        """保存手动备注。"""
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        notes_path = NOTES_DIR / f"{persona_name}.json"
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_notes(persona_name: str) -> list[str]:
        """加载手动备注（静态方法，供外部调用）。"""
        notes_path = NOTES_DIR / f"{persona_name}.json"
        if not notes_path.exists():
            return []
        try:
            return json.loads(notes_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    @property
    def session_loaded(self) -> bool:
        return self._session_loaded

    async def start(
        self,
        on_message: Callable[[list[str], str], None],
        on_typing: Callable[[bool], None] | None = None,
        no_greet: bool = False,
    ):
        """初始化引擎并启动主动消息循环。

        on_message(msgs, msg_type): msg_type 为 "reply" | "proactive" | "greet"
        on_typing(is_typing): 显示/隐藏输入状态
        """
        self._on_message = on_message
        self._on_typing = on_typing
        self._running = True

        # 延迟导入避免循环依赖
        from remember_me.analyzer.persona import Persona
        from remember_me.engine.chat import ChatEngine
        from remember_me.engine.topic_starter import TopicStarter
        from remember_me.memory.store import MemoryStore

        profile_path = PROFILES_DIR / f"{self._name}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"找不到 {self._name} 的人格档案")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY 未设置")

        persona = Persona.load(profile_path)

        # 加载记忆
        chroma_path = CHROMA_DIR / self._name
        if chroma_path.exists():
            self._memory = MemoryStore(chroma_path, persona_name=self._name)

        # 加载表情包库
        from remember_me.analyzer.sticker import StickerLibrary
        sticker_lib = None
        sticker_path = DATA_DIR / "stickers" / f"{self._name}.json"
        if sticker_path.exists():
            try:
                sticker_lib = StickerLibrary.load(sticker_path)
            except Exception as e:
                logger.debug("加载表情包库失败: %s", e)

        # 加载手动备注
        notes = self._load_notes()

        # 加载知识库
        from remember_me.knowledge.store import KnowledgeStore
        knowledge_store = None
        kb_dir = KNOWLEDGE_DIR / self._name
        chroma_path = CHROMA_DIR / self._name
        if kb_dir.exists():
            try:
                knowledge_store = KnowledgeStore(
                    chroma_dir=chroma_path, knowledge_dir=kb_dir,
                    persona_name=self._name,
                )
            except Exception as e:
                logger.debug("加载知识库失败: %s", e)

        self._engine = ChatEngine(
            persona=persona, memory=self._memory,
            api_key=self._api_key, sticker_lib=sticker_lib,
            notes=notes, knowledge_store=knowledge_store,
        )

        # 加载上次对话
        session_path = SESSIONS_DIR / f"{self._name}.json"
        self._session_loaded = self._engine.load_session(session_path)
        self._history_start_index = len(self._engine._history)

        self._topic_starter = TopicStarter(persona=persona, client=self._engine.client)
        self._has_topics = bool(getattr(persona, "topic_interests", None))

        # 待跟进事件追踪器
        from remember_me.engine.pending_events import PendingEventTracker
        self._event_tracker = PendingEventTracker(persona_name=self._name, data_dir=DATA_DIR)
        self._event_extract_index = len(self._engine._history)

        self._update_activity()
        self._next_proactive_at = time.time() + random.randint(20, 45)

        # 主动开场
        if not no_greet and self._has_topics:
            self._greeting_task = asyncio.create_task(self._send_greeting())

        # 启动后台主动消息循环
        self._proactive_task = asyncio.create_task(self._proactive_loop())

    async def _send_greeting(self):
        """发送主动开场消息。"""
        if self._on_typing:
            self._on_typing(True)
        try:
            ctx = self._engine.get_recent_context() if self._session_loaded else ""
            loop = asyncio.get_event_loop()
            greet_msgs = await loop.run_in_executor(
                None, lambda: self._topic_starter.generate(recent_context=ctx)
            )
            greet_msgs = [m for m in greet_msgs if m and m.strip()]
            if greet_msgs:
                self._engine.inject_proactive_message(greet_msgs)
                self._update_activity()
                self._next_proactive_at = time.time() + self._proactive_cooldown + random.randint(0, 30)
                if self._on_message:
                    self._on_message(greet_msgs, "greet")
                # 后台预缓存
                loop.run_in_executor(None, self._topic_starter.prefetch)
            else:
                logger.warning("开场消息生成为空（可能 BRAVE_API_KEY 未设置，已使用降级方案）")
        except Exception as e:
            logger.warning("开场消息生成失败: %s", e)
        finally:
            if self._on_typing:
                self._on_typing(False)

    def _update_activity(self):
        self._last_activity = time.time()

    def _get_idle_seconds(self) -> float:
        return time.time() - self._last_activity

    async def send_message(self, text: str,
                           image: tuple[bytes, str] | None = None) -> list[str]:
        """发送用户消息，返回回复列表。

        image: 可选 (bytes, mime_type) 图片。
        """
        if not self._engine:
            raise RuntimeError("引擎未初始化，请先调用 start()")

        self._topic_starter.on_user_replied()
        self._update_activity()
        self._fresh_session = False

        if self._on_typing:
            self._on_typing(True)
        try:
            loop = asyncio.get_event_loop()
            replies = await loop.run_in_executor(
                None, lambda: self._engine.send_multi(text, image=image)
            )
            self._update_activity()

            # 异步提取待跟进事件（每 6 轮检查一次，与 scratchpad 同步）
            history_len = len(self._engine._history)
            if history_len - self._event_extract_index >= 6:
                asyncio.create_task(self._extract_pending_events())

            return replies
        finally:
            if self._on_typing:
                self._on_typing(False)

    async def _extract_pending_events(self):
        """异步从近期对话中提取待跟进事件。"""
        if not self._event_tracker or not self._engine:
            return
        try:
            messages = []
            for h in self._engine._history[self._event_extract_index:]:
                if h.parts and h.parts[0].text:
                    messages.append({"role": h.role, "text": h.parts[0].text})
            self._event_extract_index = len(self._engine._history)

            if not messages:
                return

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self._event_tracker.extract_events(
                    self._engine.client, messages
                )
            )
        except Exception as e:
            logger.debug("事件提取失败: %s", e)

    async def _proactive_loop(self):
        """后台主动消息循环（替代 cli.py 中的 proactive_worker 线程）。"""
        while self._running:
            await asyncio.sleep(2)
            if not self._running:
                continue
            now = time.time()
            idle = self._get_idle_seconds()
            if idle <= 15 or now <= self._next_proactive_at:
                continue
            # 新 GUI 会话跳过"对话已结束"检查（用户主动打开页面说明想聊）
            if not self._fresh_session and self._engine.is_conversation_ended():
                continue

            try:
                if self._on_typing:
                    self._on_typing(True)

                loop = asyncio.get_event_loop()
                msgs = None

                # 优先级 1：检查待跟进事件
                if self._event_tracker:
                    due_events = self._event_tracker.get_due_events()
                    if due_events:
                        event = due_events[0]
                        logger.info("触发事件追问: %s", event.event)
                        msgs = await loop.run_in_executor(
                            None, lambda: self._topic_starter.generate_event_followup(
                                event.event, event.context, event.followup_hint
                            )
                        )
                        if msgs:
                            self._event_tracker.mark_done(event.id)

                # 优先级 2：常规主动话题
                if not msgs and self._has_topics and self._topic_starter.should_send_proactive():
                    ctx = self._engine.get_recent_context()
                    if self._topic_starter._last_proactive:
                        msgs = await loop.run_in_executor(
                            None, lambda: self._topic_starter.generate_followup(recent_context=ctx)
                        )
                    else:
                        msgs = self._topic_starter.pop_cached()
                        if not msgs:
                            msgs = await loop.run_in_executor(
                                None, lambda: self._topic_starter.generate(recent_context=ctx)
                            )

                if msgs:
                    # 再次检查：发送期间用户可能回复了
                    if self._get_idle_seconds() < 15:
                        continue
                    if not self._fresh_session and self._engine.is_conversation_ended():
                        continue
                    self._engine.inject_proactive_message(msgs)
                    self._update_activity()
                    self._fresh_session = False  # 第一条主动消息发出后恢复正常检查
                    self._next_proactive_at = now + self._proactive_cooldown + random.randint(0, 30)
                    if self._on_message:
                        self._on_message(msgs, "proactive")

                if self._has_topics and not self._topic_starter._last_proactive:
                    try:
                        await loop.run_in_executor(None, self._topic_starter.prefetch)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("主动消息生成失败: %s", e)
            finally:
                if self._on_typing:
                    self._on_typing(False)

    async def stop(self):
        """停止控制器，保存会话。"""
        self._running = False
        for task in (self._greeting_task, self._proactive_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._greeting_task = None
        self._proactive_task = None
        self._save_session()

    def _save_session(self):
        if not self._engine:
            return
        try:
            session_path = SESSIONS_DIR / f"{self._name}.json"
            self._engine.save_session(session_path)
            new_msgs = self._engine.get_new_messages(self._history_start_index)
            if new_msgs and self._memory:
                self._memory.add_messages(new_msgs)
            # 会话结束时提取待跟进事件（确保不遗漏）
            if self._event_tracker:
                remaining = []
                for h in self._engine._history[self._event_extract_index:]:
                    if h.parts and h.parts[0].text:
                        remaining.append({"role": h.role, "text": h.parts[0].text})
                if remaining:
                    try:
                        self._event_tracker.extract_events(self._engine.client, remaining)
                    except Exception as e:
                        logger.debug("会话结束事件提取失败: %s", e)
        except Exception as e:
            logger.warning("保存会话失败: %s", e)

    def get_total_messages(self) -> int:
        """获取历史消息总数。"""
        from remember_me.analyzer.persona import Persona
        profile_path = PROFILES_DIR / f"{self._name}.json"
        persona = Persona.load(profile_path)
        total = persona.total_messages
        history_path = HISTORY_DIR / f"{self._name}.json"
        if history_path.exists():
            try:
                data = json.loads(history_path.read_text(encoding="utf-8"))
                total = data.get("total_messages", total)
            except Exception:
                pass
        return total

    @staticmethod
    def list_personas() -> list[dict]:
        """列出所有人格档案。返回 [{name, total_messages}]。"""
        from remember_me.analyzer.persona import Persona
        if not PROFILES_DIR.exists():
            return []
        result = []
        for p in PROFILES_DIR.glob("*.json"):
            try:
                persona = Persona.load(p)
                result.append({
                    "name": persona.name,
                    "total_messages": persona.total_messages,
                    "style_summary": persona.style_summary,
                })
            except Exception:
                continue
        return result

    @staticmethod
    async def import_chat_file(
        file_path: str,
        fmt: str,
        target_name: str,
        user_name: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ):
        """导入聊天记录并生成人格档案 + 记忆索引。"""
        from remember_me.analyzer.persona import analyze
        from remember_me.importers import json_parser, plain_text, wechat
        from remember_me.memory.store import MemoryStore

        importers = {"text": plain_text, "json": json_parser, "wechat": wechat}
        importer = importers.get(fmt)
        if not importer:
            raise ValueError(f"不支持的格式: {fmt}")

        if on_progress:
            on_progress("正在解析聊天记录...")

        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(
            None, lambda: importer.parse(file_path, target_name=target_name, user_name=user_name)
        )

        if not history.messages:
            raise ValueError("未找到任何消息，请检查文件格式和目标名字。")

        target_count = len(history.target_messages)
        if on_progress:
            on_progress(f"导入 {len(history.messages)} 条消息（{target_name}：{target_count} 条）")

        # 保存聊天记录
        history_path = HISTORY_DIR / f"{target_name}.json"
        history.save(history_path)

        # 分析人格
        if on_progress:
            on_progress("正在分析人格特征...")
        persona = await loop.run_in_executor(None, lambda: analyze(history))
        profile_path = PROFILES_DIR / f"{target_name}.json"
        persona.save(profile_path)

        if on_progress:
            on_progress(f"人格分析完成 — {persona.style_summary[:60]}")

        # 建立记忆索引
        if on_progress:
            on_progress("正在建立记忆索引...")
        store = MemoryStore(CHROMA_DIR / target_name, persona_name=target_name)
        await loop.run_in_executor(None, lambda: store.index_history(history))

        if on_progress:
            on_progress(f"[OK] {target_name} 导入完成！共 {len(history.messages)} 条消息。")

        return {
            "name": target_name,
            "total_messages": len(history.messages),
            "target_count": target_count,
            "style_summary": persona.style_summary,
        }
