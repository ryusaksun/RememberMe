"""网易云音乐私信聊天记录解析 - 将 API 返回数据转换为 ChatHistory。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import requests

from .base import ChatHistory, ChatMessage
from .netease_api import NeteaseAPI

IMAGES_DIR = Path("data/images")


def _download_image(url: str, target_name: str, msg_id: int) -> str | None:
    """下载图片到本地，返回本地路径。"""
    if not url:
        return None
    try:
        save_dir = IMAGES_DIR / target_name
        save_dir.mkdir(parents=True, exist_ok=True)

        ext = "jpg"
        for e in (".png", ".gif", ".webp"):
            if e in url:
                ext = e[1:]
                break

        filepath = save_dir / f"{msg_id}.{ext}"
        if filepath.exists():
            return str(filepath)

        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        return str(filepath)
    except Exception:
        return None


def _parse_inner_msg(raw_msg: str | dict) -> dict | None:
    """解析 msg 字段中的嵌套 JSON。"""
    if isinstance(raw_msg, dict):
        return raw_msg
    if not raw_msg or not isinstance(raw_msg, str):
        return None
    try:
        return json.loads(raw_msg)
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_text(msg: dict, target_name: str = "") -> str | None:
    """从消息中提取文本内容。

    网易云私信的消息结构：顶层 type 为 null，
    实际内容在 msg 字段的嵌套 JSON 中，内部 type 决定消息类型：
      type=6:  文字消息  {"msg": "文字", "type": 6}
      type=16: 图片消息  {"picInfo": {"picUrl": "..."}, "type": 16}
      type=1:  歌曲分享  {"song": {...}, "type": 1}
      type=18: 表情消息
    """
    raw = msg.get("msg", "")
    payload = _parse_inner_msg(raw)

    if not payload:
        # 纯文本（没有嵌套 JSON 的情况）
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    inner_type = payload.get("type")

    # ── 文字消息 (type=6) ──
    if inner_type == 6:
        text = payload.get("msg", "").strip()
        return text if text else None

    # ── 图片消息 (type=16) ──
    if inner_type == 16:
        pic_info = payload.get("picInfo", {})
        pic_url = pic_info.get("picUrl", "")
        if pic_url:
            msg_id = msg.get("id", 0)
            local_path = _download_image(pic_url, target_name, msg_id)
            if local_path:
                return f"[图片: {local_path}]"
        return "[图片]"

    # ── 歌曲分享 (type=1) ──
    if inner_type == 1:
        song = payload.get("song", {})
        name = song.get("name", "未知歌曲")
        artists = song.get("artists", [])
        artist = artists[0].get("name", "") if artists and isinstance(artists[0], dict) else ""
        if artist:
            return f"[分享歌曲] {name} - {artist}"
        return f"[分享歌曲] {name}"

    # ── 歌单分享 (type=2) ──
    if inner_type == 2:
        playlist = payload.get("playlist", {})
        name = playlist.get("name", "未知歌单")
        return f"[分享歌单] {name}"

    # ── 专辑分享 (type=3) ──
    if inner_type == 3:
        album = payload.get("album", {})
        name = album.get("name", "未知专辑")
        return f"[分享专辑] {name}"

    # ── 表情消息 (type=18) ──
    if inner_type == 18:
        return "[表情]"

    # ── 其他类型，尝试提取文字 ──
    for key in ("msg", "text", "content", "title"):
        val = payload.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()

    return None


def fetch_and_parse(
    api: NeteaseAPI,
    uid: int,
    target_name: str,
    user_name: str | None = None,
    on_progress: callable = None,
) -> ChatHistory:
    """从网易云 API 拉取聊天记录并转换为 ChatHistory。"""
    raw_msgs = api.fetch_all_messages(uid, on_progress=on_progress)

    if not user_name:
        user_name = api.my_nickname or "我"

    messages: list[ChatMessage] = []

    for msg in raw_msgs:
        from_user = msg.get("fromUser", {})
        sender_uid = from_user.get("userId")

        content = _extract_text(msg, target_name=target_name)
        if not content:
            continue

        ts = None
        if time_ms := msg.get("time"):
            try:
                ts = datetime.fromtimestamp(time_ms / 1000)
            except (OSError, ValueError):
                pass

        is_target = sender_uid == uid

        messages.append(
            ChatMessage(
                sender=target_name if is_target else user_name,
                content=content,
                timestamp=ts,
                is_target=is_target,
            )
        )

    # API 返回的消息是从新到旧，翻转为时间正序
    messages.reverse()

    return ChatHistory(target_name=target_name, user_name=user_name, messages=messages)
