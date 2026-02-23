"""表情包分类与情感标注 - 从下载的图片中识别表情包并标注使用场景。"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from remember_me.importers.base import ChatHistory

# 情感关键词映射
_EMOTION_PATTERNS = {
    "搞笑": re.compile(r"(哈|笑|搞笑|笑死|好笑|离谱|绝了|鬼|神|牛|6|666|秀)"),
    "喜爱": re.compile(r"(爱|心|宝贝|想你|喜欢|可爱|好看|亲|么么|mua)"),
    "愤怒": re.compile(r"(靠|妈的|吗的|操|气|烦|傻|几把|屎|炸)"),
    "难过": re.compile(r"(呜|哭|难过|伤心|惨|寄|完蛋|废|裂开)"),
    "震惊": re.compile(r"(卧槽|我靠|天|啊|什么|啥|真的假的|不会吧)"),
    "赞同": re.compile(r"(对|是|没错|确实|真的|可以|好|行|嗯)"),
}

# 表情包判定阈值
_MAX_FILE_SIZE_KB = 100
_MAX_DIMENSION = 400


@dataclass
class Sticker:
    path: str
    size_kb: float
    width: int
    height: int
    context: str  # 发送前的文字消息
    emotion: str  # 情感标签


@dataclass
class StickerLibrary:
    name: str
    stickers: list[Sticker] = field(default_factory=list)

    def by_emotion(self, emotion: str) -> list[Sticker]:
        return [s for s in self.stickers if s.emotion == emotion]

    def random_sticker(self, emotion: str | None = None) -> Sticker | None:
        pool = self.by_emotion(emotion) if emotion else self.stickers
        if not pool:
            pool = self.stickers
        return random.choice(pool) if pool else None

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "stickers": [asdict(s) for s in self.stickers]}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> StickerLibrary:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        stickers = [Sticker(**s) for s in raw.get("stickers", [])]
        return cls(name=raw.get("name", ""), stickers=stickers)


def _classify_emotion(context: str) -> str:
    """根据上下文文本判断情感。"""
    if not context:
        return "通用"
    for emotion, pattern in _EMOTION_PATTERNS.items():
        if pattern.search(context):
            return emotion
    return "通用"


def _is_sticker(filepath: Path) -> tuple[bool, int, int]:
    """判断文件是否是表情包。返回 (是否表情包, 宽, 高)。"""
    try:
        size_kb = filepath.stat().st_size / 1024
        if size_kb > _MAX_FILE_SIZE_KB:
            return False, 0, 0

        with Image.open(filepath) as img:
            w, h = img.size
            if w <= _MAX_DIMENSION and h <= _MAX_DIMENSION:
                return True, w, h
            return False, w, h
    except Exception:
        return False, 0, 0


def classify_stickers(
    images_dir: str | Path,
    history: ChatHistory,
    target_name: str,
) -> StickerLibrary:
    """从图片目录中分类表情包并标注情感。"""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return StickerLibrary(name=target_name)

    # 建立消息 ID → 上下文映射（图片消息前的文字消息）
    context_map: dict[str, str] = {}
    for i, m in enumerate(history.messages):
        if m.content.startswith("[图片:"):
            # 往前找最近的文字消息（同一个人发的）
            ctx = ""
            for j in range(i - 1, max(i - 5, -1), -1):
                prev = history.messages[j]
                if prev.is_target == m.is_target and not prev.content.startswith("["):
                    ctx = prev.content
                    break
            # 用文件名做 key（不依赖完整路径，CWD 变化时也能匹配）
            path_match = re.search(r"\[图片: (.+?)\]", m.content)
            if path_match:
                filename = Path(path_match.group(1)).name
                context_map[filename] = ctx

    stickers: list[Sticker] = []
    for filepath in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.gif")) + sorted(images_dir.glob("*.webp")):
        is_stk, w, h = _is_sticker(filepath)
        if not is_stk:
            continue

        path_str = str(filepath)
        context = context_map.get(filepath.name, "")
        emotion = _classify_emotion(context)

        stickers.append(Sticker(
            path=path_str,
            size_kb=round(filepath.stat().st_size / 1024, 1),
            width=w,
            height=h,
            context=context,
            emotion=emotion,
        ))

    return StickerLibrary(name=target_name, stickers=stickers)
