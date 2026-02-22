"""直接导入指定联系人的聊天记录（非交互式）。"""
import os
import sys
from pathlib import Path

from rich.console import Console

from remember_me.importers.netease_api import NeteaseAPI
from remember_me.importers.netease import fetch_and_parse
from remember_me.analyzer.persona import analyze
from remember_me.memory.store import MemoryStore

console = Console()

cookie = os.environ.get("NETEASE_COOKIE", "")
if not cookie:
    console.print("[red]请设置 NETEASE_COOKIE 环境变量[/]")
    sys.exit(1)

api = NeteaseAPI()
api.login_with_cookie(cookie)
profile = api.login_status()
if not profile:
    console.print("[red]登录失败[/]")
    sys.exit(1)
console.print(f"  [green]✓[/] 登录: {api.my_nickname}")

target_name = "阴暗扭曲爬行_-_-"
target_uid = 1545867891

console.print(f"\n  正在拉取与 [bold cyan]{target_name}[/] 的聊天记录...\n")


def on_progress(count):
    console.print(f"\r  已获取 {count} 条消息...", end="")


history = fetch_and_parse(api, target_uid, target_name, user_name=api.my_nickname, on_progress=on_progress)
console.print()

img_count = sum(1 for m in history.messages if m.content.startswith("[图片:"))
target_count = len(history.target_messages)
console.print(f"  [green]✓[/] 导入 {len(history.messages)} 条消息（{target_name}：{target_count} 条）")
if img_count:
    console.print(f"  [green]✓[/] 下载了 {img_count} 张图片到 data/images/{target_name}/")

# 保存聊天记录
history_path = Path("data/history") / f"{target_name}.json"
history.save(history_path)
console.print(f"  [green]✓[/] 聊天记录已保存: {history_path}")

# 分析人格
persona = analyze(history)
profile_path = Path("data/profiles") / f"{target_name}.json"
persona.save(profile_path)
console.print(f"  [green]✓[/] 人格分析完成")
console.print(f"    说话风格: {persona.style_summary}")

# 建立记忆索引
store = MemoryStore(Path("data/chroma") / target_name, persona_name=target_name)
store.index_history(history)
console.print(f"  [green]✓[/] 记忆索引已建立")

console.print(f"\n  现在可以运行 [bold cyan]remember-me chat {target_name}[/] 开始对话\n")
