"""RememberMe CLI - 通过聊天记录复活记忆中的人。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from remember_me.analyzer.persona import Persona, analyze
from remember_me.engine.chat import ChatEngine
from remember_me.importers import json_parser, plain_text, wechat
from remember_me.memory.store import MemoryStore

console = Console()

DATA_DIR = Path("data")
PROFILES_DIR = DATA_DIR / "profiles"
CHROMA_DIR = DATA_DIR / "chroma"
HISTORY_DIR = DATA_DIR / "history"

IMPORTERS = {
    "text": plain_text,
    "json": json_parser,
    "wechat": wechat,
}


@click.group()
def cli():
    """RememberMe - 通过聊天记录复活记忆中的人"""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt", type=click.Choice(list(IMPORTERS.keys())), default="text", help="聊天记录格式")
@click.option("--target", "-t", required=True, help="目标人物的名字（聊天记录中的显示名）")
@click.option("--user", "-u", default=None, help="你自己的名字（可选，自动检测）")
def import_chat(file: str, fmt: str, target: str, user: str | None):
    """导入聊天记录并生成人格档案。"""
    console.print(f"\n  正在导入 [bold cyan]{target}[/] 的聊天记录...\n")

    # 解析聊天记录
    importer = IMPORTERS[fmt]
    history = importer.parse(file, target_name=target, user_name=user)

    if not history.messages:
        console.print("  [red]未找到任何消息，请检查文件格式和目标名字是否正确。[/]")
        sys.exit(1)

    target_count = len(history.target_messages)
    console.print(f"  [green]✓[/] 导入 {len(history.messages)} 条消息（{target}：{target_count} 条）")

    # 保存完整聊天记录
    history_path = HISTORY_DIR / f"{target}.json"
    history.save(history_path)
    console.print(f"  [green]✓[/] 聊天记录已保存: {history_path}")

    # 分析人格
    persona = analyze(history)
    profile_path = PROFILES_DIR / f"{target}.json"
    persona.save(profile_path)
    console.print(f"  [green]✓[/] 人格分析完成")
    console.print(f"    说话风格: {persona.style_summary}")

    # 建立记忆索引
    store = MemoryStore(CHROMA_DIR / target, persona_name=target)
    store.index_history(history)
    console.print(f"  [green]✓[/] 记忆索引已建立")

    console.print(f"\n  人格档案已保存: [bold]{profile_path}[/]")
    console.print(f"  现在可以运行 [bold cyan]remember-me chat {target}[/] 开始对话\n")


@cli.command()
@click.argument("name")
@click.option("--api-key", envvar="GEMINI_API_KEY", help="Gemini API Key（或设置 GEMINI_API_KEY 环境变量）")
def chat(name: str, api_key: str | None):
    """与记忆中的人对话。"""
    profile_path = PROFILES_DIR / f"{name}.json"
    if not profile_path.exists():
        console.print(f"\n  [red]找不到 {name} 的人格档案。[/]")
        console.print(f"  请先运行 [bold]remember-me import-chat --target {name} <聊天记录文件>[/]\n")
        sys.exit(1)

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        console.print("\n  [red]请设置 GEMINI_API_KEY 环境变量或使用 --api-key 参数。[/]\n")
        sys.exit(1)

    persona = Persona.load(profile_path)

    # 加载记忆
    memory: MemoryStore | None = None
    chroma_path = CHROMA_DIR / name
    if chroma_path.exists():
        memory = MemoryStore(chroma_path, persona_name=name)

    engine = ChatEngine(persona=persona, memory=memory, api_key=api_key)

    # 加载历史消息数
    from remember_me.importers.base import ChatHistory
    history_path = HISTORY_DIR / f"{name}.json"
    total_msg_count = persona.total_messages
    if history_path.exists():
        try:
            hist_data = json.loads(history_path.read_text(encoding="utf-8"))
            total_msg_count = hist_data.get("total_messages", total_msg_count)
        except Exception:
            pass

    # 欢迎界面
    title = Text(f" {name} ", style="bold white on blue")
    console.print()
    console.print(Panel(
        f"正在连接 [bold]{name}[/] 的记忆...\n"
        f"共有 {total_msg_count} 条历史消息\n\n"
        f"输入消息开始对话，输入 [bold]quit[/] 退出",
        title=title,
        border_style="blue",
    ))
    console.print()

    while True:
        try:
            user_input = console.input("[bold green]你: [/]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print(f"\n  [dim]再见，{name} 会一直在这里等你。[/]\n")
            break

        # 多条消息回复（模拟连发）
        import random
        import time as _time

        try:
            replies = engine.send_multi(user_input)
            for i, reply in enumerate(replies):
                console.print(f"[bold cyan]{name}[/]: {reply}", highlight=False)
                if i < len(replies) - 1:
                    _time.sleep(0.4 + random.random() * 0.8)
        except Exception as e:
            console.print(f"  [red]出错了: {e}[/]")

        console.print()


@cli.command()
@click.option("--cookie", envvar="NETEASE_COOKIE", default=None,
              help="MUSIC_U cookie 值（或设置 NETEASE_COOKIE 环境变量）")
def import_netease(cookie: str | None):
    """从网易云音乐导入私信聊天记录。"""
    from remember_me.importers.netease import fetch_and_parse
    from remember_me.importers.netease_api import NeteaseAPI

    api = NeteaseAPI()

    # ── 获取 Cookie ──
    if not cookie:
        cookie = os.environ.get("NETEASE_COOKIE")

    if not cookie:
        console.print()
        console.print("  需要你的网易云音乐 [bold]MUSIC_U[/] cookie 来登录。")
        console.print()
        console.print("  [bold]获取方法:[/]")
        console.print("    1. 用浏览器打开 [cyan]https://music.163.com[/] 并登录")
        console.print("    2. 按 F12 打开开发者工具")
        console.print("    3. 切换到 Application（应用）标签页")
        console.print("    4. 左侧 Cookies → https://music.163.com")
        console.print("    5. 找到 [bold]MUSIC_U[/] 那一行，复制它的值")
        console.print()
        try:
            cookie = console.input("  请粘贴 MUSIC_U 的值: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n")
            sys.exit(0)

    if not cookie:
        console.print("  [red]未提供 cookie，退出。[/]\n")
        sys.exit(1)

    # ── 用 cookie 登录 ──
    console.print("\n  正在验证登录状态...", end="")
    api.login_with_cookie(cookie)
    profile = api.login_status()

    if not profile:
        console.print(" [red]失败[/]")
        console.print("  [red]Cookie 无效或已过期，请重新获取。[/]\n")
        sys.exit(1)

    console.print(f" [green]✓[/]")
    console.print(f"  [green]✓[/] 登录成功: [bold]{api.my_nickname}[/]\n")

    # ── 获取私信联系人 ──
    console.print("  正在获取私信列表...\n")
    conversations = api.get_private_msg_users(limit=100)
    if not conversations:
        console.print("  [red]没有找到任何私信记录。[/]\n")
        sys.exit(1)

    # 展示联系人列表
    contacts = []
    for conv in conversations:
        from_user = conv.get("fromUser", {})
        to_user = conv.get("toUser", {})
        # 判断对方是谁（不是自己的那个）
        if from_user.get("userId") == api.my_uid:
            other = to_user
        else:
            other = from_user
        contacts.append({
            "uid": other.get("userId"),
            "nickname": other.get("nickname", "未知"),
            "last_msg": conv.get("lastMsg", ""),
        })

    # 去重（同一个人可能出现多次）
    seen = set()
    unique_contacts = []
    for c in contacts:
        if c["uid"] not in seen:
            seen.add(c["uid"])
            unique_contacts.append(c)
    contacts = unique_contacts

    console.print("  [bold]私信联系人:[/]\n")
    for i, c in enumerate(contacts, 1):
        console.print(f"    {i}. [cyan]{c['nickname']}[/]")
    console.print()

    # 让用户选择
    while True:
        try:
            choice = console.input("  请选择要导入的联系人编号: ")
            idx = int(choice.strip()) - 1
            if 0 <= idx < len(contacts):
                break
            console.print(f"  [red]请输入 1-{len(contacts)} 之间的数字。[/]")
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("\n")
            sys.exit(0)

    selected = contacts[idx]
    target_name = selected["nickname"]
    target_uid = selected["uid"]

    console.print(f"\n  正在拉取与 [bold cyan]{target_name}[/] 的聊天记录...\n")

    # 拉取聊天记录（带进度显示）
    def on_progress(count):
        console.print(f"\r  已获取 {count} 条消息...", end="")

    history = fetch_and_parse(
        api, target_uid, target_name,
        user_name=api.my_nickname,
        on_progress=on_progress,
    )
    console.print()

    if not history.messages:
        console.print("  [red]没有找到任何文字消息。[/]\n")
        sys.exit(1)

    # 统计图片数量
    img_count = sum(1 for m in history.messages if m.content.startswith("[图片:"))
    target_count = len(history.target_messages)
    console.print(f"  [green]✓[/] 导入 {len(history.messages)} 条消息（{target_name}：{target_count} 条）")
    if img_count:
        console.print(f"  [green]✓[/] 下载了 {img_count} 张图片到 data/images/{target_name}/")

    # 保存完整聊天记录
    history_path = HISTORY_DIR / f"{target_name}.json"
    history.save(history_path)
    console.print(f"  [green]✓[/] 聊天记录已保存: {history_path}")

    # 分析人格
    persona = analyze(history)
    profile_path = PROFILES_DIR / f"{target_name}.json"
    persona.save(profile_path)
    console.print(f"  [green]✓[/] 人格分析完成")
    console.print(f"    说话风格: {persona.style_summary}")

    # 建立记忆索引
    store = MemoryStore(CHROMA_DIR / target_name, persona_name=target_name)
    store.index_history(history)
    console.print(f"  [green]✓[/] 记忆索引已建立")

    console.print(f"\n  人格档案已保存: [bold]{profile_path}[/]")
    console.print(f"  现在可以运行 [bold cyan]remember-me chat {target_name}[/] 开始对话\n")


@cli.command()
def list_personas():
    """列出所有已创建的人格档案。"""
    if not PROFILES_DIR.exists():
        console.print("\n  [dim]还没有任何人格档案。[/]\n")
        return

    profiles = list(PROFILES_DIR.glob("*.json"))
    if not profiles:
        console.print("\n  [dim]还没有任何人格档案。[/]\n")
        return

    console.print("\n  [bold]已保存的人格档案:[/]\n")
    for p in profiles:
        persona = Persona.load(p)
        console.print(f"    [cyan]{persona.name}[/]  ({persona.total_messages} 条消息)")
    console.print()


if __name__ == "__main__":
    cli()
