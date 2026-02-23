"""首页 — 终端风格人格列表。"""

from __future__ import annotations

from nicegui import ui

from remember_me.controller import ChatController
from remember_me.gui.theme import (
    BG_PRIMARY, BG_SECONDARY, NEON_CYAN, NEON_GREEN, NEON_MAGENTA, TEXT_DIM, TEXT_PRIMARY,
)

TITLE_HTML = (
    '<div style="text-align: center; padding: 10px 0;">'
    '<div style="font-size: 42px; font-weight: 700; letter-spacing: 8px; '
    'color: {cyan}; '
    'text-shadow: 0 0 7px {cyan}, 0 0 20px rgba(0,255,213,0.5), '
    '0 0 40px rgba(0,255,213,0.3), 0 0 80px rgba(0,255,213,0.1);">'
    'REMEMBER ME</div>'
    '<div style="font-size: 11px; color: {dim}; letter-spacing: 5px; margin-top: 10px;">'
    'v0.1.0 // NEURAL LINK INTERFACE</div>'
    '</div>'
)


def create_home_page():
    """创建首页。"""
    ui.query("body").style(f"background: {BG_PRIMARY};")

    with ui.column().classes("w-full items-center").style(
        "min-height: 100vh; padding: 40px 20px;"
    ):
        # 标题
        ui.html(TITLE_HTML.format(cyan=NEON_CYAN, dim=TEXT_DIM))

        ui.html(
            f'<div style="color: {TEXT_DIM}; text-align: center; margin-top: 12px; '
            f'font-size: 12px; letter-spacing: 3px;">// 通过聊天记录复活记忆中的人 //</div>'
        )

        ui.html("").style("height: 30px;")

        # 人格列表
        personas = ChatController.list_personas()

        with ui.column().classes("w-full items-center gap-3").style("max-width: 700px;"):
            if personas:
                ui.html(
                    f'<span style="color: {NEON_CYAN}; font-size: 12px; letter-spacing: 3px;">'
                    f'[ SAVED TARGETS ]</span>'
                ).classes("text-center")

                ui.html("").style("height: 6px;")

                for i, p in enumerate(personas, 1):
                    _persona_card(i, p)
            else:
                ui.html(
                    f'<span style="color: {TEXT_DIM}; font-size: 13px;">'
                    f'[ NO TARGETS FOUND ]</span>'
                ).classes("text-center")
                ui.html(
                    f'<span style="color: {TEXT_DIM}; font-size: 12px; margin-top: 8px;">'
                    f'导入聊天记录以开始</span>'
                ).classes("text-center")

        ui.html("").style("height: 30px;")

        # 底部命令
        ui.button(
            "[ IMPORT NEW TARGET ]",
            on_click=lambda: ui.navigate.to("/import"),
        ).props("flat no-caps").style(
            f"color: {NEON_MAGENTA}; border: 1px solid rgba(255,0,170,0.3); "
            f"font-size: 12px; letter-spacing: 1px; padding: 8px 24px;"
        )

        ui.html(
            f'<div style="color: {TEXT_DIM}; font-size: 11px; margin-top: 30px; text-align: center;">'
            f'&gt; click target to initiate link // import to add new target</div>'
        )


def _persona_card(index: int, persona: dict):
    """渲染一个人格卡片。"""
    name = persona["name"]
    count = persona["total_messages"]
    summary = persona.get("style_summary", "")[:100]

    with ui.card().classes("w-full cursor-pointer").style(
        f"background: {BG_SECONDARY}; border: 1px solid rgba(0,255,213,0.15); "
        f"border-radius: 2px; padding: 14px 20px; box-shadow: none; "
        f"transition: border-color 0.2s, box-shadow 0.2s;"
    ).on("click", lambda n=name: ui.navigate.to(f"/chat/{n}")) as card:
        # CSS hover (via JS)
        card.on("mouseenter", lambda e, c=card: c.style(add=
            f"border-color: rgba(0,255,213,0.5); "
            f"box-shadow: 0 0 12px rgba(0,255,213,0.12);"
        ))
        card.on("mouseleave", lambda e, c=card: c.style(
            f"background: {BG_SECONDARY}; border: 1px solid rgba(0,255,213,0.15); "
            f"border-radius: 2px; padding: 14px 20px; box-shadow: none; "
            f"transition: border-color 0.2s, box-shadow 0.2s;"
        ))

        with ui.row().classes("w-full items-center justify-between no-wrap"):
            with ui.row().classes("items-center gap-3 no-wrap").style("overflow: hidden; min-width: 0;"):
                ui.html(
                    f'<span style="color: {TEXT_DIM}; font-size: 12px; flex-shrink: 0;">[{index:02d}]</span>'
                )
                ui.html(
                    f'<span style="color: {NEON_CYAN}; font-size: 14px; font-weight: 500; '
                    f'white-space: nowrap;">{_escape(name)}</span>'
                )
                ui.html(
                    f'<span style="color: {TEXT_DIM}; font-size: 12px; white-space: nowrap;">'
                    f'// {count} messages</span>'
                )
            ui.html(
                f'<span style="color: {NEON_GREEN}; font-size: 11px; letter-spacing: 1px; '
                f'flex-shrink: 0; text-shadow: 0 0 6px rgba(57,255,20,0.4);">ONLINE</span>'
            )

        if summary:
            ui.html(
                f'<div style="color: {TEXT_DIM}; font-size: 11px; margin-top: 8px; '
                f'line-height: 1.5; overflow: hidden; text-overflow: ellipsis; '
                f'display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">'
                f'&gt; {_escape(summary)}</div>'
            )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
