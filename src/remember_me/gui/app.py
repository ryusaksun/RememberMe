"""NiceGUI 应用入口 — 赛博朋克终端风格聊天界面。"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from nicegui import app, ui

from remember_me.gui.theme import GLOBAL_CSS


def setup_routes():
    """注册路由。"""

    @ui.page("/")
    def home_page():
        ui.add_css(GLOBAL_CSS)
        from remember_me.gui.pages.home import create_home_page
        create_home_page()

    @ui.page("/chat/{name}")
    def chat_page(name: str):
        ui.add_css(GLOBAL_CSS)
        from remember_me.gui.pages.chat import create_chat_page
        create_chat_page(name)

    @ui.page("/import")
    def import_page():
        ui.add_css(GLOBAL_CSS)
        from remember_me.gui.pages.import_flow import create_import_page
        create_import_page()


def main():
    """GUI 入口点。"""
    setup_routes()
    ui.run(
        title="RememberMe",
        port=8080,
        favicon="🧠",
        dark=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
