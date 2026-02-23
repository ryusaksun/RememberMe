"""终端风格 "正在输入" 指示器。"""

from __future__ import annotations

from nicegui import ui


class TypingIndicator:
    """显示 `[name is typing...]  █` 闪烁动画。"""

    def __init__(self, name: str):
        self._name = name
        self._container: ui.element | None = None
        self._visible = False

    def create(self, parent: ui.element) -> ui.element:
        """在 parent 中创建指示器（默认隐藏）。"""
        with parent:
            self._container = ui.html("").style("padding: 1px 0;")
            self._container.set_visibility(False)
        return self._container

    def show(self):
        if self._container and not self._visible:
            self._visible = True
            self._container.content = (
                f'<span class="msg-target">'
                f'[{_escape(self._name)} is typing...]'
                f'</span>'
                f'<span class="cursor-blink msg-target"> █</span>'
            )
            self._container.set_visibility(True)

    def hide(self):
        if self._container and self._visible:
            self._visible = False
            self._container.set_visibility(False)

    @property
    def visible(self) -> bool:
        return self._visible


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
