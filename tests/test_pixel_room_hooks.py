from remember_me.gui.components.pixel_room import _JS_ENGINE


def test_js_engine_exports_playwright_hooks() -> None:
    assert "window.render_game_to_text=function" in _JS_ENGINE
    assert "window.advanceTime=function" in _JS_ENGINE
    assert "window.pixelRoomGetSnapshot=function" in _JS_ENGINE
    assert "window.pixelRoomSetSeed=function" in _JS_ENGINE


def test_js_engine_uses_seeded_random() -> None:
    assert "function rand()" in _JS_ENGINE
    assert "setSeed(_seed);" in _JS_ENGINE
    assert "Math.random(" not in _JS_ENGINE


def test_js_engine_exposes_test_ids() -> None:
    assert '[data-testid="pixel-room-status"]' in _JS_ENGINE
