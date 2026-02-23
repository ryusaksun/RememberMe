#!/usr/bin/env python3
"""PixelRoom Playwright 回归脚本。

用途：
1) 运行桌面基线截图
2) 运行状态切换截图序列（idle/thinking/at_computer/excited/reading/sleeping）
3) 运行移动端布局截图
4) 导出 render_game_to_text JSON 与 console/pageerror 错误日志
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_SEED = 20260223


def _wait_http_ready(url: str, timeout_s: float = 30.0) -> None:
    start = time.monotonic()
    last_err: str | None = None
    while time.monotonic() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except urllib.error.URLError as exc:
            last_err = str(exc)
        except Exception as exc:  # pragma: no cover - 防御性
            last_err = str(exc)
        time.sleep(0.4)
    raise RuntimeError(f"GUI 未在 {timeout_s:.0f}s 内就绪: {url}; last_error={last_err}")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _capture_state(page: Page, out_json: Path) -> None:
    state_text = page.evaluate(
        """
        () => {
          if (typeof window.render_game_to_text === 'function') return window.render_game_to_text();
          return null;
        }
        """
    )
    if not state_text:
        _write_json(out_json, {"error": "window.render_game_to_text not found"})
        return
    try:
        _write_json(out_json, json.loads(state_text))
    except json.JSONDecodeError:
        _write_json(out_json, {"raw": state_text})


def _ensure_canvas_ready(page: Page) -> None:
    page.wait_for_selector('[data-testid="pixel-room-canvas"]', timeout=15000)
    page.wait_for_function(
        """
        () => {
          const c = document.querySelector('[data-testid="pixel-room-canvas"]');
          return !!(c && c.width > 0 && c.height > 0 && window.render_game_to_text);
        }
        """,
        timeout=15000,
    )


def _prepare_scene(page: Page, seed: int = DEFAULT_SEED) -> None:
    page.evaluate(
        """
        ([seed]) => {
          if (window.pixelRoomSetSeed) window.pixelRoomSetSeed(seed);
          if (window.pixelRoomSetState) window.pixelRoomSetState('idle');
        }
        """,
        [seed],
    )
    page.evaluate(
        """
        async () => {
          if (window.advanceTime) await window.advanceTime(1200);
        }
        """
    )


def _run_desktop_idle(page: Page, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _prepare_scene(page)
    canvas = page.locator('[data-testid="pixel-room-canvas"]')
    canvas.screenshot(path=str(out_dir / "desktop_idle_canvas.png"))
    page.screenshot(path=str(out_dir / "desktop_idle_page.png"), full_page=True)
    _capture_state(page, out_dir / "desktop_idle_state.json")


def _run_state_sequence(page: Page, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = [
        ("thinking", 1400),
        ("at_computer", 1600),
        ("excited", 900),
        ("idle", 1200),
        ("reading", 2200),
        ("sleeping", 2400),
        ("idle", 1500),
    ]
    for idx, (state, ms) in enumerate(steps, start=1):
        page.evaluate(
            """
            async ([state, ms]) => {
              if (window.pixelRoomSetState) window.pixelRoomSetState(state);
              if (window.advanceTime) await window.advanceTime(ms);
            }
            """,
            [state, ms],
        )
        canvas = page.locator('[data-testid="pixel-room-canvas"]')
        canvas.screenshot(path=str(out_dir / f"state_{idx:02d}_{state}.png"))
        _capture_state(page, out_dir / f"state_{idx:02d}_{state}.json")


def _run_mobile_layout(page: Page, out_dir: Path, url: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto(url, wait_until="domcontentloaded")
    _ensure_canvas_ready(page)
    _prepare_scene(page)
    page.screenshot(path=str(out_dir / "mobile_page.png"), full_page=True)
    page.locator('[data-testid="pixel-room-canvas"]').screenshot(path=str(out_dir / "mobile_canvas.png"))
    _capture_state(page, out_dir / "mobile_state.json")


def _first_persona_name() -> str:
    profiles_dir = ROOT / "data" / "profiles"
    candidates = sorted(profiles_dir.glob("*.json"))
    if not candidates:
        raise RuntimeError("未找到 data/profiles/*.json，无法进入聊天页")
    return candidates[0].stem


def main() -> int:
    parser = argparse.ArgumentParser(description="PixelRoom Playwright regression runner")
    parser.add_argument("--persona", help="人格名称，默认自动选择 data/profiles 第一个")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--output-dir",
        default="output/pixel_room",
        help="输出目录根路径（默认 output/pixel_room）",
    )
    parser.add_argument(
        "--round",
        default=datetime.now().strftime("round_%Y%m%d_%H%M%S"),
        help="本轮标记，用于输出目录归档",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="不自动启动 GUI，默认会在后台启动 uv run remember-me-gui",
    )
    parser.add_argument(
        "--headless",
        default="true",
        choices=["true", "false"],
        help="是否无头运行（默认 true）",
    )
    args = parser.parse_args()
    if not args.skip_server and args.port != DEFAULT_PORT:
        raise ValueError(
            "自动启动模式当前仅支持 8080；如需其他端口，请先手动启动 GUI 并使用 --skip-server"
        )

    persona = args.persona or _first_persona_name()
    encoded = urllib.parse.quote(persona, safe="")
    base_url = f"http://{args.host}:{args.port}"
    chat_url = f"{base_url}/chat/{encoded}"

    run_dir = ROOT / args.output_dir / args.round
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_meta.json",
        {
            "persona": persona,
            "chat_url": chat_url,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    )

    gui_proc: subprocess.Popen[str] | None = None
    gui_log_fh = None
    try:
        if not args.skip_server:
            gui_log_fh = (run_dir / "gui.log").open("w", encoding="utf-8")
            gui_proc = subprocess.Popen(
                ["uv", "run", "remember-me-gui"],
                cwd=ROOT,
                stdout=gui_log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )

        _wait_http_ready(base_url, timeout_s=35)

        console_errors: list[dict[str, str]] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=args.headless == "true")
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            page.on(
                "console",
                lambda msg: console_errors.append({"type": "console.error", "text": msg.text})
                if msg.type == "error"
                else None,
            )
            page.on(
                "pageerror",
                lambda exc: console_errors.append({"type": "pageerror", "text": str(exc)}),
            )

            page.goto(chat_url, wait_until="domcontentloaded")
            _ensure_canvas_ready(page)

            _run_desktop_idle(page, run_dir / "desktop_idle")
            _run_state_sequence(page, run_dir / "state_sequence")
            _run_mobile_layout(page, run_dir / "mobile", chat_url)

            browser.close()

        _write_json(run_dir / "console_errors.json", console_errors)
        if console_errors:
            print(f"[warn] 检测到 {len(console_errors)} 条错误，详情见: {run_dir/'console_errors.json'}")
        else:
            print("[ok] 无 console/pageerror")

        runner_error = run_dir / "runner_error.txt"
        if runner_error.exists():
            runner_error.unlink()

        print(f"[ok] 回归产物目录: {run_dir}")
        return 0
    except Exception as exc:
        _write_text(run_dir / "runner_error.txt", str(exc))
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if gui_proc:
            gui_proc.terminate()
            try:
                gui_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gui_proc.kill()
        if gui_log_fh:
            gui_log_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
