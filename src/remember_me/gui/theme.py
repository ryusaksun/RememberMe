"""赛博朋克终端主题常量。"""

# 色板
BG_PRIMARY = "#0a0a0f"
BG_SECONDARY = "#12121a"
BG_INPUT = "#0d0d15"
NEON_CYAN = "#00ffd5"
NEON_MAGENTA = "#ff00aa"
NEON_GREEN = "#39ff14"
TEXT_DIM = "#4a5568"
TEXT_PRIMARY = "#e0e0e0"
BORDER_GLOW = "0 0 10px rgba(0,255,213,0.3)"

# 字体
FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Source Code Pro', 'Cascadia Code', monospace"

# 效果
SCANLINE_OPACITY = 0.03

# 全局 CSS
GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');

:root {
    --bg-primary: %(bg_primary)s;
    --bg-secondary: %(bg_secondary)s;
    --bg-input: %(bg_input)s;
    --neon-cyan: %(neon_cyan)s;
    --neon-magenta: %(neon_magenta)s;
    --neon-green: %(neon_green)s;
    --text-dim: %(text_dim)s;
    --text-primary: %(text_primary)s;
}

*:not(.material-icons):not(.q-icon) {
    font-family: %(font_mono)s !important;
}

body, .q-page, .nicegui-content {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

/* NiceGUI 默认 padding 清理 */
.nicegui-content {
    padding: 0 !important;
}
.q-page {
    padding: 0 !important;
    min-height: auto !important;
}

/* 扫描线叠加 */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 213, %(scanline_opacity)s) 2px,
        rgba(0, 255, 213, %(scanline_opacity)s) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* 滚动条 */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(0, 255, 213, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 255, 213, 0.5); }

/* Quasar 覆盖 */
.q-field__control { background: var(--bg-input) !important; }
.q-field__native, .q-field__input { color: var(--neon-green) !important; }
.q-field__label { color: var(--text-dim) !important; }

/* 上传组件主题 */
.q-uploader { background: var(--bg-secondary) !important; border: 1px solid rgba(0,255,213,0.2) !important; }
.q-uploader__header { background: rgba(0,255,213,0.08) !important; }
.q-uploader__title, .q-uploader__subtitle { color: var(--neon-cyan) !important; }
.q-uploader__list { background: var(--bg-secondary) !important; }
.q-btn--flat .q-icon { color: var(--neon-cyan) !important; }

/* 下拉选择框 */
.q-menu { background: var(--bg-secondary) !important; border: 1px solid rgba(0,255,213,0.2) !important; }
.q-item { color: var(--text-primary) !important; }
.q-item--active { color: var(--neon-cyan) !important; background: rgba(0,255,213,0.08) !important; }

/* 卡片 */
.q-card { box-shadow: none !important; }

/* 分隔线 */
.q-separator { background: rgba(0,255,213,0.15) !important; }

/* 动画 */
@keyframes blink {
    0%%, 50%% { opacity: 1; }
    51%%, 100%% { opacity: 0; }
}

@keyframes glitch {
    0%% { transform: translate(0); }
    20%% { transform: translate(-2px, 1px); }
    40%% { transform: translate(2px, -1px); }
    60%% { transform: translate(-1px, 2px); }
    80%% { transform: translate(1px, -2px); }
    100%% { transform: translate(0); }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.msg-appear {
    animation: fadeInUp 0.3s ease-out;
}

.cursor-blink {
    animation: blink 1s step-end infinite;
}

.glitch-text {
    animation: glitch 0.3s ease-in-out;
}

/* 消息样式 */
.msg-target {
    color: var(--neon-cyan);
}
.msg-user {
    color: var(--neon-green);
}
.msg-system {
    color: var(--neon-magenta);
}
.msg-time {
    color: var(--text-dim);
    font-size: 0.85em;
}
.msg-sticker {
    max-width: 120px;
    max-height: 120px;
    border: 1px solid rgba(0, 255, 213, 0.3);
    border-radius: 4px;
    margin: 4px 0;
}
""" % {
    "bg_primary": BG_PRIMARY,
    "bg_secondary": BG_SECONDARY,
    "bg_input": BG_INPUT,
    "neon_cyan": NEON_CYAN,
    "neon_magenta": NEON_MAGENTA,
    "neon_green": NEON_GREEN,
    "text_dim": TEXT_DIM,
    "text_primary": TEXT_PRIMARY,
    "font_mono": FONT_MONO,
    "scanline_opacity": SCANLINE_OPACITY,
}
