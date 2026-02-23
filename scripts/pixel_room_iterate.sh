#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ROUND="${1:-round_manual}"
PERSONA="${2:-}"

CMD=(uv run python scripts/pixel_room_playwright_regression.py --round "$ROUND")
if [[ -n "$PERSONA" ]]; then
  CMD+=(--persona "$PERSONA")
fi

"${CMD[@]}"
