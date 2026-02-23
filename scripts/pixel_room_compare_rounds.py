#!/usr/bin/env python3
"""比较两个 PixelRoom 回归轮次的截图差异。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "output" / "pixel_room"


def _collect_pngs(round_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(round_dir.rglob("*.png")):
        rel = path.relative_to(round_dir).as_posix()
        files[rel] = path
    return files


def _mae(img: Image.Image) -> float:
    stat = ImageStat.Stat(img)
    means = stat.mean
    return float(sum(means) / len(means))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two pixel room regression rounds")
    parser.add_argument("--base", required=True, help="基线轮次名")
    parser.add_argument("--candidate", required=True, help="候选轮次名")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="回归输出根目录")
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="平均像素差阈值（默认 3.0）",
    )
    args = parser.parse_args()

    root = Path(args.root)
    base_dir = root / args.base
    cand_dir = root / args.candidate
    if not base_dir.exists() or not cand_dir.exists():
        raise SystemExit("指定轮次目录不存在")

    out_dir = root / f"diff_{args.base}_vs_{args.candidate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_pngs = _collect_pngs(base_dir)
    cand_pngs = _collect_pngs(cand_dir)
    common = sorted(set(base_pngs) & set(cand_pngs))

    report: dict[str, object] = {
        "base": args.base,
        "candidate": args.candidate,
        "threshold": args.threshold,
        "compared": len(common),
        "files": [],
    }

    changed = 0
    for rel in common:
        base_img = Image.open(base_pngs[rel]).convert("RGBA")
        cand_img = Image.open(cand_pngs[rel]).convert("RGBA")
        if base_img.size != cand_img.size:
            report["files"].append(
                {
                    "file": rel,
                    "status": "size_mismatch",
                    "base_size": base_img.size,
                    "candidate_size": cand_img.size,
                }
            )
            continue

        diff = ImageChops.difference(base_img, cand_img)
        score = _mae(diff)
        is_changed = score >= args.threshold
        if is_changed:
            changed += 1
            diff_out = out_dir / rel
            diff_out.parent.mkdir(parents=True, exist_ok=True)
            diff.save(diff_out)

        report["files"].append(
            {
                "file": rel,
                "mae": round(score, 4),
                "changed": is_changed,
            }
        )

    report["changed_count"] = changed
    report["unchanged_count"] = len(common) - changed

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] compared={len(common)} changed={changed} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
