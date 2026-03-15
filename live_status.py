#!/usr/bin/env python3
"""
Live status checker for tuner/master runs.

Why this version:
- best_result.json updates at generation boundaries only.
- results.jsonl updates per candidate, so it is the real live feed.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def fmt_time(ts):
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return "?"


def age_sec(path: Path):
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def bar(wr: float, width: int = 24) -> str:
    wr = max(0.0, min(1.0, wr))
    filled = int(round(wr * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {wr*100:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="Check live training status (non-blocking).")
    ap.add_argument("--workspace", default="/home/tufat/Desktop/WinterChallenge2026-Exotec")
    ap.add_argument("--artifacts-dir", default="tools/bot_merged_tuning")
    args = ap.parse_args()

    artifacts = (Path(args.workspace) / args.artifacts_dir).resolve()
    master_path = artifacts / "master_status.json"
    best_path = artifacts / "best_result.json"
    jsonl_path = artifacts / "results.jsonl"

    master = load_json(master_path)
    best = load_json(best_path)
    rows = load_jsonl(jsonl_path)

    if not master and not best and not rows:
        print(f"No status data found in {artifacts}")
        print("Run master/tuner first or pass --artifacts-dir correctly.")
        return 0

    print("=" * 72)
    print("LIVE STATUS")
    print(f"artifacts: {artifacts}")
    print("=" * 72)

    m_age = age_sec(master_path)
    b_age = age_sec(best_path)
    r_age = age_sec(jsonl_path)
    print(
        "freshness: "
        f"master_status={'n/a' if m_age is None else f'{int(m_age)}s'}  "
        f"best_result={'n/a' if b_age is None else f'{int(b_age)}s'}  "
        f"results_jsonl={'n/a' if r_age is None else f'{int(r_age)}s'}"
    )
    if b_age is not None and r_age is not None and (b_age - r_age) > 30:
        print("note: best_result.json older than results.jsonl -> generation still running (this is normal)")

    if master:
        print("\n[MASTER]")
        print(
            f"cycle={master.get('current_cycle', '?')}/{master.get('target_cycles', '?')}  "
            f"phase={master.get('phase', '?')}  updated={fmt_time(master.get('updated_at', 0))}"
        )

    if rows:
        latest = rows[-1]
        lm = latest.get("metrics", {})
        print("\n[LIVE FROM results.jsonl]")
        print(
            f"latest row: gen={latest.get('generation', '?')} idx={latest.get('index', '?')}  "
            f"WR={lm.get('win_rate', 0):.3f} {bar(float(lm.get('win_rate', 0)))}  "
            f"fit={lm.get('fitness', 0):.0f}  pts={lm.get('avg_score_diff', 0):+.1f}"
        )

        top = max(rows, key=lambda r: (
            float(r.get("metrics", {}).get("fitness", -1e18)),
            float(r.get("metrics", {}).get("win_rate", -1.0)),
        ))
        tm = top.get("metrics", {})
        print(
            f"best row:   gen={top.get('generation', '?')} idx={top.get('index', '?')}  "
            f"WR={tm.get('win_rate', 0):.3f} {bar(float(tm.get('win_rate', 0)))}  "
            f"fit={tm.get('fitness', 0):.0f}  pts={tm.get('avg_score_diff', 0):+.1f}"
        )

    if best:
        be = best.get("best_eval") or {}
        bm = be.get("metrics") or {}
        print("\n[SNAPSHOT FROM best_result.json]")
        print(
            f"best_eval: gen={be.get('generation', '?')} idx={be.get('index', '?')}  "
            f"WR={bm.get('win_rate', 0):.3f} {bar(float(bm.get('win_rate', 0)))}  "
            f"fit={bm.get('fitness', 0):.0f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
