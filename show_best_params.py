#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Extract the best GA parameters found so far without stopping.")
    ap.add_argument("--workspace", default="/home/tufat/Desktop/WinterChallenge2026-Exotec")
    ap.add_argument("--artifacts-dir", default="tools/bot_merged_tuning")
    args = ap.parse_args()

    artifacts = (Path(args.workspace) / args.artifacts_dir).resolve()
    jsonl_path = artifacts / "results.jsonl"

    if not jsonl_path.exists():
        print(f"Log file not found: {jsonl_path}")
        return 1
        
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
                
    if not rows:
        print("No valid rows found in log.")
        return 1
        
    top = max(rows, key=lambda r: (
        float(r.get("metrics", {}).get("fitness", -1e18)),
        float(r.get("metrics", {}).get("win_rate", -1.0)),
    ))
    
    tm = top.get("metrics", {})
    gen = top.get("generation", "?")
    idx = top.get("index", "?")
    params = top.get("params", {})
    
    print("=========================================")
    print(f"🏆 BEST CANDIDATE (Gen: {gen}, Index: {idx})")
    print("=========================================")
    print(f"Win Rate:  {tm.get('win_rate', 0):.3%} ({tm.get('wins', 0)}W, {tm.get('ties', 0)}T, {tm.get('losses', 0)}L)")
    print(f"Fitness:   {tm.get('fitness', 0):.0f}")
    print(f"Score Adv: {tm.get('avg_score_diff', 0):+.2f} points")
    print("\nCopy and paste these macros over your defaults in bot.cpp:\n")
    
    for k, v in params.items():
        if isinstance(v, float):
            print(f"#define {k:<18} {v:.6f}")
        else:
            print(f"#define {k:<18} {v}")
    
    print("\n(Note: The trainer is still running in the background!)")

if __name__ == "__main__":
    sys.exit(main())
