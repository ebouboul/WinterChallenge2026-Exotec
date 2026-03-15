#!/usr/bin/env python3
"""
Master iterative self-play trainer for bot_merged.cpp.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

def log(msg: str, log_file: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_status(status_path: Path, payload: dict) -> None:
    payload["updated_at"] = time.time()
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Master iterative self-play trainer — runs tune_bot_merged_ga.py in evolving cycles."
    )
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--target-cycles", type=int, default=10)
    ap.add_argument("--max-generations", type=int, default=20)
    ap.add_argument("--population",    type=int,   default=12)
    ap.add_argument("--elite",         type=int,   default=3)
    ap.add_argument("--seeds-per-eval",type=int,   default=30)
    ap.add_argument("--verify-seeds",  type=int,   default=40)
    ap.add_argument("--verify-rounds", type=int,   default=2)
    ap.add_argument("--target-winrate",type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--artifacts-dir", default="tools/bot_merged_tuning")
    ap.add_argument("--opponents", nargs="*", default=["other_bot.cpp"])
    args = ap.parse_args()

    ws           = Path(args.workspace).resolve()
    artifacts    = ws / args.artifacts_dir
    cycles_dir   = artifacts / "cycles"
    artifacts.mkdir(parents=True, exist_ok=True)
    cycles_dir.mkdir(exist_ok=True)

    master_log    = artifacts / "master_training_log.txt"
    master_status = artifacts / "master_status.json"
    best_result   = artifacts / "best_result.json"
    tuner_script  = ws / "tune_bot_merged_ga.py"

    def L(msg):
        log(msg, master_log)

    L("=" * 60)
    L("MASTER TRAINER START")
    L(f"Target cycles={args.target_cycles}  max-gen/cycle={args.max_generations}  workers={args.workers}")
    L("=" * 60)

    base_params_file: Path | None = None   
    base_params: dict | None      = None
    best_overall: dict | None     = None

    if best_result.exists():
        try:
            br = json.loads(best_result.read_text(encoding="utf-8"))
            if "best_eval" in br and "params" in br["best_eval"]:
                base_params = br["best_eval"]["params"]
                base_params_file = best_result
                best_overall = {
                    "cycle": "resumed",
                    "params": base_params,
                    "metrics": br["best_eval"].get("metrics", {}),
                    "compile_command": br.get("compile_command", "")
                }
                L(f"Resuming from existing {best_result.name} (WR: {best_overall['metrics'].get('win_rate', 0):.3f})")
        except Exception as e:
            L(f"Failed to resume from {best_result.name}: {e}")

    for cycle in range(1, args.target_cycles + 1):
        L(f"--- CYCLE {cycle}/{args.target_cycles} ---")
        if base_params:
            L(f"Baseline: {json.dumps(base_params, sort_keys=True)}")
        else:
            L("Baseline: hardcoded defaults (first cycle)")

        write_status(master_status, {
            "current_cycle":           cycle,
            "target_cycles":           args.target_cycles,
            "phase":                   "tuning",
            "current_baseline_params": base_params or {},
            "best_overall":            best_overall or {},
            "best_result_path":        str(best_result),
            "master_log_path":         str(master_log),
        })

        cmd = [
            sys.executable, str(tuner_script),
            "--workspace",      str(ws),
            "--population",     str(args.population),
            "--elite",          str(args.elite),
            "--seeds-per-eval", str(args.seeds_per_eval),
            "--verify-seeds",   str(args.verify_seeds),
            "--verify-rounds",  str(args.verify_rounds),
            "--target-winrate", str(args.target_winrate),
            "--workers",        str(args.workers),
            "--max-generations",str(args.max_generations),
            "--artifacts-dir",  args.artifacts_dir,
            "--seed",           str(20260314 + cycle),
        ]
        if args.opponents:
            cmd += ["--opponents", *args.opponents]
        if base_params_file:
            cmd += ["--base-params", str(base_params_file)]

        L(f"Launching: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ws))
        elapsed = round(time.time() - t0, 1)
        exit_code = proc.returncode

        if exit_code == 0:
            if not best_result.exists():
                L(f"ERROR: tuner exited 0 but {best_result} not found. Stopping.")
                break

            br = json.loads(best_result.read_text())
            winner_params  = br["best_eval"]["params"]
            winner_metrics = br["best_eval"]["metrics"]
            compile_cmd    = br.get("compile_command", "")

            L(f"CYCLE {cycle} SUCCESS in {elapsed}s: "
              f"WR={winner_metrics['win_rate']:.3f}  "
              f"fit={winner_metrics['fitness']:.0f}  "
              f"pts={winner_metrics['avg_score_diff']:+.1f}")
            L(f"Winner params: {json.dumps(winner_params, sort_keys=True)}")

            cycle_file = cycles_dir / f"cycle_{cycle:03d}_winner.json"
            cycle_file.write_text(json.dumps({
                "cycle":           cycle,
                "ts":              time.time(),
                "elapsed_sec":     elapsed,
                "params":          winner_params,
                "metrics":         winner_metrics,
                "compile_command": compile_cmd,
            }, indent=2), encoding="utf-8")

            best_overall      = {"cycle": cycle, "params": winner_params,
                                 "metrics": winner_metrics, "compile_command": compile_cmd}
            base_params       = winner_params
            base_params_file  = cycle_file

            write_status(master_status, {
                "current_cycle":           cycle,
                "target_cycles":           args.target_cycles,
                "phase":                   "promoting",
                "current_baseline_params": base_params,
                "best_overall":            best_overall,
                "best_result_path":        str(best_result),
                "master_log_path":         str(master_log),
            })
            L(f"Promoted: cycle {cycle} winner → baseline for cycle {cycle + 1}")

        elif exit_code == 2:
            L(f"CYCLE {cycle}: Nash Equilibrium — no mutant beat the baseline after "
              f"{args.max_generations} generations. Training converged.")
            break

        else:
            L(f"CYCLE {cycle}: Tuner exited with unexpected code {exit_code}. Stopping.")
            break

    L("=" * 60)
    L("MASTER TRAINER DONE")
    if best_overall:
        L(f"Best params overall (cycle {best_overall['cycle']}):")
        L(json.dumps(best_overall["params"], sort_keys=True))
    else:
        L("No winner found across any cycle.")
    L("=" * 60)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)