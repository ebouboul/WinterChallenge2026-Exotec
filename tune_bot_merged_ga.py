#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

RESULT_RE = re.compile(
    r"REF_STAT\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
)
MAX_TURNS = 200


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # int|float
    lo: float
    hi: float
    default: float

# SYNCHRONIZED WITH THE MACROS IN C++
PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("MAX_SEARCH_DEPTH", "int", 5, 14, 10),
    ParamSpec("W_SCORE", "float", 0.1, 1.0, 0.40),
    ParamSpec("W_BOT", "float", 0.1, 1.0, 0.30),
    ParamSpec("W_DIST_MY", "float", 0.0, 0.5, 0.05),
    ParamSpec("W_DIST_OPP", "float", 0.0, 0.5, 0.05),
    ParamSpec("W_SAFETY", "float", 0.0, 0.5, 0.10),
    ParamSpec("W_GROUND", "float", 0.0, 0.5, 0.15),
    ParamSpec("W_MOB", "float", 0.0, 0.2, 0.02),
    ParamSpec("W_TERR", "float", 0.0, 0.8, 0.15), 
    ParamSpec("UCB_EXPLORATION", "float", 0.05, 2.0, 0.50),
]

def run_cmd(cmd: List[str], timeout: int = 300, cwd=None) -> str:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout[-3000:]}"
        )
    return proc.stdout.strip()


def parse_result_line(output: str) -> Tuple[str, int, int, int, int, int]:
    m = RESULT_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not parse referee output. Tail:\n{output[-1000:]}")
        
    p0_score = int(m.group(1))
    p1_score = int(m.group(2))
    p0_losses = int(m.group(3))
    p1_losses = int(m.group(4))
    turns = int(m.group(5))
    
    if "WINNER: 0" in output:
        result = "P0_WIN"
    elif "WINNER: 1" in output:
        result = "P1_WIN"
    else:
        result = "TIE"
        
    return result, p0_score, p1_score, p0_losses, p1_losses, turns


def clamp(spec: ParamSpec, v: float):
    vv = max(spec.lo, min(spec.hi, v))
    if spec.kind == "int":
        return int(round(vv))
    return round(float(vv), 6)

def defaults() -> Dict[str, object]:
    out: Dict[str, object] = {}
    for s in PARAM_SPECS:
        out[s.name] = int(s.default) if s.kind == "int" else float(s.default)
    return out

def random_candidate(rng: random.Random) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for s in PARAM_SPECS:
        if s.kind == "int":
            out[s.name] = rng.randint(int(s.lo), int(s.hi))
        else:
            out[s.name] = round(rng.uniform(s.lo, s.hi), 6)
    return out

def mutate_candidate(rng: random.Random, base: Dict[str, object], strength: float) -> Dict[str, object]:
    c = dict(base)
    for s in PARAM_SPECS:
        if rng.random() < 0.55:
            span = s.hi - s.lo
            if s.kind == "int":
                step = max(1, int(round(strength * span * 0.12)))
                c[s.name] = clamp(s, int(c[s.name]) + rng.randint(-step, step))
            else:
                sigma = max(1e-6, span * 0.12 * strength)
                c[s.name] = clamp(s, float(c[s.name]) + rng.gauss(0.0, sigma))
    return c

def crossover(rng: random.Random, a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    child: Dict[str, object] = {}
    for s in PARAM_SPECS:
        av = a[s.name]
        bv = b[s.name]
        if s.kind == "int":
            if rng.random() < 0.5:
                child[s.name] = int(av)
            else:
                child[s.name] = int(bv)
            if rng.random() < 0.2:
                child[s.name] = int(round((int(av) + int(bv)) / 2.0))
            child[s.name] = clamp(s, child[s.name])
        else:
            w = rng.random()
            mixed = float(av) * w + float(bv) * (1.0 - w)
            child[s.name] = clamp(s, mixed)
    return child

def params_key(params: Dict[str, object]) -> str:
    parts = []
    for s in PARAM_SPECS:
        v = params[s.name]
        if s.kind == "int":
            parts.append(f"{s.name}={int(v)}")
        else:
            parts.append(f"{s.name}={float(v):.6f}")
    return "|".join(parts)

def params_hash(params: Dict[str, object]) -> str:
    return hashlib.sha1(params_key(params).encode("utf-8")).hexdigest()[:16]

def compile_bot(src: Path, out_bin: Path, params: Dict[str, object]) -> None:
    cmd = ["g++", "-std=c++17", "-O3", "-pipe", str(src), "-o", str(out_bin)]
    for s in PARAM_SPECS:
        v = params[s.name]
        if s.kind == "int":
            cmd.append(f"-D{s.name}={int(v)}")
        else:
            cmd.append(f"-D{s.name}={float(v):.6f}")
    run_cmd(cmd, timeout=300)

def ensure_referee(ref_src: Path, ref_bin: Path) -> None:
    # Compile the Java referee using Maven instead of compiling C++
    ws_dir = ref_src.parent.parent.parent.parent.parent.parent
    if not (ws_dir / "pom.xml").exists():
        ws_dir = ws_dir.parent
    run_cmd(["mvn", "compile"], timeout=300, cwd=str(ws_dir))

def run_one_game(ref_bin: Path, p0_bin: Path, p1_bin: Path, seed: int) -> Dict[str, object]:
    # We use Maven to run the Java referee. 
    ws_dir = p0_bin.parent.parent.parent.parent  # rough estimate
    
    # We use a wrapper or just direct mvn depending on how the system is set up.
    # The p_bins are absolute paths.
    cmd = [
        "mvn", "-q", "exec:java",
        "-Dexec.mainClass=Main",
        "-Dexec.classpathScope=test",
        f"-Dexec.args='{p0_bin.resolve()}' '{p1_bin.resolve()}' {seed}"
    ]
    # To get workspace dir perfectly, let's just find the parent with pom.xml
    cwd_path = p0_bin.parent
    while cwd_path != cwd_path.parent:
        if (cwd_path / "pom.xml").exists():
            break
        cwd_path = cwd_path.parent
        
    out = run_cmd(cmd, timeout=240, cwd=str(cwd_path))
    result, p0_score, p1_score, p0_losses, p1_losses, turns = parse_result_line(out)
    return {
        "result": result,
        "p0_score": p0_score,
        "p1_score": p1_score,
        "p0_losses": p0_losses,
        "p1_losses": p1_losses,
        "turns": turns,
    }


def game_fitness(
    cand_win: bool,
    cand_tie: bool,
    turns: int,
    score_diff: int,
    losses_diff: int,
    cand_score: int,
) -> float:
    suicide_penalty = -800_000.0 if cand_score == 0 else 0.0
    if cand_win:
        base = (
            900_000.0
            + score_diff * 30_000.0
            + losses_diff * 14_000.0
            + (MAX_TURNS - turns) * 2_500.0
        )
        return base + suicide_penalty
    if cand_tie:
        base = 40_000.0 + score_diff * 34_000.0 + losses_diff * 26_000.0 - turns * 40.0
        return base + suicide_penalty
    base = -700_000.0 + score_diff * 24_000.0 + losses_diff * 18_000.0 + turns * 200.0
    return base + suicide_penalty

def evaluate_candidate(
    ref_bin: Path,
    cand_bin: Path,
    base_bin: Path,
    seeds: List[int],
    workers: int,
) -> Dict[str, float]:
    tasks = []
    for sd in seeds:
        tasks.append((sd, True))
        tasks.append((sd, False))

    wins = 0
    ties = 0
    losses = 0
    turns_sum = 0
    diff_sum = 0
    losses_diff_sum = 0
    fit_sum = 0.0

    def run_task(task: Tuple[int, bool]):
        sd, cand_is_p0 = task
        p0 = cand_bin if cand_is_p0 else base_bin
        p1 = base_bin if cand_is_p0 else cand_bin
        g = run_one_game(ref_bin, p0, p1, sd)

        if cand_is_p0:
            cand_score = int(g["p0_score"])
            opp_score = int(g["p1_score"])
            cand_losses = int(g["p0_losses"])
            opp_losses = int(g["p1_losses"])
            cand_win = g["result"] == "P0_WIN"
        else:
            cand_score = int(g["p1_score"])
            opp_score = int(g["p0_score"])
            cand_losses = int(g["p1_losses"])
            opp_losses = int(g["p0_losses"])
            cand_win = g["result"] == "P1_WIN"

        cand_tie = g["result"] == "TIE"
        score_diff = cand_score - opp_score
        losses_diff = opp_losses - cand_losses
        return cand_win, cand_tie, int(g["turns"]), score_diff, losses_diff, cand_score

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_task, t) for t in tasks]
        for f in as_completed(futures):
            cand_win, cand_tie, turns, score_diff, losses_diff, cand_score = f.result()
            if cand_win:
                wins += 1
            elif cand_tie:
                ties += 1
            else:
                losses += 1
            turns_sum += turns
            diff_sum += score_diff
            losses_diff_sum += losses_diff
            fit_sum += game_fitness(cand_win, cand_tie, turns, score_diff, losses_diff, cand_score)

    games = max(1, len(tasks))
    wr = (wins + 0.5 * ties) / games
    return {
        "games": games,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wr,
        "avg_turns": turns_sum / games,
        "avg_score_diff": diff_sum / games,
        "avg_losses_diff": losses_diff_sum / games,
        "fitness": fit_sum / games,
    }

def evaluate_candidate_vs_pool(
    ref_bin: Path,
    cand_bin: Path,
    opp_bins: List[Path],
    seeds: List[int],
    workers: int,
) -> Dict[str, object]:
    per_opp: List[Dict[str, object]] = []
    
    # ALWAYS evaluate against baseline itself (which is critical if opp_bins is empty)
    if not opp_bins:
        m = evaluate_candidate(ref_bin, cand_bin, cand_bin.parent / "baseline_opponent", seeds, workers)
        per_opp.append(m)
    else:
        for opp_bin in opp_bins:
            m = evaluate_candidate(ref_bin, cand_bin, opp_bin, seeds, workers)
            per_opp.append(m)

    total_games = sum(m["games"] for m in per_opp)
    total_wins  = sum(m["wins"]  for m in per_opp)
    total_ties  = sum(m["ties"]  for m in per_opp)
    total_losses = sum(m["losses"] for m in per_opp)
    wr = (total_wins + 0.5 * total_ties) / max(1, total_games)
    avg_score_diff  = sum(m["avg_score_diff"]  * m["games"] for m in per_opp) / max(1, total_games)
    avg_losses_diff = sum(m["avg_losses_diff"] * m["games"] for m in per_opp) / max(1, total_games)
    avg_turns       = sum(m["avg_turns"]       * m["games"] for m in per_opp) / max(1, total_games)
    fitness         = sum(m["fitness"]         * m["games"] for m in per_opp) / max(1, total_games)
    return {
        "games": total_games,
        "wins":  total_wins,
        "ties":  total_ties,
        "losses": total_losses,
        "win_rate": wr,
        "avg_turns": avg_turns,
        "avg_score_diff": avg_score_diff,
        "avg_losses_diff": avg_losses_diff,
        "fitness": fitness,
        "per_opponent": [
            {"name": opp_bins[i].stem if opp_bins else "baseline", **per_opp[i]}
            for i in range(len(per_opp))
        ],
    }

def stable_seed_block(rng: random.Random, n: int) -> List[int]:
    return [rng.randint(1, 10**9 - 1) for _ in range(n)]

def log_jsonl(path: Path, item: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")

def format_compile_command(src: Path, out_bin: Path, params: Dict[str, object]) -> str:
    flags = []
    for s in PARAM_SPECS:
        v = params[s.name]
        if s.kind == "int":
            flags.append(f"-D{s.name}={int(v)}")
        else:
            flags.append(f"-D{s.name}={float(v):.6f}")
    return "g++ -std=c++17 -O3 -pipe " + " ".join(flags) + f" {src} -o {out_bin}"

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Autonomous GA hyperparameter tuner for src/bot_merged.cpp against baseline."
    )
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--seeds-per-eval", type=int, default=30)
    ap.add_argument("--verify-seeds", type=int, default=40)
    ap.add_argument("--verify-rounds", type=int, default=2)
    ap.add_argument("--target-winrate", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--seed", type=int, default=20260314)
    ap.add_argument("--max-generations", type=int, default=0)
    ap.add_argument("--artifacts-dir", default="tools/bot_merged_tuning")
    ap.add_argument(
        "--opponents",
        nargs="*",
        default=[],
        help="Opponent source files. If empty, plays against baseline only.",
    )
    ap.add_argument(
        "--base-params",
        default=None,
        help="Path to JSON file with baseline params.",
    )
    args = ap.parse_args()
# /home/tufat/Desktop/WinterChallenge2026-Exotec/src
# /home/tufat/Desktop/WinterChallenge2026-Exotec/src/main/java/com/codingame/game/Referee.java
    ws = Path(args.workspace).resolve()
    src = ws / "src" / "bot_merged.cpp"
    ref_src = ws / "src/main/java/com/codingame/game" / "Referee.java"
    artifacts = ws / args.artifacts_dir
    bins_dir = artifacts / "bins"
    artifacts.mkdir(parents=True, exist_ok=True)
    bins_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifacts / "results.jsonl"
    best_path = artifacts / "best_result.json"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if not ref_src.exists():
        raise FileNotFoundError(f"Referee source not found: {ref_src}")

    ref_bin = bins_dir / "referee_cpp"
    ensure_referee(ref_src, ref_bin)

    opp_bins: List[Path] = []
    for opp_rel in args.opponents:
        opp_src = ws / opp_rel
        if not opp_src.exists():
            raise FileNotFoundError(f"Opponent source not found: {opp_src}")
        opp_bin = bins_dir / f"opp_{opp_src.stem}"
        if not opp_bin.exists() or opp_bin.stat().st_mtime < opp_src.stat().st_mtime:
            print(f"[build] compiling opponent {opp_src.name}", flush=True)
            run_cmd(["g++", "-std=c++17", "-O3", "-pipe", str(opp_src), "-o", str(opp_bin)], timeout=300)
        opp_bins.append(opp_bin)

    if args.base_params:
        raw = json.loads(Path(args.base_params).read_text())
        if "best_eval" in raw and "params" in raw["best_eval"]:
            param_dict = raw["best_eval"]["params"]
        elif "params" in raw:
            param_dict = raw["params"]
        else:
            param_dict = raw

        base_params = defaults()
        base_params.update({k: v for k, v in param_dict.items() if k in base_params})
        print(f"[base] loaded baseline params from {args.base_params}", flush=True)
    else:
        base_params = defaults()

    base_hash = params_hash(base_params)
    baseline_bin = bins_dir / f"bot_merged_baseline_{base_hash}"
    src_mtime = src.stat().st_mtime
    if not baseline_bin.exists() or baseline_bin.stat().st_mtime < src_mtime:
        print("[build] compiling baseline bot_merged", flush=True)
        compile_bot(src, baseline_bin, base_params)

    # Provide a baseline opponent if no explicit opponents are provided
    if not opp_bins:
        baseline_opp_bin = bins_dir / "baseline_opponent"
        if not baseline_opp_bin.exists() or baseline_opp_bin.stat().st_mtime < src_mtime:
            compile_bot(src, baseline_opp_bin, base_params)

    rng = random.Random(args.seed)
    compiled_cache: Dict[str, Path] = {}

    def get_or_compile(params: Dict[str, object]) -> Path:
        key = params_key(params)
        if key in compiled_cache and compiled_cache[key].exists() \
                and compiled_cache[key].stat().st_mtime >= src_mtime:
            return compiled_cache[key]
        bh = params_hash(params)
        out_bin = bins_dir / f"bot_merged_{bh}"
        if not out_bin.exists() or out_bin.stat().st_mtime < src_mtime:
            compile_bot(src, out_bin, params)
        compiled_cache[key] = out_bin
        return out_bin

    def evaluate_and_record(gen: int, idx: int, params: Dict[str, object], seeds: List[int]) -> Dict[str, object]:
        t0 = time.time()
        bin_path = get_or_compile(params)
        metrics = evaluate_candidate_vs_pool(ref_bin, bin_path, opp_bins, seeds=seeds, workers=args.workers)
        row = {
            "ts": time.time(),
            "generation": gen,
            "index": idx,
            "params": params,
            "metrics": metrics,
            "bin": str(bin_path),
            "seeds": len(seeds),
            "games": int(metrics["games"]),
            "elapsed_sec": round(time.time() - t0, 3),
        }
        log_jsonl(log_path, row)
        return row

    pop: List[Dict[str, object]] = []
    pop.append(base_params)
    for _ in range(max(0, args.population - 1)):
        if rng.random() < 0.5:
            pop.append(mutate_candidate(rng, base_params, strength=rng.uniform(0.6, 1.4)))
        else:
            pop.append(random_candidate(rng))

    generation = 0
    best_row = None

    while True:
        generation += 1
        eval_seeds = stable_seed_block(rng, args.seeds_per_eval)
        print(
            f"[gen {generation}] evaluating population={len(pop)} with {2 * len(eval_seeds)} games/candidate",
            flush=True,
        )

        scored: List[Dict[str, object]] = []
        for i, cand in enumerate(pop):
            try:
                row = evaluate_and_record(generation, i, cand, eval_seeds)
                m = row["metrics"]
                per_opp_str = "  ".join(
                    f"{o['name']}:{o['win_rate']:.2f}"
                    for o in m.get("per_opponent", [])
                )
                print(
                    f"  cand {i:02d} WR={m['win_rate']:.3f} "
                    f"W={m['wins']} T={m['ties']} L={m['losses']} "
                    f"pts={m['avg_score_diff']:+.1f} loss_adv={m['avg_losses_diff']:+.1f} "
                    f"turns={m['avg_turns']:.1f} fit={m['fitness']:.0f}"
                    + (f"  [{per_opp_str}]" if per_opp_str else ""),
                    flush=True,
                )
                scored.append(row)
            except Exception as e:
                print(f"  cand {i:02d} failed: {e}", flush=True)

        if not scored:
            raise RuntimeError("No candidates evaluated successfully in this generation.")

        scored.sort(
            key=lambda r: (
                float(r["metrics"]["fitness"]),
                float(r["metrics"]["win_rate"]),
                float(r["metrics"]["avg_score_diff"]),
            ),
            reverse=True,
        )

        gen_best = scored[0]
        if best_row is None:
            best_row = gen_best
        else:
            cur = gen_best["metrics"]
            bst = best_row["metrics"]
            if (
                (cur["fitness"], cur["win_rate"], cur["avg_score_diff"]) >
                (bst["fitness"], bst["win_rate"], bst["avg_score_diff"])
            ):
                best_row = gen_best

        top = gen_best
        tm = top["metrics"]
        print(
            f"[gen {generation}] best WR={tm['win_rate']:.3f} fit={tm['fitness']:.1f} "
            f"diff={tm['avg_score_diff']:.2f}",
            flush=True,
        )

        verify_ok = True
        verify_rows = []
        for vr in range(args.verify_rounds):
            verify_seeds = stable_seed_block(rng, args.verify_seeds)
            vm = evaluate_candidate_vs_pool(
                ref_bin,
                Path(top["bin"]),
                opp_bins,
                seeds=verify_seeds,
                workers=args.workers,
            )
            verify_rows.append(vm)
            per_opp_str = "  ".join(
                f"{o['name']}:{o['win_rate']:.2f}"
                for o in vm.get("per_opponent", [])
            )
            print(
                f"  verify {vr + 1}/{args.verify_rounds}: WR={vm['win_rate']:.3f} "
                f"fit={vm['fitness']:.1f} diff={vm['avg_score_diff']:.2f} "
                f"games={int(vm['games'])}"
                + (f"  [{per_opp_str}]" if per_opp_str else ""),
                flush=True,
            )
            if vm["win_rate"] < args.target_winrate:
                verify_ok = False

        payload = {
            "updated_at": time.time(),
            "target_winrate": args.target_winrate,
            "best_eval": best_row,
            "latest_generation": generation,
            "latest_gen_best": gen_best,
            "latest_verify": verify_rows,
            "baseline_bin": str(baseline_bin),
            "ref_bin": str(ref_bin),
            "compile_command": format_compile_command(
                src,
                bins_dir / "bot_merged_production",
                best_row["params"],
            ),
        }
        best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        if verify_ok:
            print("[success] candidate passed all verification rounds above target win rate", flush=True)
            print("[success] final params:", json.dumps(top["params"], sort_keys=True), flush=True)
            return 0

        if args.max_generations > 0 and generation >= args.max_generations:
            print("[stop] reached max generations without satisfying target", flush=True)
            print("[best-so-far] params:", json.dumps(best_row["params"], sort_keys=True), flush=True)
            return 2

        elite_n = max(1, min(args.elite, len(scored)))
        elites = [scored[i]["params"] for i in range(elite_n)]

        new_pop: List[Dict[str, object]] = []
        for e in elites:
            new_pop.append(dict(e))

        while len(new_pop) < args.population:
            if rng.random() < 0.15:
                new_pop.append(random_candidate(rng))
                continue

            pa = rng.choice(elites)
            pb = rng.choice(elites)
            child = crossover(rng, pa, pb)
            child = mutate_candidate(rng, child, strength=rng.uniform(0.5, 1.3))
            new_pop.append(child)

        pop = new_pop

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)