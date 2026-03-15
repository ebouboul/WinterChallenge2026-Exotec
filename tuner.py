import subprocess
import random
import time
import concurrent.futures
import copy
import sys

# The baseline weights we are starting with
best_weights = {
    "W_SCORE": 0.40,
    "W_BOT": 0.30,
    "W_DIST_MY": 0.05,
    "W_DIST_OPP": 0.05,
    "W_SAFETY": 0.10,
    "W_GROUND": 0.15,
    "W_MOB": 0.02,
    "W_TERR": 0.15
}

MATCHES_PER_GENERATION = 20  # Must be an even number to swap Player 1 / Player 2 sides
MAX_THREADS = 8

def compile_bot(weights, output_name):
    # Inject the weights into the g++ compiler flags
    flags = ["g++", "-Ofast", "bot.cpp", "-o", output_name]
    for key, value in weights.items():
        flags.append(f"-D{key}={value:.5f}")
    
    subprocess.run(flags, check=True)

def play_match(seed, p0_bot, p1_bot):
    cmd = [
        "mvn", "-q", "exec:java",
        "-Dexec.mainClass=Main",
        "-Dexec.classpathScope=test",
        f"-Dexec.args='./{p0_bot}' './{p1_bot}' {seed}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    if "WINNER: 0" in output:
        return 0
    elif "WINNER: 1" in output:
        return 1
    return -1 # Tie or crash

def run_generation(mutant_weights):
    print("Compiling bots...")
    compile_bot(best_weights, "bot_best")
    compile_bot(mutant_weights, "bot_mutant")
    
    mutant_wins = 0
    best_wins = 0
    ties = 0
    
    print(f"Running {MATCHES_PER_GENERATION} Deathmatches...", end="", flush=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Tag each future with whether mutant was P0
        future_to_side = {}
        for i in range(MATCHES_PER_GENERATION):
            seed = random.randint(-9000000000000000000, 9000000000000000000)
            mutant_is_p0 = (i % 2 == 0)
            if mutant_is_p0:
                f = executor.submit(play_match, seed, "bot_mutant", "bot_best")
            else:
                f = executor.submit(play_match, seed, "bot_best", "bot_mutant")
            future_to_side[f] = mutant_is_p0
                
        completed = 0
        for future in concurrent.futures.as_completed(future_to_side):
            completed += 1
            winner = future.result()
            mutant_was_p0 = future_to_side[future]
            
            if winner == -1:
                ties += 1
            elif (winner == 0 and mutant_was_p0) or (winner == 1 and not mutant_was_p0):
                mutant_wins += 1
            else:
                best_wins += 1
                
            print(f"\rMatches: {completed}/{MATCHES_PER_GENERATION} | Mutant: {mutant_wins} | Baseline: {best_wins} | Ties: {ties}", end="", flush=True)

    print() # New line
    return mutant_wins, best_wins

if __name__ == "__main__":
    generation = 1
    while True:
        print(f"\n================ GENERATION {generation} ================")
        
        # 1. Mutate the weights
        mutant_weights = copy.deepcopy(best_weights)
        
        # Pick 1 to 3 random weights to mutate
        num_mutations = random.randint(1, 3)
        keys_to_mutate = random.sample(list(mutant_weights.keys()), num_mutations)
        
        for key in keys_to_mutate:
            # Shift the weight up or down by a random factor between -30% and +30%
            multiplier = random.uniform(0.7, 1.3)
            mutant_weights[key] = mutant_weights[key] * multiplier
            
        print("Mutated stats:")
        for key in keys_to_mutate:
            print(f"  {key}: {best_weights[key]:.4f} -> {mutant_weights[key]:.4f}")
            
        # 2. Battle
        mutant_wins, best_wins = run_generation(mutant_weights)
        
        # 3. Assess
        # The mutant must decisively beat the baseline to replace it
        if mutant_wins > best_wins:
            print(">>> NEW CHAMPION FOUND! <<<")
            best_weights = mutant_weights
            print("\nCopy these updated weights into your bot.cpp if you want to submit now:")
            for k, v in best_weights.items():
                print(f"#define {k} {v:.5f}")
        else:
            print("Mutant was weak. Discarding.")
            
        generation += 1