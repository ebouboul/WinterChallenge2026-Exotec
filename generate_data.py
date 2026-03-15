import subprocess
import random
import csv
import concurrent.futures
import time
import os
import glob

FEATURE_NAMES = [
    "score_diff", "bot_diff", "avg_my_dist", "avg_opp_dist",
    "safety_diff", "ground_diff", "mobility_diff", "territory_diff"
]
NUM_FEATURES = len(FEATURE_NAMES)

# 1. Compile the bot WITH data generation enabled
print("Compiling bot with -DDATA_GEN...")
subprocess.run(["g++", "-Ofast", "-DDATA_GEN", "bot.cpp", "-o", "bot_data_gen"], check=True)
print("Compilation OK!")

# 2. Create the Python wrapper to intercept stderr
wrapper_code = """import sys
import subprocess
match_id = sys.argv[1]
bot_exe = sys.argv[2]
with open(f"{match_id}.txt", "a") as f:
    subprocess.run([bot_exe], stdin=sys.stdin, stdout=sys.stdout, stderr=f)
"""
with open("wrapper.py", "w") as f:
    f.write(wrapper_code)

# 3. Clean up old match data files
for f in glob.glob("match_data_*.txt"):
    os.remove(f)

NUM_MATCHES = 500
MAX_THREADS = 16

def play_match_for_data(match_id):
    random_seed = random.randint(-9000000000000000000, 9000000000000000000)
    
    p0_cmd = f"python3 wrapper.py match_data_{match_id}_0 ./bot_data_gen"
    p1_cmd = f"python3 wrapper.py match_data_{match_id}_1 ./bot_data_gen"
    
    cmd = [
        "mvn", "-q", "exec:java",
        "-Dexec.mainClass=Main",
        "-Dexec.classpathScope=test",
        f"-Dexec.args='{p0_cmd}' '{p1_cmd}' {random_seed}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    winner = -1
    if "WINNER: 0" in output:
        winner = 0
    elif "WINNER: 1" in output:
        winner = 1
        
    if winner == -1:
        for p in [0, 1]:
            fname = f"match_data_{match_id}_{p}.txt"
            if os.path.exists(fname):
                os.remove(fname)
        return []
        
    dataset_rows = []
    for p in [0, 1]:
        fname = f"match_data_{match_id}_{p}.txt"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                lines = f.readlines()
            
            for line in lines:
                if "NN_DATA_P" in line:
                    try:
                        data_string = line.split(":", 1)[1].strip()
                        features = data_string.split(",")
                        
                        label = 1 if winner == p else 0
                        
                        if len(features) == NUM_FEATURES:
                            dataset_rows.append(features + [label])
                    except Exception:
                        pass
            
            os.remove(fname)
            
    return dataset_rows

if __name__ == "__main__":
    print(f"Generating data from {NUM_MATCHES} self-play matches (strong bot vs itself)...")
    print(f"Features: {FEATURE_NAMES}")
    all_data = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(play_match_for_data, i) for i in range(NUM_MATCHES)]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            rows = future.result()
            all_data.extend(rows)
            print(f"\rMatches: {completed}/{NUM_MATCHES} | Data points: {len(all_data)}", end="", flush=True)
            
    print(f"\nDone in {time.time() - start_time:.1f} seconds!")
    
    if len(all_data) > 0:
        with open("dataset.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES + ["label"])
            writer.writerows(all_data)
        print(f"Saved {len(all_data)} rows to dataset.csv with {NUM_FEATURES} features. Ready for training!")
    else:
        print("\n[ERROR] 0 data points collected. Check that the Java runner works.")