import sys
import subprocess
match_id = sys.argv[1]
bot_exe = sys.argv[2]
with open(f"{match_id}.txt", "a") as f:
    subprocess.run([bot_exe], stdin=sys.stdin, stdout=sys.stdout, stderr=f)
