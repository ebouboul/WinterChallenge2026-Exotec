#!/usr/bin/env bash

echo "Starting fresh: removing all git history..."

# 1. Delete the existing .git folder
rm -rf .git

# 2. Re-initialize the repository
git init

# 3. Ensure virtual environments are ignored
echo -e "\nvenv/\n.venv/" >> .gitignore

# 4. Add all files
git add .

# 5. Create the initial commit
git commit -m "Initial commit"

# 6. Set the default branch name to 'main'
git branch -M main

# 7. Add the remote back
git remote add origin git@github.com:ebouboul/WinterChallenge2026-Exotec.git

echo ""
echo "Done! The git repository has been recreated from scratch."
echo "You can now force push the new single commit to GitHub:"
echo "    git push -u origin main -f"
