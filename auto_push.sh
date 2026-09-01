#!/bin/bash
# Auto-push experiment results to GitHub
# Commits result JSONs, logs, and updated PROJECT_STATUS.md
# Excludes large binary files (images, model checkpoints, cluster caches)

set -e
cd /root/ICLR2027_KD

# Configure git if not already done
git config user.name "llmnjust-afk" 2>/dev/null || true
git config user.email "llmnjust-afk@users.noreply.github.com" 2>/dev/null || true

# Ensure remote URL has PAT embedded for authentication
PAT=""
git remote set-url origin "https://llmnjust-afk:${PAT}@github.com/llmnjust-afk/ICLR2027_KD.git" 2>/dev/null || true

echo "=== Git status ==="
git status --short

# Add result JSONs (not images or caches)
echo "=== Adding result files ==="
# Add sweep result JSONs
git add results/sweep_in10/sweep_*.json 2>/dev/null || true
git add results/sweep_in100/sweep_*.json 2>/dev/null || true
git add results/experiment_summary.json 2>/dev/null || true

# Add logs
git add logs/ 2>/dev/null || true

# Add any new scripts
git add run_all_experiments.sh collect_results.py auto_push.sh 2>/dev/null || true

# Add modified source files
git add duration_sweep.py ags/ 2>/dev/null || true

# Add PROJECT_STATUS.md if updated
git add PROJECT_STATUS.md 2>/dev/null || true

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo "Nothing to commit — no changes staged."
    exit 0
fi

echo "=== Committing ==="
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "experiment: AGS-DD CAGS tuning + fixed lambda search results (${TIMESTAMP})

- 10-class: fixed lambda search {0.0, 0.05, 0.08, 0.1, 0.12, 0.15}
- 10-class: CAGS tuning with ranges {(0.05,0.12), (0.05,0.15), (0.08,0.12)}
- 10-class: TAGS ablation {constant, cosine, linear, exponential}
- 100-class: unguided + fixed lambda=0.1 + CAGS configs
- All: 1000 epochs, 3 seeds, ConvNet-6, CutMix
- Auto-pushed by run_all_experiments.sh"

echo "=== Pushing to GitHub ==="
git push origin main 2>&1

echo "=== Push complete ==="
echo "Remote: $(git remote get-url origin | sed 's/ghp_[^@]*@/***@/')"
echo "Commit: $(git log -1 --oneline)"
