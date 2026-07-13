#!/usr/bin/env bash
# =====================================================================
# Persist experiment RESULTS back to GitHub from Colab (or anywhere).
# Commits ONLY the small result JSONs (+ optional run logs). Never
# datasets, checkpoints, PDFs, images or secrets.
#
# Auth: pass a GitHub Personal Access Token via the environment, never
# on the command line or in a file:
#     export GITHUB_TOKEN=ghp_xxx           # a fine-grained PAT with
#                                           # "Contents: read/write" on this repo
#     export GIT_AUTHOR_NAME="Pratham Kailasiya"
#     export GIT_AUTHOR_EMAIL="you@example.com"
#     bash push_results.sh                  # -> pushes to branch results/auto
#
# The token is used only to build the remote URL for a single push and is
# NOT written to disk or the git config.
# =====================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BRANCH="${RESULTS_BRANCH:-results/auto}"
REPO_SLUG="prathamkailashya/deep-hedging"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: set GITHUB_TOKEN (fine-grained PAT, Contents:write) in the environment." >&2
  exit 1
fi

git config user.name  "${GIT_AUTHOR_NAME:-colab-runner}"
git config user.email "${GIT_AUTHOR_EMAIL:-colab@localhost}"

git checkout -B "$BRANCH"

# Stage ONLY result artefacts (whitelist; never data/checkpoints/secrets).
shopt -s nullglob
ARTIFACTS=( *_results.json *_bootstrap.json *_validation_results.json \
            walk_forward_ci_results.json experiment_manifest.json seeds.json )
if compgen -G "results/logs/*.log" > /dev/null; then ARTIFACTS+=( results/logs/*.log ); fi
git add -f "${ARTIFACTS[@]}" 2>/dev/null || true

if git diff --cached --quiet; then
  echo "[push_results] no new result artefacts to commit."; exit 0
fi

git commit -m "results: automated experiment artefacts ($(date -u +%Y-%m-%dT%H:%M:%SZ))"

# One-shot authenticated push; token never persisted to config.
REMOTE="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git"
git push "$REMOTE" "$BRANCH" 2>&1 | sed "s/${GITHUB_TOKEN}/***/g"
echo "[push_results] pushed result artefacts to branch $BRANCH"
