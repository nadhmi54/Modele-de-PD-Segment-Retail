#!/bin/bash
# -----------------------------------------------------------------------------
# update.sh - Switch to develop and pull latest from origin.
#
# Use when:  Syncing your local develop branch before starting new work.
# Do NOT use: To create/push a new feature branch (use new-push.sh instead).
# Remote:     origin
# Branch:     develop (integration branch for ContrAgri)
# -----------------------------------------------------------------------------

set -e

if ! git show-ref --verify --quiet refs/heads/develop; then
  echo "[ERROR] Branch 'develop' does not exist locally." >&2
  echo "[HINT]  Run: git fetch origin && git switch develop" >&2
  exit 1
fi

echo "[INFO] Switching to 'develop'..."
git switch develop

echo "[INFO] Pulling from origin/develop..."
git pull origin develop

echo "[SUCCESS] 'develop' is up to date with origin."
exit 0