#!/bin/bash
# -----------------------------------------------------------------------------
# new-push.sh - Create a new branch, commit all changes, and push to origin.
#
# Use when:  Starting a new feature/fix branch with your first commit.
# Do NOT use: To sync develop with remote (use update.sh instead).
# Remote:     origin
# -----------------------------------------------------------------------------

set -e

read -r -p "Enter new branch name: " BRANCH
BRANCH="${BRANCH#"${BRANCH%%[![:space:]]*}"}"
BRANCH="${BRANCH%"${BRANCH##*[![:space:]]}"}"

if [ -z "$BRANCH" ]; then
  echo "[ERROR] Branch name cannot be empty." >&2
  exit 1
fi

echo "[INFO] Creating branch '$BRANCH'..."
git checkout -b "$BRANCH"

echo "[INFO] Staging all changes..."
git add .

echo "[INFO] Creating initial commit..."
git commit -m "Initial commit on $BRANCH"

echo "[INFO] Pushing to origin/$BRANCH..."
git push origin "$BRANCH"

echo "[SUCCESS] Branch '$BRANCH' created and pushed to origin."
exit 0