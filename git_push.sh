#!/bin/bash

# MorphoFlow Git Push Script
# Usage: ./git_push.sh "your commit message"
#    or: ./git_push.sh  (will prompt for message)

cd "$(dirname "$0")"

# Check if repo is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
    git remote add origin git@github.com:Louis-Leee/MorphoFlow.git
fi

# Get commit message
if [ -z "$1" ]; then
    echo -n "Enter commit message: "
    read COMMIT_MSG
else
    COMMIT_MSG="$1"
fi

# Default message if empty
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="update $(date '+%Y-%m-%d %H:%M:%S')"
fi

# Git operations
echo "=========================================="
echo "Adding files..."
git add .

echo "=========================================="
echo "Commit message: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

echo "=========================================="
echo "Pushing to origin/main..."
git push -u origin main

echo "=========================================="
echo "Done!"
