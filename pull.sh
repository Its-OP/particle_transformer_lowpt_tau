#!/bin/bash
# =============================================================================
# Pull latest changes for the part repository.
#
# Usage:
#   bash pull.sh              # pull from master
#   bash pull.sh my-branch    # pull from a specific branch
# =============================================================================
set -euo pipefail

BRANCH="${1:-master}"

echo "Pulling branch '${BRANCH}' for part..."
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"
echo "Done. part is now on branch '${BRANCH}'."
