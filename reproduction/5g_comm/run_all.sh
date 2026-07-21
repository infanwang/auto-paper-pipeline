#!/bin/bash
# Run all 5G communication paper reproductions
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Running 5G Communication Paper Reproductions..."
echo "================================================"

for script in "$DIR"/reproduce_*.py; do
    echo ""
    echo "Running: $(basename $script)"
    python3 "$script" 2>&1 || echo "WARN: $script exited with code $?"
done

echo ""
echo "All scripts completed."
