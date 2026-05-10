#!/bin/bash
# ============================================================
# clean_submission.sh — Prepare Ukraine Counterfactual for Hand-in
# ============================================================
# Usage:   cd ukraine_counterfactual && bash clean_submission.sh
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_NAME="ukraine_counterfactual_submission.zip"
ZIP_PATH="$PROJECT_DIR/../$ZIP_NAME"

echo "=== Cleaning repository for submission ==="

# Remove cache directories
echo "  Removing cache directories..."
rm -rf "$PROJECT_DIR/.mypy_cache"
rm -rf "$PROJECT_DIR/.pytest_cache"
rm -rf "$PROJECT_DIR/.matplotlib"
rm -rf "$PROJECT_DIR/.claude"
rm -rf "$PROJECT_DIR/.venv"
rm -rf "$PROJECT_DIR/__MACOSX"

# Remove OS metadata files
echo "  Removing OS metadata files..."
find "$PROJECT_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true

# Remove Python bytecode
echo "  Removing Python bytecode..."
find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR" -name "*.pyc" -type f -delete 2>/dev/null || true
find "$PROJECT_DIR" -name "*.pyo" -type f -delete 2>/dev/null || true

# Remove notebook checkpoints if any
echo "  Removing notebook checkpoints..."
find "$PROJECT_DIR" -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove any temporary logs
echo "  Removing temporary logs..."
find "$PROJECT_DIR" -name "*.log" -type f -delete 2>/dev/null || true

# Remove any existing submission zips
echo "  Removing existing submission zips..."
rm -f "$PROJECT_DIR"/*.zip
rm -f "$ZIP_PATH"

echo "=== Creating submission zip ==="
cd "$PROJECT_DIR/.."
zip -r "$ZIP_NAME" \
    ukraine_counterfactual/scripts/ \
    ukraine_counterfactual/data/ \
    ukraine_counterfactual/figures/ \
    ukraine_counterfactual/outputs/ \
    ukraine_counterfactual/README.md \
    ukraine_counterfactual/requirements.txt \
    ukraine_counterfactual/run.sh \
    ukraine_counterfactual/clean_submission.sh \
    ukraine_counterfactual/Part_A_Ukraine_Monetary_Regime.md \
    ukraine_counterfactual/Part_A_Ukraine_Monetary_Regime.docx \
    ukraine_counterfactual/Part_B_counterfactual_interpretation.md \
    ukraine_counterfactual/.gitignore \
    ukraine_counterfactual/LICENSE \
    -x "*/.git/*" \
    -x "*/.venv/*" \
    -x "*/__MACOSX/*" \
    -x "*/.claude/*" \
    -x "*/.pytest_cache/*" \
    -x "*/.matplotlib/*" \
    -x "*.DS_Store" \
    -x "*__pycache__*" \
    -x "*/.ipynb_checkpoints/*" \
    -x "*.pyc" \
    -x "*.pyo" \
    -x "*.log" \
    -x "*.zip"

ZIP_SIZE=$(ls -lh "$ZIP_PATH" | awk '{print $5}')

echo ""
echo "=== Done ==="
echo "  Submission zip: $(dirname "$ZIP_PATH")/$ZIP_NAME"
echo "  Size: $ZIP_SIZE"
echo ""
echo "Contents:"
unzip -l "$ZIP_PATH" | tail -n +4 | sed '$d' | sed '$d'
