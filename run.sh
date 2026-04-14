#!/bin/bash
# ============================================================
# Ukraine Counterfactual Inflation — Run All Steps
# ============================================================
# Usage:   cd ukraine_counterfactual && bash run.sh
# Prereqs: pip install -r requirements.txt
# ============================================================

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "=== Step 1: Data Pipeline ==="
"$PYTHON_BIN" scripts/step1_data_pipeline.py

echo ""
echo "=== Step 2: Counterfactual (SVAR + LP + donor benchmarks) ==="
"$PYTHON_BIN" scripts/step2_counterfactual.py

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  data/data_clean_panel.csv             — merged inflation panel"
echo "  data/data_counterfactual_results.csv   — counterfactual series + CIs"
echo "  figures/                               — all figures"
echo "  Part_A_Ukraine_Monetary_Regime.docx    — regime chronology"
