#!/bin/bash
# ============================================================
# Ukraine Counterfactual Inflation — Run All Steps
# ============================================================
# Usage:   cd ukraine_counterfactual && bash run.sh
# Prereqs: pip install -r requirements.txt
#
# Bootstrap modes:
#   Default (fast):    n_boot = 100
#   Full:              FULL_BOOTSTRAP=1 bash run.sh   (n_boot = 500)
# ============================================================

set -e

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

# Create output directories
mkdir -p outputs figures data

echo "=== Step 1: Data Pipeline ==="
"$PYTHON_BIN" scripts/step1_data_pipeline.py

echo ""
echo "=== Step 2: Counterfactual (main SVAR + robustness checks) ==="
"$PYTHON_BIN" scripts/step2_counterfactual.py

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  data/data_clean_panel.csv                   — merged inflation panel"
echo "  data/data_counterfactual_results.csv         — counterfactual series + decompositions"
echo "  data/data_ascm_weights.csv                   — synthetic control donor weights"
echo "  figures/fig_counterfactual_main_svar.png     — MAIN result (SVAR)"
echo "  figures/fig_counterfactual_robustness.png    — robustness comparison"
echo "  figures/fig_svar_stationarity_robustness.png — stationarity robustness"
echo "  outputs/svar_diagnostics.csv                 — baseline SVAR diagnostics"
echo "  outputs/svar_diagnostics.json                — baseline SVAR diagnostics (JSON)"
echo "  outputs/svar_stationary_robustness_diagnostics.csv  — robustness SVAR diagnostics"
echo "  outputs/svar_stationary_robustness_diagnostics.json — robustness SVAR diagnostics (JSON)"
echo "  outputs/lp_diagnostics.csv                   — reduced-form projection coefficients"
echo "  outputs/model_summary.md                     — methodological hierarchy + flags"
echo "  Part_A_Ukraine_Monetary_Regime.docx          — regime chronology"
echo "  Part_B_counterfactual_interpretation.md      — written interpretation"
