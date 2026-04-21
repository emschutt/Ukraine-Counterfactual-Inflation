# Ukraine Counterfactual Inflation Exam

This repository contains a reproducible take-home exam submission for the question: *What would Ukraine's inflation path have looked like under Euro Area membership?*

## Deliverables

- `Part_A_Ukraine_Monetary_Regime.docx` — documented chronology of Ukraine's monetary regime (2000–2025), including the summary table and sovereignty argument required in Part A.
- `figures/fig_counterfactual_main.png` — main Part B figure comparing actual and counterfactual inflation.
- `figures/fig_counterfactual_methods.png` — method-by-method comparison (LP, SVAR, ASCM, factor benchmark).
- `Part_B_counterfactual_interpretation.md` — written interpretation for Part B.

## Reproducibility

Install dependencies and run:

```bash
pip install -r requirements.txt
bash run.sh
```

The pipeline produces:

- `data/data_clean_panel.csv` — merged 12-country inflation panel
- `data/data_external_macro.csv` — external macro series (exchange rates, policy rates, commodities, industrial production)
- `data/data_extended_hicp_panel.csv` — expanded 24-country HICP donor panel for synthetic control
- `data/data_counterfactual_results.csv` — all counterfactual series + confidence intervals
- `data/data_ascm_weights.csv` — synthetic control donor weights
- `figures/` — all output figures

## External Data and Why They Matter

All external series are downloaded programmatically in `scripts/external_data.py` and cached under `data/external_cache/`. To force a fresh download, run with `REFRESH_EXTERNAL_DATA=1 bash run.sh`.

- **ECB main refinancing rate**: monetary-policy spread between Ukraine and the Euro Area.
- **ECB industrial production**: Euro Area output growth for the bivariate structural VAR (Blanchard-Quah identification).
- **NBU key policy rate**: Ukraine's domestic monetary stance relative to the ECB.
- **NBU UAH/USD and UAH/EUR exchange rates**: exchange-rate adjustment and devaluation episodes.
- **Ukraine industrial production** (NBU macro workbook): Ukraine output series for the structural VAR.
- **FRED Brent crude, wheat, and European natural gas prices**: external supply-side controls in the local projection benchmark.
- **Extended ECB HICP panel** (24 countries): synthetic-control donor pool beyond the base 11-country panel.

## Method Summary

- **Part A** establishes that Euro Area membership is a time-varying treatment: large during devaluation crises and the post-2016 inflation-targeting period, smaller during de facto peg periods.
- **Part B** combines four perspectives:
  1. A Euro Area common-factor benchmark (Ciccarelli and Mojon, 2010)
  2. A projection-based benchmark calibrated on stable periods with commodity controls (Jorda, 2005)
  3. A Blanchard-Quah structural VAR with demand-shock replacement (Bayoumi and Eichengreen, 1993)
  4. An augmented synthetic control with an expanded donor pool (Abadie et al., 2010)
- The main counterfactual is the **median ensemble** of the LP, SVAR, and synthetic-control series, with 90% bootstrap confidence intervals from the SVAR.
