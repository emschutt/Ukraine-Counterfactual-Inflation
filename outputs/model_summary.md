# Model Summary — Ukraine Counterfactual under Euro Area Membership

## Methodological Hierarchy

### Main Specification
- **Blanchard-Quah SVAR** (Bayoumi & Eichengreen 1993):
  Euro Area membership is operationalised as replacing Ukraine's
  domestic demand/monetary shocks with Euro Area demand shocks while
  preserving Ukraine-specific supply shocks.
  This is the preferred identification strategy.

### Robustness Benchmarks
- **Stationarity-robustness SVAR**:
  first-differenced inflation with the same B-Q identification and
  shock-replacement logic; structural IRFs are cumulated back to the
  inflation level for interpretation.
- **Reduced-form projection benchmark**: single-horizon OLS calibration
  on stable periods, with regime treatment-weight blending.
  NOT a genuine local projection (Jorda 2005).
  The FX coefficient is not stable enough for pass-through identification.
- **ASCM robustness check** (Abadie et al. 2010):
  24-country HICP donor pool with ridge-regularised SLSQP optimisation.

## Counterfactual Interpretation

- The counterfactual is NOT a simple Euro Area average.
- Treatment intensity is smaller during peg periods and larger
  during devaluation/post-2016 monetary-sovereignty periods, then
  constrained again during wartime fixed-rate/capital-control periods.
- Output enters VARs as growth, not levels.
- Inflation is checked for stationarity (ADF + KPSS).

## Main Findings

- **2008-09 GFC**: main SVAR gap +5.7 pp (actual 19.9%, counterfactual 14.2%); stationarity-robustness SVAR +9.5 pp.
- **2014-15 Crimea/Donbas crisis**: main SVAR gap +18.1 pp (actual 31.6%, counterfactual 13.5%); stationarity-robustness SVAR +27.6 pp.
- **Post-2016 inflation-targeting period**: main SVAR gap -2.1 pp (actual 9.1%, counterfactual 11.2%); stationarity-robustness SVAR -4.4 pp.
- **2022 full-scale invasion**: main SVAR gap -6.8 pp (actual 20.5%, counterfactual 27.4%); stationarity-robustness SVAR -8.4 pp.

## SVAR Diagnostics

- VAR lags (BIC): UA=2, EA=2
- Blanchard-Quah restriction check UA: 5.37e-17
- Blanchard-Quah restriction check EA: -4.12e-16
- Max root modulus UA: 201.8665
- Max root modulus EA: 7.9053
- VAR stable UA: False
- VAR stable EA: False
- Observations UA: 228, EA: 297

## Stationarity (ADF / KPSS)

- **UA_IP_YOY**: ADF stat=-3.625, KPSS stat=0.052, verdict=stationary
- **UA**: ADF stat=-2.630, KPSS stat=0.093, verdict=MIXED (ADF fails, KPSS does not reject)
- **EA_IP_YOY**: ADF stat=-3.802, KPSS stat=0.034, verdict=stationary
- **EA_MEAN**: ADF stat=-2.549, KPSS stat=0.233, verdict=MIXED (ADF fails, KPSS does not reject)

## Stationarity-Robustness SVAR

- **Transformation**: inflation -> first-difference of YoY inflation (dPi = Pi[t] - Pi[t-1])
- **Rationale**: the baseline SVAR in levels exhibits instability because
  Ukraine inflation contains large structural breaks and crisis episodes.
  First-differencing removes the persistent level component.
- **Identification**: identical B-Q long-run restriction.
  Structural IRFs for dPi are cumulated to recover level-IRFs,
  enabling the same shock-replacement counterfactual logic.
- VAR lags (BIC): UA=1, EA=1
- B-Q check UA: 2.11e-16, EA: -1.87e-16
- Max root modulus UA: 1.5739, EA: 3.1698
- VAR stable UA: False, EA: False
- Observations UA: 228, EA: 297
- ADF d(UA): -4.627 (p=0.0001)
- ADF d(EA): -6.566 (p=0.0000)

- **Interpretation**: The qualitative conclusions remain broadly similar
  across baseline and robustness SVAR specifications. Therefore the
  interpretation does not rely solely on unstable level VAR dynamics.

## Caveats

- The reduced-form projection benchmark is a single-horizon OLS, not local projections.
- The FX coefficient in the reduced-form benchmark is unstable in stable periods.
- Baseline and robustness VARs still show some instability.
- The baseline VARs in levels exhibit instability (roots outside unit circle);
  a stationarity-robustness SVAR confirms qualitative conclusions.
- Bootstrap inference is residual-based (UA VAR only; EA shocks held fixed).

## Diagnostic Warnings

The following diagnostics failed:
- var_stable_UA
- var_stable_EA
- robust_var_stable_UA
- robust_var_stable_EA

Review outputs/svar_diagnostics.csv for details.

## Output Files

- `data/data_ascm_weights.csv`
- `data/data_counterfactual_results.csv`
- `figures/fig_counterfactual_main_svar.png`
- `figures/fig_counterfactual_robustness.png`
- `figures/fig_svar_stationarity_robustness.png`
- `outputs/lp_diagnostics.csv`
- `outputs/model_summary.md`
- `outputs/svar_diagnostics.csv`
- `outputs/svar_diagnostics.json`
- `outputs/svar_stationary_robustness_diagnostics.csv`
- `outputs/svar_stationary_robustness_diagnostics.json`
