#!/usr/bin/env python3
"""
Part B -- Multi-Method Counterfactual Inflation Analysis
=======================================================

Counterfactual interpretation:
  1. Euro adoption removes the hryvnia devaluation channel.
  2. Euro adoption eliminates independent nominal-policy spread versus the ECB.
  3. Euro adoption imports credibility, reducing Ukraine's inflation premium.

Methods:
  - Factor benchmark  (Ciccarelli & Mojon 2010)
  - Local projections (Jorda 2005) -- stable-period calibration + regime weighting
  - Structural VAR    (Sims 1980; Kilian & Lutkepohl 2017) -- Cholesky identification
  - Augmented synthetic control (Abadie et al. 2010)
  - Median ensemble of LP / SVAR / ASCM
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

try:
    from statsmodels.tools.sm_exceptions import ValueWarning
except ImportError:
    ValueWarning = UserWarning


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ===================================================================
# Helpers
# ===================================================================

def fit_pca_standardized(df: pd.DataFrame):
    x = df.to_numpy(dtype=float)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    eigenvalues = (s ** 2) / (x.shape[0] - 1)
    explained_ratio = eigenvalues / eigenvalues.sum()
    scores = u * s
    return eigenvalues, explained_ratio, scores


def compute_ea_factor(ea_panel: pd.DataFrame):
    ea_clean = ea_panel.dropna()
    ea_z = (ea_clean - ea_clean.mean()) / ea_clean.std()
    eigenvalues, explained_ratio, scores = fit_pca_standardized(ea_z)
    factor_raw = pd.Series(scores[:, 0], index=ea_clean.index, name="EA_FACTOR_RAW")
    ea_mean = ea_clean.mean(axis=1)
    factor = (factor_raw - factor_raw.mean()) / factor_raw.std() * ea_mean.std() + ea_mean.mean()
    factor.name = "EA_FACTOR"
    return factor, eigenvalues, explained_ratio


def stable_period_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Mask selecting calm periods suitable for calibration (excluding crises)."""
    return (
        ((index >= "2001-01-01") & (index <= "2008-08-01"))
        | ((index >= "2010-01-01") & (index <= "2014-01-01"))
        | ((index >= "2016-01-01") & (index <= "2021-12-01"))
    )


# ===================================================================
# Regime map -- from Part A chronology
# ===================================================================

def build_regime_map(index: pd.Index) -> pd.DataFrame:
    idx = pd.to_datetime(index)
    regimes = pd.DataFrame(index=idx)
    regimes["regime"] = "peg_pre_2008"

    regimes.loc[(idx >= "2008-09-01") & (idx <= "2009-12-01"), "regime"] = "gfc_devaluation"
    regimes.loc[(idx >= "2010-01-01") & (idx <= "2014-01-01"), "regime"] = "repeg"
    regimes.loc[(idx >= "2014-02-01") & (idx <= "2015-07-01"), "regime"] = "crimea_float_crisis"
    regimes.loc[(idx >= "2015-08-01") & (idx <= "2022-01-01"), "regime"] = "inflation_targeting"
    regimes.loc[(idx >= "2022-02-01") & (idx <= "2023-09-01"), "regime"] = "wartime_fixed"
    regimes.loc[idx >= "2023-10-01", "regime"] = "managed_flexibility"

    # Channel weights: how much the euro treatment alters each channel per regime.
    # Low during peg periods (Ukraine already constrained), high during crises.
    fx_w = {"peg_pre_2008": 0.20, "gfc_devaluation": 1.00, "repeg": 0.20,
            "crimea_float_crisis": 1.00, "inflation_targeting": 0.70,
            "wartime_fixed": 0.15, "managed_flexibility": 0.50}
    pol_w = {"peg_pre_2008": 0.15, "gfc_devaluation": 0.65, "repeg": 0.15,
             "crimea_float_crisis": 0.85, "inflation_targeting": 0.80,
             "wartime_fixed": 0.10, "managed_flexibility": 0.55}
    cred_w = {"peg_pre_2008": 1.00, "gfc_devaluation": 1.00, "repeg": 0.95,
              "crimea_float_crisis": 0.90, "inflation_targeting": 0.45,
              "wartime_fixed": 0.35, "managed_flexibility": 0.40}

    regimes["fx_channel_weight"] = regimes["regime"].map(fx_w).astype(float)
    regimes["policy_channel_weight"] = regimes["regime"].map(pol_w).astype(float)
    regimes["credibility_gain_weight"] = regimes["regime"].map(cred_w).astype(float)
    regimes["euro_treatment_weight"] = (
        0.5 * regimes["fx_channel_weight"]
        + 0.3 * regimes["policy_channel_weight"]
        + 0.2 * regimes["credibility_gain_weight"]
    )
    regimes.index.name = "date"
    return regimes


# ===================================================================
# Method 1: Factor benchmark (Ciccarelli & Mojon 2010)
# ===================================================================

def factor_benchmark(df: pd.DataFrame) -> tuple[pd.Series, OLS]:
    """
    CF = fitted value from regressing UA on EA_FACTOR during stable periods.
    The intercept captures the average credibility premium.
    """
    mask = stable_period_mask(df.index) & df["UA"].notna() & df["EA_FACTOR"].notna()
    model = OLS(
        df.loc[mask, "UA"],
        add_constant(df.loc[mask, "EA_FACTOR"]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    X_full = add_constant(df["EA_FACTOR"].dropna())
    cf = pd.Series(model.predict(X_full), index=X_full.index, name="CF_factor_benchmark")
    return cf, model


# ===================================================================
# Method 2: Local projections (Jorda 2005)
# ===================================================================

def local_projection_counterfactual(df: pd.DataFrame) -> tuple[pd.Series, OLS]:
    """
    Stable-period calibration approach:

    1. Regress UA on [EA_FACTOR, BRENT_YOY] during stable periods
       (peg + re-peg + IT, excluding crisis windows). This estimates what
       Ukraine's inflation would look like tracking the EA common factor plus
       supply-side drivers (oil prices).

    2. Project out-of-sample for all periods to get cf_fitted -- the "Euro-
       Area-like" inflation path.

    3. Blend with regime weights:
       CF = (1 - treatment_weight) * UA + treatment_weight * cf_fitted

       During peg (low treatment ~0.2): CF ~ UA (small change)
       During crises (high treatment ~0.8-1.0): CF ~ cf_fitted (large change)

    This avoids endogeneity in FX pass-through estimation by estimating the
    counterfactual relationship during stable periods and applying it via
    regime weights from Part A.
    """
    controls = ["EA_FACTOR", "BRENT_YOY"]
    mask = (
        stable_period_mask(df.index)
        & df["UA"].notna()
        & df[controls].notna().all(axis=1)
    )

    model = OLS(
        df.loc[mask, "UA"],
        add_constant(df.loc[mask, controls]),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": 12})

    # Project for all periods
    full_mask = df[controls].notna().all(axis=1)
    cf_fitted = pd.Series(
        model.predict(add_constant(df.loc[full_mask, controls])),
        index=df.loc[full_mask].index,
    )

    # Blend: CF = (1 - treatment) * actual + treatment * fitted
    treatment = df["euro_treatment_weight"].reindex(cf_fitted.index)
    actual = df["UA"].reindex(cf_fitted.index)
    cf = (1 - treatment) * actual + treatment * cf_fitted
    cf.name = "CF_local_projections"
    return cf, model


# ===================================================================
# Method 3: Structural VAR (Sims 1980)
# ===================================================================

def svar_counterfactual(df: pd.DataFrame, maxlags: int = 6, ma_horizon: int = 12):
    """
    Cholesky ordering: [EA_FACTOR, BRENT_YOY, FX_DEPR, POLICY_SHOCK, UA]

    External/slow-moving variables first (EA common factor, oil prices),
    then faster financial variables (FX depreciation, policy shock),
    with Ukraine's inflation last. Consistent with small-open-economy SVAR
    orderings (Kilian & Lutkepohl 2017, ch. 10).

    The counterfactual removes only *inflationary* FX and policy structural
    shocks (clamped at zero). Under Euro membership Ukraine loses the
    devaluation channel and independent policy, but we do not reverse periods
    where those channels were disinflationary — the counterfactual should
    never exceed actual inflation.
    """
    var_cols = ["EA_FACTOR", "BRENT_YOY", "FX_DEPR", "POLICY_SHOCK", "UA"]
    sample = df[var_cols].dropna().copy()
    model = VAR(sample)
    selected = model.select_order(maxlags=maxlags).selected_orders["bic"]
    lag_order = max(1, selected or 1)
    res = model.fit(lag_order)

    sigma = res.sigma_u.to_numpy()
    chol = np.linalg.cholesky(sigma)
    residuals = res.resid.to_numpy()
    effective_index = res.resid.index
    sample_eff = sample.loc[effective_index]
    shocks = np.linalg.solve(chol, residuals.T).T

    ma = res.ma_rep(ma_horizon)
    structural_irf = np.array([ma_h @ chol for ma_h in ma])

    idx_ua = var_cols.index("UA")
    idx_fx = var_cols.index("FX_DEPR")
    idx_policy = var_cols.index("POLICY_SHOCK")

    fx_weight = df.loc[effective_index, "fx_channel_weight"].to_numpy()
    policy_weight = df.loc[effective_index, "policy_channel_weight"].to_numpy()

    removed_fx = np.zeros(len(sample_eff))
    removed_policy = np.zeros(len(sample_eff))

    for t in range(len(sample_eff)):
        for h in range(min(ma_horizon, t) + 1):
            source = t - h
            removed_fx[t] += structural_irf[h, idx_ua, idx_fx] * shocks[source, idx_fx] * fx_weight[source]
            removed_policy[t] += structural_irf[h, idx_ua, idx_policy] * shocks[source, idx_policy] * policy_weight[source]

    # Only remove inflationary contributions (positive = added inflation)
    removed_fx = np.clip(removed_fx, 0, None)
    removed_policy = np.clip(removed_policy, 0, None)
    removed_fx = pd.Series(removed_fx, index=effective_index, name="SVAR_removed_fx")
    removed_policy = pd.Series(removed_policy, index=effective_index, name="SVAR_removed_policy")
    cf = sample_eff["UA"] - removed_fx - removed_policy
    cf.name = "CF_svar"
    return cf, removed_fx, removed_policy, res


# ===================================================================
# Method 4: Augmented synthetic control (Abadie et al. 2010)
# ===================================================================

def augmented_synthetic_control(df: pd.DataFrame, donor_panel: pd.DataFrame):
    donors = donor_panel.copy().sort_index()
    donors = donors.loc[:, donors.notna().mean() >= 0.95]
    common_idx = df.index.intersection(donors.index)
    donors = donors.reindex(common_idx)
    target = df.reindex(common_idx)["UA"]

    calibration_mask = (
        target.notna() & donors.notna().all(axis=1)
        & stable_period_mask(common_idx)
    )

    X = donors.loc[calibration_mask].to_numpy(dtype=float)
    y = target.loc[calibration_mask].to_numpy(dtype=float)
    n_donors = X.shape[1]
    ridge = 1e-4

    def objective(w):
        return np.mean((y - X @ w) ** 2) + ridge * np.sum(w ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.repeat(1.0 / n_donors, n_donors)
    opt = minimize(objective, w0, bounds=bounds, constraints=cons, method="SLSQP")
    weights = pd.Series(opt.x, index=donors.columns, name="weight")
    cf = (donors @ weights).rename("CF_augmented_synthetic_control")
    return cf, weights.sort_values(ascending=False)


# ===================================================================
# Ensemble + diagnostics
# ===================================================================

def median_counterfactual(series_list: list[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=1).median(axis=1).rename("CF_main")


def run_stationarity_tests(df: pd.DataFrame, var_cols: list[str]):
    print("\nStationarity diagnostics (ADF test, 5% critical value ~ -2.86):")
    for col in var_cols:
        series = df[col].dropna()
        if len(series) < 20:
            continue
        stat, pval = adfuller(series, maxlag=12, autolag="AIC")[:2]
        verdict = "stationary" if pval < 0.05 else "NON-STATIONARY"
        print(f"  {col:<18s} ADF={stat:+.3f}  p={pval:.4f}  -> {verdict}")
    print("  Note: VAR in levels valid for IRF (Sims, Stock & Watson 1990).")


def svar_bootstrap_ci(df, svar_result, var_cols,
                      n_boot=500, alpha=0.10, ma_horizon=12):
    """Residual bootstrap for SVAR counterfactual confidence bands."""
    rng = np.random.default_rng(42)
    effective_index = svar_result.resid.index
    lag_order = svar_result.k_ar
    fitted_values = svar_result.fittedvalues.to_numpy()
    residuals = svar_result.resid.to_numpy()
    n_obs = len(residuals)

    fx_weight = df.loc[effective_index, "fx_channel_weight"].to_numpy()
    policy_weight = df.loc[effective_index, "policy_channel_weight"].to_numpy()
    idx_ua, idx_fx, idx_policy = var_cols.index("UA"), var_cols.index("FX_DEPR"), var_cols.index("POLICY_SHOCK")

    boot_cfs = np.full((n_boot, n_obs), np.nan)
    for b in range(n_boot):
        boot_resid = residuals[rng.integers(0, n_obs, size=n_obs)]
        boot_df = pd.DataFrame(fitted_values + boot_resid, index=effective_index, columns=var_cols)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ValueWarning)
                bm = VAR(boot_df).fit(lag_order)
        except Exception:
            continue
        try:
            chol_b = np.linalg.cholesky(bm.sigma_u.to_numpy())
        except np.linalg.LinAlgError:
            continue
        shocks_b = np.linalg.solve(chol_b, bm.resid.to_numpy().T).T
        sirf_b = np.array([m @ chol_b for m in bm.ma_rep(ma_horizon)])
        n_eff = len(bm.resid)
        rem_fx = np.zeros(n_eff)
        rem_pol = np.zeros(n_eff)
        for t in range(n_eff):
            for h in range(min(ma_horizon, t) + 1):
                s = t - h
                rem_fx[t] += sirf_b[h, idx_ua, idx_fx] * shocks_b[s, idx_fx] * fx_weight[s]
                rem_pol[t] += sirf_b[h, idx_ua, idx_policy] * shocks_b[s, idx_policy] * policy_weight[s]
        removed = np.clip(rem_fx, 0, None) + np.clip(rem_pol, 0, None)
        boot_actual = boot_df.iloc[lag_order:, idx_ua].to_numpy()
        boot_cfs[b, lag_order:lag_order + n_eff] = boot_actual - removed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanpercentile(boot_cfs, 100 * alpha / 2, axis=0)
        hi = np.nanpercentile(boot_cfs, 100 * (1 - alpha / 2), axis=0)
    return (pd.Series(lo, index=effective_index, name="CF_svar_ci_lo"),
            pd.Series(hi, index=effective_index, name="CF_svar_ci_hi"))


# ===================================================================
# MAIN EXECUTION
# ===================================================================

panel = pd.read_csv(os.path.join(DATA_DIR, "data_clean_panel.csv"), index_col=0, parse_dates=True)
panel.index.name = "date"
macro = pd.read_csv(os.path.join(DATA_DIR, "data_external_macro.csv"), index_col=0, parse_dates=True)
macro.index.name = "date"
donor_panel = pd.read_csv(os.path.join(DATA_DIR, "data_extended_hicp_panel.csv"), index_col=0, parse_dates=True)
donor_panel.index.name = "date"

EA_COLS = [c for c in panel.columns if c != "UA"]
ea = panel[EA_COLS].copy()

ea_factor, eigenvalues, explained_ratio = compute_ea_factor(ea)
regimes = build_regime_map(panel.index)

analysis = panel.join(ea_factor, how="left").join(macro, how="left").join(regimes, how="left")
analysis["EA_MEAN"] = ea.mean(axis=1)
analysis["UA_IP_GAP"] = analysis["UA_IP_YOY"] - 100.0
analysis["POLICY_SHOCK"] = analysis["POLICY_SPREAD_CHG"].fillna(0.0)

# --- Run all methods ---
cf_factor, factor_model = factor_benchmark(analysis)
cf_lp, lp_model = local_projection_counterfactual(analysis)
cf_svar, svar_removed_fx, svar_removed_policy, svar_model = svar_counterfactual(analysis)
svar_var_cols = ["EA_FACTOR", "BRENT_YOY", "FX_DEPR", "POLICY_SHOCK", "UA"]
svar_ci_lo, svar_ci_hi = svar_bootstrap_ci(analysis, svar_model, svar_var_cols, n_boot=500)
cf_ascm, donor_weights = augmented_synthetic_control(analysis, donor_panel)
cf_main = median_counterfactual([cf_lp, cf_svar, cf_ascm])


# ===================================================================
# Console output
# ===================================================================

print("=" * 72)
print("PART B: MULTI-METHOD COUNTERFACTUAL ANALYSIS")
print("=" * 72)
print(f"Base panel: {panel.shape[0]} months, {len(EA_COLS)} EA countries + Ukraine")
print(f"Expanded donor pool: {donor_panel.shape[1]} countries")
print(f"Period: {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")

print(f"\nEA factor benchmark:")
print(f"  PC1 explains {explained_ratio[0]:.1%} of EA inflation variance")
print(f"  Corr(EA factor, EA mean) = {ea_factor.corr(analysis['EA_MEAN']):.4f}")

print(f"\nFactor benchmark regression (stable periods):")
print(f"  const (credibility premium) = {factor_model.params['const']:.2f} pp")
print(f"  EA_FACTOR loading = {factor_model.params['EA_FACTOR']:.4f}")

print(f"\nLP regression (stable periods, UA ~ EA_FACTOR + BRENT):")
for v in lp_model.params.index:
    print(f"  {v:<14s} = {lp_model.params[v]:+.4f}  (p={lp_model.pvalues[v]:.3f})")

print("\nExplicit regime mapping from Part A:")
for rn, sub in regimes.groupby("regime"):
    print(f"  {rn:<22s} FX={sub['fx_channel_weight'].iloc[0]:.2f}"
          f" Policy={sub['policy_channel_weight'].iloc[0]:.2f}"
          f" Cred={sub['credibility_gain_weight'].iloc[0]:.2f}"
          f" -> treatment={sub['euro_treatment_weight'].iloc[0]:.2f}")

run_stationarity_tests(analysis, svar_var_cols)

print(f"\nSVAR: lag order (BIC) = {svar_model.k_ar}, MA horizon = 12")

print("\nCounterfactual gaps in key episodes:")
for label, s, e in [
    ("GFC", "2008-09-01", "2009-06-01"),
    ("Crimea/Donbas", "2014-02-01", "2015-12-01"),
    ("IT period", "2017-01-01", "2021-12-01"),
    ("Full-scale invasion", "2022-02-01", "2023-06-01"),
]:
    actual = analysis["UA"].loc[s:e].mean()
    for name, cf in [("LP", cf_lp), ("SVAR", cf_svar), ("ASCM", cf_ascm), ("Main", cf_main)]:
        m = cf.loc[s:e].mean()
        print(f"  {label:<18s} {name:<5s} actual={actual:5.1f}%  cf={m:5.1f}%  gap={actual - m:+5.1f} pp")
    print()

print("Top ASCM donor weights:")
print(donor_weights.head(10).round(4).to_string())


# ===================================================================
# Figures
# ===================================================================

def _shade_events(ax):
    for s, e, lbl, clr in [
        ("2008-09-01", "2009-06-01", "GFC", "orange"),
        ("2014-02-01", "2015-12-01", "Crimea/Donbas", "purple"),
        ("2022-02-01", "2023-06-01", "Full-scale invasion", "red"),
    ]:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=clr)

ea_mean = analysis["EA_MEAN"]

# Figure 1: 2x2 individual method panels
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
methods = [
    ("Local Projections", cf_lp, "teal", False),
    ("Structural VAR", cf_svar, "navy", True),
    ("Augmented Synthetic Control", cf_ascm, "darkorange", False),
    ("Factor Benchmark", cf_factor, "grey", False),
]
for ax, (name, cf, color, show_ci) in zip(axes.flat, methods):
    ax.plot(analysis.index, analysis["UA"], color="red", linewidth=1.8, label="Ukraine (actual)")
    ax.plot(ea_mean.index, ea_mean, color="blue", linewidth=1.0, linestyle="--", alpha=0.5, label="EA-11 mean")
    ax.plot(cf.index, cf, color=color, linewidth=1.8, label=f"Counterfactual")
    if show_ci:
        ax.fill_between(svar_ci_lo.index, svar_ci_lo, svar_ci_hi, color=color, alpha=0.12, label="90% CI")
    ax.axhline(0, color="black", linewidth=0.4, linestyle=":")
    _shade_events(ax)
    ax.set_ylabel("YoY inflation (%)")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.15)
    ax.set_ylim(-5, 65)
axes[1, 0].set_xlabel("Date")
axes[1, 1].set_xlabel("Date")
fig.suptitle("Ukraine Counterfactual Inflation: Individual Methods vs Actual & EA Mean",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_counterfactual_methods.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig_counterfactual_methods.png")

# Figure 2: Main result -- actual vs median ensemble + decomposition + regime weights
fig2 = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.08, figure=fig2)
ax1 = fig2.add_subplot(gs[0])
ax2 = fig2.add_subplot(gs[1], sharex=ax1)
ax3 = fig2.add_subplot(gs[2], sharex=ax1)

ax1.plot(analysis.index, analysis["UA"], color="red", linewidth=2.2, label="Ukraine (actual)")
ax1.plot(cf_main.index, cf_main, color="darkgreen", linewidth=2.0, label="Counterfactual (median ensemble)")
ax1.plot(ea_mean.index, ea_mean, color="blue", linewidth=1.0, linestyle="--", alpha=0.5, label="EA-11 mean")
ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
_shade_events(ax1)
ax1.set_ylabel("Year-on-year inflation (%)")
ax1.set_title("Ukraine Counterfactual Under Euro-Area Membership")
ax1.legend(loc="upper left", fontsize=9, frameon=True)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(-5, 65)
plt.setp(ax1.get_xticklabels(), visible=False)

gap = analysis["UA"] - cf_main
ax2.fill_between(gap.index, 0, gap, where=gap >= 0, color="salmon", alpha=0.4, label="Actual > CF")
ax2.fill_between(gap.index, 0, gap, where=gap < 0, color="steelblue", alpha=0.4, label="Actual < CF")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Gap (pp)")
ax2.legend(loc="upper left", fontsize=8, frameon=True)
ax2.grid(True, alpha=0.2)
plt.setp(ax2.get_xticklabels(), visible=False)

ax3.plot(regimes.index, regimes["fx_channel_weight"], color="darkorange", label="FX channel")
ax3.plot(regimes.index, regimes["policy_channel_weight"], color="steelblue", label="Policy channel")
ax3.plot(regimes.index, regimes["credibility_gain_weight"], color="purple", linestyle="--", label="Credibility")
ax3.set_ylabel("Treatment weight")
ax3.set_xlabel("Date")
ax3.set_ylim(-0.05, 1.05)
ax3.grid(True, alpha=0.2)
ax3.legend(loc="upper right", ncol=3, fontsize=8, frameon=True)

fig2.savefig(os.path.join(FIG_DIR, "fig_counterfactual_main.png"), dpi=200, bbox_inches="tight")
plt.close(fig2)
print("Saved: fig_counterfactual_main.png")


# ===================================================================
# Save outputs
# ===================================================================

results = pd.DataFrame({
    "UA_actual": analysis["UA"], "EA_factor": analysis["EA_FACTOR"],
    "EA_mean": analysis["EA_MEAN"],
    "FX_depreciation": analysis["FX_DEPR"], "Policy_spread": analysis["POLICY_SPREAD"],
    "Policy_shock": analysis["POLICY_SHOCK"],
    "CF_factor_benchmark": cf_factor, "CF_local_projections": cf_lp,
    "CF_svar": cf_svar, "CF_svar_ci_lo": svar_ci_lo, "CF_svar_ci_hi": svar_ci_hi,
    "CF_augmented_synthetic_control": cf_ascm, "CF_main": cf_main,
    "SVAR_removed_fx": svar_removed_fx, "SVAR_removed_policy": svar_removed_policy,
    "fx_channel_weight": regimes["fx_channel_weight"],
    "policy_channel_weight": regimes["policy_channel_weight"],
    "credibility_gain_weight": regimes["credibility_gain_weight"],
    "euro_treatment_weight": regimes["euro_treatment_weight"],
}).round(4)
results.to_csv(os.path.join(DATA_DIR, "data_counterfactual_results.csv"))
donor_weights.to_csv(os.path.join(DATA_DIR, "data_ascm_weights.csv"), header=True)
print("Saved: data_counterfactual_results.csv")
print("Saved: data_ascm_weights.csv")

print("\nInterpretation:")
print("  Euro Area membership would have most significantly reduced Ukraine's inflation")
print("  during the 2008-09 GFC and 2014-15 Crimea/Donbas crises, when devaluation of")
print("  the hryvnia was the primary inflation driver. During the post-2016 IT period,")
print("  the gap narrows as the NBU achieved partial credibility convergence. The 2022")
print("  wartime period shows a moderate gap, consistent with the wartime peg already")
print("  constraining monetary autonomy (low treatment intensity per Part A).")
