#!/usr/bin/env python3
"""
Part B — Multi-Method Counterfactual Inflation Analysis
=======================================================

This script reworks Part B around the requirements of the exam:
  - external macro data are fetched and merged in Step 1
  - SVAR and Local Projections are the structural core methods
  - factor models are kept as a benchmark
  - an augmented synthetic-control style donor model is added as a modern
    comparative benchmark
  - Part A regime distinctions enter transparently through explicit regime
    weights for the FX channel, policy channel, and credibility import

Counterfactual interpretation:
  1. Euro adoption removes the hryvnia devaluation channel.
  2. Euro adoption eliminates independent nominal-policy spread versus the ECB.
  3. Euro adoption imports credibility, reducing Ukraine's inflation premium.
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

warnings.filterwarnings("ignore")
np.random.seed(42)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def fit_pca_standardized(df: pd.DataFrame):
    x = df.to_numpy(dtype=float)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    eigenvalues = (s ** 2) / (x.shape[0] - 1)
    explained_ratio = eigenvalues / eigenvalues.sum()
    scores = u * s
    return eigenvalues, explained_ratio, scores


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

    # What the euro changes, channel by channel.
    fx_weights = {
        "peg_pre_2008": 0.20,
        "gfc_devaluation": 1.00,
        "repeg": 0.20,
        "crimea_float_crisis": 1.00,
        "inflation_targeting": 0.70,
        "wartime_fixed": 0.15,
        "managed_flexibility": 0.50,
    }
    policy_weights = {
        "peg_pre_2008": 0.15,
        "gfc_devaluation": 0.65,
        "repeg": 0.15,
        "crimea_float_crisis": 0.85,
        "inflation_targeting": 0.80,
        "wartime_fixed": 0.10,
        "managed_flexibility": 0.55,
    }
    credibility_weights = {
        "peg_pre_2008": 1.00,
        "gfc_devaluation": 1.00,
        "repeg": 0.95,
        "crimea_float_crisis": 0.90,
        "inflation_targeting": 0.45,
        "wartime_fixed": 0.35,
        "managed_flexibility": 0.40,
    }

    regimes["fx_channel_weight"] = regimes["regime"].map(fx_weights).astype(float)
    regimes["policy_channel_weight"] = regimes["regime"].map(policy_weights).astype(float)
    regimes["credibility_gain_weight"] = regimes["regime"].map(credibility_weights).astype(float)
    regimes["euro_treatment_weight"] = (
        0.5 * regimes["fx_channel_weight"] + 0.3 * regimes["policy_channel_weight"] + 0.2 * regimes["credibility_gain_weight"]
    )
    regimes.index.name = "date"
    return regimes


def compute_ea_factor(ea_panel: pd.DataFrame):
    ea_clean = ea_panel.dropna()
    ea_z = (ea_clean - ea_clean.mean()) / ea_clean.std()
    eigenvalues, explained_ratio, scores = fit_pca_standardized(ea_z)
    factor_raw = pd.Series(scores[:, 0], index=ea_clean.index, name="EA_FACTOR_RAW")
    ea_mean = ea_clean.mean(axis=1)
    factor = (factor_raw - factor_raw.mean()) / factor_raw.std() * ea_mean.std() + ea_mean.mean()
    factor.name = "EA_FACTOR"
    return factor, eigenvalues, explained_ratio


def estimate_credibility_shift(df: pd.DataFrame, regimes: pd.DataFrame):
    rolling_vol = df["UA"].rolling(12).std()
    stable_mask = (
        (regimes["fx_channel_weight"] <= 0.20)
        & (df["UA"].notna())
        & (df["EA_FACTOR"].notna())
        & (df.index < pd.Timestamp("2015-08-01"))
        & (rolling_vol <= rolling_vol.quantile(0.40))
    )
    anchor = df["EA_FACTOR"].rename("FACTOR_ANCHOR")
    raw_gap = (df["UA"] - anchor).where(stable_mask)
    premium = raw_gap.clip(lower=0).median()
    credibility_shift = regimes["credibility_gain_weight"] * premium
    credibility_shift.name = "credibility_shift"
    return anchor, premium, credibility_shift


def factor_benchmark(df: pd.DataFrame, credibility_shift: pd.Series):
    stable_mask = (df["UA"].notna()) & (df["EA_FACTOR"].notna()) & (df["fx_channel_weight"] <= 0.35)
    model = OLS(df.loc[stable_mask, "UA"], add_constant(df.loc[stable_mask, "EA_FACTOR"])).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    fitted = pd.Series(model.predict(add_constant(df["EA_FACTOR"], has_constant="add")), index=df.index, name="CF_factor_benchmark")
    cf = fitted - credibility_shift
    cf.name = "CF_factor_benchmark"
    return cf, model


def make_lagged_controls(df: pd.DataFrame, cols: list[str], nlags: int) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        for lag in range(1, nlags + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def local_projection_counterfactual(df: pd.DataFrame, credibility_shift: pd.Series, horizons: int = 6, nlags: int = 3):
    lp = make_lagged_controls(
        df[[
            "UA", "EA_FACTOR", "BRENT_YOY", "UA_IP_GAP", "EA_IP_YOY",
            "FX_DEPR", "POLICY_SHOCK", "fx_channel_weight", "policy_channel_weight"
        ]],
        ["UA", "EA_FACTOR", "BRENT_YOY", "UA_IP_GAP", "EA_IP_YOY"],
        nlags,
    )
    lp["FX_DEPR"] = lp["FX_DEPR"].clip(lp["FX_DEPR"].quantile(0.02), lp["FX_DEPR"].quantile(0.98))
    lp["BRENT_YOY"] = lp["BRENT_YOY"].clip(lp["BRENT_YOY"].quantile(0.02), lp["BRENT_YOY"].quantile(0.98))
    lp["FX_STATE"] = lp["FX_DEPR"] * lp["fx_channel_weight"]
    lp["POLICY_STATE"] = lp["POLICY_SHOCK"] * lp["policy_channel_weight"]

    base_cols = [
        "FX_DEPR", "FX_STATE", "POLICY_SHOCK", "POLICY_STATE",
        "EA_FACTOR", "BRENT_YOY", "UA_IP_GAP", "EA_IP_YOY",
    ]
    lag_cols = [c for c in lp.columns if "_lag" in c and not c.startswith(("FX_DEPR", "POLICY_SHOCK"))]
    design_cols = base_cols + lag_cols

    beta_fx = []
    beta_fx_state = []
    beta_policy = []
    beta_policy_state = []

    for h in range(horizons + 1):
        tmp = lp.copy()
        tmp[f"UA_lead_{h}"] = tmp["UA"].shift(-h)
        sample = tmp.dropna(subset=[f"UA_lead_{h}"] + design_cols)
        X = add_constant(sample[design_cols], has_constant="add")
        y = sample[f"UA_lead_{h}"]
        res = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
        beta_fx.append(res.params["FX_DEPR"])
        beta_fx_state.append(res.params["FX_STATE"])
        beta_policy.append(res.params["POLICY_SHOCK"])
        beta_policy_state.append(res.params["POLICY_STATE"])

    removed_fx = pd.Series(0.0, index=df.index, name="LP_removed_fx")
    removed_policy = pd.Series(0.0, index=df.index, name="LP_removed_policy")
    for h in range(horizons + 1):
        fx_term = (beta_fx[h] * df["FX_DEPR"] + beta_fx_state[h] * df["FX_DEPR"] * df["fx_channel_weight"]).shift(h)
        pol_term = (beta_policy[h] * df["POLICY_SHOCK"] + beta_policy_state[h] * df["POLICY_SHOCK"] * df["policy_channel_weight"]).shift(h)
        removed_fx = removed_fx.add(fx_term, fill_value=0.0)
        removed_policy = removed_policy.add(pol_term, fill_value=0.0)

    cf = df["UA"] - removed_fx - removed_policy - credibility_shift
    cf.name = "CF_local_projections"
    return cf, removed_fx, removed_policy


def svar_counterfactual(df: pd.DataFrame, credibility_shift: pd.Series, maxlags: int = 3, ma_horizon: int = 24):
    var_cols = ["EA_FACTOR", "BRENT_YOY", "UA_IP_GAP", "FX_DEPR", "POLICY_SHOCK", "UA"]
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

    removed_fx = pd.Series(removed_fx, index=effective_index, name="SVAR_removed_fx")
    removed_policy = pd.Series(removed_policy, index=effective_index, name="SVAR_removed_policy")
    cf = sample_eff["UA"] - removed_fx - removed_policy - credibility_shift.reindex(effective_index)
    cf.name = "CF_svar"
    return cf, removed_fx, removed_policy, res


def augmented_synthetic_control(df: pd.DataFrame, donor_panel: pd.DataFrame):
    donors = donor_panel.copy().sort_index()
    donors = donors.loc[:, donors.notna().mean() >= 0.95]
    common_idx = df.index.intersection(donors.index)
    donors = donors.reindex(common_idx)
    target = df.reindex(common_idx)["UA"]

    calibration_mask = (
        target.notna()
        & donors.notna().all(axis=1)
        & (
            ((common_idx >= pd.Timestamp("2001-01-01")) & (common_idx <= pd.Timestamp("2008-08-01")))
            | ((common_idx >= pd.Timestamp("2010-01-01")) & (common_idx <= pd.Timestamp("2014-01-01")))
            | ((common_idx >= pd.Timestamp("2016-01-01")) & (common_idx <= pd.Timestamp("2021-12-01")))
        )
    )

    X = donors.loc[calibration_mask].to_numpy(dtype=float)
    y = target.loc[calibration_mask].to_numpy(dtype=float)
    n_donors = X.shape[1]
    ridge = 1e-4

    def objective(w):
        resid = y - X @ w
        return np.mean(resid ** 2) + ridge * np.sum(w ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_donors
    w0 = np.repeat(1.0 / n_donors, n_donors)
    opt = minimize(objective, w0, bounds=bounds, constraints=cons, method="SLSQP")
    weights = pd.Series(opt.x, index=donors.columns, name="weight")

    cf = donors @ weights
    cf.name = "CF_augmented_synthetic_control"
    return cf, weights.sort_values(ascending=False)


def median_counterfactual(series_list: list[pd.Series]) -> pd.Series:
    stacked = pd.concat(series_list, axis=1)
    median = stacked.median(axis=1)
    median.name = "CF_main"
    return median


panel = pd.read_csv(os.path.join(DATA_DIR, "data_clean_panel.csv"), index_col=0, parse_dates=True)
panel.index.name = "date"
macro = pd.read_csv(os.path.join(DATA_DIR, "data_external_macro.csv"), index_col=0, parse_dates=True)
macro.index.name = "date"
donor_panel = pd.read_csv(os.path.join(DATA_DIR, "data_extended_hicp_panel.csv"), index_col=0, parse_dates=True)
donor_panel.index.name = "date"

EA_COLS = [c for c in panel.columns if c != "UA"]
ua = panel["UA"].copy()
ea = panel[EA_COLS].copy()

ea_factor, eigenvalues, explained_ratio = compute_ea_factor(ea)
regimes = build_regime_map(panel.index)

analysis = panel.join(ea_factor, how="left").join(macro, how="left").join(regimes, how="left")
analysis["EA_MEAN"] = ea.mean(axis=1)
analysis["UA_IP_GAP"] = analysis["UA_IP_YOY"] - 100.0
analysis["POLICY_SHOCK"] = analysis["POLICY_SPREAD_CHG"].fillna(0.0)

anchor, credibility_premium, credibility_shift = estimate_credibility_shift(analysis, regimes)
analysis["FACTOR_ANCHOR"] = anchor
analysis["credibility_shift"] = credibility_shift

cf_factor, factor_model = factor_benchmark(analysis, credibility_shift)
cf_lp, lp_removed_fx, lp_removed_policy = local_projection_counterfactual(analysis, credibility_shift)
cf_svar, svar_removed_fx, svar_removed_policy, svar_model = svar_counterfactual(analysis, credibility_shift)
cf_ascm, donor_weights = augmented_synthetic_control(analysis, donor_panel)
cf_main = median_counterfactual([cf_lp, cf_svar, cf_ascm])

decomp_fx = pd.concat([lp_removed_fx.rename("lp"), svar_removed_fx.rename("svar")], axis=1).mean(axis=1).rename("FX_removed")
decomp_policy = pd.concat([lp_removed_policy.rename("lp"), svar_removed_policy.rename("svar")], axis=1).mean(axis=1).rename("Policy_removed")
decomp_cred = credibility_shift.rename("Credibility_import")


print("=" * 72)
print("PART B: MULTI-METHOD COUNTERFACTUAL ANALYSIS")
print("=" * 72)
print(f"Base panel: {panel.shape[0]} months, {len(EA_COLS)} EA countries + Ukraine")
print(f"Expanded donor pool: {donor_panel.shape[1]} countries")
print(f"Period: {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")

print(f"\nEA factor benchmark:")
print(f"  PC1 explains {explained_ratio[0]:.1%} of EA inflation variance")
print(f"  Corr(EA factor, EA mean) = {ea_factor.corr(analysis['EA_MEAN']):.4f}")
print(f"  Credibility premium estimated on stable pre-IT periods = {credibility_premium:.2f} pp")

print("\nExplicit regime mapping from Part A:")
for regime_name, sub in regimes.groupby("regime"):
    print(
        f"  {regime_name:<22s}"
        f" FX={sub['fx_channel_weight'].iloc[0]:.2f}"
        f" Policy={sub['policy_channel_weight'].iloc[0]:.2f}"
        f" Cred={sub['credibility_gain_weight'].iloc[0]:.2f}"
    )

print("\nSVAR summary:")
print(f"  Variables: EA factor, Brent YoY, UA industry gap, FX depreciation, policy-spread shock, UA inflation")
print(f"  Selected lag order (BIC): {svar_model.k_ar}")

print("\nCounterfactual gaps in key episodes:")
for label, start, end in [
    ("GFC", "2008-09-01", "2009-06-01"),
    ("Crimea/Donbas", "2014-02-01", "2015-12-01"),
    ("Inflation targeting", "2017-01-01", "2021-12-01"),
    ("Full-scale invasion", "2022-02-01", "2023-06-01"),
]:
    actual = analysis["UA"].loc[start:end].mean()
    main = cf_main.loc[start:end].mean()
    lp_gap = (analysis["UA"] - cf_lp).loc[start:end].mean()
    svar_gap = (analysis["UA"] - cf_svar).loc[start:end].mean()
    print(
        f"  {label:<18s} actual={actual:6.2f}%"
        f" main_cf={main:6.2f}%"
        f" lp_gap={lp_gap:+6.2f} pp"
        f" svar_gap={svar_gap:+6.2f} pp"
    )

print("\nTop augmented-synthetic-control donor weights:")
print(donor_weights.head(10).round(4).to_string())


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 1, height_ratios=[3, 1.2, 1], hspace=0.08, figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

ax1.plot(analysis.index, analysis["UA"], color="red", linewidth=2.2, label="Ukraine (actual)")
ax1.plot(cf_main.index, cf_main, color="darkgreen", linewidth=2.0, label="Counterfactual (main median)")
ax1.plot(cf_lp.index, cf_lp, color="teal", linewidth=1.1, alpha=0.85, label="Local projections")
ax1.plot(cf_svar.index, cf_svar, color="navy", linewidth=1.1, alpha=0.85, label="SVAR")
ax1.plot(cf_ascm.index, cf_ascm, color="darkorange", linewidth=1.1, alpha=0.85, label="Augmented synthetic control")
ax1.plot(cf_factor.index, cf_factor, color="grey", linewidth=1.0, linestyle="--", alpha=0.9, label="Factor benchmark")
ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
for s, e, lbl, clr in [
    ("2008-09-01", "2009-06-01", "GFC", "orange"),
    ("2014-02-01", "2015-12-01", "Crimea/\nDonbas", "purple"),
    ("2022-02-01", "2023-06-01", "Full-scale\ninvasion", "red"),
]:
    ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.07, color=clr)
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
    ax1.text(mid, 61, lbl, ha="center", va="top", fontsize=8, color=clr, fontweight="bold", alpha=0.75)
ax1.set_ylabel("Year-on-year inflation (%)")
ax1.set_title("Ukraine Counterfactual Under Euro-Area Membership: Structural Core + Benchmarks")
ax1.legend(loc="upper left", ncol=2, fontsize=8, frameon=True)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(-5, 65)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.fill_between(analysis.index, 0, decomp_fx, color="orange", alpha=0.35, label="FX channel removed")
ax2.fill_between(analysis.index, 0, decomp_policy, color="steelblue", alpha=0.35, label="Policy harmonization")
ax2.fill_between(analysis.index, 0, -decomp_cred, color="purple", alpha=0.15, label="Credibility import")
ax2.plot((analysis["UA"] - cf_main).index, (analysis["UA"] - cf_main), color="black", linewidth=1.0, label="Actual - main CF")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Gap / contribution")
ax2.legend(loc="upper left", ncol=2, fontsize=8, frameon=True)
ax2.grid(True, alpha=0.2)
plt.setp(ax2.get_xticklabels(), visible=False)

ax3.plot(regimes.index, regimes["fx_channel_weight"], color="darkorange", label="FX channel weight")
ax3.plot(regimes.index, regimes["policy_channel_weight"], color="steelblue", label="Policy channel weight")
ax3.plot(regimes.index, regimes["credibility_gain_weight"], color="purple", linestyle="--", label="Credibility weight")
ax3.set_ylabel("0-1 weight")
ax3.set_xlabel("Date")
ax3.set_ylim(-0.05, 1.05)
ax3.grid(True, alpha=0.2)
ax3.legend(loc="upper right", ncol=3, fontsize=8, frameon=True)

fig.savefig(os.path.join(FIG_DIR, "fig_counterfactual_main.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved: fig_counterfactual_main.png")


fig2, ax = plt.subplots(figsize=(14, 6))
gap_table = pd.concat(
    [
        (analysis["UA"] - cf_lp).rename("LP"),
        (analysis["UA"] - cf_svar).rename("SVAR"),
        (analysis["UA"] - cf_ascm).rename("ASCM"),
        (analysis["UA"] - cf_factor).rename("Factor"),
    ],
    axis=1,
)
for col, color in [("LP", "teal"), ("SVAR", "navy"), ("ASCM", "darkorange"), ("Factor", "grey")]:
    ax.plot(gap_table.index, gap_table[col], linewidth=1.1, color=color, label=f"Actual - {col}")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Method Comparison: Inflation Gap Between Actual Ukraine and Each Counterfactual")
ax.set_ylabel("Percentage points")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.2)
ax.legend(ncol=2, fontsize=8, frameon=True)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_counterfactual_methods.png"), dpi=180)
plt.close(fig2)
print("Saved: fig_counterfactual_methods.png")


results = pd.DataFrame({
    "UA_actual": analysis["UA"],
    "EA_factor": analysis["EA_FACTOR"],
    "EA_mean": analysis["EA_MEAN"],
    "FX_depreciation": analysis["FX_DEPR"],
    "Policy_spread": analysis["POLICY_SPREAD"],
    "Policy_shock": analysis["POLICY_SHOCK"],
    "Ukraine_industry_yoy": analysis["UA_IP_YOY"],
    "Ukraine_industry_gap": analysis["UA_IP_GAP"],
    "EA_industry_yoy": analysis["EA_IP_YOY"],
    "Brent_yoy": analysis["BRENT_YOY"],
    "CF_factor_benchmark": cf_factor,
    "CF_local_projections": cf_lp,
    "CF_svar": cf_svar,
    "CF_augmented_synthetic_control": cf_ascm,
    "CF_main": cf_main,
    "LP_removed_fx": lp_removed_fx,
    "LP_removed_policy": lp_removed_policy,
    "SVAR_removed_fx": svar_removed_fx,
    "SVAR_removed_policy": svar_removed_policy,
    "credibility_shift": credibility_shift,
    "fx_channel_weight": regimes["fx_channel_weight"],
    "policy_channel_weight": regimes["policy_channel_weight"],
    "credibility_gain_weight": regimes["credibility_gain_weight"],
    "euro_treatment_weight": regimes["euro_treatment_weight"],
}).round(4)
results.to_csv(os.path.join(DATA_DIR, "data_counterfactual_results.csv"))
donor_weights.to_csv(os.path.join(DATA_DIR, "data_ascm_weights.csv"), header=True)

print("Saved: data_counterfactual_results.csv")
print("Saved: data_ascm_weights.csv")

print("\nInterpretation")
print("- The structural core now removes the FX and policy-autonomy channels explicitly instead of blending back actual Ukraine inflation.")
print("- The imported-credibility channel is strongest before the 2015-08 inflation-targeting transition and weaker thereafter, in line with Part A.")
print("- Factor and donor-based methods are retained as benchmarks, not as the sole identification strategy.")
