#!/usr/bin/env python3
"""
Step 1 — Data Pipeline for the Ukraine Counterfactual Inflation Exam
====================================================================

This script:
  1. Loads Ukraine CPI (month-over-month index, base prev month = 100) from repo CSV
  2. Converts it to year-on-year (YoY) inflation by chaining 12 monthly factors
  3. Loads the ECB HICP panel (11 Euro Area countries, already in YoY %)
  4. Merges everything into a single clean panel DataFrame
  5. Produces a quick diagnostic plot

Data sources (already cached as CSVs in the repo):
  - Ukraine: SSSU SDMX API — CPI prev month = 100
  - ECB: HICP annual rate of change for AT, BE, DE, ES, FI, FR, GR, IE, IT, NL, PT

Author: Eduardo Schutt
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")      # non-interactive backend for saving figures
import matplotlib.pyplot as plt

from external_data import build_external_macro_panel, get_extended_hicp_panel

# =====================================================================
# CONFIGURATION — adjust these paths to point at your local repo clone
# =====================================================================
REPO_DIR = "data"  # absolute path to repo clone
UA_CSV   = f"{REPO_DIR}/data_ukraine_cpi_raw.csv"
ECB_CSV  = f"{REPO_DIR}/data_ecb_hicp_panel.csv"

# Time window for analysis
START = "2000-01-01"
END   = "2025-12-01"


# =====================================================================
# PART 1 — UKRAINE CPI → Year-on-Year Inflation
# =====================================================================
def load_ukraine_cpi_raw(path: str) -> pd.Series:
    """
    Read the SSSU SDMX-CSV dump and return a clean monthly series.

    The raw file contains many metadata columns. We only need:
      - TIME_PERIOD: formatted like '2000-M01'
      - OBS_VALUE:   CPI index with previous month = 100
                     e.g., 104.6 means prices rose 4.6% vs. the previous month

    Returns:
        pd.Series with DatetimeIndex (month-start) and name 'ua_cpi_mom'
    """
    raw = pd.read_csv(path, dtype=str)

    # Keep only rows that look like valid monthly observations
    mask = (
        raw["TIME_PERIOD"].str.match(r"^\d{4}-M\d{2}$", na=False)
        & raw["OBS_VALUE"].notna()
    )
    df = raw.loc[mask, ["TIME_PERIOD", "OBS_VALUE"]].copy()

    # Parse dates: '2000-M01' → datetime 2000-01-01
    df["date"] = pd.to_datetime(
        df["TIME_PERIOD"].str.replace(r"^(\d{4})-M(\d{2})$", r"\1-\2-01", regex=True)
    )

    # Parse values (some locales use comma as decimal separator)
    df["value"] = pd.to_numeric(
        df["OBS_VALUE"].str.replace(",", ".", regex=False),
        errors="coerce"
    )

    series = (
        df.dropna(subset=["date", "value"])
          .sort_values("date")
          .set_index("date")["value"]
          .rename("ua_cpi_mom")
    )

    # Deduplicate (keep last if duplicates exist)
    series = series.groupby(level=0).last()
    return series


def mom_index_to_yoy_inflation(mom_series: pd.Series) -> pd.Series:
    """
    Convert a month-over-month CPI index (prev month = 100) to
    year-on-year inflation (%).

    Method:
      1. Divide by 100 to get the monthly growth factor
         (e.g., 104.6 → 1.046)
      2. Take a rolling product of the last 12 factors
         (this gives the cumulative price change over 12 months)
      3. Subtract 1 and multiply by 100 to get the YoY % change

    Example:
      If every month had CPI = 101.0 (1% monthly inflation),
      the rolling 12-month product = 1.01^12 ≈ 1.1268,
      so YoY inflation ≈ 12.68%.

    The first 11 observations will be NaN (need 12 months to compute).
    """
    monthly_factor = mom_series / 100.0          # e.g., 1.046
    yoy_factor = monthly_factor.rolling(12).apply(np.prod, raw=True)
    yoy_pct = (yoy_factor - 1.0) * 100.0        # e.g., 28.2%
    return yoy_pct.rename("UA")


# =====================================================================
# PART 2 — ECB HICP PANEL (already in YoY %)
# =====================================================================
def load_ecb_hicp_panel(path: str) -> pd.DataFrame:
    """
    Load the cached ECB HICP panel CSV.

    The file has:
      - First column: TIME_PERIOD as dates (e.g., '2000-01-01')
      - Remaining columns: country codes (AT, BE, DE, ...)
      - Values: YoY HICP inflation rate in %

    Returns:
        pd.DataFrame with DatetimeIndex (month-start) and country columns.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Standardize index to month-start timestamps
    df.index = pd.to_datetime(df.index).to_period("M").to_timestamp(how="start")
    df.index.name = "date"

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


# =====================================================================
# PART 3 — BUILD THE MERGED PANEL
# =====================================================================
def build_inflation_panel(ua_csv: str, ecb_csv: str,
                          start: str, end: str) -> pd.DataFrame:
    """
    Full pipeline: load both sources, compute Ukraine YoY, merge.

    Returns:
        pd.DataFrame — columns are country codes (AT, BE, ..., UA)
                        index is monthly datetime
    """
    # --- Ukraine ---
    ua_mom = load_ukraine_cpi_raw(ua_csv)
    ua_yoy = mom_index_to_yoy_inflation(ua_mom)

    # Standardize Ukraine index to month-start
    ua_yoy.index = pd.to_datetime(ua_yoy.index).to_period("M").to_timestamp(how="start")

    # --- ECB ---
    ecb = load_ecb_hicp_panel(ecb_csv)

    # --- Merge ---
    panel = ecb.join(ua_yoy, how="outer").sort_index()

    # Restrict to window
    panel = panel.loc[start:end]

    return panel


# =====================================================================
# PART 4 — DIAGNOSTICS
# =====================================================================
def print_diagnostics(panel: pd.DataFrame):
    """Print summary stats so you can sanity-check the data."""

    print("=" * 60)
    print("INFLATION PANEL — SUMMARY")
    print("=" * 60)
    print(f"Shape: {panel.shape[0]} months × {panel.shape[1]} countries")
    print(f"Period: {panel.index.min().strftime('%Y-%m')} to "
          f"{panel.index.max().strftime('%Y-%m')}")
    print(f"\nMissing values per country:\n{panel.isna().sum()}\n")

    print("Descriptive statistics (YoY inflation, %):")
    print(panel.describe().round(2).to_string())
    print()

    # Flag notable episodes
    ua = panel["UA"].dropna()
    print(f"Ukraine max inflation: {ua.max():.1f}% in {ua.idxmax().strftime('%Y-%m')}")
    print(f"Ukraine min inflation: {ua.min():.1f}% in {ua.idxmin().strftime('%Y-%m')}")

    # Check for the key structural breaks you'll discuss in Part A
    for year in [2008, 2014, 2015, 2022]:
        subset = ua.loc[f"{year}"]
        if len(subset) > 0:
            print(f"  {year}: mean={subset.mean():.1f}%, "
                  f"max={subset.max():.1f}%, min={subset.min():.1f}%")


def plot_panel(panel: pd.DataFrame, save_path: str = "fig_inflation_panel.png"):
    """
    Plot all inflation series. Ukraine is highlighted in bold red.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot EA countries in grey
    ea_cols = [c for c in panel.columns if c != "UA"]
    for col in ea_cols:
        ax.plot(panel.index, panel[col], color="grey", alpha=0.4, linewidth=0.8,
                label=col if col == ea_cols[0] else None)

    # Plot Ukraine in bold red
    ax.plot(panel.index, panel["UA"], color="red", linewidth=2.0, label="Ukraine")

    # EA simple average
    ea_mean = panel[ea_cols].mean(axis=1)
    ax.plot(panel.index, ea_mean, color="blue", linewidth=1.5,
            linestyle="--", label="EA-11 average")

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")

    # Shade key events
    ax.axvspan(pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-01"),
               alpha=0.08, color="orange", label="GFC")
    ax.axvspan(pd.Timestamp("2014-02-01"), pd.Timestamp("2015-02-01"),
               alpha=0.08, color="purple", label="Crimea/Maidan")
    ax.axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2023-06-01"),
               alpha=0.08, color="red", label="Full-scale invasion")

    ax.set_xlabel("Date")
    ax.set_ylabel("Year-on-year inflation (%)")
    ax.set_title("Inflation Panel: Ukraine vs. Euro Area Countries (2000–2025)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, "..")
    data_dir = os.path.join(project_dir, "data")
    fig_dir = os.path.join(project_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    panel = build_inflation_panel(UA_CSV, ECB_CSV, START, END)
    print_diagnostics(panel)
    plot_panel(panel, save_path=os.path.join(fig_dir, "fig_inflation_panel.png"))

    # Save the clean panel for use in subsequent steps
    panel.to_csv(os.path.join(data_dir, "data_clean_panel.csv"))
    print("\nClean panel saved to: data_clean_panel.csv")
    print("Columns:", list(panel.columns))

    print("\nFetching and caching external macro series...")
    macro = build_external_macro_panel(START, END)
    macro.to_csv(os.path.join(data_dir, "data_external_macro.csv"))
    print("Saved: data_external_macro.csv")
    print("External macro columns:", list(macro.columns))

    print("\nFetching and caching expanded HICP donor panel...")
    donor_panel = get_extended_hicp_panel(START, END)
    donor_panel.to_csv(os.path.join(data_dir, "data_extended_hicp_panel.csv"))
    print("Saved: data_extended_hicp_panel.csv")
    print("Expanded donor countries:", list(donor_panel.columns))

    print("\nData pipeline complete. Ready for Step 2 (SVAR, LP, and donor-based counterfactuals).")
