#!/usr/bin/env python3
"""
External data helpers for the Ukraine counterfactual project.

This module fetches and caches macro series that are not bundled with the
original repository:
  - NBU key policy rate
  - NBU official UAH exchange rates
  - ECB key policy rate and industrial production
  - FRED Brent crude prices
  - Ukraine industry indicators from the NBU macro workbook
  - Expanded HICP donor panel from the ECB Data Portal

All fetchers fall back to a local cache when the network is unavailable.
"""

from __future__ import annotations

import os
import time
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import requests


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "external_cache")

os.makedirs(CACHE_DIR, exist_ok=True)


NBU_BASE = "https://bank.gov.ua"
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

BASE_EA_COUNTRIES = ["AT", "BE", "DE", "ES", "FI", "FR", "GR", "IE", "IT", "NL", "PT"]
EXTENDED_DONOR_COUNTRIES = BASE_EA_COUNTRIES + [
    "BG", "CY", "CZ", "EE", "HR", "HU", "LT", "LV", "MT", "PL", "RO", "SI", "SK"
]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def _maybe_refresh() -> bool:
    return os.environ.get("REFRESH_EXTERNAL_DATA", "").strip().lower() in {"1", "true", "yes"}


def _save_frame(df: pd.DataFrame | pd.Series, path: str) -> None:
    if isinstance(df, pd.Series):
        df.to_frame().to_csv(path)
    else:
        df.to_csv(path)


def _load_frame(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _fetch_or_cache(cache_name: str, fetch_fn):
    path = _cache_path(cache_name)
    refresh = _maybe_refresh()

    if os.path.exists(path) and not refresh:
        return _load_frame(path)

    try:
        fresh = fetch_fn()
        _save_frame(fresh, path)
        return fresh if isinstance(fresh, pd.DataFrame) else fresh.to_frame()
    except Exception:
        if os.path.exists(path):
            return _load_frame(path)
        raise


def _ecb_fetch_csv(dataset: str, key: str, start: str, end: str) -> pd.DataFrame:
    url = f"{ECB_BASE}/{dataset}/{key}"
    params = {"format": "csvdata", "startPeriod": start, "endPeriod": end}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def fetch_ecb_hicp_panel(countries: list[str], start: str, end: str) -> pd.DataFrame:
    key = f"M.{'+'.join(countries)}.N.000000.4.ANR"
    raw = _ecb_fetch_csv("ICP", key, start, end)
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    panel = (
        raw.pivot_table(index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="last")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    panel.index = panel.index.to_period("M").to_timestamp(how="start")
    panel.index.name = "date"
    return panel


def get_extended_hicp_panel(start: str, end: str) -> pd.DataFrame:
    return _fetch_or_cache(
        "data_extended_hicp_panel.csv",
        lambda: fetch_ecb_hicp_panel(EXTENDED_DONOR_COUNTRIES, start, end),
    )


def _ecb_single_series(dataset: str, key: str, start: str, end: str, date_freq: str) -> pd.Series:
    raw = _ecb_fetch_csv(dataset, key, start, end)
    raw["TIME_PERIOD"] = pd.to_datetime(raw["TIME_PERIOD"])
    raw["OBS_VALUE"] = pd.to_numeric(raw["OBS_VALUE"], errors="coerce")
    series = raw.set_index("TIME_PERIOD")["OBS_VALUE"].sort_index()
    if date_freq == "M":
        series.index = series.index.to_period("M").to_timestamp(how="start")
    series.index.name = "date"
    return series


def get_ecb_mro_rate(start: str, end: str) -> pd.Series:
    df = _fetch_or_cache(
        "ecb_mro_daily.csv",
        lambda: _ecb_single_series("FM", "D.U2.EUR.4F.KR.MRR_RT.LEV", start, end, date_freq="D"),
    )
    series = df.iloc[:, 0].rename("EA_MRO")
    monthly = series.resample("MS").last()
    monthly.index.name = "date"
    return monthly


def get_ecb_industrial_production(start: str, end: str) -> pd.Series:
    # Euro area industrial production, total industry excluding construction.
    df = _fetch_or_cache(
        "ecb_industrial_production.csv",
        lambda: _ecb_single_series("STS", "M.I9.Y.PROD.NS0020.4.000", start, end, date_freq="M"),
    )
    series = df.iloc[:, 0].rename("EA_IP_INDEX")
    series = series.loc[start:end]
    series.index.name = "date"
    return series


def get_fred_series(series_id: str) -> pd.Series:
    def _fetch() -> pd.Series:
        response = requests.get(FRED_BASE, params={"id": series_id}, timeout=60)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        series = df.set_index("observation_date")[series_id].rename(series_id)
        series.index.name = "date"
        return series

    df = _fetch_or_cache(f"fred_{series_id.lower()}.csv", _fetch)
    return df.iloc[:, 0].rename(series_id)


def get_nbu_key_rate() -> pd.Series:
    def _fetch() -> pd.Series:
        response = requests.get(f"{NBU_BASE}/NBUStatService/v1/statdirectory/key?json", timeout=60)
        response.raise_for_status()
        records = response.json()
        df = pd.DataFrame(records)
        df = df.loc[df["id_api"] == "KEY_PolicyRate", ["dt", "value"]].copy()
        df["date"] = pd.to_datetime(df["dt"], format="%Y%m%d")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        series = df.dropna().drop_duplicates("date").set_index("date")["value"].sort_index()
        series.index.name = "date"
        return series.rename("NBU_KEY_RATE")

    df = _fetch_or_cache("nbu_key_rate_daily.csv", _fetch)
    series = df.iloc[:, 0].rename("NBU_KEY_RATE")
    monthly = series.resample("MS").last()
    monthly.index.name = "date"
    return monthly


def get_nbu_exchange_monthly(currency: str, start: str, end: str) -> pd.Series:
    cache_name = f"nbu_fx_{currency.lower()}_monthly.csv"

    def _fetch() -> pd.Series:
        values = []
        end_month = pd.Timestamp(end) + pd.offsets.MonthEnd(0)
        for dt in pd.date_range(start=start, end=end_month, freq="ME"):
            date_str = dt.strftime("%Y%m%d")
            url = (
                f"{NBU_BASE}/NBUStatService/v1/statdirectory/exchange"
                f"?valcode={currency.upper()}&date={date_str}&json"
            )
            rate = np.nan
            for _ in range(3):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    if data:
                        rate = pd.to_numeric(data[0]["rate"], errors="coerce")
                    break
                except requests.RequestException:
                    time.sleep(0.5)
            values.append((dt.to_period("M").to_timestamp(how="start"), rate))
            time.sleep(0.02)

        series = pd.Series(dict(values), name=f"UAH_{currency.upper()}").sort_index().ffill()
        series.index.name = "date"
        return series

    df = _fetch_or_cache(cache_name, _fetch)
    return df.iloc[:, 0].rename(f"UAH_{currency.upper()}")


def get_ukraine_industry_yoy() -> pd.Series:
    def _fetch() -> pd.Series:
        response = requests.get(f"{NBU_BASE}/files/macro/Indust_m.xlsx", timeout=60)
        response.raise_for_status()
        workbook = BytesIO(response.content)
        sheet = pd.read_excel(workbook, sheet_name="4", header=None)
        dates = pd.to_datetime(sheet.iloc[1, 2:], format="%m.%Y", errors="coerce")
        # The workbook contains two overlapping aggregate series:
        # row 2 with the older 2010-base publication block and row 3 with the
        # later 2016-base continuation. We stitch them so the sample extends to
        # the latest release instead of stopping in 2018.
        values_old = pd.to_numeric(sheet.iloc[2, 2:], errors="coerce")
        values_new = pd.to_numeric(sheet.iloc[3, 2:], errors="coerce")
        values = values_old.combine_first(values_new)
        series = pd.Series(values.values, index=dates, name="UA_IP_YOY").dropna()
        series.index = series.index.to_period("M").to_timestamp(how="start")
        series.index.name = "date"
        return series.sort_index()

    df = _fetch_or_cache("ua_industry_yoy.csv", _fetch)
    return df.iloc[:, 0].rename("UA_IP_YOY")


def build_external_macro_panel(start: str, end: str) -> pd.DataFrame:
    ecb_mro = get_ecb_mro_rate(start, end)
    ecb_ip = get_ecb_industrial_production(start, end)
    nbu_rate = get_nbu_key_rate().loc[start:end]
    fx_usd = get_nbu_exchange_monthly("USD", start, end)
    fx_eur = get_nbu_exchange_monthly("EUR", start, end)
    ua_ip_yoy = get_ukraine_industry_yoy().loc[start:end]
    brent = get_fred_series("DCOILBRENTEU")
    brent = brent.resample("MS").mean().rename("BRENT")
    wheat = get_fred_series("PWHEAMTUSDM").rename("WHEAT")
    gas = get_fred_series("PNGASEUUSDM").rename("GAS")

    panel = pd.concat(
        [
            ecb_mro.rename("EA_MRO"),
            ecb_ip.rename("EA_IP_INDEX"),
            nbu_rate.rename("NBU_KEY_RATE"),
            fx_usd.rename("UAH_USD"),
            fx_eur.rename("UAH_EUR"),
            ua_ip_yoy.rename("UA_IP_YOY"),
            brent.rename("BRENT"),
            wheat,
            gas,
        ],
        axis=1,
    ).sort_index()

    panel = panel.loc[start:end]
    panel["EA_IP_YOY"] = 100.0 * np.log(panel["EA_IP_INDEX"] / panel["EA_IP_INDEX"].shift(12))
    panel["BRENT_YOY"] = 100.0 * np.log(panel["BRENT"] / panel["BRENT"].shift(12))
    panel["WHEAT_YOY"] = 100.0 * np.log(panel["WHEAT"] / panel["WHEAT"].shift(12))
    panel["GAS_YOY"] = 100.0 * np.log(panel["GAS"] / panel["GAS"].shift(12))
    panel["FX_USD_DEPR"] = 100.0 * np.log(panel["UAH_USD"] / panel["UAH_USD"].shift(1))
    panel["FX_EUR_DEPR"] = 100.0 * np.log(panel["UAH_EUR"] / panel["UAH_EUR"].shift(1))
    panel["FX_DEPR"] = panel["FX_EUR_DEPR"]  # EUR more relevant for Euro counterfactual
    panel["POLICY_SPREAD"] = panel["NBU_KEY_RATE"] - panel["EA_MRO"]
    panel["POLICY_SPREAD_CHG"] = panel["POLICY_SPREAD"].diff()

    panel.index.name = "date"
    return panel
