from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml
import yfinance as yf


REFRESH_BUFFER_DAYS = 10


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_download(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            return raw
        raw.columns = raw.columns.get_level_values(0)
    return raw


def download_close_history(
    tickers: Iterable[str],
    start: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    tickers = list(tickers)

    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=auto_adjust,
        progress=False,
    )
    raw = _normalize_download(raw)

    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw.copy()
        if "Close" in px.columns:
            px = px[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px.index = pd.to_datetime(px.index)
    return px.sort_index().ffill().dropna(how="all")


def download_ohlc_history(
    ticker: str,
    start: str,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        auto_adjust=auto_adjust,
        progress=False,
    )
    raw = _normalize_download(raw)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    out = raw[cols].dropna().copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def cache_prices(df: pd.DataFrame, filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_index().to_csv(path)


def load_cached_prices(filepath: str | Path) -> pd.DataFrame:
    path = Path(filepath)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def _merge_cached_and_fresh(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if cached is None or cached.empty:
        out = fresh.copy()
    elif fresh is None or fresh.empty:
        out = cached.copy()
    else:
        out = pd.concat([cached, fresh], axis=0)
        out = out[~out.index.duplicated(keep="last")].sort_index()

    return out.ffill().dropna(how="all")


def _refresh_start_from_cache(cached: pd.DataFrame, fallback_start: str) -> str:
    if cached is None or cached.empty:
        return fallback_start

    last_dt = pd.to_datetime(cached.index.max())
    refresh_start = last_dt - pd.Timedelta(days=REFRESH_BUFFER_DAYS)
    return refresh_start.strftime("%Y-%m-%d")


def get_close_history(
    tickers: Iterable[str],
    settings: dict,
    cache_name: str,
) -> pd.DataFrame:
    tickers = list(tickers)
    cache_dir = Path(settings["data"]["cache_dir"])
    cache_path = cache_dir / cache_name
    use_cache = settings["data"].get("use_cache", True)
    start_date = settings["data"]["start_date"]
    auto_adjust = settings["data"].get("auto_adjust", True)

    cached = None
    if use_cache and cache_path.exists():
        cached = load_cached_prices(cache_path)

    refresh_start = _refresh_start_from_cache(cached, start_date)

    fresh = download_close_history(
        tickers=tickers,
        start=refresh_start,
        auto_adjust=auto_adjust,
    )

    df = _merge_cached_and_fresh(cached, fresh)

    if df.empty:
        df = download_close_history(
            tickers=tickers,
            start=start_date,
            auto_adjust=auto_adjust,
        )

    cache_prices(df, cache_path)
    return df


def get_ohlc_history(
    ticker: str,
    settings: dict,
    cache_name: str,
) -> pd.DataFrame:
    cache_dir = Path(settings["data"]["cache_dir"])
    cache_path = cache_dir / cache_name
    use_cache = settings["data"].get("use_cache", True)
    start_date = settings["data"]["start_date"]
    auto_adjust = settings["data"].get("auto_adjust", True)

    cached = None
    if use_cache and cache_path.exists():
        cached = load_cached_prices(cache_path)

    refresh_start = _refresh_start_from_cache(cached, start_date)

    fresh = download_ohlc_history(
        ticker=ticker,
        start=refresh_start,
        auto_adjust=auto_adjust,
    )

    df = _merge_cached_and_fresh(cached, fresh)

    if df.empty:
        df = download_ohlc_history(
            ticker=ticker,
            start=start_date,
            auto_adjust=auto_adjust,
        )

    cache_prices(df, cache_path)
    return df
