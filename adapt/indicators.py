from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def realized_vol(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window).std() * np.sqrt(252.0)


def consecutive_up_days(series: pd.Series) -> pd.Series:
    up = series.diff() > 0
    out = []
    count = 0

    for flag in up.fillna(False):
        if flag:
            count += 1
        else:
            count = 0
        out.append(count)

    return pd.Series(out, index=series.index, dtype="int64")


def pct_distance_from_ma(price: pd.Series, ma: pd.Series) -> pd.Series:
    return (price / ma) - 1.0
