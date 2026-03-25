from __future__ import annotations

import pandas as pd


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * (252 ** 0.5)


def vol_target_scale(
    returns: pd.Series,
    target_vol: float = 0.20,
    window: int = 20,
    max_leverage: float = 1.0,
) -> pd.Series:
    vol = realized_volatility(returns, window=window)
    scale = target_vol / vol
    scale = scale.replace([float("inf"), -float("inf")], pd.NA)
    scale = scale.clip(lower=0.0, upper=max_leverage)
    return scale.fillna(1.0)


def apply_vol_target(
    returns: pd.Series,
    target_vol: float = 0.20,
    window: int = 20,
    max_leverage: float = 1.0,
) -> pd.Series:
    scale = vol_target_scale(
        returns=returns,
        target_vol=target_vol,
        window=window,
        max_leverage=max_leverage,
    )
    return returns * scale.shift(1).fillna(1.0)
