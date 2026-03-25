from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from adapt.indicators import sma, rsi, consecutive_up_days, pct_distance_from_ma


@dataclass
class CoreState:
    peak_value: float
    trough_value: float
    circuit_breaker_active: bool
    prev_signal_regime: int | None = None


def build_core_features(prices: pd.DataFrame, core_cfg: dict) -> pd.DataFrame:
    p = core_cfg["parameters"]
    tqqq = prices["TQQQ"]

    out = pd.DataFrame(index=prices.index)
    out["close"] = tqqq
    out["sma"] = sma(tqqq, int(p["sma_period"]))
    out["rsi"] = rsi(tqqq, int(p["rsi_period"]))
    out["consec_up"] = consecutive_up_days(tqqq)
    out["sma_pct"] = pct_distance_from_ma(out["close"], out["sma"])

    return out.dropna().copy()


def update_circuit_breaker(
    state: CoreState,
    portfolio_value: float,
    core_cfg: dict,
) -> None:
    p = core_cfg["parameters"]

    dd = (portfolio_value / state.peak_value) - 1.0

    if dd < float(p["drawdown_threshold"]):
        state.circuit_breaker_active = True
        state.trough_value = min(state.trough_value, portfolio_value)

    if state.circuit_breaker_active:
        recovery = (portfolio_value / state.trough_value) - 1.0
        if recovery >= float(p["drawdown_recovery"]):
            state.circuit_breaker_active = False
            state.trough_value = portfolio_value


def get_historical_r4(date_ts: pd.Timestamp, core_cfg: dict) -> Dict[str, float]:
    high_rate_start = pd.Timestamp(core_cfg["thresholds"]["high_rate_start"])
    if pd.Timestamp(date_ts) >= high_rate_start:
        return dict(core_cfg["baskets"]["r4_high"])
    return dict(core_cfg["baskets"]["r4_low"])


def classify_regime(row: pd.Series, state: CoreState, core_cfg: dict) -> int:
    t = core_cfg["thresholds"]
    p = core_cfg["parameters"]

    if state.circuit_breaker_active:
        return 4

    close = float(row["close"])
    avg = float(row["sma"])
    rsi_val = float(row["rsi"])
    consec = int(row["consec_up"])

    if rsi_val < float(t["rsi_oversold"]) and close > avg:
        return 1

    if rsi_val > float(t["rsi_overbought"]):
        return 2

    if close > avg and consec >= int(p["consec_days"]):
        return 3

    return 4


def r3_weights(row: pd.Series, core_cfg: dict) -> Dict[str, float]:
    w = core_cfg["weights"]

    scaled = min(max(float(row["sma_pct"]), 0.0), 0.20) / 0.20
    tqqq_weight = float(w["r3_tqqq_min"]) + scaled * (
        float(w["r3_tqqq_max"]) - float(w["r3_tqqq_min"])
    )
    rem = 1.0 - tqqq_weight

    return {
        "TQQQ": tqqq_weight,
        "LQD": rem * 0.25,
        "IAU": rem * 0.25,
        "USMV": rem * 0.25,
        "UUP": rem * 0.25,
    }


def target_weights(
    regime: int,
    row: pd.Series,
    date_ts: pd.Timestamp,
    core_cfg: dict,
) -> Dict[str, float]:
    if regime == 1:
        return {"TQQQ": 1.0}
    if regime == 2:
        return {"SQQQ": 1.0}
    if regime == 3:
        return r3_weights(row, core_cfg)
    return get_historical_r4(date_ts, core_cfg)
