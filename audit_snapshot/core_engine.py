from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

from adapt.data_loader import load_settings, get_close_history
from adapt.execution import (
    apply_close_signal_next_day_return,
    compute_drawdown,
    update_portfolio_value,
)
from adapt.core.core_signal import (
    CoreState,
    build_core_features,
    classify_regime,
    target_weights,
    update_circuit_breaker,
)


def load_core_config(path: str | Path = "config/core_strategy.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_core_prices(settings: dict) -> pd.DataFrame:
    tickers = settings["tickers"]["core"]
    return get_close_history(
        tickers=tickers,
        settings=settings,
        cache_name="core_prices.csv",
    )


def run_core_backtest(
    settings: dict | None = None,
    core_cfg: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    settings = settings or load_settings()
    core_cfg = core_cfg or load_core_config()

    prices = get_core_prices(settings)
    features = build_core_features(prices, core_cfg)
    returns = prices.pct_change()

    start_value = float(settings["portfolio"]["start_value"])
    transaction_cost = float(core_cfg["parameters"]["transaction_cost"])

    first_date = features.index[0]
    first_row = features.iloc[0]

    state = CoreState(
        peak_value=start_value,
        trough_value=start_value,
        circuit_breaker_active=False,
        prev_signal_regime=None,
    )

    first_regime = classify_regime(first_row, state, core_cfg)
    active_weights = target_weights(first_regime, first_row, first_date, core_cfg)

    portfolio_value = start_value
    records = []

    for date, row in features.iloc[1:].iterrows():
        returns_row = returns.loc[date]

        state.peak_value = max(state.peak_value, portfolio_value)
        update_circuit_breaker(state, portfolio_value, core_cfg)

        signal_regime = classify_regime(row, state, core_cfg)
        target = target_weights(signal_regime, row, date, core_cfg)

        realized_return, is_trade, next_active_weights = apply_close_signal_next_day_return(
            active_weights=active_weights,
            target_weights=target,
            returns_row=returns_row,
            transaction_cost=transaction_cost,
        )

        portfolio_value = update_portfolio_value(portfolio_value, realized_return)
        state.peak_value = max(state.peak_value, portfolio_value)
        drawdown = compute_drawdown(portfolio_value, state.peak_value)

        records.append(
            {
                "date": date,
                "signal_regime": signal_regime,
                "is_trade": is_trade,
                "ret": realized_return,
                "cum_val": portfolio_value,
                "dd": drawdown,
                "close": float(row["close"]),
                "sma": float(row["sma"]),
                "rsi": float(row["rsi"]),
                "consec_up": int(row["consec_up"]),
            }
        )

        active_weights = next_active_weights
        state.prev_signal_regime = signal_regime

    df = pd.DataFrame(records).set_index("date")

    years = len(df) / 252.0
    cagr = (df["cum_val"].iloc[-1] / start_value) ** (1.0 / years) - 1.0
    maxdd = float(df["dd"].min())
    std = df["ret"].std()
    sharpe = (df["ret"].mean() * 252.0) / (std * (252.0 ** 0.5)) if std > 0 else 0.0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0.0

    metrics = {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "final_value": float(df["cum_val"].iloc[-1]),
        "trades": int(df["is_trade"].sum()),
        "years": years,
    }

    return df, metrics
