from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

from adapt.data_loader import load_settings, get_close_history, get_ohlc_history
from adapt.execution import (
    apply_close_signal_next_day_return,
    compute_drawdown,
    update_portfolio_value,
)
from adapt.alpha.alpha_signal import (
    build_alpha_features,
    classify_alpha_signal,
    entry_weights,
)


def load_alpha_config(path: str | Path = "config/alpha_strategy.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_alpha_data(settings: dict, alpha_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_ticker = alpha_cfg["execution"]["signal_asset"]
    exec_ticker = alpha_cfg["execution"]["long_asset"]
    benchmark = alpha_cfg["execution"]["benchmark_asset"]

    signal_ohlc = get_ohlc_history(
        ticker=signal_ticker,
        settings=settings,
        cache_name=f"{signal_ticker.lower()}_ohlc.csv",
    )

    closes = get_close_history(
        tickers=[exec_ticker, benchmark],
        settings=settings,
        cache_name="alpha_prices.csv",
    )

    return signal_ohlc, closes


def run_alpha_backtest(
    settings: dict | None = None,
    alpha_cfg: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    settings = settings or load_settings()
    alpha_cfg = alpha_cfg or load_alpha_config()

    signal_ohlc, closes = get_alpha_data(settings, alpha_cfg)
    features = build_alpha_features(signal_ohlc, alpha_cfg)

    exec_asset = alpha_cfg["execution"]["long_asset"]
    benchmark = alpha_cfg["execution"]["benchmark_asset"]

    joined = pd.DataFrame(index=features.index)
    joined["signal_close"] = features["Close"]
    joined["bullish_choch"] = features["bullish_choch"]
    joined["bullish_bos1"] = features["bullish_bos1"]
    joined["bearish_choch"] = features["bearish_choch"]
    joined[exec_asset] = closes[exec_asset]
    joined[benchmark] = closes[benchmark]
    joined = joined.dropna().copy()

    returns = joined[[exec_asset, benchmark]].pct_change()

    start_value = float(settings["portfolio"]["start_value"])
    transaction_cost = float(alpha_cfg["parameters"]["transaction_cost"])

    first_row = joined.iloc[0]
    first_signal = classify_alpha_signal(first_row, alpha_cfg)

    if first_signal == "entry":
        active_weights = entry_weights(alpha_cfg)
    else:
        active_weights = {}

    portfolio_value = start_value
    peak_value = start_value
    records = []

    for date, row in joined.iloc[1:].iterrows():
        returns_row = returns.loc[date]

        signal = classify_alpha_signal(row, alpha_cfg)

        if signal == "entry":
            target = entry_weights(alpha_cfg)
        elif signal == "exit":
            target = {}
        else:
            target = dict(active_weights)

        realized_return, is_trade, next_active_weights = apply_close_signal_next_day_return(
            active_weights=active_weights,
            target_weights=target,
            returns_row=returns_row,
            transaction_cost=transaction_cost,
        )

        portfolio_value = update_portfolio_value(portfolio_value, realized_return)
        peak_value = max(peak_value, portfolio_value)
        drawdown = compute_drawdown(portfolio_value, peak_value)

        records.append(
            {
                "date": date,
                "signal": signal,
                "is_trade": is_trade,
                "ret": realized_return,
                "cum_val": portfolio_value,
                "dd": drawdown,
                "weight": active_weights.get(exec_asset, 0.0),
                "target_weight": target.get(exec_asset, 0.0),
                "bullish_choch": bool(row["bullish_choch"]),
                "bullish_bos1": bool(row["bullish_bos1"]),
                "bearish_choch": bool(row["bearish_choch"]),
            }
        )

        active_weights = next_active_weights

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
        "time_in_market": float((df["weight"] > 0).mean()),
    }

    return df, metrics
