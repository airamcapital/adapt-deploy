from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def weights_changed(
    old: Dict[str, float],
    new: Dict[str, float],
    tol: float = 1e-12,
) -> bool:
    keys = set(old) | set(new)
    return any(abs(old.get(k, 0.0) - new.get(k, 0.0)) > tol for k in keys)


def portfolio_return(
    weights: Dict[str, float],
    returns_row: pd.Series,
) -> float:
    total = 0.0
    for ticker, weight in weights.items():
        if ticker in returns_row.index and not pd.isna(returns_row[ticker]):
            total += float(weight) * float(returns_row[ticker])
    return total


def apply_close_signal_next_day_return(
    active_weights: Dict[str, float],
    target_weights: Dict[str, float],
    returns_row: pd.Series,
    transaction_cost: float,
) -> Tuple[float, bool, Dict[str, float]]:
    realized_return = portfolio_return(active_weights, returns_row)
    is_trade = weights_changed(active_weights, target_weights)

    if is_trade:
        realized_return -= float(transaction_cost)

    next_active_weights = dict(target_weights)
    return realized_return, is_trade, next_active_weights


def update_portfolio_value(
    current_value: float,
    realized_return: float,
) -> float:
    return float(current_value) * (1.0 + float(realized_return))


def compute_drawdown(
    portfolio_value: float,
    peak_value: float,
) -> float:
    if peak_value <= 0:
        return 0.0
    return (float(portfolio_value) / float(peak_value)) - 1.0
