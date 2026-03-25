import numpy as np
import pandas as pd


TRADING_DAYS = 252


def total_return(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan

    days = (pd.to_datetime(equity.index[-1]) - pd.to_datetime(equity.index[0])).days
    if days <= 0:
        return np.nan

    years = days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_vol(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan

    rets = equity.pct_change().dropna()
    if rets.empty:
        return np.nan

    return float(rets.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def sharpe_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan

    rets = equity.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return np.nan

    daily_rf = rf / TRADING_DAYS
    excess = rets - daily_rf
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS))


def compute_equity_metrics(equity: pd.Series) -> dict:
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "volatility": annualized_vol(equity),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(equity),
    }


def yearly_return_table_precise(equity_map: dict) -> pd.DataFrame:
    rows = {}

    for name, s in equity_map.items():
        s = s.dropna().copy().sort_index()
        s.index = pd.to_datetime(s.index)

        yr_map = {}
        for year, grp in s.groupby(s.index.year):
            if len(grp) >= 2:
                yr_map[int(year)] = (float(grp.iloc[-1]) / float(grp.iloc[0]) - 1.0) * 100.0

        rows[name] = yr_map

    df = pd.DataFrame(rows).sort_index()
    return df.round(2)


def metrics_table_for_panel(strategy_name: str, strategy_equity: pd.Series, benchmark_equities: dict) -> pd.DataFrame:
    records = []

    strategy_metrics = compute_equity_metrics(strategy_equity)
    records.append({
        "Series": strategy_name,
        "Total Return %": round(strategy_metrics["total_return"] * 100.0, 2),
        "CAGR %": round(strategy_metrics["cagr"] * 100.0, 2),
        "Vol %": round(strategy_metrics["volatility"] * 100.0, 2),
        "Max DD %": round(strategy_metrics["max_drawdown"] * 100.0, 2),
        "Sharpe": round(strategy_metrics["sharpe"], 2) if pd.notna(strategy_metrics["sharpe"]) else np.nan,
    })

    for symbol, eq in benchmark_equities.items():
        m = compute_equity_metrics(eq)
        records.append({
            "Series": symbol,
            "Total Return %": round(m["total_return"] * 100.0, 2),
            "CAGR %": round(m["cagr"] * 100.0, 2),
            "Vol %": round(m["volatility"] * 100.0, 2),
            "Max DD %": round(m["max_drawdown"] * 100.0, 2),
            "Sharpe": round(m["sharpe"], 2) if pd.notna(m["sharpe"]) else np.nan,
        })

    return pd.DataFrame(records)
