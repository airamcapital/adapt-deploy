from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from adapt.data_loader import load_settings
from adapt.core.core_engine import run_core_backtest
from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.allocator.combined_dynamic import run_combined_dynamic


def download_benchmark_returns(start: str) -> pd.DataFrame:
    raw = yf.download(["SPY", "QQQ", "TQQQ"], start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw.copy()
    return px.pct_change().dropna(how="all")


def performance_metrics(returns: pd.Series, start_value: float = 50000.0) -> dict:
    returns = returns.dropna().copy()
    if len(returns) == 0:
        return {
            "cagr": 0.0,
            "maxdd": 0.0,
            "sharpe": 0.0,
            "final_value": start_value,
            "days": 0,
        }

    pv = start_value
    peak = start_value
    dds = []

    for r in returns:
        pv *= (1.0 + float(r))
        peak = max(peak, pv)
        dds.append((pv / peak) - 1.0)

    years = len(returns) / 252.0
    cagr = (pv / start_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    maxdd = float(min(dds)) if dds else 0.0
    std = returns.std()
    sharpe = (returns.mean() * 252.0) / (std * (252.0 ** 0.5)) if std and std > 0 else 0.0

    return {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "final_value": pv,
        "days": len(returns),
    }


def pick_strategy_df(strategy: str, settings: dict) -> pd.DataFrame:
    if strategy == "core":
        df, _ = run_core_backtest(settings=settings)
        return df[["ret"]].copy()

    if strategy == "alpha":
        df, _ = run_alpha_backtest(settings=settings)
        return df[["ret"]].copy()

    if strategy == "combined":
        df, _ = run_combined_dynamic(settings=settings)
        return df[["ret"]].copy()

    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ADAPT strategies over a chosen date range.")
    parser.add_argument("--strategy", choices=["core", "alpha", "combined"], required=True)
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    settings = load_settings()
    start_value = float(settings["portfolio"]["start_value"])

    strat_df = pick_strategy_df(args.strategy, settings)
    bench_ret = download_benchmark_returns(settings["data"]["start_date"])

    joined = pd.DataFrame(index=strat_df.index)
    joined["strategy"] = strat_df["ret"]
    joined = joined.join(bench_ret, how="inner")

    joined = joined.loc[args.start:args.end].dropna(how="all")

    if len(joined) == 0:
        print("No data available for that date range.")
        return

    strategy_metrics = performance_metrics(joined["strategy"], start_value)
    spy_metrics = performance_metrics(joined["SPY"], start_value)
    qqq_metrics = performance_metrics(joined["QQQ"], start_value)
    tqqq_metrics = performance_metrics(joined["TQQQ"], start_value)

    print("=" * 108)
    print("  ADAPT DATE-RANGE VALIDATION / BENCHMARK COMPARISON")
    print("=" * 108)
    print(f"Strategy : {args.strategy}")
    print(f"Window   : {args.start} to {args.end}")
    print(f"Days     : {len(joined)}")
    print()

    print(f"{'Model':<14} {'CAGR':>10} {'MaxDD':>10} {'Sharpe':>10} {'Final Value':>16}")
    print("-" * 108)
    for label, m in [
        ("ADAPT", strategy_metrics),
        ("SPY", spy_metrics),
        ("QQQ", qqq_metrics),
        ("TQQQ", tqqq_metrics),
    ]:
        print(
            f"{label:<14} "
            f"{m['cagr']:>9.2%} "
            f"{m['maxdd']:>9.2%} "
            f"{m['sharpe']:>10.3f} "
            f"${m['final_value']:>15,.0f}"
        )

    print()
    print("YEARLY RETURNS")
    print("-" * 108)
    years = sorted(joined.index.year.unique())
    print(f"{'Year':<6} {'ADAPT':>10} {'SPY':>10} {'QQQ':>10} {'TQQQ':>10}")
    print("-" * 108)

    for yr in years:
        sub = joined[joined.index.year == yr]
        if len(sub) == 0:
            continue

        def yr_ret(s: pd.Series) -> float:
            s = s.dropna()
            return float((1.0 + s).prod() - 1.0) if len(s) else 0.0

        print(
            f"{yr:<6} "
            f"{yr_ret(sub['strategy']):>9.1%} "
            f"{yr_ret(sub['SPY']):>9.1%} "
            f"{yr_ret(sub['QQQ']):>9.1%} "
            f"{yr_ret(sub['TQQQ']):>9.1%}"
        )


if __name__ == "__main__":
    main()
