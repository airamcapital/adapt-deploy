"""
ADAPT — Benchmark Comparison
==============================
Compares ADAPT combined strategy vs TQQQ, QQQ, SPY
starting from $50,000 on 2012-05-25 to present.

Shows: Final Value, CAGR, Max Drawdown, Sharpe, Calmar

Usage:
    cd ~/Desktop/ADAPT_DEPLOY_Claude
    ./venv/bin/python benchmark_comparison.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from adapt.allocator.combined_dynamic import run_combined_dynamic
from adapt.data_loader import load_settings, get_close_history

START_CAPITAL = 50_000.0
TRADING_DAYS  = 252
RF_DAILY      = 0.032 / TRADING_DAYS


def compute_metrics(returns: pd.Series, start_capital: float) -> dict:
    cum = (1 + returns).cumprod()
    equity = cum * start_capital

    years = len(returns) / TRADING_DAYS
    final_value = float(equity.iloc[-1])
    cagr = (final_value / start_capital) ** (1.0 / years) - 1.0

    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    maxdd = float(dd.min())

    excess = returns - RF_DAILY
    sharpe = (excess.mean() * TRADING_DAYS) / (excess.std() * TRADING_DAYS ** 0.5) if excess.std() > 0 else 0.0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0.0

    return {
        "final_value": final_value,
        "total_return": (final_value / start_capital) - 1.0,
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "years": years,
    }


def worst_drawdown_periods(equity: pd.Series, top_n: int = 3) -> list[dict]:
    """Find the top N worst drawdown periods with dates."""
    peak = equity.cummax()
    dd = (equity / peak) - 1.0

    periods = []
    in_dd = False
    start = None
    trough_date = None
    trough_val = 0.0

    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = date
            trough_date = date
            trough_val = val
        elif val < trough_val and in_dd:
            trough_date = date
            trough_val = val
        elif val == 0.0 and in_dd:
            periods.append({
                "start": start,
                "trough": trough_date,
                "end": date,
                "maxdd": trough_val,
                "duration": (date - start).days,
            })
            in_dd = False

    if in_dd:
        periods.append({
            "start": start,
            "trough": trough_date,
            "end": equity.index[-1],
            "maxdd": trough_val,
            "duration": (equity.index[-1] - start).days,
        })

    return sorted(periods, key=lambda x: x["maxdd"])[:top_n]


def main():
    print("=" * 80)
    print("  ADAPT — BENCHMARK COMPARISON")
    print(f"  Starting Capital: ${START_CAPITAL:,.0f}  |  Start: 2012-05-25  |  End: Present")
    print("=" * 80)

    settings = load_settings()

    # ── ADAPT ─────────────────────────────────────────────────────────────────
    print("\n▶ Running ADAPT combined strategy...")
    df, _ = run_combined_dynamic()
    adapt_returns = df["ret"]
    adapt_metrics = compute_metrics(adapt_returns, START_CAPITAL)
    adapt_equity = (1 + adapt_returns).cumprod() * START_CAPITAL

    # ── Benchmarks ────────────────────────────────────────────────────────────
    print("▶ Loading benchmark data...")
    benchmarks_raw = get_close_history(
        tickers=["TQQQ", "QQQ", "SPY"],
        settings=settings,
        cache_name="benchmark_prices.csv",
    )

    # Align to ADAPT start date
    start_date = adapt_returns.index[0]
    benchmarks_raw = benchmarks_raw[benchmarks_raw.index >= start_date]

    benchmark_results = {}
    for ticker in ["TQQQ", "QQQ", "SPY"]:
        prices = benchmarks_raw[ticker].dropna()
        # Align to same dates as ADAPT
        prices = prices[prices.index.isin(adapt_returns.index)]
        rets = prices.pct_change().dropna()
        rets = rets[rets.index.isin(adapt_returns.index)]
        m = compute_metrics(rets, START_CAPITAL)
        equity = (1 + rets).cumprod() * START_CAPITAL
        benchmark_results[ticker] = {"metrics": m, "equity": equity, "returns": rets}

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PERFORMANCE SUMMARY")
    print("=" * 80)

    all_results = [("ADAPT", adapt_metrics)] + [
        (t, benchmark_results[t]["metrics"]) for t in ["TQQQ", "QQQ", "SPY"]
    ]

    print(f"\n  {'Strategy':<10} {'Start':>10} {'Final Value':>14} {'Total Ret':>10} "
          f"{'CAGR':>8} {'Max DD':>9} {'Sharpe':>8} {'Calmar':>8}")
    print(f"  {'─'*79}")

    for name, m in all_results:
        marker = " ◀" if name == "ADAPT" else ""
        print(
            f"  {name:<10} "
            f"${START_CAPITAL:>9,.0f} "
            f"${m['final_value']:>13,.0f} "
            f"{m['total_return']:>10.2%} "
            f"{m['cagr']:>8.2%} "
            f"{m['maxdd']:>9.2%} "
            f"{m['sharpe']:>8.3f} "
            f"{m['calmar']:>8.3f}"
            f"{marker}"
        )

    # ── ADAPT vs each benchmark ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ADAPT EDGE OVER BENCHMARKS")
    print("=" * 80)
    print(f"\n  {'vs':<8} {'CAGR Edge':>10} {'DD Reduction':>14} {'Sharpe Edge':>13} {'Extra $':>14}")
    print(f"  {'─'*61}")
    for ticker in ["TQQQ", "QQQ", "SPY"]:
        bm = benchmark_results[ticker]["metrics"]
        print(
            f"  vs {ticker:<5} "
            f"{adapt_metrics['cagr'] - bm['cagr']:>+10.2%} "
            f"{adapt_metrics['maxdd'] - bm['maxdd']:>+14.2%} "
            f"{adapt_metrics['sharpe'] - bm['sharpe']:>+13.3f} "
            f"${adapt_metrics['final_value'] - bm['final_value']:>+13,.0f}"
        )

    # ── Yearly Returns Table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  YEARLY RETURNS")
    print("=" * 80)

    all_returns = {"ADAPT": adapt_returns}
    for t in ["TQQQ", "QQQ", "SPY"]:
        all_returns[t] = benchmark_results[t]["returns"]

    years = sorted(set(adapt_returns.index.year))
    print(f"\n  {'Year':<6} {'ADAPT':>8} {'TQQQ':>8} {'QQQ':>8} {'SPY':>8}  {'Winner':<8}")
    print(f"  {'─'*56}")

    for year in years:
        row = {}
        for name, rets in all_returns.items():
            yr = rets[rets.index.year == year]
            if len(yr) > 5:
                row[name] = float((1 + yr).prod() - 1.0)
            else:
                row[name] = None

        if row.get("ADAPT") is None:
            continue

        valid = {k: v for k, v in row.items() if v is not None}
        winner = max(valid, key=lambda k: valid[k])

        print(
            f"  {year:<6} "
            f"{row['ADAPT']:>8.2%} "
            f"{row.get('TQQQ', float('nan')):>8.2%} "
            f"{row.get('QQQ', float('nan')):>8.2%} "
            f"{row.get('SPY', float('nan')):>8.2%}  "
            f"{'★ ' + winner if winner == 'ADAPT' else winner:<8}"
        )

    # ── Worst Drawdown Periods ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  TOP 3 WORST DRAWDOWN PERIODS")
    print("=" * 80)

    all_equities = {"ADAPT": adapt_equity}
    for t in ["TQQQ", "QQQ", "SPY"]:
        all_equities[t] = benchmark_results[t]["equity"]

    for name, equity in all_equities.items():
        periods = worst_drawdown_periods(equity, top_n=3)
        print(f"\n  {name}:")
        print(f"  {'─'*60}")
        for i, p in enumerate(periods, 1):
            print(
                f"  #{i}  Start: {p['start'].date()}  "
                f"Trough: {p['trough'].date()}  "
                f"{'Recovered: ' + str(p['end'].date()) if p['end'] != equity.index[-1] else 'Ongoing    ':>22}  "
                f"MaxDD: {p['maxdd']:>8.2%}  "
                f"Duration: {p['duration']} days"
            )

    # ── Risk-Adjusted Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RISK-ADJUSTED SUMMARY")
    print("=" * 80)
    print(f"""
  ADAPT delivers {adapt_metrics['cagr']:.2%} CAGR with only {adapt_metrics['maxdd']:.2%} max drawdown.
  Calmar ratio of {adapt_metrics['calmar']:.3f} means you earn {adapt_metrics['calmar']:.1f}x your worst drawdown annually.

  Comparison:
    TQQQ: Higher raw return potential but {benchmark_results['TQQQ']['metrics']['maxdd']:.2%} max drawdown
    QQQ:  {benchmark_results['QQQ']['metrics']['cagr']:.2%} CAGR with {benchmark_results['QQQ']['metrics']['maxdd']:.2%} max drawdown
    SPY:  {benchmark_results['SPY']['metrics']['cagr']:.2%} CAGR with {benchmark_results['SPY']['metrics']['maxdd']:.2%} max drawdown

  ADAPT Calmar: {adapt_metrics['calmar']:.3f}  vs  TQQQ: {benchmark_results['TQQQ']['metrics']['calmar']:.3f}  
  QQQ: {benchmark_results['QQQ']['metrics']['calmar']:.3f}  vs  SPY: {benchmark_results['SPY']['metrics']['calmar']:.3f}
""")


if __name__ == "__main__":
    main()
