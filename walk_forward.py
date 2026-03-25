"""
ADAPT Walk-Forward Validation Framework
========================================
Tests whether fixed strategy parameters perform consistently
across rolling in-sample / out-of-sample windows.

Usage:
    cd ~/Desktop/ADAPT_DEPLOY_Claude
    ./venv/bin/python walk_forward.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from adapt.allocator.combined_dynamic import run_combined_dynamic
from adapt.data_loader import load_settings

# ── Configuration ──────────────────────────────────────────────────────────────
IS_YEARS  = 3      # In-sample window length
OOS_YEARS = 1      # Out-of-sample window length
STEP_YEARS = 1     # How far to slide each iteration
TRADING_DAYS = 252
# ──────────────────────────────────────────────────────────────────────────────


def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute CAGR, MaxDD, Sharpe from a return series."""
    if len(returns) < 5:
        return {"cagr": np.nan, "maxdd": np.nan, "sharpe": np.nan}

    cum = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS

    cagr = (cum.iloc[-1]) ** (1.0 / years) - 1.0

    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    maxdd = float(dd.min())

    rf_daily = 0.032 / TRADING_DAYS
    excess = returns - rf_daily
    sharpe = (excess.mean() * TRADING_DAYS) / (excess.std() * TRADING_DAYS ** 0.5) if excess.std() > 0 else 0.0

    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe}


def efficiency_ratio(oos_cagr: float, is_cagr: float) -> float:
    """OOS / IS efficiency. 1.0 = perfect, >0.5 = acceptable."""
    if is_cagr <= 0:
        return np.nan
    return oos_cagr / is_cagr


def robustness_score(results: list[dict]) -> str:
    """Simple robustness classification based on efficiency ratios."""
    ratios = [r["efficiency"] for r in results if not np.isnan(r["efficiency"])]
    if not ratios:
        return "INSUFFICIENT DATA"
    avg = np.mean(ratios)
    pct_positive_oos = np.mean([r["oos_cagr"] > 0 for r in results])

    if avg >= 0.70 and pct_positive_oos >= 0.80:
        return "STRONG ✅"
    elif avg >= 0.50 and pct_positive_oos >= 0.65:
        return "ACCEPTABLE ⚠️"
    else:
        return "WEAK ❌"


def main():
    print("=" * 72)
    print("  ADAPT — WALK-FORWARD VALIDATION")
    print(f"  IS: {IS_YEARS}yr  |  OOS: {OOS_YEARS}yr  |  Step: {STEP_YEARS}yr")
    print("=" * 72)

    # Run full backtest once to get the complete return series
    print("\n▶ Running full combined backtest...")
    full_df, full_metrics = run_combined_dynamic()

    print(f"  Full period:  {full_df.index[0].date()} → {full_df.index[-1].date()}")
    print(f"  Full CAGR:    {full_metrics['cagr']:.2%}")
    print(f"  Full MaxDD:   {full_metrics['maxdd']:.2%}")
    print(f"  Full Sharpe:  {full_metrics['sharpe']:.3f}")
    print(f"  Total bars:   {len(full_df)}")

    returns = full_df["ret"]
    dates = returns.index

    IS_BARS   = IS_YEARS  * TRADING_DAYS
    OOS_BARS  = OOS_YEARS * TRADING_DAYS
    STEP_BARS = STEP_YEARS * TRADING_DAYS
    WINDOW    = IS_BARS + OOS_BARS

    n_windows = (len(returns) - WINDOW) // STEP_BARS + 1

    print(f"\n▶ Running {n_windows} walk-forward windows...\n")

    header = (
        f"{'Window':<8} {'IS Start':>10} {'IS End':>10} "
        f"{'OOS Start':>10} {'OOS End':>10} "
        f"{'IS CAGR':>8} {'OOS CAGR':>9} {'Efficiency':>11} "
        f"{'OOS DD':>8} {'OOS Sharpe':>11}"
    )
    print(header)
    print("-" * len(header))

    results = []

    for w in range(n_windows):
        start     = w * STEP_BARS
        is_end    = start + IS_BARS
        oos_end   = is_end + OOS_BARS

        if oos_end > len(returns):
            break

        is_ret  = returns.iloc[start:is_end]
        oos_ret = returns.iloc[is_end:oos_end]

        is_m  = compute_metrics(is_ret)
        oos_m = compute_metrics(oos_ret)

        eff = efficiency_ratio(oos_m["cagr"], is_m["cagr"])

        result = {
            "window":    w + 1,
            "is_start":  dates[start].date(),
            "is_end":    dates[is_end - 1].date(),
            "oos_start": dates[is_end].date(),
            "oos_end":   dates[oos_end - 1].date(),
            "is_cagr":   is_m["cagr"],
            "oos_cagr":  oos_m["cagr"],
            "efficiency": eff,
            "oos_maxdd": oos_m["maxdd"],
            "oos_sharpe": oos_m["sharpe"],
        }
        results.append(result)

        eff_str = f"{eff:.2f}" if not np.isnan(eff) else "  N/A"
        flag = ""
        if not np.isnan(eff):
            if eff < 0:
                flag = " ❌"
            elif eff < 0.50:
                flag = " ⚠️"
            else:
                flag = " ✅"

        print(
            f"  {w+1:<6} "
            f"{str(dates[start].date()):>10} {str(dates[is_end-1].date()):>10} "
            f"{str(dates[is_end].date()):>10} {str(dates[oos_end-1].date()):>10} "
            f"{is_m['cagr']:>8.2%} {oos_m['cagr']:>9.2%} "
            f"{eff_str:>11}{flag} "
            f"{oos_m['maxdd']:>8.2%} {oos_m['sharpe']:>11.3f}"
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY STATISTICS")
    print("=" * 72)

    is_cagrs  = [r["is_cagr"]  for r in results]
    oos_cagrs = [r["oos_cagr"] for r in results]
    effs      = [r["efficiency"] for r in results if not np.isnan(r["efficiency"])]
    oos_dds   = [r["oos_maxdd"] for r in results]

    print(f"\n  IS  CAGR  —  Mean: {np.mean(is_cagrs):.2%}  |  Min: {np.min(is_cagrs):.2%}  |  Max: {np.max(is_cagrs):.2%}")
    print(f"  OOS CAGR  —  Mean: {np.mean(oos_cagrs):.2%}  |  Min: {np.min(oos_cagrs):.2%}  |  Max: {np.max(oos_cagrs):.2%}")
    print(f"  OOS MaxDD —  Mean: {np.mean(oos_dds):.2%}  |  Worst: {np.min(oos_dds):.2%}")
    print(f"\n  Efficiency Ratio — Mean: {np.mean(effs):.2f}  |  Min: {np.min(effs):.2f}  |  Max: {np.max(effs):.2f}")
    print(f"  OOS Windows Positive: {sum(c > 0 for c in oos_cagrs)} / {len(oos_cagrs)}")
    print(f"  OOS Windows > 10%%:   {sum(c > 0.10 for c in oos_cagrs)} / {len(oos_cagrs)}")

    score = robustness_score(results)
    print(f"\n  ROBUSTNESS SCORE: {score}")

    print("\n" + "=" * 72)
    print("  INTERPRETATION GUIDE")
    print("=" * 72)
    print("""
  Efficiency Ratio (OOS CAGR / IS CAGR):
    >= 0.70  → Strong robustness, strategy generalises well
    0.50-0.70 → Acceptable, some degradation expected
    0.30-0.50 → Weak, significant overfitting likely
    < 0.30   → Strategy is curve-fitted to historical data

  What to look for:
    • Consistent OOS CAGR across windows = robust
    • Large IS/OOS gap in specific windows = regime sensitivity
    • Negative OOS CAGR windows = parameter fragility
    • OOS MaxDD >> full-period MaxDD = tail risk underestimated
""")


if __name__ == "__main__":
    main()
