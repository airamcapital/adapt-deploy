"""
ADAPT — RSI Threshold Fix Test
================================
Tests statistically-grounded RSI thresholds for RSI(40).

Current thresholds were designed for RSI(14) and are unreachable
with RSI(40). This script tests corrected thresholds based on
the actual percentile distribution of RSI(40) on TQQQ.

New thresholds (statistically grounded, NOT optimized):
  rsi_oversold  : 35 → 45  (10th percentile of RSI(40))
  rsi_overbought: 85 → 63  (90th percentile of RSI(40))

Usage:
    cd ~/Desktop/ADAPT_DEPLOY_Claude
    ./venv/bin/python rsi_fix_test.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import copy
import yaml
import numpy as np
import pandas as pd


def load_configs():
    with open("config/core_strategy.yaml") as f:
        core_cfg = yaml.safe_load(f)
    with open("config/alpha_strategy.yaml") as f:
        alpha_cfg = yaml.safe_load(f)
    return core_cfg, alpha_cfg


def run_combined_with_core_cfg(core_cfg, settings=None):
    """Run full combined backtest with a custom core config."""
    from adapt.core.core_engine import run_core_backtest
    from adapt.alpha.alpha_engine import run_alpha_backtest
    from adapt.allocator.allocation_logic import load_allocator_config, choose_weights
    from adapt.data_loader import load_settings

    IBKR_CASH_YIELD = 0.032
    settings = settings or load_settings()

    core_df, _ = run_core_backtest(settings=settings, core_cfg=core_cfg)
    alpha_df, _ = run_alpha_backtest(settings=settings)

    df = pd.DataFrame({
        "core_ret":     core_df["ret"],
        "core_regime":  core_df["signal_regime"],
        "alpha_ret":    alpha_df["ret"],
        "alpha_weight": alpha_df["weight"],
    }).dropna()

    allocator_cfg = load_allocator_config()
    daily_cash = IBKR_CASH_YIELD / 252.0
    combined = []

    for _, row in df.iterrows():
        cw, aw, _ = choose_weights(
            core_regime=int(row["core_regime"]),
            alpha_in_market=float(row["alpha_weight"]) > 0.0,
            allocator_cfg=allocator_cfg,
        )
        a_ret = float(row["alpha_ret"]) if float(row["alpha_weight"]) > 0.0 else daily_cash
        combined.append(cw * float(row["core_ret"]) + aw * a_ret)

    returns = pd.Series(combined, index=df.index)

    # Regime breakdown
    regime_counts = core_df["signal_regime"].value_counts().sort_index()

    return returns, regime_counts


def compute_metrics(returns: pd.Series) -> dict:
    cum = (1 + returns).cumprod()
    years = len(returns) / 252.0
    cagr = cum.iloc[-1] ** (1.0 / years) - 1.0
    peak = cum.cummax()
    maxdd = float(((cum / peak) - 1.0).min())
    rf = 0.032 / 252
    excess = returns - rf
    sharpe = (excess.mean() * 252) / (excess.std() * 252 ** 0.5) if excess.std() > 0 else 0.0
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0.0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "calmar": calmar}


def print_result(label, metrics, regime_counts, base_metrics=None):
    print(f"\n  {'─'*70}")
    print(f"  {label}")
    print(f"  {'─'*70}")
    print(f"  CAGR:   {metrics['cagr']:>8.2%}", end="")
    if base_metrics:
        delta = metrics['cagr'] - base_metrics['cagr']
        print(f"  ({delta:>+.2%} vs baseline)", end="")
    print()
    print(f"  MaxDD:  {metrics['maxdd']:>8.2%}", end="")
    if base_metrics:
        delta = metrics['maxdd'] - base_metrics['maxdd']
        print(f"  ({delta:>+.2%} vs baseline)", end="")
    print()
    print(f"  Sharpe: {metrics['sharpe']:>8.3f}", end="")
    if base_metrics:
        delta = metrics['sharpe'] - base_metrics['sharpe']
        print(f"  ({delta:>+.3f} vs baseline)", end="")
    print()
    print(f"  Calmar: {metrics['calmar']:>8.3f}")
    print(f"\n  Regime breakdown:")
    total = regime_counts.sum()
    for regime, count in regime_counts.items():
        names = {1: "RSI Oversold", 2: "RSI Overbought", 3: "Momentum", 4: "Defensive"}
        name = names.get(int(regime), f"Regime {regime}")
        print(f"    Regime {int(regime)} ({name:<15}): {count:>5} days ({count/total:.1%})")


def main():
    print("=" * 72)
    print("  ADAPT — RSI THRESHOLD FIX TEST")
    print("=" * 72)
    print("""
  RSI(40) on TQQQ distribution:
    5th  pct: 42.3  |  10th pct: 44.7  |  25th pct: 49.4
    75th pct: 58.9  |  90th pct: 62.4  |  95th pct: 64.7

  Current thresholds (designed for RSI(14), unreachable with RSI(40)):
    rsi_oversold:   35  →  RSI min ever: 34.8  →  fired 0 times
    rsi_overbought: 85  →  RSI max ever: 73.8  →  fired 0 times

  Proposed thresholds (statistically grounded on RSI(40) distribution):
    rsi_oversold:   45  (10th percentile)
    rsi_overbought: 63  (90th percentile)

  NOTE: These thresholds are set by percentile, NOT by optimizing CAGR.
""")

    from adapt.data_loader import load_settings
    settings = load_settings()
    base_core_cfg, _ = load_configs()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("▶ Running baseline (current thresholds: oversold=35, overbought=85)...")
    base_returns, base_regimes = run_combined_with_core_cfg(base_core_cfg, settings)
    base_metrics = compute_metrics(base_returns)
    print_result("BASELINE", base_metrics, base_regimes)

    # ── Test 1: Both thresholds corrected ─────────────────────────────────────
    print("\n▶ Test 1: Both thresholds corrected (oversold=45, overbought=63)...")
    cfg1 = copy.deepcopy(base_core_cfg)
    cfg1["thresholds"]["rsi_oversold"] = 45
    cfg1["thresholds"]["rsi_overbought"] = 63
    ret1, reg1 = run_combined_with_core_cfg(cfg1, settings)
    m1 = compute_metrics(ret1)
    print_result("CORRECTED THRESHOLDS (oversold=45, overbought=63)", m1, reg1, base_metrics)

    # ── Test 2: Only oversold corrected ───────────────────────────────────────
    print("\n▶ Test 2: Only oversold corrected (oversold=45, overbought=85)...")
    cfg2 = copy.deepcopy(base_core_cfg)
    cfg2["thresholds"]["rsi_oversold"] = 45
    ret2, reg2 = run_combined_with_core_cfg(cfg2, settings)
    m2 = compute_metrics(ret2)
    print_result("OVERSOLD ONLY (oversold=45, overbought=85)", m2, reg2, base_metrics)

    # ── Test 3: Only overbought corrected ─────────────────────────────────────
    print("\n▶ Test 3: Only overbought corrected (oversold=35, overbought=63)...")
    cfg3 = copy.deepcopy(base_core_cfg)
    cfg3["thresholds"]["rsi_overbought"] = 63
    ret3, reg3 = run_combined_with_core_cfg(cfg3, settings)
    m3 = compute_metrics(ret3)
    print_result("OVERBOUGHT ONLY (oversold=35, overbought=63)", m3, reg3, base_metrics)

    # ── Test 4: Wider thresholds (5th/95th pct) ───────────────────────────────
    print("\n▶ Test 4: Wider thresholds — 5th/95th pct (oversold=42, overbought=65)...")
    cfg4 = copy.deepcopy(base_core_cfg)
    cfg4["thresholds"]["rsi_oversold"] = 42
    cfg4["thresholds"]["rsi_overbought"] = 65
    ret4, reg4 = run_combined_with_core_cfg(cfg4, settings)
    m4 = compute_metrics(ret4)
    print_result("WIDER THRESHOLDS (oversold=42, overbought=65)", m4, reg4, base_metrics)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    results = [
        ("Baseline (35/85)",        base_metrics),
        ("Corrected (45/63)",        m1),
        ("Oversold only (45/85)",    m2),
        ("Overbought only (35/63)",  m3),
        ("Wider (42/65)",            m4),
    ]

    print(f"\n  {'Scenario':<30} {'CAGR':>8} {'MaxDD':>9} {'Sharpe':>8} {'Calmar':>8}")
    print(f"  {'─'*65}")
    for label, m in results:
        marker = " ◀ BASELINE" if label.startswith("Baseline") else ""
        print(f"  {label:<30} {m['cagr']:>8.2%} {m['maxdd']:>9.2%} {m['sharpe']:>8.3f} {m['calmar']:>8.3f}{marker}")

    print("""
  DECISION GUIDE:
    • If corrected thresholds improve CAGR meaningfully → update config
    • If they make no difference → the Regime 1/2 logic is still not firing
      (structural contradiction: oversold RSI rarely occurs above the SMA)
    • If they hurt performance → leave thresholds as documented dead code
    • In all cases: no CAGR-chasing. Accept the result as-is.
""")


if __name__ == "__main__":
    main()
