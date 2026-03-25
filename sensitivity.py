"""
ADAPT Parameter Sensitivity Analysis
======================================
Tests how much each parameter affects CAGR, MaxDD, and Sharpe
when varied ±1 and ±2 steps around the current value.

A robust strategy should show a PLATEAU — small changes produce
small performance changes. A knife-edge shows large swings.

Usage:
    cd ~/Desktop/ADAPT_DEPLOY_Claude
    ./venv/bin/python sensitivity.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import copy
import yaml
import numpy as np
import pandas as pd
from adapt.allocator.combined_dynamic import run_combined_dynamic
from adapt.data_loader import load_settings


def load_configs():
    with open("config/core_strategy.yaml") as f:
        core_cfg = yaml.safe_load(f)
    with open("config/alpha_strategy.yaml") as f:
        alpha_cfg = yaml.safe_load(f)
    return core_cfg, alpha_cfg


def run_with_core_param(param: str, value, settings, base_core_cfg, base_alpha_cfg=None):
    """Run combined backtest with one core parameter overridden."""
    core_cfg = copy.deepcopy(base_core_cfg)
    core_cfg["parameters"][param] = value

    from adapt.core.core_engine import run_core_backtest
    from adapt.alpha.alpha_engine import run_alpha_backtest
    from adapt.allocator.allocation_logic import load_allocator_config, choose_weights

    IBKR_CASH_YIELD = 0.032

    core_df, _ = run_core_backtest(settings=settings, core_cfg=core_cfg)
    alpha_df, _ = run_alpha_backtest(settings=settings)

    df = pd.DataFrame({
        "core_ret": core_df["ret"],
        "core_regime": core_df["signal_regime"],
        "alpha_ret": alpha_df["ret"],
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

    returns = pd.Series(combined)
    return compute_metrics(returns)


def run_with_alpha_param(param: str, value, settings, base_alpha_cfg):
    """Run combined backtest with one alpha parameter overridden."""
    alpha_cfg = copy.deepcopy(base_alpha_cfg)
    alpha_cfg["parameters"][param] = value

    from adapt.core.core_engine import run_core_backtest
    from adapt.alpha.alpha_engine import run_alpha_backtest
    from adapt.allocator.allocation_logic import load_allocator_config, choose_weights

    IBKR_CASH_YIELD = 0.032

    core_df, _ = run_core_backtest(settings=settings)
    alpha_df, _ = run_alpha_backtest(settings=settings, alpha_cfg=alpha_cfg)

    df = pd.DataFrame({
        "core_ret": core_df["ret"],
        "core_regime": core_df["signal_regime"],
        "alpha_ret": alpha_df["ret"],
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

    returns = pd.Series(combined)
    return compute_metrics(returns)


def compute_metrics(returns: pd.Series) -> dict:
    cum = (1 + returns).cumprod()
    years = len(returns) / 252.0
    cagr = cum.iloc[-1] ** (1.0 / years) - 1.0
    peak = cum.cummax()
    maxdd = float(((cum / peak) - 1.0).min())
    rf = 0.032 / 252
    excess = returns - rf
    sharpe = (excess.mean() * 252) / (excess.std() * 252 ** 0.5) if excess.std() > 0 else 0.0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe}


def sensitivity_row(param, current_val, test_val, base_metrics, result_metrics):
    delta = test_val - current_val
    sign = f"+{delta}" if delta > 0 else str(delta)
    cagr_chg = result_metrics["cagr"] - base_metrics["cagr"]
    dd_chg = result_metrics["maxdd"] - base_metrics["maxdd"]
    sharpe_chg = result_metrics["sharpe"] - base_metrics["sharpe"]

    flag = ""
    if abs(cagr_chg) > 0.05:
        flag = " ⚠️  SENSITIVE"
    elif abs(cagr_chg) > 0.02:
        flag = " 〰️  MODERATE"
    else:
        flag = " ✅ STABLE"

    return (
        f"  {param:<22} {str(current_val):>8} → {str(test_val):<8} ({sign:>4})  "
        f"CAGR: {result_metrics['cagr']:>7.2%} ({cagr_chg:>+7.2%})  "
        f"DD: {result_metrics['maxdd']:>7.2%} ({dd_chg:>+7.2%})  "
        f"Sharpe: {result_metrics['sharpe']:>6.3f} ({sharpe_chg:>+6.3f})"
        f"{flag}"
    )


def main():
    print("=" * 100)
    print("  ADAPT — PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 100)

    settings = load_settings()
    core_cfg, alpha_cfg = load_configs()

    # Baseline
    print("\n▶ Running baseline...")
    from adapt.allocator.combined_dynamic import run_combined_dynamic
    _, base_m = run_combined_dynamic()
    print(f"  Baseline CAGR: {base_m['cagr']:.2%}  MaxDD: {base_m['maxdd']:.2%}  Sharpe: {base_m['sharpe']:.3f}")

    base_metrics = {"cagr": base_m["cagr"], "maxdd": base_m["maxdd"], "sharpe": base_m["sharpe"]}

    # ── CORE Parameters ────────────────────────────────────────────────────────
    core_tests = {
        "sma_period":    [130, 140, 150, 160, 170],
        "rsi_period":    [30,  35,  40,  45,  50],
        "consec_days":   [1,   2,   3,   4],
        "rsi_oversold":  [25,  30,  35,  40,  45],
        "rsi_overbought":[75,  80,  85,  90,  95],
    }

    print("\n" + "=" * 100)
    print("  CORE PARAMETERS")
    print("=" * 100)

    for param, values in core_tests.items():
        current = core_cfg["parameters"].get(param) or core_cfg["thresholds"].get(param)
        print(f"\n  ── {param} (current: {current}) ──")

        for val in values:
            if val == current:
                print(f"  {'[BASELINE]':<22} {str(current):>8}          "
                      f"CAGR: {base_metrics['cagr']:>7.2%}           "
                      f"DD: {base_metrics['maxdd']:>7.2%}           "
                      f"Sharpe: {base_metrics['sharpe']:>6.3f}")
                continue
            try:
                m = run_with_core_param(param, val, settings, core_cfg)
                print(sensitivity_row(param, current, val, base_metrics, m))
            except Exception as e:
                print(f"  {param:<22} {str(current):>8} → {str(val):<8}  ERROR: {e}")

    # ── ALPHA Parameters ───────────────────────────────────────────────────────
    alpha_tests = {
        "swing_left":  [1, 2, 3, 4],
        "swing_right": [1, 2, 3, 4],
    }

    print("\n" + "=" * 100)
    print("  ALPHA PARAMETERS")
    print("=" * 100)

    for param, values in alpha_tests.items():
        current = alpha_cfg["parameters"][param]
        print(f"\n  ── {param} (current: {current}) ──")

        for val in values:
            if val == current:
                print(f"  {'[BASELINE]':<22} {str(current):>8}          "
                      f"CAGR: {base_metrics['cagr']:>7.2%}           "
                      f"DD: {base_metrics['maxdd']:>7.2%}           "
                      f"Sharpe: {base_metrics['sharpe']:>6.3f}")
                continue
            try:
                m = run_with_alpha_param(param, val, settings, alpha_cfg)
                print(sensitivity_row(param, current, val, base_metrics, m))
            except Exception as e:
                print(f"  {param:<22} {str(current):>8} → {str(val):<8}  ERROR: {e}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  INTERPRETATION")
    print("=" * 100)
    print("""
  ✅ STABLE    — CAGR change < 2%  → parameter sitting on a robust plateau
  〰️  MODERATE  — CAGR change 2-5%  → some sensitivity, worth monitoring
  ⚠️  SENSITIVE — CAGR change > 5%  → parameter is load-bearing, handle carefully

  What you want to see:
    • Most parameters STABLE across the range
    • No single parameter causes > 5% CAGR swing
    • The current values sit near the middle of the best range (plateau center)
    • If a parameter is SENSITIVE, do NOT tune it further — you risk overfitting
""")


if __name__ == "__main__":
    main()
