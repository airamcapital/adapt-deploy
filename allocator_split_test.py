"""
ADAPT — Allocator Neutral State Split Test
===========================================
The neutral state currently covers two fundamentally different situations
treated identically at 50/50 core/alpha:

  neutral_r3: Core=Regime3 (Momentum) + Alpha OUT of market  →  +80.75% annualized
  neutral_r4: Core=Regime4 (Defensive) + Alpha IN market     →  -45.22% annualized

This script tests whether splitting these into separate weight allocations
improves performance. Logic is structural — weights are set by directional
reasoning, not CAGR optimization.

Reasoning:
  neutral_r3 → Core is in momentum, trust it more → test higher core weight
  neutral_r4 → Core is defensive, Alpha fighting it and losing → reduce alpha weight

Usage:
    cd ~/Desktop/ADAPT_DEPLOY_Claude
    ./venv/bin/python allocator_split_test.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from adapt.core.core_engine import run_core_backtest
from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.data_loader import load_settings


IBKR_CASH_YIELD = 0.032


def build_combined(core_df, alpha_df, weights_risk_on, weights_neutral_r3,
                   weights_neutral_r4, weights_risk_off):
    """Build combined return series with custom allocator weights."""
    df = pd.DataFrame({
        "core_ret":     core_df["ret"],
        "core_regime":  core_df["signal_regime"],
        "alpha_ret":    alpha_df["ret"],
        "alpha_weight": alpha_df["weight"],
    }).dropna()

    daily_cash = IBKR_CASH_YIELD / 252.0
    combined = []
    states = []

    for _, row in df.iterrows():
        cr = int(row["core_regime"])
        ai = float(row["alpha_weight"]) > 0.0
        a_ret = float(row["alpha_ret"]) if ai else daily_cash

        if cr == 3 and ai:
            cw, aw = weights_risk_on
            state = "risk_on"
        elif cr == 3 and not ai:
            cw, aw = weights_neutral_r3
            state = "neutral_r3"
        elif cr == 4 and ai:
            cw, aw = weights_neutral_r4
            state = "neutral_r4"
        else:
            cw, aw = weights_risk_off
            state = "risk_off"

        combined.append(cw * float(row["core_ret"]) + aw * a_ret)
        states.append(state)

    returns = pd.Series(combined, index=df.index)
    return returns, states


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


def walk_forward_score(returns: pd.Series) -> tuple[float, int]:
    """Quick walk-forward: mean OOS CAGR and positive window count."""
    IS, OOS = 756, 252
    oos_cagrs = []
    for w in range(10):
        s = w * 252
        is_r  = returns.iloc[s:s+IS]
        oos_r = returns.iloc[s+IS:s+IS+OOS]
        if len(oos_r) < 200:
            break
        cum = (1 + oos_r).cumprod()
        oos_cagrs.append(cum.iloc[-1] ** (252 / len(oos_r)) - 1.0)
    if not oos_cagrs:
        return 0.0, 0
    return float(np.mean(oos_cagrs)), sum(c > 0 for c in oos_cagrs)


def print_result(label, metrics, wf_mean, wf_pos, base_metrics=None):
    cagr_d  = f"({metrics['cagr']-base_metrics['cagr']:>+.2%})" if base_metrics else ""
    maxdd_d = f"({metrics['maxdd']-base_metrics['maxdd']:>+.2%})" if base_metrics else ""
    sharpe_d= f"({metrics['sharpe']-base_metrics['sharpe']:>+.3f})" if base_metrics else ""
    print(f"\n  {label}")
    print(f"  {'─'*65}")
    print(f"  CAGR:      {metrics['cagr']:>8.2%}  {cagr_d}")
    print(f"  MaxDD:     {metrics['maxdd']:>8.2%}  {maxdd_d}")
    print(f"  Sharpe:    {metrics['sharpe']:>8.3f}  {sharpe_d}")
    print(f"  Calmar:    {metrics['calmar']:>8.3f}")
    print(f"  WF OOS CAGR (mean): {wf_mean:>7.2%}  |  Positive windows: {wf_pos}/10")


def main():
    print("=" * 72)
    print("  ADAPT — ALLOCATOR NEUTRAL STATE SPLIT TEST")
    print("=" * 72)
    print("""
  Current neutral state (50/50) covers two opposite situations:
    neutral_r3  Core=R3 + Alpha OUT  →  +80.75% annualized  (222 days)
    neutral_r4  Core=R4 + Alpha IN   →  -45.22% annualized  (1555 days)

  Test logic (structural reasoning, not CAGR optimization):
    neutral_r3: Core in momentum, trust it more → increase core weight
    neutral_r4: Core defensive, Alpha fighting it → reduce alpha weight
""")

    settings = load_settings()
    print("▶ Loading engine data...")
    core_df, _ = run_core_backtest(settings=settings)
    alpha_df, _ = run_alpha_backtest(settings=settings)

    # ── Baseline (current allocator) ──────────────────────────────────────────
    print("\n▶ Running baseline...")
    base_ret, _ = build_combined(
        core_df, alpha_df,
        weights_risk_on    = (0.40, 0.60),
        weights_neutral_r3 = (0.50, 0.50),   # current: no split
        weights_neutral_r4 = (0.50, 0.50),   # current: no split
        weights_risk_off   = (0.80, 0.20),
    )
    base_m = compute_metrics(base_ret)
    base_wf, base_pos = walk_forward_score(base_ret)
    print_result("BASELINE (current: neutral=50/50 for both sub-states)",
                 base_m, base_wf, base_pos)

    # ── Test scenarios ────────────────────────────────────────────────────────
    # Reasoning:
    # neutral_r3 → Core momentum dominant, Alpha sitting out → trust Core more
    # neutral_r4 → Core defensive, Alpha fighting it → reduce Alpha weight
    scenarios = [
        {
            "label": "Test 1 — R3_out=70/30, R4_in=70/30 (trust Core in both)",
            "neutral_r3": (0.70, 0.30),
            "neutral_r4": (0.70, 0.30),
        },
        {
            "label": "Test 2 — R3_out=70/30, R4_in=80/20 (Core dominant when defensive)",
            "neutral_r3": (0.70, 0.30),
            "neutral_r4": (0.80, 0.20),
        },
        {
            "label": "Test 3 — R3_out=60/40, R4_in=80/20 (mild R3 shift, strong R4 shift)",
            "neutral_r3": (0.60, 0.40),
            "neutral_r4": (0.80, 0.20),
        },
        {
            "label": "Test 4 — R3_out=70/30, R4_in=90/10 (max Core trust when defensive)",
            "neutral_r3": (0.70, 0.30),
            "neutral_r4": (0.90, 0.10),
        },
        {
            "label": "Test 5 — R3_out=80/20, R4_in=80/20 (equal trust both)",
            "neutral_r3": (0.80, 0.20),
            "neutral_r4": (0.80, 0.20),
        },
        {
            "label": "Test 6 — R3_out=50/50, R4_in=80/20 (only fix the problem sub-state)",
            "neutral_r3": (0.50, 0.50),
            "neutral_r4": (0.80, 0.20),
        },
    ]

    results = [("BASELINE (neutral=50/50)", base_m, base_wf, base_pos)]

    for s in scenarios:
        print(f"\n▶ {s['label']}...")
        ret, _ = build_combined(
            core_df, alpha_df,
            weights_risk_on    = (0.40, 0.60),
            weights_neutral_r3 = s["neutral_r3"],
            weights_neutral_r4 = s["neutral_r4"],
            weights_risk_off   = (0.80, 0.20),
        )
        m = compute_metrics(ret)
        wf, pos = walk_forward_score(ret)
        print_result(s["label"], m, wf, pos, base_m)
        results.append((s["label"][:45], m, wf, pos))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"\n  {'Scenario':<46} {'CAGR':>8} {'MaxDD':>9} {'Sharpe':>8} {'WF OOS':>8} {'WF+':>5}")
    print(f"  {'─'*86}")
    for label, m, wf, pos in results:
        marker = " ◀" if label.startswith("BASELINE") else ""
        print(f"  {label:<46} {m['cagr']:>8.2%} {m['maxdd']:>9.2%} "
              f"{m['sharpe']:>8.3f} {wf:>8.2%} {pos:>4}/10{marker}")

    print("""
  DECISION GUIDE:
    Accept a change only if:
      1. CAGR improves meaningfully
      2. MaxDD does not worsen significantly
      3. Walk-forward OOS CAGR improves or holds
      4. The change has a logical structural justification

    Do NOT accept a change purely because CAGR is higher in-sample.
    Walk-forward column is the deciding vote.
""")


if __name__ == "__main__":
    main()
