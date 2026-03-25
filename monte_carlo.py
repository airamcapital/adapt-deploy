"""
ADAPT — Monte Carlo Simulation
================================
Resamples actual daily returns to generate 10,000 possible equity paths
and computes updated underperformance probabilities against the new
26.66% CAGR baseline.

Methodology:
  - Block bootstrap resampling (blocks of 20 days) to preserve
    autocorrelation and volatility clustering in returns
  - Each simulation draws from actual historical daily returns
  - No parametric assumptions about return distribution

Usage:
    cd ~/Desktop/ADAPT_DEPLOY_Claude
    ./venv/bin/python monte_carlo.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from adapt.allocator.combined_dynamic import run_combined_dynamic

# ── Configuration ──────────────────────────────────────────────────────────────
N_SIMULATIONS   = 10_000
BLOCK_SIZE      = 20        # days per resampling block (preserves autocorrelation)
HORIZON_1Y      = 252
HORIZON_2Y      = 504
HORIZON_3Y      = 756
HORIZON_5Y      = 1260
TRADING_DAYS    = 252
SEED            = 42
# ──────────────────────────────────────────────────────────────────────────────


def run_simulation(returns: np.ndarray, horizon: int, n_sims: int,
                   block_size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Block bootstrap Monte Carlo.
    Returns array of shape (n_sims,) with terminal CAGR for each path.
    """
    n = len(returns)
    n_blocks = int(np.ceil(horizon / block_size))
    terminal_cagrs = np.empty(n_sims)

    for i in range(n_sims):
        # Draw random block start indices
        starts = rng.integers(0, n - block_size, size=n_blocks)
        path = np.concatenate([returns[s:s + block_size] for s in starts])[:horizon]
        cum = np.prod(1.0 + path)
        years = horizon / TRADING_DAYS
        terminal_cagrs[i] = cum ** (1.0 / years) - 1.0

    return terminal_cagrs


def prob_below(cagrs: np.ndarray, threshold: float) -> float:
    return float((cagrs < threshold).mean())


def max_drawdown_path(returns: np.ndarray) -> float:
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    return float(dd.min())


def run_drawdown_simulation(returns: np.ndarray, horizon: int, n_sims: int,
                            block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Returns array of max drawdowns across simulated paths."""
    n = len(returns)
    n_blocks = int(np.ceil(horizon / block_size))
    max_dds = np.empty(n_sims)

    for i in range(n_sims):
        starts = rng.integers(0, n - block_size, size=n_blocks)
        path = np.concatenate([returns[s:s + block_size] for s in starts])[:horizon]
        max_dds[i] = max_drawdown_path(path)

    return max_dds


def percentile_str(arr: np.ndarray, p: float) -> str:
    return f"{np.percentile(arr, p):.2%}"


def main():
    print("=" * 72)
    print("  ADAPT — MONTE CARLO SIMULATION")
    print(f"  {N_SIMULATIONS:,} simulations | Block size: {BLOCK_SIZE} days")
    print("=" * 72)

    print("\n▶ Running full combined backtest...")
    df, metrics = run_combined_dynamic()

    returns = df["ret"].values
    actual_cagr = metrics["cagr"]
    actual_maxdd = metrics["maxdd"]
    actual_sharpe = metrics["sharpe"]

    print(f"  Actual CAGR:   {actual_cagr:.2%}")
    print(f"  Actual MaxDD:  {actual_maxdd:.2%}")
    print(f"  Actual Sharpe: {actual_sharpe:.3f}")
    print(f"  Daily returns: {len(returns):,} observations")

    rng = np.random.default_rng(SEED)

    # ── Run simulations for each horizon ──────────────────────────────────────
    horizons = {
        "1-Year":  (HORIZON_1Y,  run_simulation),
        "2-Year":  (HORIZON_2Y,  run_simulation),
        "3-Year":  (HORIZON_3Y,  run_simulation),
        "5-Year":  (HORIZON_5Y,  run_simulation),
    }

    sim_results = {}
    for label, (horizon, fn) in horizons.items():
        print(f"\n▶ Simulating {label} paths ({N_SIMULATIONS:,} runs)...")
        cagrs = fn(returns, horizon, N_SIMULATIONS, BLOCK_SIZE, rng)
        sim_results[label] = cagrs

    # ── Drawdown simulations ───────────────────────────────────────────────────
    print(f"\n▶ Simulating 2-Year drawdown paths ({N_SIMULATIONS:,} runs)...")
    dd_2y = run_drawdown_simulation(returns, HORIZON_2Y, N_SIMULATIONS, BLOCK_SIZE, rng)

    print(f"\n▶ Simulating 5-Year drawdown paths ({N_SIMULATIONS:,} runs)...")
    dd_5y = run_drawdown_simulation(returns, HORIZON_5Y, N_SIMULATIONS, BLOCK_SIZE, rng)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CAGR DISTRIBUTION BY HORIZON")
    print("=" * 72)

    print(f"\n  {'Horizon':<10} {'5th pct':>9} {'25th pct':>9} {'Median':>9} "
          f"{'75th pct':>9} {'95th pct':>9} {'Mean':>9}")
    print(f"  {'─'*66}")

    for label, cagrs in sim_results.items():
        print(
            f"  {label:<10} "
            f"{percentile_str(cagrs, 5):>9} "
            f"{percentile_str(cagrs, 25):>9} "
            f"{percentile_str(cagrs, 50):>9} "
            f"{percentile_str(cagrs, 75):>9} "
            f"{percentile_str(cagrs, 95):>9} "
            f"{np.mean(cagrs):>9.2%}"
        )

    # ── Underperformance probabilities ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  UNDERPERFORMANCE PROBABILITIES")
    print("=" * 72)

    thresholds = [0.0, 0.05, 0.08, 0.10, 0.15, 0.20]

    for label, cagrs in sim_results.items():
        print(f"\n  {label}:")
        print(f"  {'─'*50}")
        for t in thresholds:
            p = prob_below(cagrs, t)
            bar = "█" * int(p * 40)
            print(f"  Prob(CAGR < {t:>5.0%}): {p:>6.2%}  {bar}")

    # ── Drawdown analysis ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  DRAWDOWN DISTRIBUTION")
    print("=" * 72)

    print(f"\n  Actual max drawdown (full history): {actual_maxdd:.2%}")

    print(f"\n  2-Year simulated drawdown:")
    for p in [50, 75, 90, 95, 99]:
        print(f"    {p}th percentile worst DD: {np.percentile(dd_2y, p):.2%}")
    print(f"    Prob(2Y DD worse than -20%): {prob_below(dd_2y, -0.20):.2%}")
    print(f"    Prob(2Y DD worse than -30%): {prob_below(dd_2y, -0.30):.2%}")
    print(f"    Prob(2Y DD worse than -40%): {prob_below(dd_2y, -0.40):.2%}")

    print(f"\n  5-Year simulated drawdown:")
    for p in [50, 75, 90, 95, 99]:
        print(f"    {p}th percentile worst DD: {np.percentile(dd_5y, p):.2%}")
    print(f"    Prob(5Y DD worse than -20%): {prob_below(dd_5y, -0.20):.2%}")
    print(f"    Prob(5Y DD worse than -30%): {prob_below(dd_5y, -0.30):.2%}")
    print(f"    Prob(5Y DD worse than -40%): {prob_below(dd_5y, -0.40):.2%}")

    # ── Recovery from drawdown ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MEDIAN / WORST CASE SCENARIOS")
    print("=" * 72)

    cagrs_2y = sim_results["2-Year"]
    cagrs_5y = sim_results["5-Year"]

    print(f"""
  Starting with $100,000:

  2-Year outcomes:
    Best case  (95th pct): ${100000 * (1 + np.percentile(cagrs_2y, 95))**2:>12,.0f}
    Good case  (75th pct): ${100000 * (1 + np.percentile(cagrs_2y, 75))**2:>12,.0f}
    Median               : ${100000 * (1 + np.percentile(cagrs_2y, 50))**2:>12,.0f}
    Weak case  (25th pct): ${100000 * (1 + np.percentile(cagrs_2y, 25))**2:>12,.0f}
    Worst case  (5th pct): ${100000 * (1 + np.percentile(cagrs_2y,  5))**2:>12,.0f}

  5-Year outcomes:
    Best case  (95th pct): ${100000 * (1 + np.percentile(cagrs_5y, 95))**5:>12,.0f}
    Good case  (75th pct): ${100000 * (1 + np.percentile(cagrs_5y, 75))**5:>12,.0f}
    Median               : ${100000 * (1 + np.percentile(cagrs_5y, 50))**5:>12,.0f}
    Weak case  (25th pct): ${100000 * (1 + np.percentile(cagrs_5y, 25))**5:>12,.0f}
    Worst case  (5th pct): ${100000 * (1 + np.percentile(cagrs_5y,  5))**5:>12,.0f}
""")

    # ── Monte Carlo vs actual ─────────────────────────────────────────────────
    print("=" * 72)
    print("  MONTE CARLO MEDIAN vs ACTUAL CAGR")
    print("=" * 72)
    print(f"""
  This is the key robustness check. If Monte Carlo median ≈ actual CAGR,
  the strategy is not dependent on a specific sequence of lucky years.

  Actual full-period CAGR:     {actual_cagr:.2%}
  Monte Carlo 2Y median CAGR:  {np.percentile(cagrs_2y, 50):.2%}
  Monte Carlo 5Y median CAGR:  {np.percentile(cagrs_5y, 50):.2%}

  Verdict: {"✅ ROBUST — median tracks actual CAGR closely" 
            if abs(np.percentile(cagrs_5y, 50) - actual_cagr) < 0.05 
            else "⚠️  GAP DETECTED — median diverges from actual CAGR"}
""")


if __name__ == "__main__":
    main()
