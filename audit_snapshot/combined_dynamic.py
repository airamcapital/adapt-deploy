from __future__ import annotations

from typing import Tuple

import pandas as pd

from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.allocator.allocation_logic import load_allocator_config, choose_weights
from adapt.core.core_engine import run_core_backtest
from adapt.data_loader import load_settings

IBKR_CASH_YIELD = 0.032


def run_combined_dynamic(
    settings: dict | None = None,
    allocator_cfg: dict | None = None,
) -> Tuple[pd.DataFrame, dict]:
    settings = settings or load_settings()
    allocator_cfg = allocator_cfg or load_allocator_config()

    core_df, _ = run_core_backtest(settings=settings)
    alpha_df, _ = run_alpha_backtest(settings=settings)

    df = pd.DataFrame(
        {
            "core_ret": core_df["ret"],
            "core_regime": core_df["signal_regime"],
            "alpha_ret": alpha_df["ret"],
            "alpha_weight": alpha_df["weight"],
        }
    ).dropna()

    core_w = []
    alpha_w = []
    state = []
    alpha_effective_ret = []
    combined_ret = []

    daily_cash_rate = IBKR_CASH_YIELD / 252.0

    for _, row in df.iterrows():
        cw, aw, st = choose_weights(
            core_regime=int(row["core_regime"]),
            alpha_in_market=float(row["alpha_weight"]) > 0.0,
            allocator_cfg=allocator_cfg,
        )

        core_w.append(cw)
        alpha_w.append(aw)
        state.append(st)

        if float(row["alpha_weight"]) > 0.0:
            a_ret = float(row["alpha_ret"])
        else:
            a_ret = daily_cash_rate

        alpha_effective_ret.append(a_ret)

        combined_ret.append(
            cw * float(row["core_ret"]) + aw * a_ret
        )

    df["core_w"] = core_w
    df["alpha_w"] = alpha_w
    df["allocator_state"] = state
    df["alpha_effective_ret"] = alpha_effective_ret
    df["ret"] = combined_ret

    start_value = float(settings["portfolio"]["start_value"])

    pv = start_value
    peak = start_value

    vals = []
    dds = []

    for r in df["ret"]:
        pv *= (1.0 + float(r))
        vals.append(pv)

        peak = max(peak, pv)
        dds.append((pv / peak) - 1.0)

    df["cum_val"] = vals
    df["dd"] = dds

    years = len(df) / 252.0

    cagr = (df["cum_val"].iloc[-1] / start_value) ** (1.0 / years) - 1.0
    maxdd = float(df["dd"].min())

    std = df["ret"].std()

    sharpe = (
        (df["ret"].mean() * 252.0) / (std * (252.0 ** 0.5))
        if std > 0
        else 0.0
    )

    calmar = cagr / abs(maxdd) if maxdd != 0 else 0.0

    metrics = {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "final_value": float(df["cum_val"].iloc[-1]),
        "years": years,
        "risk_on_pct": float((df["allocator_state"] == "risk_on").mean()),
        "neutral_pct": float((df["allocator_state"] == "neutral").mean()),
        "risk_off_pct": float((df["allocator_state"] == "risk_off").mean()),
    }

    return df, metrics
