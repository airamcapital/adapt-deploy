from __future__ import annotations

import numpy as np
import pandas as pd


def mark_swings(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:
    """
    Look-ahead safe swing detection.

    A swing high at t is only confirmed at t+right.
    """

    out = df.copy()

    swing_high = [False] * len(out)
    swing_low = [False] * len(out)

    for i in range(left, len(out) - right):
        center_high = out["High"].iloc[i]
        center_low = out["Low"].iloc[i]

        left_highs = out["High"].iloc[i-left:i]
        right_highs = out["High"].iloc[i+1:i+1+right]

        left_lows = out["Low"].iloc[i-left:i]
        right_lows = out["Low"].iloc[i+1:i+1+right]

        if center_high > left_highs.max() and center_high > right_highs.max():
            swing_high[i + right] = True

        if center_low < left_lows.min() and center_low < right_lows.min():
            swing_low[i + right] = True

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low

    return out


def build_structure_signals(
    df: pd.DataFrame,
    left: int = 2,
    right: int = 2,
) -> pd.DataFrame:

    out = mark_swings(df, left=left, right=right)

    last_swing_high = np.nan
    last_swing_low = np.nan
    structure_state = "neutral"
    bos_count_since_choch = 0

    rows = []

    for i in range(len(out)):
        row = out.iloc[i]
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])

        if bool(row["swing_high"]):
            last_swing_high = high
        if bool(row["swing_low"]):
            last_swing_low = low

        bullish_choch = False
        bullish_bos1 = False
        bearish_choch = False

        if not np.isnan(last_swing_high) and close > last_swing_high:
            if structure_state in ("bearish", "neutral"):
                bullish_choch = True
                structure_state = "bullish"
                bos_count_since_choch = 0
            elif structure_state == "bullish":
                bos_count_since_choch += 1
                if bos_count_since_choch == 1:
                    bullish_bos1 = True

        if not np.isnan(last_swing_low) and close < last_swing_low:
            if structure_state == "bullish":
                bearish_choch = True
            structure_state = "bearish"
            bos_count_since_choch = 0

        rows.append(
            {
                "bullish_choch": bullish_choch,
                "bullish_bos1": bullish_bos1,
                "bearish_choch": bearish_choch,
                "structure_state": structure_state,
                "last_swing_high": last_swing_high,
                "last_swing_low": last_swing_low,
            }
        )

    sig = pd.DataFrame(rows, index=out.index)

    return pd.concat([out, sig], axis=1)
