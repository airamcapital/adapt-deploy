from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from adapt.structure import build_structure_signals


@dataclass
class AlphaState:
    in_market: bool
    prev_signal: str | None = None


def build_alpha_features(ohlc: pd.DataFrame, alpha_cfg: dict) -> pd.DataFrame:
    p = alpha_cfg["parameters"]
    return build_structure_signals(
        ohlc,
        left=int(p["swing_left"]),
        right=int(p["swing_right"]),
    ).copy()


def classify_alpha_signal(row: pd.Series, alpha_cfg: dict) -> str:
    if bool(row["bearish_choch"]):
        return "exit"
    if bool(row["bullish_bos1"]):
        return "entry"
    return "hold"


def entry_weights(alpha_cfg: dict) -> Dict[str, float]:
    long_asset = alpha_cfg["execution"]["long_asset"]
    return {long_asset: 1.0}
