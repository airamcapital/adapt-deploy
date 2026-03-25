from __future__ import annotations

from pathlib import Path
import yaml


def load_allocator_config(path: str | Path = "config/allocator.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_weights(core_regime: int, alpha_in_market: bool, allocator_cfg: dict) -> tuple[float, float, str]:
    weights = allocator_cfg["weights"]

    if core_regime == 3 and alpha_in_market:
        return (
            float(weights["risk_on"]["core"]),
            float(weights["risk_on"]["alpha"]),
            "risk_on",
        )

    if core_regime == 4 and not alpha_in_market:
        return (
            float(weights["risk_off"]["core"]),
            float(weights["risk_off"]["alpha"]),
            "risk_off",
        )

    return (
        float(weights["neutral"]["core"]),
        float(weights["neutral"]["alpha"]),
        "neutral",
    )
