from __future__ import annotations

from typing import Dict


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    out = {k: float(v) for k, v in weights.items() if abs(float(v)) > 1e-12}
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def core_target_allocation(signal_regime: int) -> Dict[str, float]:
    # Broker-ready summary for current production interpretation:
    # R3 = trend-on sleeve
    # R4 = defensive sleeve
    #
    # For exact basket composition, this mirrors the deploy logic
    # using the currently active production configuration.
    if signal_regime == 1:
        return {"TQQQ": 1.0}

    if signal_regime == 2:
        return {"SQQQ": 1.0}

    if signal_regime == 3:
        # Production R3 summary: trend sleeve
        # This is simplified broker-ready output. It can be expanded later
        # if you want the exact dynamic R3 basket by SMA distance.
        return {
            "TQQQ": 0.60,
            "LQD": 0.10,
            "IAU": 0.10,
            "USMV": 0.10,
            "UUP": 0.10,
        }

    # Default defensive sleeve summary
    return {
        "TLT": 0.10,
        "BIL": 0.40,
        "BTAL": 0.20,
        "USMV": 0.20,
        "UUP": 0.10,
    }


def alpha_target_allocation(alpha_in_market: bool) -> Dict[str, float]:
    if alpha_in_market:
        return {"TQQQ": 1.0}
    return {"CASH": 1.0}


def scale_weights(weights: Dict[str, float], scale: float) -> Dict[str, float]:
    return {k: float(v) * float(scale) for k, v in weights.items()}


def combine_allocations(*allocations: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for alloc in allocations:
        for k, v in alloc.items():
            out[k] = out.get(k, 0.0) + float(v)
    return out


def combined_target_allocation(
    core_regime: int,
    alpha_in_market: bool,
    core_weight: float,
    alpha_weight: float,
) -> Dict[str, float]:
    core_alloc = scale_weights(core_target_allocation(core_regime), core_weight)
    alpha_alloc = scale_weights(alpha_target_allocation(alpha_in_market), alpha_weight)
    return normalize_weights(combine_allocations(core_alloc, alpha_alloc))
