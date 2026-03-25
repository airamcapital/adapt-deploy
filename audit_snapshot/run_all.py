from adapt.core.core_engine import run_core_backtest
from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.allocator.combined_dynamic import run_combined_dynamic
from adapt.data_loader import load_settings
from adapt.reporting import write_json_signal, archive_json_signal, write_daily_summary
from adapt.allocation_summary import (
    core_target_allocation,
    alpha_target_allocation,
    combined_target_allocation,
)


def print_summary(title, metrics):
    print("====================================================================================")
    print(f"  {title}")
    print("====================================================================================")
    print(f"CAGR         : {metrics['cagr']:.2%}")
    print(f"MaxDD        : {metrics['maxdd']:.2%}")
    print(f"Sharpe       : {metrics['sharpe']:.3f}")
    print(f"Calmar       : {metrics['calmar']:.3f}")
    print(f"Final Value  : ${metrics['final_value']:,.2f}")

    if "trades" in metrics:
        print(f"Trades       : {metrics['trades']}")
    if "time_in_market" in metrics:
        print(f"Time in mkt  : {metrics['time_in_market']:.2%}")
    if "risk_on_pct" in metrics:
        print(f"Risk On      : {metrics['risk_on_pct']:.2%}")
        print(f"Neutral      : {metrics['neutral_pct']:.2%}")
        print(f"Risk Off     : {metrics['risk_off_pct']:.2%}")

    print()


def core_regime_text(regime: int) -> str:
    return {
        1: "R1 (oversold bullish)",
        2: "R2 (overbought inverse)",
        3: "R3 (trend-on)",
        4: "R4 (defensive)",
    }.get(regime, f"Unknown regime {regime}")


def combined_posture_text(state: str, core_w: float, alpha_w: float) -> str:
    return f"{state} posture — CORE {core_w:.0%} / ALPHA {alpha_w:.0%}"


def format_alloc(alloc: dict) -> str:
    if not alloc:
        return "none"
    return ", ".join(f"{k} {v:.1%}" for k, v in alloc.items())


def main():
    settings = load_settings()

    core_df, core_metrics = run_core_backtest(settings)
    alpha_df, alpha_metrics = run_alpha_backtest(settings)
    comb_df, comb_metrics = run_combined_dynamic(settings)

    core_last = core_df.iloc[-1]
    alpha_last = alpha_df.iloc[-1]
    comb_last = comb_df.iloc[-1]
    signal_date = str(comb_df.index[-1].date())

    core_regime = int(core_last["signal_regime"])
    alpha_in_market = bool(float(alpha_last["weight"]) > 0.0)

    core_alloc = core_target_allocation(core_regime)
    alpha_alloc = alpha_target_allocation(alpha_in_market)
    combined_alloc = combined_target_allocation(
        core_regime=core_regime,
        alpha_in_market=alpha_in_market,
        core_weight=float(comb_last["core_w"]),
        alpha_weight=float(comb_last["alpha_w"]),
    )

    print_summary("CORE", core_metrics)
    print_summary("ALPHA", alpha_metrics)
    print_summary("COMBINED_DYNAMIC", comb_metrics)

    print("Latest Core Row")
    print("------------------------------------------------------------------------------------")
    print(core_df.tail(1).to_string())
    print()

    print("Latest Alpha Row")
    print("------------------------------------------------------------------------------------")
    print(alpha_df.tail(1).to_string())
    print()

    print("Latest Combined Row")
    print("------------------------------------------------------------------------------------")
    print(comb_df.tail(1).to_string())
    print()

    print("BROKER-READY ALLOCATION SUMMARY")
    print("------------------------------------------------------------------------------------")
    print(f"CORE     : {format_alloc(core_alloc)}")
    print(f"ALPHA    : {format_alloc(alpha_alloc)}")
    print(f"COMBINED : {format_alloc(combined_alloc)}")
    print()

    core_payload = {
        "date": str(core_df.index[-1].date()),
        "strategy": "ADAPT_CORE",
        "signal_regime": core_regime,
        "signal_regime_text": core_regime_text(core_regime),
        "is_trade": bool(core_last["is_trade"]),
        "daily_return": float(core_last["ret"]),
        "equity_value": float(core_last["cum_val"]),
        "drawdown": float(core_last["dd"]),
        "close": float(core_last["close"]),
        "sma": float(core_last["sma"]),
        "rsi": float(core_last["rsi"]),
        "consec_up": int(core_last["consec_up"]),
        "target_allocation": core_alloc,
        "metrics": core_metrics,
    }

    alpha_payload = {
        "date": str(alpha_df.index[-1].date()),
        "strategy": "ADAPT_ALPHA_STRUCTURE",
        "signal": str(alpha_last["signal"]),
        "is_trade": bool(alpha_last["is_trade"]),
        "daily_return": float(alpha_last["ret"]),
        "equity_value": float(alpha_last["cum_val"]),
        "drawdown": float(alpha_last["dd"]),
        "weight": float(alpha_last["weight"]),
        "target_weight": float(alpha_last["target_weight"]),
        "bullish_choch": bool(alpha_last["bullish_choch"]),
        "bullish_bos1": bool(alpha_last["bullish_bos1"]),
        "bearish_choch": bool(alpha_last["bearish_choch"]),
        "target_allocation": alpha_alloc,
        "metrics": alpha_metrics,
    }

    combined_payload = {
        "date": str(comb_df.index[-1].date()),
        "strategy": "ADAPT_COMBINED_DYNAMIC",
        "allocator_state": str(comb_last["allocator_state"]),
        "core_weight": float(comb_last["core_w"]),
        "alpha_weight": float(comb_last["alpha_w"]),
        "core_regime": int(comb_last["core_regime"]),
        "alpha_in_market": alpha_in_market,
        "daily_return": float(comb_last["ret"]),
        "equity_value": float(comb_last["cum_val"]),
        "drawdown": float(comb_last["dd"]),
        "target_allocation": combined_alloc,
        "metrics": comb_metrics,
    }

    core_path = write_json_signal("core_signal_latest.json", core_payload, settings)
    alpha_path = write_json_signal("alpha_signal_latest.json", alpha_payload, settings)
    combined_path = write_json_signal("combined_signal_latest.json", combined_payload, settings)

    core_archive = archive_json_signal("core_signal", core_payload, signal_date, settings)
    alpha_archive = archive_json_signal("alpha_signal", alpha_payload, signal_date, settings)
    combined_archive = archive_json_signal("combined_signal", combined_payload, signal_date, settings)

    core_alloc_path = write_json_signal(
        "core_allocation_latest.json",
        {
            "date": signal_date,
            "strategy": "ADAPT_CORE",
            "signal_regime": core_regime,
            "allocation": core_alloc,
        },
        settings,
    )
    alpha_alloc_path = write_json_signal(
        "alpha_allocation_latest.json",
        {
            "date": signal_date,
            "strategy": "ADAPT_ALPHA_STRUCTURE",
            "alpha_in_market": alpha_in_market,
            "allocation": alpha_alloc,
        },
        settings,
    )
    combined_alloc_path = write_json_signal(
        "combined_allocation_latest.json",
        {
            "date": signal_date,
            "strategy": "ADAPT_COMBINED_DYNAMIC",
            "allocator_state": str(comb_last["allocator_state"]),
            "allocation": combined_alloc,
        },
        settings,
    )

    archive_json_signal(
        "core_allocation",
        {"date": signal_date, "strategy": "ADAPT_CORE", "allocation": core_alloc},
        signal_date,
        settings,
    )
    archive_json_signal(
        "alpha_allocation",
        {"date": signal_date, "strategy": "ADAPT_ALPHA_STRUCTURE", "allocation": alpha_alloc},
        signal_date,
        settings,
    )
    archive_json_signal(
        "combined_allocation",
        {"date": signal_date, "strategy": "ADAPT_COMBINED_DYNAMIC", "allocation": combined_alloc},
        signal_date,
        settings,
    )

    summary_lines = [
        f"ADAPT DAILY SUMMARY — {signal_date}",
        "=" * 72,
        "",
        "CORE",
        "-" * 72,
        f"Regime: {core_regime_text(core_regime)}",
        f"Close: {float(core_last['close']):.2f}",
        f"SMA150: {float(core_last['sma']):.2f}",
        f"RSI: {float(core_last['rsi']):.2f}",
        f"Consecutive up days: {int(core_last['consec_up'])}",
        f"Target allocation: {format_alloc(core_alloc)}",
        f"Equity value: ${float(core_last['cum_val']):,.2f}",
        f"Drawdown: {float(core_last['dd']):.2%}",
        "",
        "ALPHA",
        "-" * 72,
        f"Signal: {str(alpha_last['signal'])}",
        f"In market: {'yes' if alpha_in_market else 'no'}",
        f"Current weight: {float(alpha_last['weight']):.0%}",
        f"Target allocation: {format_alloc(alpha_alloc)}",
        f"Bullish BoS1 today: {bool(alpha_last['bullish_bos1'])}",
        f"Bearish ChoC today: {bool(alpha_last['bearish_choch'])}",
        f"Equity value: ${float(alpha_last['cum_val']):,.2f}",
        f"Drawdown: {float(alpha_last['dd']):.2%}",
        "",
        "COMBINED_DYNAMIC",
        "-" * 72,
        f"Allocator state: {str(comb_last['allocator_state'])}",
        f"Posture: {combined_posture_text(str(comb_last['allocator_state']), float(comb_last['core_w']), float(comb_last['alpha_w']))}",
        f"Target allocation: {format_alloc(combined_alloc)}",
        f"Equity value: ${float(comb_last['cum_val']):,.2f}",
        f"Drawdown: {float(comb_last['dd']):.2%}",
        "",
        "FILES WRITTEN",
        "-" * 72,
        f"Latest core JSON: {core_path}",
        f"Latest alpha JSON: {alpha_path}",
        f"Latest combined JSON: {combined_path}",
        f"Latest core allocation JSON: {core_alloc_path}",
        f"Latest alpha allocation JSON: {alpha_alloc_path}",
        f"Latest combined allocation JSON: {combined_alloc_path}",
        f"Archived core JSON: {core_archive}",
        f"Archived alpha JSON: {alpha_archive}",
        f"Archived combined JSON: {combined_archive}",
    ]

    summary_text = "\n".join(summary_lines)
    summary_path = write_daily_summary(signal_date, summary_text, settings)

    print("JSON SIGNAL FILES WRITTEN")
    print("------------------------------------------------------------------------------------")
    print(core_path)
    print(alpha_path)
    print(combined_path)
    print()

    print("ALLOCATION FILES WRITTEN")
    print("------------------------------------------------------------------------------------")
    print(core_alloc_path)
    print(alpha_alloc_path)
    print(combined_alloc_path)
    print()

    print("ARCHIVE FILES WRITTEN")
    print("------------------------------------------------------------------------------------")
    print(core_archive)
    print(alpha_archive)
    print(combined_archive)
    print()

    print("DAILY SUMMARY WRITTEN")
    print("------------------------------------------------------------------------------------")
    print(summary_path)


if __name__ == "__main__":
    main()
