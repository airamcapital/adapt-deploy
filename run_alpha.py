from adapt.alpha.alpha_engine import run_alpha_backtest


def main() -> None:
    df, metrics = run_alpha_backtest()

    print("=" * 84)
    print("  ADAPT_DEPLOY — ALPHA RUNNER")
    print("=" * 84)
    print(f"CAGR         : {metrics['cagr']:.2%}")
    print(f"MaxDD        : {metrics['maxdd']:.2%}")
    print(f"Sharpe       : {metrics['sharpe']:.3f}")
    print(f"Calmar       : {metrics['calmar']:.3f}")
    print(f"Final Value  : ${metrics['final_value']:,.2f}")
    print(f"Trades       : {metrics['trades']}")
    print(f"Time in mkt  : {metrics['time_in_market']:.2%}")
    print()

    print("LATEST SIGNAL ROW")
    print("-" * 84)
    print(df.tail(1).to_string())


if __name__ == "__main__":
    main()
