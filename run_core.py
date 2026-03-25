from adapt.core.core_engine import run_core_backtest


def main() -> None:
    df, metrics = run_core_backtest()

    print("=" * 84)
    print("  ADAPT_DEPLOY — CORE RUNNER")
    print("=" * 84)
    print(f"CAGR         : {metrics['cagr']:.2%}")
    print(f"MaxDD        : {metrics['maxdd']:.2%}")
    print(f"Sharpe       : {metrics['sharpe']:.3f}")
    print(f"Calmar       : {metrics['calmar']:.3f}")
    print(f"Final Value  : ${metrics['final_value']:,.2f}")
    print(f"Trades       : {metrics['trades']}")
    print()

    print("LATEST SIGNAL ROW")
    print("-" * 84)
    print(df.tail(1).to_string())


if __name__ == "__main__":
    main()
