from adapt.allocator.combined_dynamic import run_combined_dynamic


def main() -> None:
    df, metrics = run_combined_dynamic()

    print("=" * 84)
    print("  ADAPT_DEPLOY — COMBINED DYNAMIC RUNNER")
    print("=" * 84)
    print(f"CAGR         : {metrics['cagr']:.2%}")
    print(f"MaxDD        : {metrics['maxdd']:.2%}")
    print(f"Sharpe       : {metrics['sharpe']:.3f}")
    print(f"Calmar       : {metrics['calmar']:.3f}")
    print(f"Final Value  : ${metrics['final_value']:,.2f}")
    print()
    print(f"Risk On      : {metrics['risk_on_pct']:.2%}")
    print(f"Neutral      : {metrics['neutral_pct']:.2%}")
    print(f"Risk Off     : {metrics['risk_off_pct']:.2%}")
    print()

    print("LATEST ROW")
    print("-" * 84)
    print(df.tail(1).to_string())


if __name__ == "__main__":
    main()
