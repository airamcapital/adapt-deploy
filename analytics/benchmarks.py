nano ~/Desktop/ADAPT_DEPLOY/analytics/benchmarks.py
import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    s = series.copy().dropna()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def build_benchmark_equity(strategy_equity: pd.Series, benchmark_prices: pd.Series) -> pd.Series:
    """
    Rebase benchmark equity to the same start value and dates as the strategy.
    This ensures each dashboard panel compares correctly.
    """

    strategy_equity = normalize_series(strategy_equity)
    benchmark_prices = normalize_series(benchmark_prices)

    # Align benchmark to strategy dates
    bench = benchmark_prices.reindex(strategy_equity.index).ffill()

    # Remove leading NaNs if benchmark begins later
    first_valid = bench.first_valid_index()
    strategy_equity = strategy_equity.loc[first_valid:]
    bench = bench.loc[first_valid:]

    returns = bench.pct_change().fillna(0.0)

    start_value = float(strategy_equity.iloc[0])

    benchmark_equity = start_value * (1 + returns).cumprod()

    return benchmark_equity


def build_panel_benchmarks(strategy_equity: pd.Series, benchmark_prices: dict):
    """
    Build benchmark equity curves for one dashboard panel.
    """
    output = {}

    for symbol, price_series in benchmark_prices.items():
        try:
            output[symbol] = build_benchmark_equity(strategy_equity, price_series)
        except Exception:
            continue

    return output
