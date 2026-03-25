from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.io import to_html

from adapt.data_loader import load_settings
from adapt.core.core_engine import run_core_backtest
from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.allocator.combined_dynamic import run_combined_dynamic


START_VALUE = 50000.0


def download_benchmark_prices(start: str) -> pd.DataFrame:
    raw = yf.download(["SPY", "QQQ", "TQQQ"], start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw.copy()
    return px.dropna(how="all")


def perf_from_returns(returns: pd.Series, start_value: float = START_VALUE) -> dict:
    returns = returns.dropna().copy()
    if len(returns) == 0:
        return {"cagr": 0.0, "maxdd": 0.0, "sharpe": 0.0, "final_value": start_value, "days": 0}

    pv = start_value
    peak = start_value
    dds = []

    for r in returns:
        pv *= (1.0 + float(r))
        peak = max(peak, pv)
        dds.append((pv / peak) - 1.0)

    years = len(returns) / 252.0
    cagr = (pv / start_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    maxdd = float(min(dds)) if dds else 0.0
    std = returns.std()
    sharpe = (returns.mean() * 252.0) / (std * (252.0 ** 0.5)) if std and std > 0 else 0.0

    return {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "final_value": pv,
        "days": len(returns),
    }


def curve_from_returns(returns: pd.Series, start_value: float = START_VALUE) -> tuple[list[float], list[float]]:
    returns = returns.fillna(0.0)
    pv = start_value
    peak = start_value
    equity = []
    dd = []

    for r in returns:
        pv *= (1.0 + float(r))
        peak = max(peak, pv)
        equity.append(pv)
        dd.append((pv / peak) - 1.0)

    return equity, dd


def core_regime_text(regime: int) -> str:
    return {
        1: "R1 oversold bullish",
        2: "R2 overbought inverse",
        3: "R3 trend-on",
        4: "R4 defensive",
    }.get(int(regime), f"R{regime}")


def format_alloc(d: dict) -> str:
    if not d:
        return "none"
    return ", ".join(f"{k} {v:.0%}" for k, v in d.items())


def core_alloc_from_regime(regime: int) -> dict:
    if regime == 1:
        return {"TQQQ": 1.0}
    if regime == 2:
        return {"SQQQ": 1.0}
    if regime == 3:
        return {"TQQQ": 0.60, "LQD": 0.10, "IAU": 0.10, "USMV": 0.10, "UUP": 0.10}
    return {"TLT": 0.10, "BIL": 0.40, "BTAL": 0.20, "USMV": 0.20, "UUP": 0.10}


def alpha_alloc_from_weight(weight: float) -> dict:
    if weight > 0:
        return {"TQQQ": 1.0}
    return {"CASH": 1.0}


def combine_allocs(core_alloc: dict, alpha_alloc: dict, core_w: float, alpha_w: float) -> dict:
    out = {}
    for k, v in core_alloc.items():
        out[k] = out.get(k, 0.0) + float(v) * float(core_w)
    for k, v in alpha_alloc.items():
        out[k] = out.get(k, 0.0) + float(v) * float(alpha_w)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def build_equity_fig(dates, equity, hover, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            mode="lines",
            name=title,
            hovertemplate="%{text}<extra></extra>",
            text=hover,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#141c2f",
        plot_bgcolor="#141c2f",
        margin=dict(l=40, r=20, t=50, b=40),
        height=430,
    )
    return fig


def build_drawdown_fig(dates, dd, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dd,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#141c2f",
        plot_bgcolor="#141c2f",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
        yaxis_tickformat=".0%",
    )
    return fig


def benchmark_table_html(strategy_name: str, rets: pd.Series, bench_df: pd.DataFrame) -> str:
    rows = []

    strat = perf_from_returns(rets, START_VALUE)
    rows.append(
        f"<tr><td><b>{strategy_name}</b></td><td><b>{strat['cagr']:.2%}</b></td>"
        f"<td><b>{strat['maxdd']:.2%}</b></td><td><b>{strat['sharpe']:.3f}</b></td>"
        f"<td><b>${strat['final_value']:,.0f}</b></td></tr>"
    )

    for col in ["SPY", "QQQ", "TQQQ"]:
        if col not in bench_df.columns:
            continue
        m = perf_from_returns(bench_df[col], START_VALUE)
        rows.append(
            f"<tr><td>{col}</td><td>{m['cagr']:.2%}</td><td>{m['maxdd']:.2%}</td>"
            f"<td>{m['sharpe']:.3f}</td><td>${m['final_value']:,.0f}</td></tr>"
        )

    return f"""
    <div class="table-wrap">
      <h3>{strategy_name} vs Benchmarks</h3>
      <table>
        <thead>
          <tr><th>Series</th><th>CAGR</th><th>MaxDD</th><th>Sharpe</th><th>Final Value</th></tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """


def metrics_cards_html(m: dict) -> str:
    return f"""
    <div class="cards">
      <div class="card"><div class="label">CAGR</div><div class="value">{m['cagr']:.2%}</div></div>
      <div class="card"><div class="label">MaxDD</div><div class="value">{m['maxdd']:.2%}</div></div>
      <div class="card"><div class="label">Sharpe</div><div class="value">{m['sharpe']:.3f}</div></div>
      <div class="card"><div class="label">Final Value</div><div class="value">${m['final_value']:,.0f}</div></div>
    </div>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ADAPT HTML dashboard")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output", default=None, help="Optional output HTML path")
    args = parser.parse_args()

    settings = load_settings()

    core_df, _ = run_core_backtest(settings)
    alpha_df, _ = run_alpha_backtest(settings)
    combined_df, _ = run_combined_dynamic(settings)
    bench_px = download_benchmark_prices(settings["data"]["start_date"])
    bench_ret = bench_px.pct_change().dropna(how="all")

    core = core_df.loc[args.start:args.end].copy()
    alpha = alpha_df.loc[args.start:args.end].copy()
    combined = combined_df.loc[args.start:args.end].copy()

    core_bench = bench_ret.reindex(core.index).dropna()
    alpha_bench = bench_ret.reindex(alpha.index).dropna()
    combined_bench = bench_ret.reindex(combined.index).dropna()

    core = core.loc[core_bench.index]
    alpha = alpha.loc[alpha_bench.index]
    combined = combined.loc[combined_bench.index]

    if len(core) == 0 or len(alpha) == 0 or len(combined) == 0:
        raise SystemExit("No data available for the chosen date range.")

    core_eq, core_dd = curve_from_returns(core["ret"])
    alpha_eq, alpha_dd = curve_from_returns(alpha["ret"])
    combined_eq, combined_dd = curve_from_returns(combined["ret"])

    core_hover = [
        (
            f"Date: {idx.strftime('%Y-%m-%d')}<br>"
            f"Equity: ${eq:,.0f}<br>"
            f"Drawdown: {dd:.2%}<br>"
            f"Regime: {core_regime_text(int(row.signal_regime))}<br>"
            f"Allocation: {format_alloc(core_alloc_from_regime(int(row.signal_regime)))}"
        )
        for idx, row, eq, dd in zip(core.index, core.itertuples(), core_eq, core_dd)
    ]
    alpha_hover = [
        (
            f"Date: {idx.strftime('%Y-%m-%d')}<br>"
            f"Equity: ${eq:,.0f}<br>"
            f"Drawdown: {dd:.2%}<br>"
            f"Signal: {row.signal}<br>"
            f"In market: {'yes' if row.weight > 0 else 'no'}<br>"
            f"Allocation: {format_alloc(alpha_alloc_from_weight(float(row.weight)))}"
        )
        for idx, row, eq, dd in zip(alpha.index, alpha.itertuples(), alpha_eq, alpha_dd)
    ]
    combined_hover = [
        (
            f"Date: {idx.strftime('%Y-%m-%d')}<br>"
            f"Equity: ${eq:,.0f}<br>"
            f"Drawdown: {dd:.2%}<br>"
            f"Allocator State: {row.allocator_state}<br>"
            f"Core Weight: {row.core_w:.0%}<br>"
            f"Alpha Weight: {row.alpha_w:.0%}<br>"
            f"Allocation: {format_alloc(combine_allocs(core_alloc_from_regime(int(row.core_regime)), alpha_alloc_from_weight(float(row.alpha_weight)), float(row.core_w), float(row.alpha_w)))}"
        )
        for idx, row, eq, dd in zip(combined.index, combined.itertuples(), combined_eq, combined_dd)
    ]

    core_fig = build_equity_fig(core.index, core_eq, core_hover, "CORE Equity Curve")
    core_dd_fig = build_drawdown_fig(core.index, core_dd, "CORE Drawdown")
    alpha_fig = build_equity_fig(alpha.index, alpha_eq, alpha_hover, "ALPHA Equity Curve")
    alpha_dd_fig = build_drawdown_fig(alpha.index, alpha_dd, "ALPHA Drawdown")
    combined_fig = build_equity_fig(combined.index, combined_eq, combined_hover, "COMBINED Equity Curve")
    combined_dd_fig = build_drawdown_fig(combined.index, combined_dd, "COMBINED Drawdown")

    core_eq_html = to_html(core_fig, include_plotlyjs=True, full_html=False)
    core_dd_html = to_html(core_dd_fig, include_plotlyjs=False, full_html=False)
    alpha_eq_html = to_html(alpha_fig, include_plotlyjs=False, full_html=False)
    alpha_dd_html = to_html(alpha_dd_fig, include_plotlyjs=False, full_html=False)
    combined_eq_html = to_html(combined_fig, include_plotlyjs=False, full_html=False)
    combined_dd_html = to_html(combined_dd_fig, include_plotlyjs=False, full_html=False)

    core_metrics = perf_from_returns(core["ret"])
    alpha_metrics = perf_from_returns(alpha["ret"])
    combined_metrics = perf_from_returns(combined["ret"])

    if args.output:
        outpath = Path(args.output)
    else:
        outpath = Path("outputs/reports") / f"dashboard_{args.start}_to_{args.end}.html"

    outpath.parent.mkdir(parents=True, exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ADAPT Dashboard</title>
<style>
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
  background: #0b1020;
  color: #e8ecf3;
}}
.header {{
  padding: 20px 24px 8px 24px;
}}
.sub {{
  color: #aab4c5;
}}
.tabs {{
  display: flex;
  gap: 10px;
  padding: 12px 24px;
}}
.tab-btn {{
  padding: 10px 16px;
  border: 1px solid #2b3750;
  background: #141c2f;
  color: #e8ecf3;
  cursor: pointer;
  border-radius: 10px;
}}
.tab-btn.active {{
  background: #233252;
}}
.panel {{
  display: none;
  padding: 8px 24px 24px 24px;
}}
.panel.active {{
  display: block;
}}
.cards {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 12px;
  margin-bottom: 18px;
}}
.card {{
  background: #141c2f;
  border: 1px solid #25324b;
  border-radius: 14px;
  padding: 14px;
}}
.card .label {{
  color: #9aa7bc;
  font-size: 12px;
  margin-bottom: 8px;
}}
.card .value {{
  font-size: 22px;
  font-weight: 700;
}}
.chart {{
  background: #141c2f;
  border: 1px solid #25324b;
  border-radius: 14px;
  padding: 8px;
  margin-bottom: 18px;
}}
.table-wrap {{
  background: #141c2f;
  border: 1px solid #25324b;
  border-radius: 14px;
  padding: 14px;
}}
table {{
  width: 100%;
  border-collapse: collapse;
}}
th, td {{
  text-align: right;
  padding: 8px 6px;
  border-bottom: 1px solid #22304a;
}}
th:first-child, td:first-child {{
  text-align: left;
}}
</style>
<script>
function showTab(id, btn) {{
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  btn.classList.add("active");
}}
</script>
</head>
<body>
<div class="header">
  <h1>ADAPT Dashboard / Backtest Explorer</h1>
  <div class="sub">Window: {args.start} to {args.end}</div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="showTab('core', this)">CORE</button>
  <button class="tab-btn" onclick="showTab('combined', this)">COMBINED</button>
  <button class="tab-btn" onclick="showTab('alpha', this)">ALPHA</button>
</div>

<div id="core" class="panel active">
  {metrics_cards_html(core_metrics)}
  <div class="chart">{core_eq_html}</div>
  <div class="chart">{core_dd_html}</div>
  {benchmark_table_html("CORE", core["ret"], core_bench)}
</div>

<div id="combined" class="panel">
  {metrics_cards_html(combined_metrics)}
  <div class="chart">{combined_eq_html}</div>
  <div class="chart">{combined_dd_html}</div>
  {benchmark_table_html("COMBINED", combined["ret"], combined_bench)}
</div>

<div id="alpha" class="panel">
  {metrics_cards_html(alpha_metrics)}
  <div class="chart">{alpha_eq_html}</div>
  <div class="chart">{alpha_dd_html}</div>
  {benchmark_table_html("ALPHA", alpha["ret"], alpha_bench)}
</div>

</body>
</html>
"""

    outpath.write_text(html, encoding="utf-8")
    print(f"Dashboard written to: {outpath}")


if __name__ == "__main__":
    main()
