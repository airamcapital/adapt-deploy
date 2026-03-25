from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.allocator.combined_dynamic import run_combined_dynamic
from adapt.core.core_engine import run_core_backtest
from adapt.data_loader import load_settings


DEFAULT_START_CAPITAL = 50000.0
TRADING_DAYS = 252


@st.cache_data(show_spinner=False)
def get_price_history(symbols: list[str], start: str = "1990-01-01") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(symbols, start=start, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        px = raw["Close"].copy()
    else:
        px = raw.copy()
        if len(symbols) == 1 and "Close" in px.columns:
            px = px.rename(columns={"Close": symbols[0]})

    if isinstance(px, pd.Series):
        px = px.to_frame()

    return px.dropna(how="all")


@st.cache_data(show_spinner=False)
def get_strategy_bundle() -> dict[str, pd.DataFrame]:
    settings = load_settings()

    core_df, _ = run_core_backtest(settings)
    alpha_df, _ = run_alpha_backtest(settings)
    combined_df, _ = run_combined_dynamic(settings)

    return {
        "CORE": core_df,
        "ALPHA": alpha_df,
        "COMBINED": combined_df,
    }


def first_valid_dates(px: pd.DataFrame) -> dict[str, pd.Timestamp]:
    out: dict[str, pd.Timestamp] = {}
    if px.empty:
        return out

    for col in px.columns:
        s = px[col].dropna()
        if not s.empty:
            out[str(col)] = pd.to_datetime(s.index[0]).normalize()
    return out


@st.cache_data(show_spinner=False)
def get_strategy_ranges() -> dict[str, pd.Timestamp]:
    data_map = get_strategy_bundle()
    return {name: pd.to_datetime(df.index.min()).normalize() for name, df in data_map.items()}


def perf_from_returns(returns: pd.Series, start_value: float) -> dict:
    returns = returns.dropna().copy()
    if len(returns) == 0:
        return {
            "cagr": 0.0,
            "maxdd": 0.0,
            "sharpe": 0.0,
            "final_value": start_value,
            "days": 0,
            "calmar": 0.0,
            "win_rate": 0.0,
        }

    pv = start_value
    peak = start_value
    dds = []

    for r in returns:
        pv *= (1.0 + float(r))
        peak = max(peak, pv)
        dds.append((pv / peak) - 1.0)

    years = len(returns) / TRADING_DAYS
    cagr = (pv / start_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    maxdd = float(min(dds)) if dds else 0.0
    std = returns.std()
    sharpe = (returns.mean() * TRADING_DAYS) / (std * math.sqrt(TRADING_DAYS)) if std and std > 0 else 0.0
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0
    win_rate = float((returns > 0).mean())

    return {
        "cagr": cagr,
        "maxdd": maxdd,
        "sharpe": sharpe,
        "final_value": pv,
        "days": len(returns),
        "calmar": calmar,
        "win_rate": win_rate,
    }


def contribution_flags(index: pd.Index, frequency: str) -> pd.Series:
    idx = pd.to_datetime(index)
    flags = pd.Series(False, index=idx)

    if len(idx) == 0 or frequency == "None":
        return flags

    if frequency == "Monthly":
        month_key = pd.Series(idx.to_period("M"), index=idx)
        first_days = ~month_key.duplicated()
        flags.loc[first_days.index] = first_days.values
    elif frequency == "Yearly":
        year_key = pd.Series(idx.to_period("Y"), index=idx)
        first_days = ~year_key.duplicated()
        flags.loc[first_days.index] = first_days.values

    return flags


def curve_from_returns_with_contributions(
    returns: pd.Series,
    start_value: float,
    contribution_amount: float = 0.0,
    contribution_frequency: str = "None",
) -> tuple[list[float], list[float], float]:
    returns = returns.fillna(0.0)
    idx = pd.to_datetime(returns.index)

    pv = start_value
    peak = start_value
    equity = []
    dd = []

    flags = contribution_flags(idx, contribution_frequency)
    total_contributions = 0.0

    for dt, r in zip(idx, returns):
        if bool(flags.loc[dt]) and contribution_amount > 0:
            pv += float(contribution_amount)
            total_contributions += float(contribution_amount)

        pv *= (1.0 + float(r))
        peak = max(peak, pv)
        equity.append(pv)
        dd.append((pv / peak) - 1.0)

    return equity, dd, total_contributions


def recovery_stats(drawdown_series: pd.Series) -> tuple[float | None, float | None, float | None]:
    if drawdown_series.empty:
        return None, None, None

    dd = drawdown_series.fillna(0.0).astype(float)
    underwater = dd < 0
    if not underwater.any():
        return 0.0, 0.0, 0.0

    lengths = []
    current = 0
    for flag in underwater:
        if flag:
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)

    if not lengths:
        return 0.0, 0.0, 0.0

    s = pd.Series(lengths, dtype=float)
    return float(s.median()), float(s.mean()), float(s.max())


def ulcer_index(drawdown_series: pd.Series) -> float:
    if drawdown_series.empty:
        return 0.0
    return float((drawdown_series.pow(2).mean()) ** 0.5)


def rolling_cagr_series(returns: pd.Series, window_days: int) -> pd.Series:
    returns = returns.dropna().copy()
    if len(returns) < window_days:
        return pd.Series(dtype=float)

    growth = (1.0 + returns).cumprod()
    rolling = growth / growth.shift(window_days) - 1.0
    years = window_days / TRADING_DAYS
    rolling = (1.0 + rolling).pow(1.0 / years) - 1.0
    return rolling.dropna()


def pretty_symbol(symbol: str) -> str:
    return "NDX" if symbol == "^NDX" else symbol


def core_alloc_from_regime(regime: int) -> dict:
    if regime == 1:
        return {"TQQQ": 1.0}
    if regime == 2:
        return {"SQQQ": 1.0}
    if regime == 3:
        return {"TQQQ": 0.60, "LQD": 0.10, "IAU": 0.10, "USMV": 0.10, "UUP": 0.10}
    return {"TLT": 0.10, "BIL": 0.40, "BTAL": 0.20, "USMV": 0.20, "UUP": 0.10}


def alpha_alloc_from_weight(weight: float) -> dict:
    return {"TQQQ": 1.0} if float(weight) > 0 else {"CASH": 1.0}


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


def alloc_to_text(d: dict) -> str:
    if not d:
        return "No holdings"
    items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
    return "<br>".join(f"{k}: {v:.1%}" for k, v in items)


def build_holdings_text(strategy_name: str, df: pd.DataFrame) -> list[str]:
    texts: list[str] = []

    if strategy_name == "CORE":
        for _, row in df.iterrows():
            alloc = core_alloc_from_regime(int(row["signal_regime"]))
            texts.append(alloc_to_text(alloc))
    elif strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else "weight"
        for _, row in df.iterrows():
            alloc = alpha_alloc_from_weight(float(row[weight_col]))
            texts.append(alloc_to_text(alloc))
    elif strategy_name == "COMBINED":
        for _, row in df.iterrows():
            core_alloc = core_alloc_from_regime(int(row["core_regime"]))
            alpha_alloc = alpha_alloc_from_weight(float(row["alpha_weight"]))
            alloc = combine_allocs(core_alloc, alpha_alloc, float(row["core_w"]), float(row["alpha_w"]))
            texts.append(alloc_to_text(alloc))
    else:
        texts = ["No holdings"] * len(df)

    return texts


def build_allocation_frame(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dt, row in df.iterrows():
        if strategy_name == "CORE":
            alloc = core_alloc_from_regime(int(row["signal_regime"]))
        elif strategy_name == "ALPHA":
            weight_col = "target_weight" if "target_weight" in df.columns else "weight"
            alloc = alpha_alloc_from_weight(float(row[weight_col]))
        elif strategy_name == "COMBINED":
            core_alloc = core_alloc_from_regime(int(row["core_regime"]))
            alpha_alloc = alpha_alloc_from_weight(float(row["alpha_weight"]))
            alloc = combine_allocs(core_alloc, alpha_alloc, float(row["core_w"]), float(row["alpha_w"]))
        else:
            alloc = {}

        alloc["date"] = dt
        rows.append(alloc)

    if not rows:
        return pd.DataFrame()

    alloc_df = pd.DataFrame(rows).fillna(0.0).set_index("date").sort_index()
    return alloc_df


def current_allocation_table(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame(columns=["Asset", "Weight"])

    latest = alloc_df.iloc[-1]
    latest = latest[latest > 0].sort_values(ascending=False)

    out = latest.rename_axis("Asset").reset_index(name="Weight")
    out["Weight"] = out["Weight"].map(lambda x: f"{x:.1%}")
    return out


def allocation_change_log(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame()

    rounded = alloc_df.round(6)
    changed = rounded.ne(rounded.shift(1)).any(axis=1)
    changes = rounded.loc[changed].copy()

    if changes.empty:
        return pd.DataFrame()

    changes = changes.loc[:, (changes != 0).any(axis=0)]
    changes.insert(0, "Date", changes.index.strftime("%Y-%m-%d"))
    changes = changes.reset_index(drop=True)

    value_cols = [c for c in changes.columns if c != "Date"]
    for col in value_cols:
        changes[col] = changes[col].map(lambda x: "" if pd.isna(x) or abs(float(x)) < 1e-12 else f"{float(x):.1%}")

    return changes


def benchmark_table(
    strategy_name: str,
    strategy_ret: pd.Series,
    bench_ret: pd.DataFrame,
    start_value: float,
    contribution_amount: float,
    contribution_frequency: str,
) -> pd.DataFrame:
    rows = []

    s = perf_from_returns(strategy_ret, start_value)
    strategy_eq, _, strategy_contrib = curve_from_returns_with_contributions(
        strategy_ret, start_value, contribution_amount, contribution_frequency
    )
    rows.append({
        "Series": strategy_name,
        "CAGR": f"{s['cagr']:.2%}",
        "MaxDD": f"{s['maxdd']:.2%}",
        "Sharpe": round(s["sharpe"], 3),
        "Calmar": round(s["calmar"], 3),
        "Final Value": f"${strategy_eq[-1]:,.0f}" if strategy_eq else f"${start_value:,.0f}",
        "Total Contributions": f"${strategy_contrib:,.0f}",
    })

    for col in bench_ret.columns:
        m = perf_from_returns(bench_ret[col], start_value)
        b_eq, _, b_contrib = curve_from_returns_with_contributions(
            bench_ret[col], start_value, contribution_amount, contribution_frequency
        )
        rows.append({
            "Series": col,
            "CAGR": f"{m['cagr']:.2%}",
            "MaxDD": f"{m['maxdd']:.2%}",
            "Sharpe": round(m["sharpe"], 3),
            "Calmar": round(m["calmar"], 3),
            "Final Value": f"${b_eq[-1]:,.0f}" if b_eq else f"${start_value:,.0f}",
            "Total Contributions": f"${b_contrib:,.0f}",
        })

    return pd.DataFrame(rows)


def yearly_return_table(strategy_name: str, strategy_ret: pd.Series, bench_ret: pd.DataFrame) -> pd.DataFrame:
    series_map = {strategy_name: strategy_ret}
    for col in bench_ret.columns:
        series_map[col] = bench_ret[col]

    out = {}
    for name, rets in series_map.items():
        rets = rets.dropna().copy()
        if rets.empty:
            continue
        eq = (1.0 + rets).cumprod()
        yr_map = {}
        for year, grp in eq.groupby(eq.index.year):
            if len(grp) >= 2:
                yr_map[int(year)] = (grp.iloc[-1] / grp.iloc[0] - 1.0) * 100.0
        out[name] = yr_map

    if not out:
        return pd.DataFrame()

    return pd.DataFrame(out).sort_index().round(2)


def build_equity_drawdown_fig(
    strategy_name: str,
    df: pd.DataFrame,
    bench_ret: pd.DataFrame,
    start_value: float,
    contribution_amount: float,
    contribution_frequency: str,
    hover_mode: str = "Performance",
) -> tuple[go.Figure, float, pd.Series, pd.Series]:
    strategy_equity, strategy_dd, total_contributions = curve_from_returns_with_contributions(
        df["ret"], start_value, contribution_amount, contribution_frequency
    )
    strategy_equity_s = pd.Series(strategy_equity, index=df.index, dtype=float)
    strategy_dd_s = pd.Series(strategy_dd, index=df.index, dtype=float)
    holdings_text = build_holdings_text(strategy_name, df)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
        subplot_titles=(f"{strategy_name} Equity vs Benchmarks", "Drawdown"),
    )

    if hover_mode == "Holdings %":
        strategy_hover = [
            f"Date: {dt:%Y-%m-%d}<br>{strategy_name} Equity: ${eq:,.0f}<br><br>{hold}"
            for dt, eq, hold in zip(df.index, strategy_equity_s, holdings_text)
        ]
    else:
        strategy_hover = [
            f"Date: {dt:%Y-%m-%d}<br>{strategy_name} Equity: ${eq:,.0f}"
            for dt, eq in zip(df.index, strategy_equity_s)
        ]

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=strategy_equity_s,
            mode="lines",
            name=strategy_name,
            text=strategy_hover,
            hovertemplate="%{text}<extra></extra>",
            line=dict(width=3.4, color="#fbbf24"),
        ),
        row=1,
        col=1,
    )

    if not bench_ret.empty:
        aligned_holdings = pd.Series(holdings_text, index=df.index).reindex(bench_ret.index).ffill()
        benchmark_colors = {
            "SPY": "#34d399",
            "QQQ": "#60a5fa",
            "TQQQ": "#c084fc",
            "NDX": "#f87171",
        }
        for col in bench_ret.columns:
            b_eq, _, _ = curve_from_returns_with_contributions(
                bench_ret[col], start_value, contribution_amount, contribution_frequency
            )

            if hover_mode == "Holdings %":
                bench_hover = [
                    f"Date: {dt:%Y-%m-%d}<br>{col}: ${eq:,.0f}<br><br>{hold}"
                    for dt, eq, hold in zip(bench_ret.index, b_eq, aligned_holdings.fillna("No holdings"))
                ]
            else:
                bench_hover = [f"Date: {dt:%Y-%m-%d}<br>{col}: ${eq:,.0f}" for dt, eq in zip(bench_ret.index, b_eq)]

            fig.add_trace(
                go.Scatter(
                    x=bench_ret.index,
                    y=b_eq,
                    mode="lines",
                    name=col,
                    text=bench_hover,
                    hovertemplate="%{text}<extra></extra>",
                    line=dict(width=1.7, color=benchmark_colors.get(col, "#94a3b8")),
                    opacity=0.66,
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=strategy_dd_s.index,
            y=strategy_dd_s,
            mode="lines",
            fill="tozeroy",
            line=dict(color="#f87171", width=2.0),
            name="Drawdown",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=760,
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=24, r=24, t=78, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
    )
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig, total_contributions, strategy_equity_s, strategy_dd_s


def build_rolling_fig(returns: pd.Series) -> tuple[go.Figure, pd.Series, pd.Series]:
    rolling_3y = rolling_cagr_series(returns, 756)
    rolling_2y = rolling_cagr_series(returns, 504)

    fig = go.Figure()
    if not rolling_3y.empty:
        fig.add_trace(
            go.Scatter(
                x=rolling_3y.index,
                y=rolling_3y,
                mode="lines",
                name="Rolling 3Y CAGR",
                line=dict(color="#34d399", width=2.4),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Rolling 3Y CAGR: %{y:.2%}<extra></extra>",
            )
        )
    if not rolling_2y.empty:
        fig.add_trace(
            go.Scatter(
                x=rolling_2y.index,
                y=rolling_2y,
                mode="lines",
                name="Rolling 2Y CAGR",
                line=dict(color="#60a5fa", width=2.0, dash="dot"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Rolling 2Y CAGR: %{y:.2%}<extra></extra>",
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.25)")
    fig.update_layout(
        title="Rolling CAGR Stability",
        height=360,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=24, r=24, t=54, b=24),
        yaxis_tickformat=".0%",
        xaxis_title="Date",
        yaxis_title="CAGR",
        legend=dict(orientation="h", yanchor="bottom", y=0.98, x=0, font=dict(size=11)),
    )
    return fig, rolling_3y, rolling_2y


def build_regime_fig(strategy_name: str, df: pd.DataFrame) -> go.Figure:
    if strategy_name == "CORE" and "signal_regime" in df.columns:
        regime_series = df["signal_regime"].astype(int).rename("Regime")
    elif strategy_name == "COMBINED" and "core_regime" in df.columns:
        regime_series = df["core_regime"].astype(int).rename("Regime")
    else:
        return go.Figure()

    regime_labels = {
        1: "R1 Risk-On",
        2: "R2 Bear",
        3: "R3 Growth",
        4: "R4 Defensive",
    }
    regime_colors = {
        "R1 Risk-On": "#34d399",
        "R2 Bear": "#f87171",
        "R3 Growth": "#60a5fa",
        "R4 Defensive": "#fbbf24",
    }

    counts = regime_series.map(regime_labels).value_counts().reindex(regime_colors.keys()).fillna(0)
    counts = counts[counts > 0]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.index),
                values=list(counts.values),
                hole=0.62,
                marker=dict(colors=[regime_colors[k] for k in counts.index]),
                textinfo="percent",
                textposition="inside",
                insidetextorientation="radial",
                sort=False,
                hovertemplate="%{label}<br>Days: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Regime Distribution",
        height=390,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=24, r=24, t=54, b=24),
        legend=dict(orientation="v", x=1.02, y=0.5, font=dict(size=12)),
    )
    return fig


def build_yearly_heatmap(yr: pd.DataFrame) -> go.Figure:
    if yr.empty:
        return go.Figure()

    z = yr.T.values
    x = [str(i) for i in yr.index.tolist()]
    y = yr.columns.tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[
                [0.0, "#7f1d1d"],
                [0.45, "#ef4444"],
                [0.5, "#111827"],
                [0.55, "#22c55e"],
                [1.0, "#14532d"],
            ],
            zmid=0,
            text=np.round(z, 2),
            texttemplate="%{text:.2f}%",
            textfont={"size": 11},
            hovertemplate="Series: %{y}<br>Year: %{x}<br>Return: %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Yearly Return Heatmap",
        height=390,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        margin=dict(l=24, r=24, t=54, b=24),
        xaxis_title="Year",
        yaxis_title="Series",
    )
    return fig


st.set_page_config(
    page_title="ADAPT Strategy Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --panel: #111a2e;
        --panel-2: #0f172a;
        --border: rgba(148, 163, 184, 0.16);
        --text: #e5ecf6;
        --muted: #9fb0c7;
        --gold: #fbbf24;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(251,191,36,0.08), transparent 25%),
            linear-gradient(180deg, #07111d 0%, #0b1324 100%);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.2rem;
        max-width: 1540px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="collapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.94rem !important;
        color: #d7e2f1 !important;
        font-weight: 500;
    }

    h1 {
        font-size: 2.45rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem !important;
    }

    h2, h3 {
        font-weight: 740 !important;
        letter-spacing: -0.02em;
    }

    .hero {
        background: linear-gradient(135deg, rgba(17,26,46,0.96) 0%, rgba(12,19,36,0.98) 100%);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 22px 24px 18px 24px;
        box-shadow: 0 18px 42px rgba(0,0,0,0.22);
        margin-bottom: 16px;
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.78rem;
        color: var(--gold);
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .hero-sub {
        color: var(--muted);
        font-size: 1rem;
        margin-top: 0.25rem;
    }

    .hero-pill-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: flex-end;
    }

    .hero-pill {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 8px 12px;
        color: #dce6f4;
        font-size: 0.88rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .section-card {
        background: linear-gradient(180deg, rgba(17,26,46,0.94) 0%, rgba(15,23,42,0.96) 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 12px 14px 8px 14px;
        box-shadow: 0 12px 32px rgba(0,0,0,0.16);
        margin-bottom: 12px;
    }

    .metric-shell {
        background: linear-gradient(180deg, rgba(20,29,48,0.96) 0%, rgba(14,22,38,0.98) 100%);
        border: 1px solid rgba(251,191,36,0.12);
        border-radius: 18px;
        padding: 16px 16px 14px 16px;
        min-height: 132px;
        box-shadow: 0 14px 30px rgba(0,0,0,0.16);
    }

    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.74rem;
        color: #93a4bd;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 2.0rem;
        line-height: 1.0;
        font-weight: 800;
        color: #f8fbff;
        margin-bottom: 8px;
    }

    .metric-sub {
        font-size: 0.88rem;
        line-height: 1.35;
        color: #9fb0c7;
    }

    .metric-sub strong { color: #e5ecf6; }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 16px;
        overflow: hidden;
    }

    div.stButton > button {
        border-radius: 14px;
        font-weight: 700;
        border: 1px solid rgba(251,191,36,0.24);
    }

    .note-box {
        background: rgba(96,165,250,0.08);
        border: 1px solid rgba(96,165,250,0.18);
        color: #dce6f4;
        border-radius: 16px;
        padding: 12px 14px;
        margin-top: 6px;
        margin-bottom: 6px;
        font-size: 0.93rem;
    }

    @media (max-width: 1200px) {
        .hero-top {
            flex-direction: column;
        }
        .hero-pill-wrap {
            justify-content: flex-start;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-top">
        <div>
          <div class="eyebrow">ADAPT Strategy Terminal</div>
          <h1>Brokerage-Ready Interactive Dashboard</h1>
          <div class="hero-sub">
            Institutional-style analytics for CORE, ALPHA, and COMBINED. This dashboard is focused on display quality
            while preserving the validated ADAPT engine workflow.
          </div>
        </div>
        <div class="hero-pill-wrap">
          <div class="hero-pill">EOD signal workflow</div>
          <div class="hero-pill">Same-day close / MOC assumption</div>
          <div class="hero-pill">Cash yield framing: IBKR idle cash / SGOV / BIL / T-bills</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

strategy_ranges = get_strategy_ranges()

with st.sidebar:
    st.markdown("## ADAPT Controls")
    st.caption("Use the small arrow in the upper-left corner to collapse or reopen this panel.")

    strategy_choice = st.selectbox("Strategy", ["CORE", "ALPHA", "COMBINED"])

    start_capital = st.number_input("Starting Capital", min_value=1000.0, value=DEFAULT_START_CAPITAL, step=1000.0)
    contribution_amount = st.number_input("Contribution Amount", min_value=0.0, value=0.0, step=100.0)
    contribution_frequency = st.selectbox("Contribution Frequency", ["None", "Monthly", "Yearly"], index=0)
    hover_mode = st.radio("Hover Display", ["Performance", "Holdings %"], index=0)

    available_benchmarks = ["SPY", "QQQ", "TQQQ", "NDX"]
    selected_benchmarks = st.multiselect("Benchmarks", available_benchmarks, default=["SPY", "QQQ", "TQQQ"])

    benchmark_download_symbols = ["^NDX" if x == "NDX" else x for x in selected_benchmarks]
    px = get_price_history(benchmark_download_symbols, start="1990-01-01")
    valid_dates = first_valid_dates(px)

    strategy_min_date = strategy_ranges[strategy_choice]
    benchmark_min_dates = [valid_dates[s] for s in benchmark_download_symbols if s in valid_dates]
    min_allowed_date = max([strategy_min_date] + benchmark_min_dates) if benchmark_min_dates else strategy_min_date

    display_valid = {pretty_symbol(k): v.strftime("%Y-%m-%d") for k, v in valid_dates.items()}
    with st.expander("First valid dates"):
        st.write("Strategy earliest rows:")
        st.json({k: v.strftime("%Y-%m-%d") for k, v in strategy_ranges.items()})
        st.write("Benchmark earliest rows:")
        st.json(display_valid)

    default_start = max(pd.Timestamp("2022-01-01"), min_allowed_date)
    today = pd.Timestamp.today().normalize()

    start_date = st.date_input(
        "Start Date",
        value=default_start.date(),
        min_value=min_allowed_date.date(),
        max_value=today.date(),
    )
    end_date = st.date_input(
        "End Date",
        value=today.date(),
        min_value=min_allowed_date.date(),
        max_value=today.date(),
    )

    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    st.markdown(
        """
        <div class="note-box">
        Official dashboard note: top metric cards and benchmark comparison are driven from the ADAPT return stream.
        Final value and equity curves include scheduled contributions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    run = st.button("Run Backtest", type="primary", use_container_width=True)

if run:
    with st.spinner("Running ADAPT..."):
        data_map = get_strategy_bundle()

    raw_df = data_map[strategy_choice].copy()
    df = raw_df.loc[str(start_date):str(end_date)].copy()

    if df.empty:
        st.error(
            f"No strategy data available for the selected date range. "
            f"Earliest usable date for {strategy_choice} is {strategy_min_date.date()}."
        )
        st.stop()

    bench_px = get_price_history(benchmark_download_symbols, start=str(start_date))
    bench_px = bench_px.loc[str(start_date):str(end_date)].copy()

    if not bench_px.empty:
        bench_px = bench_px.rename(columns={c: pretty_symbol(c) for c in bench_px.columns})
        bench_ret = bench_px.pct_change().dropna(how="all")
        bench_ret = bench_ret.reindex(df.index).dropna(how="all")
    else:
        bench_ret = pd.DataFrame(index=df.index)

    perf = perf_from_returns(df["ret"], start_capital)
    equity_fig, total_contributions, strategy_equity_s, strategy_dd_s = build_equity_drawdown_fig(
        strategy_choice,
        df,
        bench_ret,
        start_capital,
        contribution_amount,
        contribution_frequency,
        hover_mode,
    )

    final_value = float(strategy_equity_s.iloc[-1]) if not strategy_equity_s.empty else start_capital
    ulcer = ulcer_index(strategy_dd_s)
    median_rec, avg_rec, max_rec = recovery_stats(strategy_dd_s)

    rolling_fig, rolling_3y, rolling_2y = build_rolling_fig(df["ret"])
    regime_fig = build_regime_fig(strategy_choice, df)

    roll3_best = float(rolling_3y.max()) if not rolling_3y.empty else 0.0
    roll3_worst = float(rolling_3y.min()) if not rolling_3y.empty else 0.0
    roll3_median = float(rolling_3y.median()) if not rolling_3y.empty else 0.0

    med_2y = float(rolling_2y.median()) if not rolling_2y.empty else 0.0
    prob_neg = float((rolling_2y < 0).mean()) if not rolling_2y.empty else 0.0
    prob_lt5 = float((rolling_2y < 0.05).mean()) if not rolling_2y.empty else 0.0
    prob_lt8 = float((rolling_2y < 0.08).mean()) if not rolling_2y.empty else 0.0
    prob_lt10 = float((rolling_2y < 0.10).mean()) if not rolling_2y.empty else 0.0

    yr = yearly_return_table(strategy_choice, df["ret"], bench_ret)
    heatmap_fig = build_yearly_heatmap(yr)

    current_alloc = current_allocation_table(strategy_choice, df)
    alloc_changes = allocation_change_log(strategy_choice, df)

    st.markdown(f"### {strategy_choice} Overview")

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">CAGR</div>
              <div class="metric-value">{perf['cagr']:.2%}</div>
              <div class="metric-sub"><strong>{perf['days']}</strong> trading days<br>Calculated from df["ret"]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top2:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">Max Drawdown</div>
              <div class="metric-value">{perf['maxdd']:.2%}</div>
              <div class="metric-sub">Ulcer Index <strong>{ulcer:.2%}</strong><br>Recovery metrics below</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top3:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">Sharpe</div>
              <div class="metric-value">{perf['sharpe']:.2f}</div>
              <div class="metric-sub">Calmar <strong>{perf['calmar']:.2f}</strong><br>Win rate <strong>{perf['win_rate']:.1%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top4:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">Final Value</div>
              <div class="metric-value">${final_value:,.0f}</div>
              <div class="metric-sub">Contributions <strong>${total_contributions:,.0f}</strong><br>Equity includes contribution schedule</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    mid1, mid2, mid3, mid4 = st.columns(4)
    with mid1:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">Median Recovery</div>
              <div class="metric-value">{0 if median_rec is None else median_rec:.0f} d</div>
              <div class="metric-sub">Average <strong>{0 if avg_rec is None else avg_rec:.1f} d</strong><br>Longest <strong>{0 if max_rec is None else max_rec:.0f} d</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid2:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">Rolling 3Y CAGR</div>
              <div class="metric-value">{roll3_median:.2%}</div>
              <div class="metric-sub">Best <strong>{roll3_best:.2%}</strong><br>Worst <strong>{roll3_worst:.2%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid3:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">2Y Underperformance</div>
              <div class="metric-value">{prob_neg:.2%}</div>
              <div class="metric-sub">Prob(2Y CAGR &lt; 0%)<br>Median 2Y CAGR <strong>{med_2y:.2%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid4:
        st.markdown(
            f"""
            <div class="metric-shell">
              <div class="metric-label">Sub-Target Risk</div>
              <div class="metric-value">{prob_lt10:.2%}</div>
              <div class="metric-sub">Prob(&lt;5%) <strong>{prob_lt5:.2%}</strong><br>Prob(&lt;8%) <strong>{prob_lt8:.2%}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(equity_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    row2_left, row2_right = st.columns([1.15, 0.85])
    with row2_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(rolling_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row2_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if strategy_choice in {"CORE", "COMBINED"}:
            st.plotly_chart(regime_fig, use_container_width=True)
        else:
            st.subheader("Current Allocation")
            st.dataframe(current_alloc, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    row3_left, row3_right = st.columns([0.95, 1.05])
    with row3_left:
        st.subheader("Current Allocation")
        st.dataframe(current_alloc, use_container_width=True, hide_index=True)

    with row3_right:
        st.subheader("Allocation Change Log")
        st.dataframe(alloc_changes, use_container_width=True, hide_index=True)

    row4_left, row4_right = st.columns([1.1, 0.9])
    with row4_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with row4_right:
        st.subheader(f"{strategy_choice} vs Benchmarks")
        st.dataframe(
            benchmark_table(
                strategy_choice,
                df["ret"],
                bench_ret,
                start_capital,
                contribution_amount,
                contribution_frequency,
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Yearly Returns (%)")
    st.dataframe(yr, use_container_width=True)

    with st.expander("Validation / Audit Notes", expanded=False):
        v1, v2 = st.columns(2)
        with v1:
            st.write(f"Earliest usable date for {strategy_choice}: **{strategy_min_date.date()}**")
            st.write(f"Earliest allowed date with selected benchmarks: **{min_allowed_date.date()}**")
            st.write(f"Selected range: **{start_date}** to **{end_date}**")
            st.write(f"Hover mode: **{hover_mode}**")
        with v2:
            st.write(f"Contribution schedule: **{contribution_frequency}**")
            st.write(f"Contribution amount: **${contribution_amount:,.0f}**")
            st.write(f"2Y CAGR < 5%: **{prob_lt5:.2%}**")
            st.write(f"2Y CAGR < 8%: **{prob_lt8:.2%}**")
            st.write(f"2Y CAGR < 10%: **{prob_lt10:.2%}**")

    st.caption(
        'Note: CAGR, MaxDD, Sharpe, rolling CAGR, and underperformance probabilities are calculated from the raw return stream '
        'using df["ret"]. Final Value and equity charts include scheduled contributions.'
    )
else:
    st.info("Choose your settings in the sidebar, then click Run Backtest.")
