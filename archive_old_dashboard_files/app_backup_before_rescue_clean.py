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


def section_label(title: str, tooltip: str) -> str:
    return (
        '<div style="display:flex;align-items:center;gap:8px;font-weight:700;">'
        f'<span>{title}</span>'
        f'<span title="{tooltip}" style="display:inline-flex;align-items:center;justify-content:center;'
        'width:18px;height:18px;border-radius:50%;border:1px solid #64748b;font-size:11px;cursor:help;">i</span>'
        '</div>'
    )


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
    return {"CORE": core_df, "ALPHA": alpha_df, "COMBINED": combined_df}


def perf_from_returns(returns: pd.Series, start_value: float) -> dict:
    returns = returns.dropna().copy()
    if len(returns) == 0:
        return {"cagr": 0.0, "maxdd": 0.0, "sharpe": 0.0, "final_value": start_value}
    pv = start_value
    peak = start_value
    dds = []
    for r in returns:
        pv *= 1.0 + float(r)
        peak = max(peak, pv)
        dds.append((pv / peak) - 1.0)
    years = len(returns) / TRADING_DAYS
    cagr = (pv / start_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    maxdd = float(min(dds)) if dds else 0.0
    std = returns.std()
    sharpe = (returns.mean() * TRADING_DAYS) / (std * math.sqrt(TRADING_DAYS)) if std and std > 0 else 0.0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "final_value": pv}


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
    out: dict[str, float] = {}
    for k, v in core_alloc.items():
        out[k] = out.get(k, 0.0) + float(v) * float(core_w)
    for k, v in alpha_alloc.items():
        out[k] = out.get(k, 0.0) + float(v) * float(alpha_w)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def build_allocation_frame(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dt, row in df.iterrows():
        if strategy_name == "CORE":
            alloc = core_alloc_from_regime(int(row["signal_regime"]))
        elif strategy_name == "ALPHA":
            weight_col = "target_weight" if "target_weight" in df.columns else "weight"
            alloc = alpha_alloc_from_weight(float(row[weight_col]))
        else:
            alloc = combine_allocs(
                core_alloc_from_regime(int(row["core_regime"])),
                alpha_alloc_from_weight(float(row["alpha_weight"])),
                float(row["core_w"]),
                float(row["alpha_w"]),
            )
        alloc["date"] = dt
        rows.append(alloc)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).fillna(0.0).set_index("date").sort_index()


def allocation_change_log(strategy_name: str, df: pd.DataFrame, newest_first: bool = True) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame()
    changed = alloc_df.round(6).ne(alloc_df.round(6).shift(1)).any(axis=1)
    changes = alloc_df.loc[changed].copy()
    if changes.empty:
        return pd.DataFrame()
    changes = changes.sort_index(ascending=not newest_first)
    changes.insert(0, "Date", changes.index.strftime("%Y-%m-%d"))
    changes = changes.reset_index(drop=True)
    for col in changes.columns[1:]:
        changes[col] = changes[col].map(lambda x: "" if abs(float(x)) < 1e-12 else f"{float(x):.1%}")
    return changes


def count_rebalance_events(strategy_name: str, df: pd.DataFrame) -> int:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty or len(alloc_df) < 2:
        return 0
    changed = alloc_df.round(6).ne(alloc_df.round(6).shift(1)).any(axis=1)
    return int(changed.iloc[1:].sum())


def get_state_distribution(strategy_name: str, df: pd.DataFrame):
    if strategy_name == "CORE":
        labels = {1: "R1 Risk-On", 2: "R2 Bear", 3: "R3 Growth", 4: "R4 Defensive"}
        counts = df["signal_regime"].astype(int).map(labels).value_counts() if "signal_regime" in df.columns else pd.Series(dtype=float)
        return counts, "CORE Regime Distribution"
    if strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if weight_col is None:
            return pd.Series(dtype=float), "ALPHA State Distribution"
        counts = pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "ALPHA Risk-On", "ALPHA Cash"), index=df.index).value_counts()
        return counts, "ALPHA State Distribution"
    label_map = {"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}
    counts = df["allocator_state"].astype(str).map(lambda x: label_map.get(x, x)).value_counts() if "allocator_state" in df.columns else pd.Series(dtype=float)
    return counts, "COMBINED Allocator State Distribution"


def build_state_fig(counts: pd.Series) -> go.Figure:
    fig = go.Figure()
    if counts.empty:
        return fig
    colors = {
        "Risk On": "#34d399",
        "Neutral": "#60a5fa",
        "Risk Off": "#fbbf24",
        "R1 Risk-On": "#34d399",
        "R2 Bear": "#f87171",
        "R3 Growth": "#60a5fa",
        "R4 Defensive": "#fbbf24",
        "ALPHA Risk-On": "#34d399",
        "ALPHA Cash": "#94a3b8",
    }
    fig.add_trace(go.Pie(
        labels=list(counts.index),
        values=list(counts.values),
        hole=0.62,
        sort=False,
        marker=dict(colors=[colors.get(x, "#94a3b8") for x in counts.index]),
        textinfo="percent",
    ))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", height=340, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def build_equity_fig(df: pd.DataFrame, bench_ret: pd.DataFrame, strategy_name: str, start_value: float) -> go.Figure:
    ret = df["ret"].fillna(0.0)
    eq = start_value * (1.0 + ret).cumprod()
    dd = eq / eq.cummax() - 1.0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08, subplot_titles=(f"{strategy_name} Equity vs Benchmarks", "Drawdown"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=strategy_name, line=dict(width=3, color="#fbbf24")), row=1, col=1)
    for col in bench_ret.columns:
        beq = start_value * (1.0 + bench_ret[col].fillna(0.0)).cumprod()
        fig.add_trace(go.Scatter(x=beq.index, y=beq.values, mode="lines", name=col, line=dict(width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown", line=dict(color="#f87171")), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", height=700, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return fig


st.set_page_config(page_title="ADAPT Stable Reset", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #050d19 0%, #091223 100%); color: white; }
    .block-container { max-width: 1500px; padding-top: 2rem; }
    .hero { background: #0f172a; border: 1px solid #334155; border-radius: 18px; padding: 20px; margin-bottom: 16px; }
    .metricbox { background: #0f172a; border: 1px solid #334155; border-radius: 14px; padding: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">ADAPT Strategy Terminal</h2>
      <div style="color:#94a3b8;margin-top:6px;">Clean reset build using allocator_state for COMBINED state display.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle = get_strategy_bundle()

with st.sidebar:
    strategy = st.selectbox("Strategy", ["COMBINED", "CORE", "ALPHA"], index=0)
    start_capital = st.number_input("Starting Capital", min_value=1000.0, value=DEFAULT_START_CAPITAL, step=1000.0)
    sort_order = st.selectbox("Allocation Trade Log Order", ["Newest to Oldest", "Oldest to Newest"], index=0)
    benchmarks = st.multiselect("Benchmarks", ["SPY", "QQQ", "TQQQ", "NDX"], default=["SPY", "QQQ", "TQQQ"])
    run = st.button("Run Backtest", type="primary", width="stretch")

if run:
    df = bundle[strategy].copy()

    bench_symbols = ["^NDX" if x == "NDX" else x for x in benchmarks]
    bench_px = get_price_history(bench_symbols, start=str(df.index.min().date()))
    if not bench_px.empty:
        bench_px = bench_px.rename(columns={"^NDX": "NDX"})
        bench_ret = bench_px.pct_change().reindex(df.index).dropna(how="all")
    else:
        bench_ret = pd.DataFrame(index=df.index)

    counts, state_title = get_state_distribution(strategy, df)
    fig_state = build_state_fig(counts)
    fig_eq = build_equity_fig(df, bench_ret, strategy, start_capital)

    perf = perf_from_returns(df["ret"], start_capital)
    rebalance_events = count_rebalance_events(strategy, df)
    alloc_log = allocation_change_log(strategy, df, newest_first=(sort_order == "Newest to Oldest"))
    alloc_now = current_allocation_table(strategy, df)
    alloc_profile = state_allocation_profile(strategy, df)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metricbox"><div>CAGR</div><h3>{perf["cagr"]:.2%}</h3></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metricbox"><div>MaxDD</div><h3>{perf["maxdd"]:.2%}</h3></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metricbox"><div>Sharpe</div><h3>{perf["sharpe"]:.2f}</h3></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metricbox"><div>Rebalance Events</div><h3>{rebalance_events:,}</h3></div>', unsafe_allow_html=True)

    st.plotly_chart(fig_eq, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.markdown(section_label(state_title, "COMBINED uses allocator_state: Risk On / Neutral / Risk Off."), unsafe_allow_html=True)
        st.plotly_chart(fig_state, width="stretch")
    with right:
        st.markdown("#### Current Allocation")
        st.dataframe(alloc_now, width="stretch", hide_index=True)

    left2, right2 = st.columns([1, 1])
    with left2:
        st.markdown("#### State Allocation Profile")
        st.dataframe(alloc_profile, width="stretch", hide_index=True)
    with right2:
        st.markdown("#### Allocation Trade Log")
        st.dataframe(alloc_log.head(100), width="stretch", hide_index=True)
else:
    st.info("Run Backtest to load the stable reset dashboard.")
