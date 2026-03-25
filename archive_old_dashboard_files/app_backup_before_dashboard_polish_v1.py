from __future__ import annotations

import math
from io import StringIO

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


def tooltip_label(title: str, tooltip: str, subtitle: str | None = None) -> str:
    subtitle_html = f'<span class="label-sub">{subtitle}</span>' if subtitle else ""
    return f"""
    <div class="label-wrap">
      <div class="label-row">
        <span class="label-title">{title}</span>
        <span class="tooltip-wrap">
          <span class="tooltip-icon">i</span>
          <span class="tooltip-box">{tooltip}</span>
        </span>
      </div>
      {subtitle_html}
    </div>
    """


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


def first_valid_dates(px: pd.DataFrame) -> dict[str, pd.Timestamp]:
    out: dict[str, pd.Timestamp] = {}
    if px.empty:
        return out
    for col in px.columns:
        s = px[col].dropna()
        if not s.empty:
            out[str(col)] = pd.to_datetime(s.index[0]).normalize()
    return out


def get_strategy_ranges(bundle: dict[str, pd.DataFrame]) -> dict[str, pd.Timestamp]:
    return {name: pd.to_datetime(df.index.min()).normalize() for name, df in bundle.items()}


def perf_from_returns(returns: pd.Series, start_value: float) -> dict:
    returns = returns.dropna().copy()
    if len(returns) == 0:
        return {"cagr": 0.0, "maxdd": 0.0, "sharpe": 0.0, "final_value": start_value, "days": 0, "calmar": 0.0}
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
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "final_value": pv, "days": len(returns), "calmar": calmar}


def contribution_flags(index: pd.Index, frequency: str) -> pd.Series:
    idx = pd.to_datetime(index)
    flags = pd.Series(False, index=idx)
    if len(idx) == 0 or frequency == "None":
        return flags
    if frequency == "Monthly":
        key = pd.Series(idx.to_period("M"), index=idx)
        flags.loc[~key.duplicated()] = True
    elif frequency == "Yearly":
        key = pd.Series(idx.to_period("Y"), index=idx)
        flags.loc[~key.duplicated()] = True
    return flags


def equity_with_contributions(
    returns: pd.Series,
    start_value: float,
    contribution_amount: float,
    contribution_frequency: str,
) -> tuple[pd.Series, pd.Series, float]:
    ret = returns.fillna(0.0).copy()
    flags = contribution_flags(ret.index, contribution_frequency)
    pv = start_value
    peak = start_value
    total_contributions = 0.0
    eq = []
    dd = []
    for dt, r in ret.items():
        if bool(flags.loc[dt]) and contribution_amount > 0:
            pv += float(contribution_amount)
            total_contributions += float(contribution_amount)
        pv *= 1.0 + float(r)
        peak = max(peak, pv)
        eq.append(pv)
        dd.append((pv / peak) - 1.0)
    return (
        pd.Series(eq, index=ret.index, dtype=float),
        pd.Series(dd, index=ret.index, dtype=float),
        total_contributions,
    )


def rolling_cagr_series(returns: pd.Series, window_days: int) -> pd.Series:
    returns = returns.dropna().copy()
    if len(returns) < window_days:
        return pd.Series(dtype=float)
    growth = (1.0 + returns).cumprod()
    rolling = growth / growth.shift(window_days) - 1.0
    years = window_days / TRADING_DAYS
    rolling = (1.0 + rolling).pow(1.0 / years) - 1.0
    return rolling.dropna()


def recovery_stats(drawdown_series: pd.Series):
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
            alloc = core_alloc_from_regime(int(row["signal_regime"])) if "signal_regime" in df.columns else {}
        elif strategy_name == "ALPHA":
            weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
            alloc = alpha_alloc_from_weight(float(row[weight_col])) if weight_col else {}
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


def current_allocation_table(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame(columns=["Asset", "Weight"])
    latest = alloc_df.iloc[-1]
    latest = latest[latest > 0].sort_values(ascending=False)
    out = latest.rename_axis("Asset").reset_index(name="Weight")
    out["Weight"] = out["Weight"].map(lambda x: f"{x:.1%}")
    return out


def allocation_change_log(strategy_name: str, df: pd.DataFrame, newest_first: bool = True) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame()
    changed = alloc_df.round(6).ne(alloc_df.round(6).shift(1)).any(axis=1)
    changes = alloc_df.loc[changed].copy()
    if changes.empty:
        return pd.DataFrame()
    changes = changes.loc[:, (changes != 0).any(axis=0)]
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
        return counts, "CORE Regime Distribution", "CORE state distribution."
    if strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if weight_col is None:
            return pd.Series(dtype=float), "ALPHA State Distribution", "ALPHA state distribution."
        counts = pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "ALPHA Risk-On", "ALPHA Cash"), index=df.index).value_counts()
        return counts, "ALPHA State Distribution", "ALPHA state distribution."
    label_map = {"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}
    counts = df["allocator_state"].astype(str).map(lambda x: label_map.get(x, x)).value_counts() if "allocator_state" in df.columns else pd.Series(dtype=float)
    return counts, "COMBINED Allocator State Distribution", "COMBINED uses allocator_state: Risk On / Neutral / Risk Off."


def current_state_label(strategy_name: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "Unavailable"
    if strategy_name == "COMBINED" and "allocator_state" in df.columns:
        return {"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}.get(str(df["allocator_state"].iloc[-1]), str(df["allocator_state"].iloc[-1]))
    if strategy_name == "CORE" and "signal_regime" in df.columns:
        return {1: "R1 Risk-On", 2: "R2 Bear", 3: "R3 Growth", 4: "R4 Defensive"}.get(int(df["signal_regime"].iloc[-1]), "Unavailable")
    weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
    if weight_col:
        return "Risk-On" if float(df[weight_col].iloc[-1]) > 0 else "Cash"
    return "Unavailable"


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


def state_allocation_profile(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame()
    if strategy_name == "COMBINED" and "allocator_state" in df.columns:
        states = df["allocator_state"].astype(str).map({"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"})
    elif strategy_name == "CORE" and "signal_regime" in df.columns:
        states = df["signal_regime"].astype(int).map({1: "R1", 2: "R2", 3: "R3", 4: "R4"})
    else:
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if not weight_col:
            return pd.DataFrame()
        states = pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "Risk-On", "Cash"), index=df.index)
    merged = alloc_df.copy()
    merged["State"] = states.reindex(alloc_df.index)
    profile = merged.groupby("State").mean(numeric_only=True)
    if profile.empty:
        return pd.DataFrame()
    out = profile.reset_index()
    for col in out.columns[1:]:
        out[col] = out[col].map(lambda x: f"{float(x):.1%}")
    return out


def build_equity_fig(
    df: pd.DataFrame,
    bench_ret: pd.DataFrame,
    strategy_name: str,
    start_value: float,
    contribution_amount: float,
    contribution_frequency: str,
) -> go.Figure:
    eq, dd, _ = equity_with_contributions(df["ret"], start_value, contribution_amount, contribution_frequency)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08,
        subplot_titles=(f"{strategy_name} Equity vs Benchmarks", "Drawdown")
    )
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=strategy_name, line=dict(width=3, color="#fbbf24")), row=1, col=1)
    color_map = {"SPY": "#34d399", "QQQ": "#60a5fa", "TQQQ": "#c084fc", "NDX": "#f87171"}
    for col in bench_ret.columns:
        beq, _, _ = equity_with_contributions(bench_ret[col].fillna(0.0), start_value, contribution_amount, contribution_frequency)
        fig.add_trace(go.Scatter(x=beq.index, y=beq.values, mode="lines", name=col, line=dict(width=1.5, color=color_map.get(col, "#94a3b8"))), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines", fill="tozeroy", name="Drawdown", line=dict(color="#f87171")), row=2, col=1)
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", height=700, margin=dict(l=20, r=20, t=60, b=20))
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    return fig


def build_rolling_fig(returns: pd.Series) -> tuple[go.Figure, pd.Series, pd.Series]:
    rolling_3y = rolling_cagr_series(returns, 756)
    rolling_2y = rolling_cagr_series(returns, 504)
    fig = go.Figure()
    if not rolling_3y.empty:
        fig.add_trace(go.Scatter(x=rolling_3y.index, y=rolling_3y, mode="lines", name="Rolling 3Y CAGR", line=dict(color="#34d399", width=2.25)))
    if not rolling_2y.empty:
        fig.add_trace(go.Scatter(x=rolling_2y.index, y=rolling_2y, mode="lines", name="Rolling 2Y CAGR", line=dict(color="#60a5fa", width=1.95, dash="dot")))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.22)")
    fig.update_layout(height=340, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=20, r=20, t=20, b=20), yaxis_tickformat=".0%", xaxis_title="Date", yaxis_title="CAGR", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig, rolling_3y, rolling_2y


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


def build_yearly_heatmap(yr: pd.DataFrame) -> go.Figure:
    if yr.empty:
        return go.Figure()
    z = yr.T.values
    x = [str(i) for i in yr.index.tolist()]
    y = yr.columns.tolist()
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0.0, "#7f1d1d"], [0.45, "#ef4444"], [0.5, "#111827"], [0.55, "#22c55e"], [1.0, "#14532d"]],
        zmid=0,
        text=np.round(z, 2),
        texttemplate="%{text:.2f}%",
        textfont={"size": 11},
    ))
    fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Year", yaxis_title="Series")
    return fig


def benchmark_table(strategy_name: str, strategy_ret: pd.Series, bench_ret: pd.DataFrame, start_value: float, contribution_amount: float, contribution_frequency: str) -> pd.DataFrame:
    rows = []
    s = perf_from_returns(strategy_ret, start_value)
    strategy_eq, _, strategy_contrib = equity_with_contributions(strategy_ret, start_value, contribution_amount, contribution_frequency)
    rows.append({
        "Series": strategy_name,
        "CAGR": f"{s['cagr']:.2%}",
        "MaxDD": f"{s['maxdd']:.2%}",
        "Sharpe": round(s["sharpe"], 3),
        "Calmar": round(s["calmar"], 3),
        "Final Value": f"${strategy_eq.iloc[-1]:,.0f}" if not strategy_eq.empty else f"${start_value:,.0f}",
        "Total Contributions": f"${strategy_contrib:,.0f}",
    })
    for col in bench_ret.columns:
        m = perf_from_returns(bench_ret[col], start_value)
        b_eq, _, b_contrib = equity_with_contributions(bench_ret[col], start_value, contribution_amount, contribution_frequency)
        rows.append({
            "Series": col,
            "CAGR": f"{m['cagr']:.2%}",
            "MaxDD": f"{m['maxdd']:.2%}",
            "Sharpe": round(m["sharpe"], 3),
            "Calmar": round(m["calmar"], 3),
            "Final Value": f"${b_eq.iloc[-1]:,.0f}" if not b_eq.empty else f"${start_value:,.0f}",
            "Total Contributions": f"${b_contrib:,.0f}",
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="ADAPT Restored Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #050d19 0%, #091223 100%); color: white; }
    .block-container { max-width: 1500px; padding-top: 2rem; }
    .hero { background: linear-gradient(135deg, #0f172a 0%, #111f3d 100%); border: 1px solid #334155; border-radius: 18px; padding: 20px; margin-bottom: 16px; }
    .metricbox { background: #0f172a; border: 1px solid #334155; border-radius: 14px; padding: 14px; min-height: 110px; }
    .metriclabel { color:#94a3b8; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; }
    .metricvalue { color:white; font-size:2rem; font-weight:700; margin-top:6px; }
    .metricsub { color:#94a3b8; font-size:0.95rem; margin-top:6px; line-height:1.35; }
    .label-wrap { margin-bottom: 8px; }
    .label-row { display:flex; align-items:center; gap:8px; }
    .label-title { font-weight:700; color:white; }
    .label-sub { display:block; color:#94a3b8; font-size:0.9rem; margin-top:4px; }
    .tooltip-wrap { position:relative; display:inline-flex; }
    .tooltip-icon { display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:50%; border:1px solid #64748b; font-size:11px; cursor:help; color:#cbd5e1; }
    .tooltip-box { visibility:hidden; opacity:0; position:absolute; top:24px; left:0; width:320px; background:#0f172a; border:1px solid #334155; color:#e2e8f0; border-radius:10px; padding:10px 12px; font-size:0.85rem; line-height:1.4; z-index:9999; transition:opacity 0.15s ease; box-shadow:0 10px 24px rgba(0,0,0,0.35); }
    .tooltip-wrap:hover .tooltip-box { visibility:visible; opacity:1; }
    .side-note { background:#13233d; border:1px solid #334155; border-radius:14px; padding:14px; color:#dbe5f3; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div style="color:#fbbf24;font-size:0.85rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">ADAPT Strategy Terminal</div>
      <h1 style="margin:8px 0 6px 0;">Restored Interactive Dashboard</h1>
      <div style="color:#94a3b8;font-size:1.05rem;">Restored dashboard panels and sidebar date visibility while keeping the working allocator_state logic for COMBINED.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle = get_strategy_bundle()
ranges = get_strategy_ranges(bundle)

with st.sidebar:
    hover_mode = st.radio("Hover Display", ["Performance", "Holdings %"], index=0)
    strategy = st.selectbox("Strategy", ["COMBINED", "CORE", "ALPHA"], index=0)
    start_capital = st.number_input("Starting Capital", min_value=1000.0, value=DEFAULT_START_CAPITAL, step=1000.0)
    contribution_amount = st.number_input("Contribution Amount", min_value=0.0, value=0.0, step=100.0)
    contribution_frequency = st.selectbox("Contribution Frequency", ["None", "Monthly", "Yearly"], index=0)
    sort_order = st.selectbox("Allocation Trade Log Order", ["Newest to Oldest", "Oldest to Newest"], index=0)
    benchmarks = st.multiselect("Benchmarks", ["SPY", "QQQ", "TQQQ", "NDX"], default=["SPY", "QQQ", "TQQQ"])

    bench_symbols = ["^NDX" if x == "NDX" else x for x in benchmarks]
    px = get_price_history(bench_symbols, start="1990-01-01")
    valid_dates = first_valid_dates(px)
    strategy_min = ranges[strategy]
    benchmark_min_dates = [valid_dates[s] for s in bench_symbols if s in valid_dates]
    min_allowed = max([strategy_min] + benchmark_min_dates) if benchmark_min_dates else strategy_min
    default_start = max(pd.Timestamp("2022-01-01"), min_allowed)
    today = pd.Timestamp.today().normalize()

    with st.expander("First valid dates"):
        st.write("Strategy earliest rows:")
        st.json({k: v.strftime("%Y-%m-%d") for k, v in ranges.items()})
        st.write("Benchmark earliest rows:")
        st.json({("NDX" if k == "^NDX" else k): v.strftime("%Y-%m-%d") for k, v in valid_dates.items()})

    start_date = st.date_input("Start Date", value=default_start.date(), min_value=min_allowed.date(), max_value=today.date())
    end_date = st.date_input("End Date", value=today.date(), min_value=min_allowed.date(), max_value=today.date())

    st.markdown(
        """
        <div class="side-note">
        Official dashboard note: top metric cards and benchmark comparison are driven from the ADAPT return stream. Final value and equity curves include scheduled contributions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    run = st.button("Run Backtest", type="primary", width="stretch")

if run:
    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    df = bundle[strategy].copy().loc[str(start_date):str(end_date)]
    if df.empty:
        st.error("No data for that date range.")
        st.stop()

    bench_symbols = ["^NDX" if x == "NDX" else x for x in benchmarks]
    bench_px = get_price_history(bench_symbols, start=str(start_date))
    bench_px = bench_px.loc[str(start_date):str(end_date)].copy()
    if not bench_px.empty:
        bench_px = bench_px.rename(columns={"^NDX": "NDX"})
        bench_ret = bench_px.pct_change().reindex(df.index).dropna(how="all")
    else:
        bench_ret = pd.DataFrame(index=df.index)

    counts, state_title, state_tooltip = get_state_distribution(strategy, df)
    fig_state = build_state_fig(counts)
    fig_eq = build_equity_fig(df, bench_ret, strategy, start_capital, contribution_amount, contribution_frequency)
    rolling_fig, rolling_3y, rolling_2y = build_rolling_fig(df["ret"])

    perf = perf_from_returns(df["ret"], start_capital)
    rebalance_events = count_rebalance_events(strategy, df)
    alloc_log = allocation_change_log(strategy, df, newest_first=(sort_order == "Newest to Oldest"))
    alloc_now = current_allocation_table(strategy, df)
    alloc_profile = state_allocation_profile(strategy, df)
    state_now = current_state_label(strategy, df)

    eq_series, dd_series, total_contrib = equity_with_contributions(df["ret"], start_capital, contribution_amount, contribution_frequency)
    median_rec, avg_rec, max_rec = recovery_stats(dd_series)
    ulcer = ulcer_index(dd_series)
    roll3_best = float(rolling_3y.max()) if not rolling_3y.empty else 0.0
    roll3_worst = float(rolling_3y.min()) if not rolling_3y.empty else 0.0
    roll3_med = float(rolling_3y.median()) if not rolling_3y.empty else 0.0
    roll2_med = float(rolling_2y.median()) if not rolling_2y.empty else 0.0
    prob_neg = float((rolling_2y < 0).mean()) if not rolling_2y.empty else 0.0
    prob_lt5 = float((rolling_2y < 0.05).mean()) if not rolling_2y.empty else 0.0

    st.markdown("## COMBINED Overview" if strategy == "COMBINED" else f"## {strategy} Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metricbox"><div class="metriclabel">CAGR</div><div class="metricvalue">{perf["cagr"]:.2%}</div><div class="metricsub">{perf["days"]:,} trading days<br>Calculated from df["ret"]</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metricbox"><div class="metriclabel">Max Drawdown</div><div class="metricvalue">{perf["maxdd"]:.2%}</div><div class="metricsub">Ulcer Index {ulcer:.2%}<br>Recovery metrics below</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metricbox"><div class="metriclabel">Sharpe</div><div class="metricvalue">{perf["sharpe"]:.2f}</div><div class="metricsub">Calmar {perf["calmar"]:.2f}<br>Return-stream metric</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metricbox"><div class="metriclabel">Final Value</div><div class="metricvalue">${eq_series.iloc[-1]:,.0f}</div><div class="metricsub">Contributions ${total_contrib:,.0f}<br>Equity includes contribution schedule</div></div>', unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    c5.markdown(f'<div class="metricbox"><div class="metriclabel">Median Recovery</div><div class="metricvalue">{0 if median_rec is None else median_rec:.0f} d</div><div class="metricsub">Average {0 if avg_rec is None else avg_rec:.1f} d<br>Longest {0 if max_rec is None else max_rec:.0f} d</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="metricbox"><div class="metriclabel">Rolling 3Y CAGR</div><div class="metricvalue">{roll3_med:.2%}</div><div class="metricsub">Best {roll3_best:.2%}<br>Worst {roll3_worst:.2%}</div></div>', unsafe_allow_html=True)
    c7.markdown(f'<div class="metricbox"><div class="metriclabel">2Y Underperformance</div><div class="metricvalue">{prob_neg:.2%}</div><div class="metricsub">Prob(2Y CAGR &lt; 0%)<br>Median 2Y CAGR {roll2_med:.2%}</div></div>', unsafe_allow_html=True)
    c8.markdown(f'<div class="metricbox"><div class="metriclabel">Sub-Target Risk</div><div class="metricvalue">{prob_lt5:.2%}</div><div class="metricsub">Prob(&lt;5%) over rolling 2Y periods<br>Rebalance events {rebalance_events:,}</div></div>', unsafe_allow_html=True)

    st.plotly_chart(fig_eq, width="stretch")

    row2_left, row2_right = st.columns([1.15, 0.85])
    with row2_left:
        st.markdown(tooltip_label("Rolling CAGR Stability", "Rolling annualized returns over 2-year and 3-year windows."), unsafe_allow_html=True)
        st.plotly_chart(rolling_fig, width="stretch")
    with row2_right:
        st.markdown(tooltip_label(state_title, state_tooltip), unsafe_allow_html=True)
        st.plotly_chart(fig_state, width="stretch")

    row3_left, row3_right = st.columns([1, 1])
    with row3_left:
        alloc_tooltip = (
            "This table shows the current live allocation for the selected strategy and end date.<br><br>"
            "<strong>Risk On</strong>: more aggressive exposure.<br>"
            "<strong>Neutral</strong>: balanced intermediate stance.<br>"
            "<strong>Risk Off</strong>: preservation-oriented stance."
        )
        st.markdown(tooltip_label("Current Allocation", alloc_tooltip, f"Current state: {state_now}"), unsafe_allow_html=True)
        st.dataframe(alloc_now, width="stretch", hide_index=True)
        st.markdown("#### State Allocation Profile")
        st.dataframe(alloc_profile, width="stretch", hide_index=True)
    with row3_right:
        st.markdown("#### Allocation Change Log")
        st.dataframe(alloc_log.head(250), width="stretch", hide_index=True)

    yearly_df = yearly_return_table(strategy, df["ret"], bench_ret)
    row4_left, row4_right = st.columns([1, 1])
    with row4_left:
        st.markdown("#### Yearly Return Heatmap")
        st.plotly_chart(build_yearly_heatmap(yearly_df), width="stretch")
    with row4_right:
        st.markdown(f"#### {strategy} vs Benchmarks")
        st.dataframe(
            benchmark_table(strategy, df["ret"], bench_ret, start_capital, contribution_amount, contribution_frequency),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Yearly Returns (%)")
    st.dataframe(yearly_df, width="stretch")
else:
    st.info("Run Backtest to load the restored dashboard.")
