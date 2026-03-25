
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
def get_strategy_bundle() -> dict[str, pd.DataFrame]:
    settings = load_settings()
    core_df, _ = run_core_backtest(settings)
    alpha_df, _ = run_alpha_backtest(settings)
    combined_df, _ = run_combined_dynamic(settings)
    return {"CORE": core_df, "ALPHA": alpha_df, "COMBINED": combined_df}


@st.cache_data(show_spinner=False)
def get_price_history(symbols: list[str], start: str) -> pd.DataFrame:
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


def pretty_symbol(symbol: str) -> str:
    return "NDX" if symbol == "^NDX" else symbol


def perf_from_returns(returns: pd.Series, start_value: float) -> dict[str, float]:
    returns = returns.dropna().astype(float)
    if returns.empty:
        return {"cagr": 0.0, "maxdd": 0.0, "sharpe": 0.0, "final_value": start_value, "calmar": 0.0}
    equity = (1.0 + returns).cumprod() * start_value
    years = len(returns) / TRADING_DAYS
    cagr = (equity.iloc[-1] / start_value) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    dd = equity / equity.cummax() - 1.0
    maxdd = float(dd.min())
    std = returns.std()
    sharpe = float((returns.mean() / std) * math.sqrt(TRADING_DAYS)) if std and std > 0 else 0.0
    calmar = cagr / abs(maxdd) if maxdd < 0 else 0.0
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "final_value": float(equity.iloc[-1]), "calmar": calmar}


def ulcer_index(drawdown: pd.Series) -> float:
    if drawdown.empty:
        return 0.0
    return float((drawdown.pow(2).mean()) ** 0.5)


def recovery_stats(drawdown: pd.Series) -> tuple[float, float, float]:
    if drawdown.empty:
        return 0.0, 0.0, 0.0
    underwater = drawdown < 0
    lengths = []
    cur = 0
    for flag in underwater:
        if flag:
            cur += 1
        elif cur > 0:
            lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    if not lengths:
        return 0.0, 0.0, 0.0
    s = pd.Series(lengths, dtype=float)
    return float(s.median()), float(s.mean()), float(s.max())


def rolling_cagr_series(returns: pd.Series, window_days: int) -> pd.Series:
    returns = returns.dropna().astype(float)
    if len(returns) < window_days:
        return pd.Series(dtype=float)
    growth = (1.0 + returns).cumprod()
    total = growth / growth.shift(window_days) - 1.0
    years = window_days / TRADING_DAYS
    return ((1.0 + total).pow(1.0 / years) - 1.0).dropna()


def core_alloc_from_regime(regime: int) -> dict[str, float]:
    if regime == 1:
        return {"TQQQ": 1.0}
    if regime == 2:
        return {"SQQQ": 1.0}
    if regime == 3:
        return {"TQQQ": 0.60, "LQD": 0.10, "IAU": 0.10, "USMV": 0.10, "UUP": 0.10}
    return {"TLT": 0.10, "BIL": 0.40, "BTAL": 0.20, "USMV": 0.20, "UUP": 0.10}


def alpha_alloc_from_weight(weight: float) -> dict[str, float]:
    return {"TQQQ": 1.0} if float(weight) > 0 else {"CASH": 1.0}


def combine_allocs(core_alloc: dict[str, float], alpha_alloc: dict[str, float], core_w: float, alpha_w: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in core_alloc.items():
        out[k] = out.get(k, 0.0) + v * float(core_w)
    for k, v in alpha_alloc.items():
        out[k] = out.get(k, 0.0) + v * float(alpha_w)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else {}


def build_allocation_frame(strategy: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dt, row in df.iterrows():
        if strategy == "CORE":
            alloc = core_alloc_from_regime(int(row["signal_regime"]))
        elif strategy == "ALPHA":
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
    return pd.DataFrame(rows).fillna(0.0).set_index("date").sort_index() if rows else pd.DataFrame()


def current_allocation_table(strategy: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy, df)
    if alloc_df.empty:
        return pd.DataFrame(columns=["Asset", "Weight"])
    latest = alloc_df.iloc[-1]
    latest = latest[latest > 0].sort_values(ascending=False)
    out = latest.rename_axis("Asset").reset_index(name="Weight")
    out["Weight"] = out["Weight"].map(lambda x: f"{x:.1%}")
    return out


def allocation_change_log(strategy: str, df: pd.DataFrame, newest_first: bool = True) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy, df)
    if alloc_df.empty:
        return pd.DataFrame()
    rounded = alloc_df.round(6)
    changed = rounded.ne(rounded.shift(1)).any(axis=1)
    changes = rounded.loc[changed].copy()
    if len(changes) > 0:
        changes = changes.iloc[1:]  # exclude initial setup row
    changes = changes.loc[:, (changes != 0).any(axis=0)]
    changes = changes.sort_index(ascending=not newest_first)
    if changes.empty:
        return pd.DataFrame(columns=["Date"])
    changes.insert(0, "Date", changes.index.strftime("%Y-%m-%d"))
    changes = changes.reset_index(drop=True)
    for c in changes.columns[1:]:
        changes[c] = changes[c].map(lambda x: "" if abs(float(x)) < 1e-12 else f"{float(x):.1%}")
    return changes


def count_rebalance_events(strategy: str, df: pd.DataFrame) -> int:
    alloc_df = build_allocation_frame(strategy, df)
    if alloc_df.empty:
        return 0
    changed = alloc_df.round(6).ne(alloc_df.round(6).shift(1)).any(axis=1)
    return int(changed.iloc[1:].sum()) if len(changed) > 1 else 0


def get_state_distribution(strategy: str, df: pd.DataFrame):
    if strategy == "CORE":
        labels = {1: "R1 Risk-On", 2: "R2 Bear", 3: "R3 Growth", 4: "R4 Defensive"}
        counts = df["signal_regime"].astype(int).map(labels).value_counts() if "signal_regime" in df.columns else pd.Series(dtype=float)
        title = "CORE Regime Distribution"
    elif strategy == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        state = np.where(df[weight_col].fillna(0.0) > 0, "ALPHA Risk-On", "ALPHA Cash") if weight_col else []
        counts = pd.Series(state, index=df.index).value_counts() if weight_col else pd.Series(dtype=float)
        title = "ALPHA State Distribution"
    else:
        label_map = {"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}
        counts = df["allocator_state"].astype(str).map(label_map).value_counts() if "allocator_state" in df.columns else pd.Series(dtype=float)
        title = "COMBINED Allocator State Distribution"
    return counts, title


def build_state_fig(counts: pd.Series) -> go.Figure:
    if counts.empty:
        return go.Figure()
    colors = {
        "R1 Risk-On": "#34d399",
        "R2 Bear": "#f87171",
        "R3 Growth": "#60a5fa",
        "R4 Defensive": "#fbbf24",
        "ALPHA Risk-On": "#34d399",
        "ALPHA Cash": "#94a3b8",
        "Risk On": "#34d399",
        "Neutral": "#60a5fa",
        "Risk Off": "#fbbf24",
    }
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(counts.index),
                values=list(counts.values),
                hole=0.62,
                marker=dict(colors=[colors.get(x, "#60a5fa") for x in counts.index]),
                textinfo="percent",
                sort=False,
                hovertemplate="%{label}<br>Days: %{value}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(height=340, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def state_timeline_frame(strategy: str, df: pd.DataFrame):
    if strategy == "CORE" and "signal_regime" in df.columns:
        return df["signal_regime"].astype(int).map({1: "R1", 2: "R2", 3: "R3", 4: "R4"}), "CORE Regime Timeline"
    if strategy == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if weight_col:
            return pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "Risk-On", "Cash"), index=df.index), "ALPHA State Timeline"
    if strategy == "COMBINED" and "allocator_state" in df.columns:
        return df["allocator_state"].astype(str).map({"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}), "COMBINED Allocator State Timeline"
    return pd.Series(dtype=str), "State Timeline"


def build_state_timeline_fig(strategy: str, df: pd.DataFrame) -> go.Figure:
    state_series, title = state_timeline_frame(strategy, df)
    if state_series.empty:
        return go.Figure()
    cmap = {"R1": "#34d399", "R2": "#f87171", "R3": "#60a5fa", "R4": "#fbbf24", "Risk On": "#34d399", "Neutral": "#60a5fa", "Risk Off": "#fbbf24", "Cash": "#94a3b8"}
    states = list(pd.unique(state_series))
    y_map = {s: i for i, s in enumerate(states)}
    fig = go.Figure()
    for state in states:
        mask = state_series == state
        fig.add_trace(go.Scatter(x=state_series.index[mask], y=[y_map[state]] * int(mask.sum()), mode="markers", marker=dict(size=7, color=cmap.get(state, "#60a5fa"), symbol="square"), name=state, hovertemplate="Date: %{x|%Y-%m-%d}<br>State: " + state + "<extra></extra>"))
    fig.update_layout(title=title, height=230, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), yaxis=dict(tickmode="array", tickvals=list(y_map.values()), ticktext=list(y_map.keys()), title="State"), xaxis_title="Date", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def build_rebalance_yearly_fig(strategy: str, df: pd.DataFrame) -> go.Figure:
    alloc_df = build_allocation_frame(strategy, df)
    if alloc_df.empty:
        return go.Figure()
    changed = alloc_df.round(6).ne(alloc_df.round(6).shift(1)).any(axis=1).iloc[1:]
    yearly = changed.groupby(changed.index.year).sum()
    fig = go.Figure([go.Bar(x=yearly.index.astype(str), y=yearly.values, hovertemplate="Year: %{x}<br>Events: %{y}<extra></extra>")])
    fig.update_layout(title="Rebalance Events by Year", height=260, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), xaxis_title="Year", yaxis_title="Events")
    return fig


def build_transition_matrix(strategy: str, df: pd.DataFrame) -> go.Figure:
    states, _ = state_timeline_frame(strategy, df)
    if states.empty or len(states) < 2:
        return go.Figure()
    tm = pd.crosstab(states.shift(1).iloc[1:], states.iloc[1:], normalize="index") * 100.0
    fig = go.Figure([go.Heatmap(z=tm.values, x=tm.columns.tolist(), y=tm.index.tolist(), colorscale="Blues", text=np.round(tm.values, 1), texttemplate="%{text:.1f}%")])
    fig.update_layout(title="State Transition Matrix", height=260, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), xaxis_title="Next State", yaxis_title="Prior State")
    return fig


def build_allocation_timeline_fig(strategy: str, df: pd.DataFrame) -> go.Figure:
    alloc_df = build_allocation_frame(strategy, df)
    if alloc_df.empty:
        return go.Figure()
    alloc_df = alloc_df.loc[:, (alloc_df.sum(axis=0) > 0)]
    top_cols = alloc_df.mean().sort_values(ascending=False).head(8).index.tolist()
    alloc_df = alloc_df[top_cols]
    fig = go.Figure()
    for col in alloc_df.columns:
        fig.add_trace(go.Scatter(x=alloc_df.index, y=alloc_df[col], stackgroup="one", mode="lines", name=col, hovertemplate="Date: %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Allocation Timeline", height=300, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), xaxis_title="Date", yaxis_title="Weight", yaxis_tickformat=".0%", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def state_allocation_profile(strategy: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy, df)
    if alloc_df.empty:
        return pd.DataFrame()
    if strategy == "CORE" and "signal_regime" in df.columns:
        states = df["signal_regime"].astype(int).map({1: "R1", 2: "R2", 3: "R3", 4: "R4"})
    elif strategy == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if not weight_col:
            return pd.DataFrame()
        states = pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "Risk-On", "Cash"), index=df.index)
    else:
        states = df["allocator_state"].astype(str).map({"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}) if "allocator_state" in df.columns else pd.Series(dtype=str)
    merged = alloc_df.copy()
    merged["State"] = states.reindex(alloc_df.index)
    profile = merged.groupby("State").mean(numeric_only=True)
    if profile.empty:
        return pd.DataFrame()
    profile = profile.loc[:, (profile.sum(axis=0) > 0)]
    out = profile.round(3).reset_index()
    for c in out.columns[1:]:
        out[c] = out[c].map(lambda x: f"{x:.1%}")
    return out


def build_equity_drawdown_fig(strategy: str, df: pd.DataFrame, bench_ret: pd.DataFrame, start_value: float, contribution_amount: float, contribution_frequency: str, hover_mode: str):
    equity, dd, total_contributions = curve_from_returns_with_contributions(df["ret"], start_value, contribution_amount, contribution_frequency)
    equity_s = pd.Series(equity, index=df.index, dtype=float)
    dd_s = pd.Series(dd, index=df.index, dtype=float)
    holdings_text = build_holdings_text(strategy, df)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.07, subplot_titles=(f"{strategy} Equity vs Benchmarks", "Drawdown"))
    hover_text = [f"Date: {dt:%Y-%m-%d}<br>{strategy} Equity: ${eq:,.0f}" if hover_mode == "Performance" else f"Date: {dt:%Y-%m-%d}<br>{strategy} Equity: ${eq:,.0f}<br><br>{hold}" for dt, eq, hold in zip(df.index, equity_s, holdings_text)]
    fig.add_trace(go.Scatter(x=df.index, y=equity_s, mode="lines", name=strategy, text=hover_text, hovertemplate="%{text}<extra></extra>", line=dict(width=3.0, color="#fbbf24")), row=1, col=1)
    for col in bench_ret.columns:
        bench_eq, _, _ = curve_from_returns_with_contributions(bench_ret[col], start_value, contribution_amount, contribution_frequency)
        fig.add_trace(go.Scatter(x=bench_ret.index, y=bench_eq, mode="lines", name=col, hovertemplate=f"{col}: %{{y:,.0f}}<extra></extra>", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=dd_s.index, y=dd_s, mode="lines", fill="tozeroy", line=dict(color="#f87171", width=1.8), hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>", showlegend=False), row=2, col=1)
    fig.update_layout(height=760, hovermode="x unified", template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=78, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig, total_contributions, equity_s, dd_s


st.markdown("""
<div class="hero">
  <div class="hero-top">
    <div>
      <span class="eyebrow">ADAPT Strategy Terminal</span>
      <h2>Institutional Analytics for the ADAPT CORE + ALPHA Allocation Engine</h2>
      <div class="hero-sub">
        Multi-panel research terminal for performance, allocator states, turnover, transitions, and allocation behavior.
        Display layer only — preserves the validated ADAPT engine workflow and return stream calculations.
      </div>
    </div>
    <div class="hero-pill-wrap">
      <div class="hero-pill">Sidebar toggle: upper-left arrow</div>
      <div class="hero-pill">End-of-day signals</div>
      <div class="hero-pill">Market-on-Close orders</div>
      <div class="hero-pill">Cash yield: IBKR idle cash / SGOV / BIL / T-bills</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
