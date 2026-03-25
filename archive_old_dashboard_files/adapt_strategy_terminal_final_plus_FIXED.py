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
    return {"CORE": core_df, "ALPHA": alpha_df, "COMBINED": combined_df}


@st.cache_data(show_spinner=False)
def get_strategy_ranges() -> dict[str, pd.Timestamp]:
    data_map = get_strategy_bundle()
    return {name: pd.to_datetime(df.index.min()).normalize() for name, df in data_map.items()}


def first_valid_dates(px: pd.DataFrame) -> dict[str, pd.Timestamp]:
    out = {}
    if px.empty:
        return out
    for col in px.columns:
        s = px[col].dropna()
        if not s.empty:
            out[str(col)] = pd.to_datetime(s.index[0]).normalize()
    return out


def perf_from_returns(returns: pd.Series, start_value: float) -> dict:
    returns = returns.dropna().copy()
    if len(returns) == 0:
        return {"cagr": 0.0, "maxdd": 0.0, "sharpe": 0.0, "final_value": start_value, "days": 0, "calmar": 0.0, "win_rate": 0.0}
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
    return {"cagr": cagr, "maxdd": maxdd, "sharpe": sharpe, "final_value": pv, "days": len(returns), "calmar": calmar, "win_rate": win_rate}


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


def curve_from_returns_with_contributions(returns: pd.Series, start_value: float, contribution_amount: float = 0.0, contribution_frequency: str = "None"):
    returns = returns.fillna(0.0)
    idx = pd.to_datetime(returns.index)
    pv = start_value
    peak = start_value
    equity, dd = [], []
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


def build_holdings_text(strategy_name: str, df: pd.DataFrame) -> list[str]:
    def alloc_to_text(d: dict) -> str:
        if not d:
            return "No holdings"
        items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
        return "<br>".join(f"{k}: {v:.1%}" for k, v in items)

    texts = []
    if strategy_name == "CORE":
        for _, row in df.iterrows():
            texts.append(alloc_to_text(core_alloc_from_regime(int(row["signal_regime"]))))
    elif strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else "weight"
        for _, row in df.iterrows():
            texts.append(alloc_to_text(alpha_alloc_from_weight(float(row[weight_col]))))
    elif strategy_name == "COMBINED":
        for _, row in df.iterrows():
            ca = core_alloc_from_regime(int(row["core_regime"]))
            aa = alpha_alloc_from_weight(float(row["alpha_weight"]))
            texts.append(alloc_to_text(combine_allocs(ca, aa, float(row["core_w"]), float(row["alpha_w"]))))
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
            ca = core_alloc_from_regime(int(row["core_regime"]))
            aa = alpha_alloc_from_weight(float(row["alpha_weight"]))
            alloc = combine_allocs(ca, aa, float(row["core_w"]), float(row["alpha_w"]))
        else:
            alloc = {}
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
    rounded = alloc_df.round(6)
    changed_days = rounded.ne(rounded.shift(1)).any(axis=1)
    changes = rounded.loc[changed_days].copy()
    if changes.empty:
        return pd.DataFrame()
    changes = changes.loc[:, (changes != 0).any(axis=0)]
    changes = changes.sort_index(ascending=not newest_first)
    changes.insert(0, "Date", changes.index.strftime("%Y-%m-%d"))
    changes = changes.reset_index(drop=True)
    for col in [c for c in changes.columns if c != "Date"]:
        changes[col] = changes[col].map(lambda x: "" if pd.isna(x) or abs(float(x)) < 1e-12 else f"{float(x):.1%}")
    return changes


def count_rebalance_events(strategy_name: str, df: pd.DataFrame) -> int:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return 0
    rounded = alloc_df.round(6)
    changed_days = rounded.ne(rounded.shift(1)).any(axis=1)
    return int(changed_days.iloc[1:].sum()) if len(changed_days) > 1 else 0


def get_state_distribution(strategy_name: str, df: pd.DataFrame):
    if strategy_name == "CORE":
        if "signal_regime" not in df.columns:
            return pd.Series(dtype=float), "CORE Regime Distribution", "Shows the share of days spent in each CORE regime state."
        labels = {1: "R1 Risk-On", 2: "R2 Bear", 3: "R3 Growth", 4: "R4 Defensive"}
        counts = df["signal_regime"].astype(int).map(labels).value_counts()
        tooltip = "Shows the share of days spent in each CORE regime state.<br><br><strong>R1 Risk-On</strong>: higher equity exposure in favorable conditions.<br><strong>R2 Bear</strong>: inverse/risk-off bear regime.<br><strong>R3 Growth</strong>: moderate growth-oriented exposure in trending conditions.<br><strong>R4 Defensive</strong>: preservation-oriented allocation in risk-off conditions."
        return counts, "CORE Regime Distribution", tooltip
    if strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if weight_col is None:
            return pd.Series(dtype=float), "ALPHA State Distribution", "Shows the share of days spent in each ALPHA state."
        state = np.where(df[weight_col].fillna(0.0) > 0, "ALPHA Risk-On", "ALPHA Cash")
        counts = pd.Series(state, index=df.index).value_counts()
        tooltip = "Shows the share of days spent in each ALPHA state.<br><br><strong>ALPHA Risk-On</strong>: ALPHA sleeve allocated to active risk exposure.<br><strong>ALPHA Cash</strong>: ALPHA sleeve parked in cash / no active long exposure."
        return counts, "ALPHA State Distribution", tooltip
    if "allocator_state" in df.columns:
        label_map = {"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}
        counts = df["allocator_state"].astype(str).map(lambda x: label_map.get(x, x)).value_counts()
        tooltip = "Shows the share of days spent in each COMBINED allocator state.<br><br><strong>Risk On</strong>: allocator favoring more aggressive exposure.<br><strong>Neutral</strong>: balanced / intermediate allocator stance.<br><strong>Risk Off</strong>: allocator emphasizing capital preservation."
        return counts, "COMBINED Allocator State Distribution", tooltip
    return pd.Series(dtype=float), "State Distribution", "Shows the share of days spent in each system state."


def build_state_fig(counts: pd.Series) -> go.Figure:
    if counts.empty:
        return go.Figure()
    color_map = {
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
    labels = list(counts.index)
    values = list(counts.values)
    colors = [color_map.get(label, "#60a5fa") for label in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.66, marker=dict(colors=colors), textinfo="percent", textposition="inside", sort=False, hovertemplate="%{label}<br>Days: %{value}<br>%{percent}<extra></extra>")])
    fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=11)))
    return fig


def state_timeline_frame(strategy_name: str, df: pd.DataFrame):
    if strategy_name == "CORE" and "signal_regime" in df.columns:
        return df["signal_regime"].astype(int).map({1: "R1", 2: "R2", 3: "R3", 4: "R4"}), "CORE Regime Timeline"
    if strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if weight_col is not None:
            return pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "Risk-On", "Cash"), index=df.index), "ALPHA State Timeline"
    if strategy_name == "COMBINED" and "allocator_state" in df.columns:
        return df["allocator_state"].astype(str).map({"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"}), "COMBINED Allocator State Timeline"
    return pd.Series(dtype=str), "State Timeline"


def build_state_timeline_fig(strategy_name: str, df: pd.DataFrame) -> go.Figure:
    state_series, title = state_timeline_frame(strategy_name, df)
    if state_series.empty:
        return go.Figure()
    color_map = {"R1": "#34d399", "R2": "#f87171", "R3": "#60a5fa", "R4": "#fbbf24", "Risk On": "#34d399", "Neutral": "#60a5fa", "Risk Off": "#fbbf24", "Cash": "#94a3b8"}
    states = list(pd.unique(state_series))
    y_map = {s: i for i, s in enumerate(states)}
    y = state_series.map(y_map)
    fig = go.Figure()
    for state in states:
        mask = state_series == state
        fig.add_trace(go.Scatter(x=state_series.index[mask], y=y[mask], mode="markers", marker=dict(size=8, color=color_map.get(state, "#60a5fa"), symbol="square"), name=state, hovertemplate="Date: %{x|%Y-%m-%d}<br>State: " + state + "<extra></extra>"))
    fig.update_layout(title=title, height=240, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), yaxis=dict(tickmode="array", tickvals=list(y_map.values()), ticktext=list(y_map.keys()), title="State"), xaxis_title="Date", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)))
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def build_rebalance_yearly_fig(strategy_name: str, df: pd.DataFrame) -> go.Figure:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return go.Figure()
    rounded = alloc_df.round(6)
    changed_days = rounded.ne(rounded.shift(1)).any(axis=1).iloc[1:]
    yearly = changed_days.groupby(changed_days.index.year).sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly.index.astype(str), y=yearly.values, hovertemplate="Year: %{x}<br>Events: %{y}<extra></extra>"))
    fig.update_layout(title="Rebalance Events by Year", height=280, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), xaxis_title="Year", yaxis_title="Events", showlegend=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
    fig.update_xaxes(showgrid=False)
    return fig


def build_transition_matrix(strategy_name: str, df: pd.DataFrame) -> go.Figure:
    states, _ = state_timeline_frame(strategy_name, df)
    if states.empty or len(states) < 2:
        return go.Figure()
    current = states.iloc[1:].reset_index(drop=True)
    prev = states.shift(1).iloc[1:].reset_index(drop=True)
    tm = pd.crosstab(prev, current, normalize="index") * 100.0
    fig = go.Figure(data=go.Heatmap(z=tm.values, x=tm.columns.tolist(), y=tm.index.tolist(), colorscale="Blues", text=np.round(tm.values, 1), texttemplate="%{text:.1f}%", hovertemplate="From: %{y}<br>To: %{x}<br>Probability: %{z:.2f}%<extra></extra>"))
    fig.update_layout(title="State Transition Matrix", height=280, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), xaxis_title="Next State", yaxis_title="Prior State")
    return fig


def build_allocation_timeline_fig(strategy_name: str, df: pd.DataFrame) -> go.Figure:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return go.Figure()
    alloc_df = alloc_df.loc[:, (alloc_df.sum(axis=0) > 0)]
    top_cols = alloc_df.mean().sort_values(ascending=False).head(8).index.tolist()
    alloc_df = alloc_df[top_cols]
    colors = ["#fbbf24", "#34d399", "#60a5fa", "#c084fc", "#f87171", "#38bdf8", "#22c55e", "#94a3b8"]
    fig = go.Figure()
    for i, col in enumerate(alloc_df.columns):
        fig.add_trace(go.Scatter(x=alloc_df.index, y=alloc_df[col], stackgroup="one", mode="lines", line=dict(width=0.8, color=colors[i % len(colors)]), name=col, hovertemplate="Date: %{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1%}<extra></extra>"))
    fig.update_layout(title="Allocation Timeline", height=320, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=46, b=20), xaxis_title="Date", yaxis_title="Weight", yaxis_tickformat=".0%", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
    fig.update_xaxes(showgrid=False)
    return fig


def state_allocation_profile(strategy_name: str, df: pd.DataFrame) -> pd.DataFrame:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        return pd.DataFrame()
    if strategy_name == "CORE" and "signal_regime" in df.columns:
        states = df["signal_regime"].astype(int).map({1: "R1", 2: "R2", 3: "R3", 4: "R4"})
    elif strategy_name == "ALPHA":
        weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
        if weight_col is None:
            return pd.DataFrame()
        states = pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "Risk-On", "Cash"), index=df.index)
    elif strategy_name == "COMBINED" and "allocator_state" in df.columns:
        states = df["allocator_state"].astype(str).map({"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"})
    else:
        return pd.DataFrame()
    merged = alloc_df.copy()
    merged["State"] = states.reindex(alloc_df.index)
    profile = merged.groupby("State").mean(numeric_only=True)
    if profile.empty:
        return pd.DataFrame()
    profile = profile.loc[:, (profile.sum(axis=0) > 0)]
    out = profile.round(3).reset_index()
    for col in [c for c in out.columns if c != "State"]:
        out[col] = out[col].map(lambda x: f"{x:.1%}")
    return out


def benchmark_table(strategy_name: str, strategy_ret: pd.Series, bench_ret: pd.DataFrame, start_value: float, contribution_amount: float, contribution_frequency: str) -> pd.DataFrame:
    rows = []
    s = perf_from_returns(strategy_ret, start_value)
    strategy_eq, _, strategy_contrib = curve_from_returns_with_contributions(strategy_ret, start_value, contribution_amount, contribution_frequency)
    rows.append({"Series": strategy_name, "CAGR": f"{s['cagr']:.2%}", "MaxDD": f"{s['maxdd']:.2%}", "Sharpe": round(s["sharpe"], 3), "Calmar": round(s["calmar"], 3), "Final Value": f"${strategy_eq[-1]:,.0f}" if strategy_eq else f"${start_value:,.0f}", "Total Contributions": f"${strategy_contrib:,.0f}"})
    for col in bench_ret.columns:
        m = perf_from_returns(bench_ret[col], start_value)
        b_eq, _, b_contrib = curve_from_returns_with_contributions(bench_ret[col], start_value, contribution_amount, contribution_frequency)
        rows.append({"Series": col, "CAGR": f"{m['cagr']:.2%}", "MaxDD": f"{m['maxdd']:.2%}", "Sharpe": round(m["sharpe"], 3), "Calmar": round(m["calmar"], 3), "Final Value": f"${b_eq[-1]:,.0f}" if b_eq else f"${start_value:,.0f}", "Total Contributions": f"${b_contrib:,.0f}"})
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


def build_yearly_heatmap(yr: pd.DataFrame) -> go.Figure:
    if yr.empty:
        return go.Figure()
    z = yr.T.values
    x = [str(i) for i in yr.index.tolist()]
    y = yr.columns.tolist()
    fig = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale=[[0.0, "#7f1d1d"], [0.45, "#ef4444"], [0.5, "#111827"], [0.55, "#22c55e"], [1.0, "#14532d"]], zmid=0, text=np.round(z, 2), texttemplate="%{text:.2f}%", textfont={"size": 11}, hovertemplate="Series: %{y}<br>Year: %{x}<br>Return: %{z:.2f}%<extra></extra>"))
    fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=24, b=20), xaxis_title="Year", yaxis_title="Series")
    return fig


def build_equity_drawdown_fig(strategy_name: str, df: pd.DataFrame, bench_ret: pd.DataFrame, start_value: float, contribution_amount: float, contribution_frequency: str, hover_mode: str = "Performance"):
    strategy_equity, strategy_dd, total_contributions = curve_from_returns_with_contributions(df["ret"], start_value, contribution_amount, contribution_frequency)
    strategy_equity_s = pd.Series(strategy_equity, index=df.index, dtype=float)
    strategy_dd_s = pd.Series(strategy_dd, index=df.index, dtype=float)
    holdings_text = build_holdings_text(strategy_name, df)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.07, subplot_titles=(f"{strategy_name} Equity vs Benchmarks", "Drawdown"))
    if hover_mode == "Holdings %":
        strategy_hover = [f"Date: {dt:%Y-%m-%d}<br>{strategy_name} Equity: ${eq:,.0f}<br><br>{hold}" for dt, eq, hold in zip(df.index, strategy_equity_s, holdings_text)]
    else:
        strategy_hover = [f"Date: {dt:%Y-%m-%d}<br>{strategy_name} Equity: ${eq:,.0f}" for dt, eq in zip(df.index, strategy_equity_s)]
    fig.add_trace(go.Scatter(x=df.index, y=strategy_equity_s, mode="lines", name=strategy_name, text=strategy_hover, hovertemplate="%{text}<extra></extra>", line=dict(width=3.2, color="#fbbf24")), row=1, col=1)
    if not bench_ret.empty:
        aligned_holdings = pd.Series(holdings_text, index=df.index).reindex(bench_ret.index).ffill()
        benchmark_colors = {"SPY": "#34d399", "QQQ": "#60a5fa", "TQQQ": "#c084fc", "NDX": "#f87171"}
        for col in bench_ret.columns:
            b_eq, _, _ = curve_from_returns_with_contributions(bench_ret[col], start_value, contribution_amount, contribution_frequency)
            if hover_mode == "Holdings %":
                bench_hover = [f"Date: {dt:%Y-%m-%d}<br>{col}: ${eq:,.0f}<br><br>{hold}" for dt, eq, hold in zip(bench_ret.index, b_eq, aligned_holdings.fillna("No holdings"))]
            else:
                bench_hover = [f"Date: {dt:%Y-%m-%d}<br>{col}: ${eq:,.0f}" for dt, eq in zip(bench_ret.index, b_eq)]
            fig.add_trace(go.Scatter(x=bench_ret.index, y=b_eq, mode="lines", name=col, text=bench_hover, hovertemplate="%{text}<extra></extra>", line=dict(width=1.55, color=benchmark_colors.get(col, "#94a3b8")), opacity=0.68), row=1, col=1)
    fig.add_trace(go.Scatter(x=strategy_dd_s.index, y=strategy_dd_s, mode="lines", fill="tozeroy", line=dict(color="#f87171", width=1.9), name="Drawdown", hovertemplate="Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>", showlegend=False), row=2, col=1)
    fig.update_layout(height=760, hovermode="x unified", template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=78, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12)))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig, total_contributions, strategy_equity_s, strategy_dd_s


def build_rolling_fig(returns: pd.Series):
    rolling_3y = rolling_cagr_series(returns, 756)
    rolling_2y = rolling_cagr_series(returns, 504)
    fig = go.Figure()
    if not rolling_3y.empty:
        fig.add_trace(go.Scatter(x=rolling_3y.index, y=rolling_3y, mode="lines", name="Rolling 3Y CAGR", line=dict(color="#34d399", width=2.25), hovertemplate="Date: %{x|%Y-%m-%d}<br>Rolling 3Y CAGR: %{y:.2%}<extra></extra>"))
    if not rolling_2y.empty:
        fig.add_trace(go.Scatter(x=rolling_2y.index, y=rolling_2y, mode="lines", name="Rolling 2Y CAGR", line=dict(color="#60a5fa", width=1.95, dash="dot"), hovertemplate="Date: %{x|%Y-%m-%d}<br>Rolling 2Y CAGR: %{y:.2%}<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.22)")
    fig.update_layout(height=360, template="plotly_dark", paper_bgcolor="#0e1627", plot_bgcolor="#0e1627", margin=dict(l=24, r=24, t=40, b=20), yaxis_tickformat=".0%", xaxis_title="Date", yaxis_title="CAGR", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.10)")
    return fig, rolling_3y, rolling_2y


st.set_page_config(page_title="ADAPT Strategy Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root { --border: rgba(148, 163, 184, 0.16); --border-strong: rgba(148, 163, 184, 0.24); --text: #e7edf7; --muted: #9fb0c7; --gold: #fbbf24; }
.stApp { background: radial-gradient(circle at top right, rgba(251,191,36,0.07), transparent 24%), radial-gradient(circle at left, rgba(96,165,250,0.06), transparent 20%), linear-gradient(180deg, #050d19 0%, #091223 100%); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif; }
.block-container { max-width: 1560px; padding-top: 2.6rem; padding-bottom: 1.4rem; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #09111f 0%, #0d1527 100%); border-right: 1px solid var(--border); }
[data-testid="collapsedControl"] { display: block !important; visibility: visible !important; opacity: 1 !important; z-index: 999999 !important; }
h2, h3 { letter-spacing: -0.03em; }
h2 { font-size: 2.8rem !important; font-weight: 820 !important; margin-bottom: 0.2rem !important; line-height: 1.05 !important; }
.hero { background: linear-gradient(135deg, rgba(16,28,50,0.98) 0%, rgba(10,19,36,0.98) 100%); border: 1px solid var(--border); border-radius: 22px; padding: 28px 24px 18px 24px; box-shadow: 0 20px 46px rgba(0,0,0,0.22); margin-bottom: 14px; overflow: visible; }
.hero-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.82rem; color: var(--gold); font-weight: 800; margin-bottom: 0.5rem; display: block; }
.hero-sub { color: var(--muted); font-size: 1.02rem; line-height: 1.55; max-width: 860px; }
.hero-pill-wrap { display: flex; flex-wrap: wrap; gap: 10px; justify-content: flex-end; max-width: 520px; }
.hero-pill { background: rgba(255,255,255,0.035); border: 1px solid var(--border); border-radius: 999px; padding: 8px 12px; color: #dce6f4; font-size: 0.86rem; font-weight: 650; white-space: nowrap; }
.floating-bar { position: sticky; top: 0.7rem; z-index: 999; background: linear-gradient(180deg, rgba(14,23,39,0.96) 0%, rgba(12,20,35,0.96) 100%); border: 1px solid var(--border-strong); border-radius: 16px; padding: 10px 14px; margin-bottom: 16px; backdrop-filter: blur(10px); box-shadow: 0 12px 28px rgba(0,0,0,0.18); font-size: 0.95rem; color: #d8e5f7; }
.section-card { background: linear-gradient(180deg, rgba(15,24,41,0.98) 0%, rgba(11,18,31,0.98) 100%); border: 1px solid var(--border); border-radius: 20px; padding: 12px 14px 10px 14px; box-shadow: 0 14px 34px rgba(0,0,0,0.16); margin-bottom: 12px; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }
.metric-shell { background: linear-gradient(180deg, rgba(17,27,46,0.96) 0%, rgba(13,21,37,0.98) 100%); border: 1px solid rgba(251,191,36,0.10); border-radius: 16px; padding: 12px 14px 11px 14px; min-height: 104px; box-shadow: 0 10px 24px rgba(0,0,0,0.14); }
.metric-label { text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.70rem; color: #94a7c0; font-weight: 800; margin-bottom: 8px; }
.metric-value { font-size: 1.55rem; line-height: 1.0; font-weight: 820; color: #f8fbff; margin-bottom: 8px; }
.metric-sub { font-size: 0.84rem; line-height: 1.34; color: #9fb0c7; }
.metric-sub strong { color: #e5ecf6; }
.mini-card { background: linear-gradient(180deg, rgba(15,24,41,0.98) 0%, rgba(11,18,31,0.98) 100%); border: 1px solid var(--border); border-radius: 18px; padding: 12px 14px; margin-bottom: 12px; box-shadow: 0 12px 24px rgba(0,0,0,0.14); }
.tooltip-label { display: inline-flex; align-items: center; gap: 8px; font-size: 1.15rem; font-weight: 760; letter-spacing: -0.02em; color: #edf3fb; margin-bottom: 8px; }
.tooltip-wrap { position: relative; display: inline-flex; align-items: center; cursor: help; }
.tooltip-icon { display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.22); color: #c9d6e8; font-size: 11px; font-weight: 800; background: rgba(255,255,255,0.04); }
.tooltip-text { visibility: hidden; opacity: 0; position: absolute; right: 24px; top: 24px; width: 320px; background: #0a1220; color: #e6eef9; border: 1px solid rgba(255,255,255,0.12); border-radius: 12px; padding: 11px 12px; box-shadow: 0 12px 28px rgba(0,0,0,0.28); font-size: 0.84rem; line-height: 1.45; z-index: 10000; transition: opacity 0.18s ease; }
.tooltip-wrap:hover .tooltip-text { visibility: visible; opacity: 1; }
.note-box { background: rgba(96,165,250,0.08); border: 1px solid rgba(96,165,250,0.18); color: #dce6f4; border-radius: 16px; padding: 12px 14px; margin-top: 6px; margin-bottom: 8px; font-size: 0.93rem; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
div.stButton > button { border-radius: 14px; font-weight: 760; border: 1px solid rgba(251,191,36,0.20); }
@media (max-width: 1200px) {
    .hero-top { flex-direction: column; }
    .hero-pill-wrap { justify-content: flex-start; max-width: unset; }
    .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .tooltip-text { right: 0; width: 280px; }
}
@media (max-width: 700px) { .metric-grid { grid-template-columns: 1fr; } }
</style>
""", unsafe_allow_html=True)

def section_label(title: str, tooltip: str) -> str:
    return '<div class="tooltip-label"><span>' + title + '</span><span class="tooltip-wrap"><span class="tooltip-icon">i</span><span class="tooltip-text">' + tooltip + '</span></span></div>'

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

strategy_ranges = get_strategy_ranges()

with st.sidebar:
    st.markdown("## ADAPT Controls")
    st.caption("Use the arrow in the upper-left corner to collapse or reopen this panel.")
    strategy_choice = st.selectbox("Strategy", ["COMBINED", "CORE", "ALPHA"], index=0)
    start_capital = st.number_input("Starting Capital", min_value=1000.0, value=DEFAULT_START_CAPITAL, step=1000.0)
    contribution_amount = st.number_input("Contribution Amount", min_value=0.0, value=0.0, step=100.0)
    contribution_frequency = st.selectbox("Contribution Frequency", ["None", "Monthly", "Yearly"], index=0)
    hover_mode = st.radio("Hover Display", ["Performance", "Holdings %"], index=0)
    change_log_sort = st.selectbox("Allocation Trade Log Order", ["Newest to Oldest", "Oldest to Newest"], index=0)
    available_benchmarks = ["SPY", "QQQ", "TQQQ", "NDX"]
    selected_benchmarks = st.multiselect("Benchmarks", available_benchmarks, default=["SPY", "QQQ", "TQQQ"])
    benchmark_download_symbols = ["^NDX" if x == "NDX" else x for x in selected_benchmarks]
    px = get_price_history(benchmark_download_symbols, start="1990-01-01")
    valid_dates = first_valid_dates(px)
    strategy_min_date = strategy_ranges[strategy_choice]
    benchmark_min_dates = [valid_dates[s] for s in benchmark_download_symbols if s in valid_dates]
    min_allowed_date = max([strategy_min_date] + benchmark_min_dates) if benchmark_min_dates else strategy_min_date
    with st.expander("First valid dates"):
        st.write("Strategy earliest rows:")
        st.json({k: v.strftime("%Y-%m-%d") for k, v in strategy_ranges.items()})
        st.write("Benchmark earliest rows:")
        st.json({pretty_symbol(k): v.strftime("%Y-%m-%d") for k, v in valid_dates.items()})
    default_start = max(pd.Timestamp("2022-01-01"), min_allowed_date)
    today = pd.Timestamp.today().normalize()
    start_date = st.date_input("Start Date", value=default_start.date(), min_value=min_allowed_date.date(), max_value=today.date())
    end_date = st.date_input("End Date", value=today.date(), min_value=min_allowed_date.date(), max_value=today.date())
    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()
    st.markdown('<div class="note-box">Top metrics and benchmark comparison are driven from the ADAPT return stream. Final value and equity curves include scheduled contributions.</div>', unsafe_allow_html=True)
    run = st.button("Run Backtest", type="primary", use_container_width=True)

if run:
    with st.spinner("Running ADAPT..."):
        data_map = get_strategy_bundle()
    raw_df = data_map[strategy_choice].copy()
    df = raw_df.loc[str(start_date):str(end_date)].copy()
    if df.empty:
        st.error(f"No strategy data available for the selected date range. Earliest usable date for {strategy_choice} is {strategy_min_date.date()}.")
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
    equity_fig, total_contributions, strategy_equity_s, strategy_dd_s = build_equity_drawdown_fig(strategy_choice, df, bench_ret, start_capital, contribution_amount, contribution_frequency, hover_mode)
    rolling_fig, rolling_3y, rolling_2y = build_rolling_fig(df["ret"])
    state_counts, state_title, state_tooltip = get_state_distribution(strategy_choice, df)
    state_fig = build_state_fig(state_counts)
    timeline_fig = build_state_timeline_fig(strategy_choice, df)
    rebalance_yearly_fig = build_rebalance_yearly_fig(strategy_choice, df)
    transition_fig = build_transition_matrix(strategy_choice, df)
    allocation_timeline_fig = build_allocation_timeline_fig(strategy_choice, df)
    heatmap_fig = build_yearly_heatmap(yearly_return_table(strategy_choice, df["ret"], bench_ret))
    current_alloc = current_allocation_table(strategy_choice, df)
    newest_first = change_log_sort == "Newest to Oldest"
    alloc_changes = allocation_change_log(strategy_choice, df, newest_first=newest_first)
    rebalance_events = count_rebalance_events(strategy_choice, df)
    state_alloc_profile = state_allocation_profile(strategy_choice, df)
    final_value = float(strategy_equity_s.iloc[-1]) if not strategy_equity_s.empty else start_capital
    ulcer = ulcer_index(strategy_dd_s)
    median_rec, avg_rec, max_rec = recovery_stats(strategy_dd_s)
    roll3_best = float(rolling_3y.max()) if not rolling_3y.empty else 0.0
    roll3_worst = float(rolling_3y.min()) if not rolling_3y.empty else 0.0
    roll3_median = float(rolling_3y.median()) if not rolling_3y.empty else 0.0
    med_2y = float(rolling_2y.median()) if not rolling_2y.empty else 0.0
    prob_neg = float((rolling_2y < 0).mean()) if not rolling_2y.empty else 0.0
    prob_lt5 = float((rolling_2y < 0.05).mean()) if not rolling_2y.empty else 0.0

    st.markdown(f'<div class="floating-bar"><strong>Backtest Information</strong> &nbsp;|&nbsp; {strategy_choice} &nbsp;|&nbsp; {start_date} → {end_date} &nbsp;|&nbsp; {len(df):,} trading days &nbsp;|&nbsp; Rebalance Events: <strong>{rebalance_events:,}</strong></div>', unsafe_allow_html=True)

    top_row_left, top_row_right = st.columns([1.22, 0.78])
    with top_row_left:
        st.markdown("### ADAPT Overview")
        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-shell"><div class="metric-label">CAGR</div><div class="metric-value">{perf['cagr']:.2%}</div><div class="metric-sub"><strong>{perf['days']}</strong> trading days<br>Calculated from df["ret"]</div></div>
          <div class="metric-shell"><div class="metric-label">Max Drawdown</div><div class="metric-value">{perf['maxdd']:.2%}</div><div class="metric-sub">Ulcer Index <strong>{ulcer:.2%}</strong><br>Recovery profile below</div></div>
          <div class="metric-shell"><div class="metric-label">Sharpe</div><div class="metric-value">{perf['sharpe']:.2f}</div><div class="metric-sub">Calmar <strong>{perf['calmar']:.2f}</strong><br>Win rate <strong>{perf['win_rate']:.1%}</strong></div></div>
          <div class="metric-shell"><div class="metric-label">Final Value</div><div class="metric-value">${final_value:,.0f}</div><div class="metric-sub">Contributions <strong>${total_contributions:,.0f}</strong><br>Equity includes contribution schedule</div></div>
          <div class="metric-shell"><div class="metric-label">Median Recovery</div><div class="metric-value">{0 if median_rec is None else median_rec:.0f} d</div><div class="metric-sub">Average <strong>{0 if avg_rec is None else avg_rec:.1f} d</strong><br>Longest <strong>{0 if max_rec is None else max_rec:.0f} d</strong></div></div>
          <div class="metric-shell"><div class="metric-label">Rolling 3Y CAGR</div><div class="metric-value">{roll3_median:.2%}</div><div class="metric-sub">Best <strong>{roll3_best:.2%}</strong><br>Worst <strong>{roll3_worst:.2%}</strong></div></div>
          <div class="metric-shell"><div class="metric-label">2Y Underperformance</div><div class="metric-value">{prob_neg:.2%}</div><div class="metric-sub">Prob(2Y CAGR &lt; 0%)<br>Median 2Y CAGR <strong>{med_2y:.2%}</strong></div></div>
          <div class="metric-shell"><div class="metric-label">Rebalance Events</div><div class="metric-value">{rebalance_events:,}</div><div class="metric-sub">1 event = 1 allocation change day<br>See trade log below</div></div>
        </div>
        """, unsafe_allow_html=True)
    with top_row_right:
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown("#### Current Allocation")
        st.dataframe(current_alloc, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        exec_tooltip = "Signals are generated using final closing data.<br><br>Trades are assumed to be executed via the closing auction using <strong>Market-on-Close orders</strong>, not at the next day’s open."
        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        st.markdown(section_label("Execution model: End-of-day signals, Market-on-Close orders", exec_tooltip), unsafe_allow_html=True)
        st.caption("Scalable end-of-day workflow with liquid ETF execution.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.plotly_chart(equity_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    row2_left, row2_right = st.columns([1.12, 0.88])
    with row2_left:
        rolling_tooltip = "Rolling compound annual growth rate over moving time windows.<br><br><strong>3Y CAGR</strong>: annualized return over rolling 3-year periods.<br><strong>2Y CAGR</strong>: annualized return over rolling 2-year periods.<br><br>This shows the stability and variability of performance through time."
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(section_label("Rolling CAGR Stability", rolling_tooltip), unsafe_allow_html=True)
        st.plotly_chart(rolling_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row2_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(section_label(state_title, state_tooltip), unsafe_allow_html=True)
        st.plotly_chart(state_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row3_left, row3_right = st.columns([1.0, 1.0])
    with row3_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row3_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(rebalance_yearly_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row4_left, row4_right = st.columns([1.0, 1.0])
    with row4_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(transition_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row4_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(allocation_timeline_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row5_left, row5_right = st.columns([0.92, 1.08])
    with row5_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Current Allocation")
        st.dataframe(current_alloc, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### State Allocation Profile")
        st.dataframe(state_alloc_profile, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row5_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Allocation Trade Log")
        st.dataframe(alloc_changes, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row6_left, row6_right = st.columns([1.0, 1.0])
    with row6_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Yearly Return Heatmap")
        st.plotly_chart(heatmap_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row6_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(f"#### {strategy_choice} vs Benchmarks")
        st.dataframe(benchmark_table(strategy_choice, df["ret"], bench_ret, start_capital, contribution_amount, contribution_frequency), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### Yearly Returns (%)")
    st.dataframe(yearly_return_table(strategy_choice, df["ret"], bench_ret), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Choose your settings in the sidebar, then click Run Backtest.")
