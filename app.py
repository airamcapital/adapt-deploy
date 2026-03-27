from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.markdown("""
<style>
div.stDownloadButton > button {
    background-color: #1a1a2e;
    color: #4da6ff;
    border: 1px solid #4da6ff;
}
div.stDownloadButton > button:hover {
    background-color: #4da6ff;
    color: #ffffff;
    border: 1px solid #4da6ff;
}
div.stButton > button {
    background-color: #1a1a2e;
    color: #4da6ff;
    border: 1px solid #4da6ff;
}
div.stButton > button:hover {
    background-color: #4da6ff;
    color: #ffffff;
    border: 1px solid #4da6ff;
}
</style>
""", unsafe_allow_html=True)
import yfinance as yf
from plotly.subplots import make_subplots

from adapt.alpha.alpha_engine import run_alpha_backtest
from adapt.allocator.combined_dynamic import run_combined_dynamic
from adapt.core.core_engine import run_core_backtest
from adapt.data_loader import load_settings

DEFAULT_START_CAPITAL = 50000.0
TRADING_DAYS = 252


def tooltip_label(title: str, tooltip: str, subtitle: str | None = None, align: str = "left") -> str:
    subtitle_html = f'<span class="label-sub">{subtitle}</span>' if subtitle else ""
    box_class = "tooltip-box tooltip-right" if align == "right" else "tooltip-box tooltip-left"
    return f"""
    <div class="label-wrap">
      <div class="label-row">
        <span class="label-title">{title}</span>
        <span class="tooltip-wrap">
          <span class="tooltip-icon">i</span>
          <span class="{box_class}">{tooltip}</span>
        </span>
      </div>
      {subtitle_html}
    </div>
    """



def metric_card_html(title: str, value: str, lines: list[str], tooltip: str | None = None, align: str = "left") -> str:
    tooltip_html = ""
    if tooltip:
        box_class = "tooltip-box tooltip-right" if align == "right" else "tooltip-box tooltip-left"
        tooltip_html = (
            '<span class="metric-tooltip-wrap">'
            '<span class="metric-tooltip-icon">i</span>'
            f'<span class="{box_class}">{tooltip}</span>'
            '</span>'
        )
    lines_html = "<br>".join(lines)
    return f"""
    <div class="metricbox">
      <div class="metrichead">
        <div class="metriclabel">{title}</div>
        {tooltip_html}
      </div>
      <div class="metricvalue">{value}</div>
      <div class="metricsub">{lines_html}</div>
    </div>
    """



def dark_dataframe(df: pd.DataFrame, hide_index: bool = True):
    styler = (
        df.style
        .set_table_styles([
            {"selector": "thead th", "props": [("background-color", "#111827"), ("color", "#e5e7eb"), ("border", "1px solid #334155")]},
            {"selector": "tbody td", "props": [("background-color", "#0b1220"), ("color", "#e5e7eb"), ("border", "1px solid #243041")]},
            {"selector": "tbody th", "props": [("background-color", "#0b1220"), ("color", "#e5e7eb"), ("border", "1px solid #243041")]},
            {"selector": "table", "props": [("background-color", "#0b1220"), ("color", "#e5e7eb")]},
        ])
    )
    if hide_index:
        try:
            styler = styler.hide(axis="index")
        except Exception:
            pass
    return styler


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


def build_holdings_text(strategy_name: str, df: pd.DataFrame) -> list[str]:
    alloc_df = build_allocation_frame(strategy_name, df)
    texts = []
    for _, row in alloc_df.iterrows():
        active = row[row > 0].sort_values(ascending=False)
        if active.empty:
            texts.append("No holdings")
        else:
            texts.append("<br>".join(f"{k}: {v:.1%}" for k, v in active.items()))
    return texts


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
    state_series = get_state_series(strategy_name, df).reindex(changes.index)
    changes = changes.sort_index(ascending=not newest_first)
    changes.insert(0, "Date", changes.index.strftime("%Y-%m-%d"))
    if not state_series.empty:
        changes.insert(1, "State", state_series.loc[changes.index].fillna("").astype(str).values)
    changes = changes.reset_index(drop=True)
    start_col = 2 if "State" in changes.columns else 1
    for col in changes.columns[start_col:]:
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
    hover_mode: str,
) -> go.Figure:
    eq, dd, _ = equity_with_contributions(df["ret"], start_value, contribution_amount, contribution_frequency)
    holdings_text = build_holdings_text(strategy_name, df)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08,
        subplot_titles=(f"{strategy_name} Equity vs Benchmarks", "Drawdown")
    )
    strategy_hover = []
    for dt, val, hold in zip(eq.index, eq.values, holdings_text):
        if hover_mode == "Holdings %":
            strategy_hover.append(f"Date: {pd.to_datetime(dt):%Y-%m-%d}<br>{strategy_name} Equity: ${val:,.0f}<br><br>{hold}")
        else:
            strategy_hover.append(f"Date: {pd.to_datetime(dt):%Y-%m-%d}<br>{strategy_name} Equity: ${val:,.0f}")
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode="lines", name=strategy_name, line=dict(width=3, color="#fbbf24"), text=strategy_hover, hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    color_map = {"SPY": "#34d399", "QQQ": "#60a5fa", "TQQQ": "#c084fc", "NDX": "#f87171"}
    aligned_holdings = pd.Series(holdings_text, index=df.index).reindex(bench_ret.index).ffill().fillna("No holdings")
    for col in bench_ret.columns:
        beq, _, _ = equity_with_contributions(bench_ret[col].fillna(0.0), start_value, contribution_amount, contribution_frequency)
        bench_hover = []
        for dt, val, hold in zip(beq.index, beq.values, aligned_holdings):
            if hover_mode == "Holdings %":
                bench_hover.append(f"Date: {pd.to_datetime(dt):%Y-%m-%d}<br>{col}: ${val:,.0f}<br><br>{hold}")
            else:
                bench_hover.append(f"Date: {pd.to_datetime(dt):%Y-%m-%d}<br>{col}: ${val:,.0f}")
        fig.add_trace(go.Scatter(x=beq.index, y=beq.values, mode="lines", name=col, line=dict(width=1.5, color=color_map.get(col, "#94a3b8")), text=bench_hover, hovertemplate="%{text}<extra></extra>"), row=1, col=1)

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




def get_state_series(strategy_name: str, df: pd.DataFrame) -> pd.Series:
    if strategy_name == "COMBINED" and "allocator_state" in df.columns:
        return df["allocator_state"].astype(str).map({"risk_on": "Risk On", "neutral": "Neutral", "risk_off": "Risk Off"})
    if strategy_name == "CORE" and "signal_regime" in df.columns:
        return df["signal_regime"].astype(int).map({1: "R1 Risk-On", 2: "R2 Bear", 3: "R3 Growth", 4: "R4 Defensive"})
    weight_col = "target_weight" if "target_weight" in df.columns else ("weight" if "weight" in df.columns else None)
    if weight_col:
        return pd.Series(np.where(df[weight_col].fillna(0.0) > 0, "Risk-On", "Cash"), index=df.index)
    return pd.Series(index=df.index, dtype=object)


def build_trade_reconstruction(strategy_name: str, df: pd.DataFrame, portfolio_value_series: pd.Series, execution_mode: str = "Fractional Shares") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    state_series = get_state_series(strategy_name, df).reindex(alloc_df.index).fillna("")
    changed = alloc_df.round(8).ne(alloc_df.round(8).shift(1)).any(axis=1)
    rebalance_dates = alloc_df.index[changed].tolist()

    traded_symbols = sorted([c for c in alloc_df.columns if c != "CASH"])
    px = get_price_history(traded_symbols, start=str(alloc_df.index.min().date()))
    px = px.reindex(df.index).ffill()

    shares: dict[str, float] = {sym: 0.0 for sym in traded_symbols}
    cash_balance = 0.0

    detail_rows = []
    summary_rows = []

    for dt in rebalance_dates:
        prices_row = px.loc[dt] if not px.empty else pd.Series(dtype=float)
        portfolio_before = float(portfolio_value_series.loc[dt])

        prev_values = {sym: shares.get(sym, 0.0) * float(prices_row.get(sym, np.nan)) for sym in traded_symbols}
        known_prev_total = float(sum(v for v in prev_values.values() if pd.notna(v)) + cash_balance)
        target_weights = alloc_df.loc[dt].fillna(0.0)
        num_trades = 0

        new_shares = {}
        new_values = {}

        explicit_cash_target = portfolio_before * float(target_weights.get("CASH", 0.0))
        rounding_cash = 0.0
        cash_target = explicit_cash_target

        for sym in traded_symbols:
            price = float(prices_row.get(sym, np.nan)) if sym in prices_row.index else np.nan
            prev_sh = float(shares.get(sym, 0.0))
            prev_val = float(prev_values.get(sym, 0.0))
            prev_w = (prev_val / known_prev_total) if known_prev_total > 0 else 0.0
            tgt_w = float(target_weights.get(sym, 0.0))
            tgt_val = portfolio_before * tgt_w

            if pd.isna(price) or price <= 0:
                ns = prev_sh
                nv = prev_val
            else:
                if execution_mode == "Whole Shares":
                    ns = math.floor(tgt_val / price)
                    nv = ns * price
                    rounding_cash += max(0.0, tgt_val - nv)
                else:
                    ns = tgt_val / price
                    nv = ns * price

            new_shares[sym] = ns
            new_values[sym] = nv
            delta_sh = ns - prev_sh

            if abs(delta_sh) > 1e-10 or (abs(tgt_w - prev_w) > 1e-10):
                num_trades += 1
                if prev_sh == 0 and ns > 0:
                    action = "Buy"
                elif prev_sh > 0 and ns == 0:
                    action = "Sell"
                elif ns > prev_sh:
                    action = "Increase"
                elif ns < prev_sh:
                    action = "Reduce"
                else:
                    action = "Hold"

                detail_rows.append({
                    "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "State": state_series.loc[dt],
                    "Symbol": sym,
                    "Action": action,
                    "Close Price": price,
                    "Prev Shares": prev_sh,
                    "New Shares": ns,
                    "Shares Delta": delta_sh,
                    "Prev Weight": prev_w,
                    "New Weight": tgt_w,
                    "Prev Value": prev_val,
                    "New Value": nv,
                })

        cash_target = explicit_cash_target + rounding_cash
        portfolio_after = float(sum(new_values.values()) + cash_target)

        summary_rows.append({
            "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "State": state_series.loc[dt],
            "Portfolio Value Before": portfolio_before,
            "Portfolio Value After": portfolio_after,
            "Cash Before": cash_balance,
            "Cash After": cash_target,
            "Explicit Cash Target": explicit_cash_target,
            "Rounding Cash": rounding_cash,
            "Execution Mode": execution_mode,
            "Number of Trades": num_trades,
        })

        shares = new_shares
        cash_balance = cash_target

    end_dt = alloc_df.index.max()
    end_prices = px.loc[end_dt] if not px.empty else pd.Series(dtype=float)
    final_rows = []
    final_total = cash_balance
    for sym in traded_symbols:
        sh = float(shares.get(sym, 0.0))
        price = float(end_prices.get(sym, np.nan)) if sym in end_prices.index else np.nan
        val = sh * price if pd.notna(price) else np.nan
        if pd.notna(val):
            final_total += val
        if sh > 1e-12:
            final_rows.append({
                "Date": pd.to_datetime(end_dt).strftime("%Y-%m-%d"),
                "State": state_series.loc[end_dt],
                "Symbol": sym,
                "Close Price": price,
                "Shares Held": sh,
                "Market Value": val,
            })
    if cash_balance > 1e-12:
        final_rows.append({
            "Date": pd.to_datetime(end_dt).strftime("%Y-%m-%d"),
            "State": state_series.loc[end_dt],
            "Symbol": "CASH",
            "Close Price": 1.0,
            "Shares Held": cash_balance,
            "Market Value": cash_balance,
        })

    summary_rows.append({
        "Date": pd.to_datetime(end_dt).strftime("%Y-%m-%d"),
        "State": state_series.loc[end_dt],
        "Portfolio Value Before": float(final_total),
        "Portfolio Value After": float(final_total),
        "Cash Before": cash_balance,
        "Cash After": cash_balance,
        "Explicit Cash Target": cash_balance,
        "Rounding Cash": 0.0,
        "Execution Mode": execution_mode,
        "Number of Trades": 0,
        "Note": "Final mark-to-market",
    })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    final_df = pd.DataFrame(final_rows)

    return detail_df, summary_df, final_df

def format_trade_reconstruction(detail_df: pd.DataFrame, summary_df: pd.DataFrame, final_df: pd.DataFrame):
    detail_fmt = detail_df.copy()
    summary_fmt = summary_df.copy()
    final_fmt = final_df.copy()

    for df_ in [detail_fmt]:
        if not df_.empty:
            for col in ["Close Price", "Prev Shares", "New Shares", "Shares Delta", "Prev Value", "New Value"]:
                if col in df_.columns:
                    df_[col] = df_[col].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}" if "Shares" in col else f"{x:,.2f}")
            for col in ["Prev Weight", "New Weight"]:
                if col in df_.columns:
                    df_[col] = df_[col].map(lambda x: f"{x:.1%}")

    if not summary_fmt.empty:
        for col in ["Portfolio Value Before", "Portfolio Value After", "Cash Before", "Cash After", "Explicit Cash Target", "Rounding Cash"]:
            if col in summary_fmt.columns:
                summary_fmt[col] = summary_fmt[col].map(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

    if not final_fmt.empty:
        for col in ["Close Price", "Shares Held", "Market Value"]:
            if col in final_fmt.columns:
                if col == "Market Value":
                    final_fmt[col] = final_fmt[col].map(lambda x: "" if pd.isna(x) else f"${x:,.2f}")
                elif col == "Shares Held":
                    final_fmt[col] = final_fmt[col].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
                else:
                    final_fmt[col] = final_fmt[col].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}")

    return detail_fmt, summary_fmt, final_fmt

def build_trade_tickets(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df.empty:
        return pd.DataFrame(columns=[
            "Date", "State", "Action", "Symbol", "Shares", "Close Price",
            "Estimated Notional", "Instruction"
        ])

    tickets = detail_df.copy()

    def normalize_action(action: str) -> str:
        if action in {"Buy", "Increase"}:
            return "BUY"
        if action in {"Sell", "Reduce"}:
            return "SELL"
        return str(action).upper()

    tickets["Action"] = tickets["Action"].map(normalize_action)
    tickets["Shares"] = tickets["Shares Delta"].abs()
    tickets["Estimated Notional"] = tickets["Shares"] * tickets["Close Price"]
    tickets = tickets[tickets["Shares"] > 1e-10].copy()

    tickets["Instruction"] = tickets.apply(
        lambda r: f'{r["Action"]} {r["Shares"]:,.4f} {r["Symbol"]} @ close ({r["Close Price"]:,.4f})',
        axis=1,
    )

    out = tickets[[
        "Date", "State", "Action", "Symbol", "Shares", "Close Price",
        "Estimated Notional", "Instruction"
    ]].copy()

    out["Shares"] = out["Shares"].map(lambda x: f"{x:,.4f}")
    out["Close Price"] = out["Close Price"].map(lambda x: f"{x:,.4f}")
    out["Estimated Notional"] = out["Estimated Notional"].map(lambda x: f"${x:,.2f}")
    return out


def ticket_csv_bytes(ticket_df: pd.DataFrame) -> bytes:
    return ticket_df.to_csv(index=False).encode("utf-8")


def parse_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")



def estimate_ibkr_commission(shares: float, trade_value: float, commission_mode: str) -> float:
    shares = max(float(shares), 0.0)
    trade_value = max(float(trade_value), 0.0)
    if shares <= 0 or trade_value <= 0:
        return 0.0

    if commission_mode == "Fixed":
        rate = 0.005
        minimum = 1.00
    else:
        rate = 0.0035
        minimum = 0.35

    commission = max(shares * rate, minimum)
    cap = trade_value * 0.01
    return float(min(commission, cap))



def estimate_daily_execution_costs(
    daily_ticket_df: pd.DataFrame,
    commission_mode: str,
    estimated_slippage_bps: float,
) -> dict:
    empty_result = {
        "ticket_count": 0,
        "estimated_commission": 0.0,
        "estimated_slippage": 0.0,
        "estimated_total_cost": 0.0,
        "estimated_commission_text": "$0.00",
        "estimated_slippage_text": "$0.00",
        "estimated_total_cost_text": "$0.00",
        "assumption_text": f"{commission_mode} | {estimated_slippage_bps:.1f} bps",
    }
    if daily_ticket_df.empty:
        return empty_result

    work = daily_ticket_df.copy()
    if "Shares" not in work.columns or "Estimated Notional" not in work.columns:
        return empty_result

    work["_shares_num"] = parse_numeric_series(work["Shares"]).fillna(0.0)
    work["_notional_num"] = parse_numeric_series(work["Estimated Notional"]).fillna(0.0)

    work["_estimated_commission"] = work.apply(
        lambda row: estimate_ibkr_commission(row["_shares_num"], row["_notional_num"], commission_mode),
        axis=1,
    )
    work["_estimated_slippage"] = work["_notional_num"] * (float(estimated_slippage_bps) / 10000.0)

    estimated_commission = float(work["_estimated_commission"].sum())
    estimated_slippage = float(work["_estimated_slippage"].sum())
    estimated_total_cost = estimated_commission + estimated_slippage

    return {
        "ticket_count": int(len(work)),
        "estimated_commission": estimated_commission,
        "estimated_slippage": estimated_slippage,
        "estimated_total_cost": estimated_total_cost,
        "estimated_commission_text": f"${estimated_commission:,.2f}",
        "estimated_slippage_text": f"${estimated_slippage:,.2f}",
        "estimated_total_cost_text": f"${estimated_total_cost:,.2f}",
        "assumption_text": f"{commission_mode} | {estimated_slippage_bps:.1f} bps",
    }


def build_daily_trade_panel(trade_detail_df: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    if df.empty:
        return pd.DataFrame(), "", "No data loaded."
    latest_dt = pd.to_datetime(df.index.max())
    latest_str = latest_dt.strftime("%Y-%m-%d")

    if trade_detail_df.empty:
        return pd.DataFrame(), latest_str, "No trades for the selected period."

    today_detail = trade_detail_df[trade_detail_df["Date"] == latest_str].copy()
    if today_detail.empty:
        return pd.DataFrame(), latest_str, "No rebalance on the latest available date."

    tickets = build_trade_tickets(today_detail)
    return tickets, latest_str, f"Trades generated for {latest_str} close."







def _safe_float(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        if cleaned == "":
            return 0.0
        try:
            return float(cleaned)
        except Exception:
            return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return 0.0


def prepare_fill_tracking_frame(daily_ticket_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "Date", "State", "Action", "Symbol",
        "Expected Close", "Expected Shares", "Expected Notional",
        "Actual Fill", "Shares Filled", "Commission",
        "Fill Notional", "Price Slippage $", "Price Slippage bps",
        "Total Cost $", "Notes",
    ]
    if daily_ticket_df.empty:
        return pd.DataFrame(columns=base_cols)

    frame = daily_ticket_df.copy()
    frame["Expected Close"] = frame["Close Price"].map(_safe_float)
    frame["Expected Shares"] = frame["Shares"].map(_safe_float)
    frame["Expected Notional"] = frame["Estimated Notional"].map(_safe_float)

    frame["Actual Fill"] = frame["Expected Close"]
    frame["Shares Filled"] = frame["Expected Shares"]
    frame["Commission"] = 0.0
    frame["Notes"] = ""

    frame["Fill Notional"] = frame["Actual Fill"] * frame["Shares Filled"]
    frame["Price Slippage $"] = (frame["Actual Fill"] - frame["Expected Close"]) * frame["Shares Filled"]
    frame["Price Slippage bps"] = np.where(
        frame["Expected Close"] != 0,
        ((frame["Actual Fill"] - frame["Expected Close"]) / frame["Expected Close"]) * 10000.0,
        0.0,
    )
    frame["Total Cost $"] = frame["Price Slippage $"] + frame["Commission"]

    out = frame[base_cols].copy()
    return out


def recompute_fill_tracking_metrics(fill_df: pd.DataFrame) -> pd.DataFrame:
    if fill_df.empty:
        return fill_df.copy()

    out = fill_df.copy()
    for col in ["Expected Close", "Expected Shares", "Expected Notional", "Actual Fill", "Shares Filled", "Commission"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].map(_safe_float)

    out["Fill Notional"] = out["Actual Fill"] * out["Shares Filled"]
    out["Price Slippage $"] = (out["Actual Fill"] - out["Expected Close"]) * out["Shares Filled"]
    out["Price Slippage bps"] = np.where(
        out["Expected Close"] != 0,
        ((out["Actual Fill"] - out["Expected Close"]) / out["Expected Close"]) * 10000.0,
        0.0,
    )
    out["Total Cost $"] = out["Price Slippage $"] + out["Commission"]

    if "Notes" not in out.columns:
        out["Notes"] = ""

    ordered_cols = [
        "Date", "State", "Action", "Symbol",
        "Expected Close", "Expected Shares", "Expected Notional",
        "Actual Fill", "Shares Filled", "Commission",
        "Fill Notional", "Price Slippage $", "Price Slippage bps",
        "Total Cost $", "Notes",
    ]
    for col in ordered_cols:
        if col not in out.columns:
            out[col] = "" if col == "Notes" else 0.0
    return out[ordered_cols].copy()


def build_fill_tracking_summary(fill_df: pd.DataFrame) -> dict:
    if fill_df.empty:
        return {
            "ticket_count": 0,
            "expected_notional": 0.0,
            "fill_notional": 0.0,
            "commission": 0.0,
            "slippage_dollars": 0.0,
            "avg_slippage_bps": 0.0,
            "total_cost": 0.0,
        }

    frame = recompute_fill_tracking_metrics(fill_df)
    return {
        "ticket_count": int(len(frame)),
        "expected_notional": float(frame["Expected Notional"].sum()),
        "fill_notional": float(frame["Fill Notional"].sum()),
        "commission": float(frame["Commission"].sum()),
        "slippage_dollars": float(frame["Price Slippage $"].sum()),
        "avg_slippage_bps": float(frame["Price Slippage bps"].mean()) if len(frame) else 0.0,
        "total_cost": float(frame["Total Cost $"].sum()),
    }


def save_fill_tracking_files(
    fill_df: pd.DataFrame,
    base_dir: Path,
    strategy_name: str,
    trade_date: str,
    execution_mode: str,
) -> tuple[Path, Path]:
    fill_dir = base_dir / "logs" / "fill_tracking"
    fill_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    safe_trade_date = str(trade_date or "na").replace(":", "-")
    safe_mode = execution_mode.replace(" ", "_")
    stem = f"ADAPT_fill_tracking_{strategy_name}_{safe_trade_date}_{safe_mode}_{stamp}"

    csv_path = fill_dir / f"{stem}.csv"
    json_path = fill_dir / f"{stem}.json"

    frame = recompute_fill_tracking_metrics(fill_df)
    frame.to_csv(csv_path, index=False)

    payload = {
        "saved_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy_name,
        "trade_date": safe_trade_date,
        "execution_mode": execution_mode,
        "summary": build_fill_tracking_summary(frame),
        "rows": frame.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def render_fill_tracking_panel(
    daily_ticket_df: pd.DataFrame,
    strategy_name: str,
    trade_date: str,
    execution_mode: str,
):
    st.markdown(tooltip_label(
        "Fill Tracking",
        "Manual execution-audit layer for today's tickets only. Enter actual fills, shares filled, and commission. "
        "Derived slippage metrics are audit-only and do not flow back into the ADAPT engine or backtest return stream.",
        f"Trade date: {trade_date or 'N/A'}",
        align="left"
    ), unsafe_allow_html=True)

    if daily_ticket_df.empty:
        st.info("No fill tracking required because there are no tickets on the latest finalized bar.")
        return

    fill_key = f"fill_tracking_editor_{strategy_name}_{trade_date}_{execution_mode.replace(' ', '_')}"
    base_df = prepare_fill_tracking_frame(daily_ticket_df)

    if fill_key not in st.session_state or len(st.session_state.get(fill_key, pd.DataFrame())) != len(base_df):
        st.session_state[fill_key] = base_df.copy()

    edited_df = st.data_editor(
        st.session_state[fill_key],
        width="stretch",
        hide_index=True,
        key=f"{fill_key}_widget",
        column_config={
            "Date": st.column_config.TextColumn(disabled=True),
            "State": st.column_config.TextColumn(disabled=True),
            "Action": st.column_config.TextColumn(disabled=True),
            "Symbol": st.column_config.TextColumn(disabled=True),
            "Expected Close": st.column_config.NumberColumn(format="%.4f", disabled=True),
            "Expected Shares": st.column_config.NumberColumn(format="%.4f", disabled=True),
            "Expected Notional": st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Actual Fill": st.column_config.NumberColumn(format="%.4f"),
            "Shares Filled": st.column_config.NumberColumn(format="%.4f"),
            "Commission": st.column_config.NumberColumn(format="$%.2f"),
            "Fill Notional": st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Price Slippage $": st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Price Slippage bps": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "Total Cost $": st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Notes": st.column_config.TextColumn(),
        },
        disabled=[
            "Date", "State", "Action", "Symbol",
            "Expected Close", "Expected Shares", "Expected Notional",
            "Fill Notional", "Price Slippage $", "Price Slippage bps", "Total Cost $",
        ],
    )

    recomputed_df = recompute_fill_tracking_metrics(pd.DataFrame(edited_df))
    st.session_state[fill_key] = recomputed_df.copy()

    summary = build_fill_tracking_summary(recomputed_df)
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(metric_card_html("Tickets", f"{summary['ticket_count']:,}", [f"Trade date {trade_date}", f"Mode: {execution_mode}"]), unsafe_allow_html=True)
    s2.markdown(metric_card_html("Expected Notional", f"${summary['expected_notional']:,.2f}", ["Model close-based", "Audit baseline only"]), unsafe_allow_html=True)
    s3.markdown(metric_card_html("Fill Notional", f"${summary['fill_notional']:,.2f}", [f"Commission ${summary['commission']:,.2f}", f"Avg slippage {summary['avg_slippage_bps']:.2f} bps"]), unsafe_allow_html=True)
    s4.markdown(metric_card_html("Total Audit Cost", f"${summary['total_cost']:,.2f}", [f"Price slippage ${summary['slippage_dollars']:,.2f}", "Does not change engine metrics"]), unsafe_allow_html=True)

    st.dataframe(dark_dataframe(recomputed_df), width="stretch")

    base_dir = Path(__file__).resolve().parent
    fill_save_state_key = "fill_tracking_last_save"
    c1, c2 = st.columns([0.38, 0.62])
    with c1:
        if st.button("Save Fill Tracking Log", width="stretch"):
            csv_path, json_path = save_fill_tracking_files(recomputed_df, base_dir, strategy_name, trade_date, execution_mode)
            st.session_state[fill_save_state_key] = {
                "csv_name": csv_path.name,
                "csv_path": str(csv_path),
                "json_path": str(json_path),
            }
    with c2:
        st.download_button(
            label="Download Fill Tracking CSV",
            data=recomputed_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ADAPT_fill_tracking_{strategy_name}_{trade_date}_{execution_mode.replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
        )

    fill_save_state = st.session_state.get(fill_save_state_key)
    if fill_save_state:
        st.success(f"Fill tracking saved: {fill_save_state['csv_name']}")
        with st.expander("Saved file details", expanded=False):
            st.caption(f"CSV: {fill_save_state['csv_path']}")
            st.caption(f"JSON: {fill_save_state['json_path']}")


def build_live_execution_snapshot(
    strategy_name: str,
    df: pd.DataFrame,
    daily_ticket_df: pd.DataFrame,
    latest_trade_date_str: str,
    execution_mode: str,
    commission_mode: str,
    estimated_slippage_bps: float,
) -> dict:
    if df.empty:
        return {
            "latest_finalized_bar_date": "",
            "current_state": "Unavailable",
            "target_allocation_text": "Unavailable",
            "rebalance_required": False,
            "trade_count": 0,
            "latest_trade_date": latest_trade_date_str,
            "estimated_cost_text": "$0.00",
            "estimated_commission_text": "$0.00",
            "estimated_slippage_text": "$0.00",
            "estimated_notional_text": "$0.00",
            "commission_mode": commission_mode,
            "estimated_slippage_bps": float(estimated_slippage_bps),
            "checklist_rows": [],
        }

    latest_dt = pd.to_datetime(df.index.max())
    current_state = current_state_label(strategy_name, df)

    alloc_df = build_allocation_frame(strategy_name, df)
    if alloc_df.empty:
        target_allocation_text = "Unavailable"
    else:
        latest_alloc = alloc_df.iloc[-1]
        latest_alloc = latest_alloc[latest_alloc > 0].sort_values(ascending=False)
        target_allocation_text = " | ".join(f"{asset} {weight:.1%}" for asset, weight in latest_alloc.items()) if not latest_alloc.empty else "No active allocation"

    rebalance_required = not daily_ticket_df.empty

    estimated_notional = 0.0
    if not daily_ticket_df.empty and "Estimated Notional" in daily_ticket_df.columns:
        estimated_notional = parse_numeric_series(daily_ticket_df["Estimated Notional"]).fillna(0.0).sum()

    cost_summary = estimate_daily_execution_costs(daily_ticket_df, commission_mode, estimated_slippage_bps)

    checklist_rows = [
        {"Check": "Latest bar finalized", "Status": "Yes", "Detail": latest_dt.strftime("%Y-%m-%d")},
        {"Check": "Allocator state confirmed", "Status": "Yes", "Detail": current_state},
        {"Check": "Execution mode confirmed", "Status": "Yes", "Detail": execution_mode},
        {"Check": "Rebalance required today", "Status": "Yes" if rebalance_required else "No", "Detail": latest_trade_date_str or latest_dt.strftime("%Y-%m-%d")},
        {"Check": "Trade tickets ready", "Status": "Yes" if rebalance_required else "No", "Detail": f"{len(daily_ticket_df):,} ticket(s)"},
        {"Check": "Execution cost estimate", "Status": "Ready", "Detail": f"{cost_summary['estimated_total_cost_text']} ({cost_summary['assumption_text']})"},
    ]

    return {
        "latest_finalized_bar_date": latest_dt.strftime("%Y-%m-%d"),
        "current_state": current_state,
        "target_allocation_text": target_allocation_text,
        "rebalance_required": rebalance_required,
        "trade_count": len(daily_ticket_df),
        "latest_trade_date": latest_trade_date_str or latest_dt.strftime("%Y-%m-%d"),
        "estimated_cost_text": cost_summary["estimated_total_cost_text"],
        "estimated_commission_text": cost_summary["estimated_commission_text"],
        "estimated_slippage_text": cost_summary["estimated_slippage_text"],
        "estimated_notional_text": f"${estimated_notional:,.2f}",
        "commission_mode": commission_mode,
        "estimated_slippage_bps": float(estimated_slippage_bps),
        "checklist_rows": checklist_rows,
    }


def render_live_execution_panel(snapshot: dict, daily_ticket_df: pd.DataFrame):
    st.markdown(tooltip_label(
        "Live Execution Panel",
        "Top-of-dashboard operational summary for the latest finalized bar. "
        "This is a display-only layer built from existing dashboard outputs and does not change ADAPT engine logic, allocation logic, reconstruction math, or the return stream.",
        f"Finalized bar: {snapshot.get('latest_finalized_bar_date', 'N/A')}",
        align="left",
    ), unsafe_allow_html=True)

    rebalance_required = bool(snapshot.get("rebalance_required", False))
    trade_count = int(snapshot.get("trade_count", 0) or 0)
    status_label = "ACTION NEEDED" if rebalance_required else "MONITOR"
    status_note = (
        f"Latest trade date {snapshot.get('latest_trade_date', 'N/A')} | {trade_count:,} ticket(s)"
        if rebalance_required
        else f"Latest trade date {snapshot.get('latest_trade_date', 'N/A')} | no rebalance"
    )

    s1, s2 = st.columns([1.2, 2.8])
    s1.markdown(
        metric_card_html(
            "Operational Status",
            status_label,
            [status_note, f"Allocator {snapshot.get('current_state', 'Unavailable')}"],
        ),
        unsafe_allow_html=True,
    )
    s2.markdown(
        f"""
        <div class="side-note" style="min-height:88px; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.80rem; text-transform:uppercase; letter-spacing:0.06em; color:#9ca3af; margin-bottom:6px;">Operator Focus</div>
            <div style="font-size:1.02rem; font-weight:700; color:#e5e7eb; margin-bottom:6px;">
                {"Rebalance session active for latest finalized bar." if rebalance_required else "No rebalance required on latest finalized bar."}
            </div>
            <div style="font-size:0.90rem; color:#cbd5e1;">
                {'Review trade tickets, cost estimate, and checklist before export or execution.' if rebalance_required else 'Review allocator state, finalized bar date, and snapshot logging status.'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4, p5 = st.columns(5)
    p1.markdown(
        metric_card_html(
            "Latest Finalized Bar",
            snapshot.get("latest_finalized_bar_date", "N/A"),
            ["Latest available model date", "Display only"],
        ),
        unsafe_allow_html=True,
    )
    p2.markdown(
        metric_card_html(
            "Allocator State",
            snapshot.get("current_state", "Unavailable"),
            ["Current live state", "From existing strategy output"],
        ),
        unsafe_allow_html=True,
    )
    p3.markdown(
        metric_card_html(
            "Rebalance Today?",
            "Yes" if rebalance_required else "No",
            [f"Trade date {snapshot.get('latest_trade_date', 'N/A')}", f"Tickets {trade_count:,}"],
        ),
        unsafe_allow_html=True,
    )
    p4.markdown(
        metric_card_html(
            "Execution Cost Estimate",
            snapshot.get("estimated_cost_text", "$0.00"),
            [
                f"Commission {snapshot.get('estimated_commission_text', '$0.00')}",
                f"Slippage {snapshot.get('estimated_slippage_text', '$0.00')} | {snapshot.get('commission_mode', 'Tiered')} | {snapshot.get('estimated_slippage_bps', 0.0):.1f} bps",
            ],
        ),
        unsafe_allow_html=True,
    )
    p5.markdown(
        metric_card_html(
            "Ticket Notional",
            snapshot.get("estimated_notional_text", "$0.00"),
            ["Sum of today's ticket notionals", "Operational sizing view"],
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Current Target Allocation")
    st.markdown(
        f"""
        <div class="side-note">
        {snapshot.get("target_allocation_text", "Unavailable")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    tcol, ccol = st.columns([2.2, 1.2])
    with tcol:
        st.markdown("#### Today's Trade Tickets")
        if not daily_ticket_df.empty:
            st.dataframe(dark_dataframe(daily_ticket_df), width="stretch")
        else:
            st.info("No rebalance required on the latest finalized bar.")
    with ccol:
        st.markdown("#### Operator Checklist")
        checklist_df = pd.DataFrame(snapshot.get("checklist_rows", []))
        if not checklist_df.empty:
            checklist_lines = []
            for _, row in checklist_df.iterrows():
                status = str(row.get("Status", "")).strip().upper()
                item = str(row.get("Check", "")).strip()
                detail = str(row.get("Detail", "")).strip()
                icon = "[DONE]" if status in {"READY", "YES", "DONE", "OK"} else "[CHECK]"
                line = f"{icon} **{item}**"
                if detail:
                    line += f"  \n<small>{detail}</small>"
                checklist_lines.append(line)
            st.markdown("\n\n".join(checklist_lines), unsafe_allow_html=True)
        else:
            st.info("No checklist items available.")



def build_snapshot_payload(
    strategy_name: str,
    start_date_value,
    end_date_value,
    execution_mode: str,
    snapshot: dict,
    alloc_now: pd.DataFrame,
    daily_ticket_df: pd.DataFrame,
    final_positions_df: pd.DataFrame,
) -> dict:
    run_ts = pd.Timestamp.now()

    allocation_rows = alloc_now.to_dict(orient="records") if not alloc_now.empty else []
    ticket_rows = daily_ticket_df.to_dict(orient="records") if not daily_ticket_df.empty else []

    latest_prices_df = pd.DataFrame()
    if not final_positions_df.empty:
        latest_prices_df = final_positions_df.copy()
        if "Close Price" in latest_prices_df.columns:
            latest_prices_df["Close Price"] = latest_prices_df["Close Price"].map(
                lambda x: None if pd.isna(x) else round(float(x), 6)
            )
        if "Shares Held" in latest_prices_df.columns:
            latest_prices_df["Shares Held"] = latest_prices_df["Shares Held"].map(
                lambda x: None if pd.isna(x) else round(float(x), 6)
            )
        if "Market Value" in latest_prices_df.columns:
            latest_prices_df["Market Value"] = latest_prices_df["Market Value"].map(
                lambda x: None if pd.isna(x) else round(float(x), 2)
            )

    payload = {
        "run_timestamp": run_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "run_timestamp_utc": run_ts.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy_name,
        "start_date": str(start_date_value),
        "end_date": str(end_date_value),
        "execution_mode": execution_mode,
        "latest_finalized_bar_date": snapshot.get("latest_finalized_bar_date", ""),
        "allocator_state": snapshot.get("current_state", "Unavailable"),
        "rebalance_required_today": bool(snapshot.get("rebalance_required", False)),
        "latest_trade_date": snapshot.get("latest_trade_date", ""),
        "ticket_count": int(snapshot.get("trade_count", 0)),
        "ticket_notional": snapshot.get("estimated_notional_text", "$0.00"),
        "commission_mode": snapshot.get("commission_mode", "Tiered"),
        "estimated_slippage_bps": float(snapshot.get("estimated_slippage_bps", 0.0)),
        "estimated_commission": snapshot.get("estimated_commission_text", "$0.00"),
        "estimated_slippage": snapshot.get("estimated_slippage_text", "$0.00"),
        "execution_cost_estimate": snapshot.get("estimated_cost_text", "$0.00"),
        "target_allocation": allocation_rows,
        "today_trade_tickets": ticket_rows,
        "latest_prices": latest_prices_df.to_dict(orient="records") if not latest_prices_df.empty else [],
        "operational_checklist": snapshot.get("checklist_rows", []),
    }
    return payload


def save_snapshot_files(payload: dict, base_dir: Path) -> tuple[Path, Path]:
    snapshot_dir = base_dir / "logs" / "daily_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    finalized_bar = str(payload.get("latest_finalized_bar_date", "na")).replace(":", "-")
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    stem = f"ADAPT_snapshot_{payload.get('strategy', 'UNKNOWN')}_{finalized_bar}_{stamp}"

    json_path = snapshot_dir / f"{stem}.json"
    tickets_path = snapshot_dir / f"{stem}_tickets.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    tickets_df = pd.DataFrame(payload.get("today_trade_tickets", []))
    if tickets_df.empty:
        tickets_df = pd.DataFrame(columns=[
            "Date", "State", "Action", "Symbol", "Shares", "Close Price",
            "Estimated Notional", "Instruction",
        ])
    tickets_df.to_csv(tickets_path, index=False)

    return json_path, tickets_path


def render_snapshot_logging_panel(payload: dict):
    st.markdown("#### Daily Snapshot Logging")
    st.caption("Writes an audit snapshot to logs/daily_snapshots inside ADAPT_DEPLOY. Save is explicit only in this step.")

    base_dir = Path(__file__).resolve().parent
    snapshot_save_state_key = "snapshot_last_save"
    c1, c2 = st.columns([0.38, 0.62])
    with c1:
        if st.button("Save Daily Snapshot", width="stretch"):
            json_path, tickets_path = save_snapshot_files(payload, base_dir)
            st.session_state[snapshot_save_state_key] = {
                "json_name": json_path.name,
                "json_path": str(json_path),
                "tickets_path": str(tickets_path),
            }
    with c2:
        st.download_button(
            label="Download Snapshot JSON",
            data=json.dumps(payload, indent=2),
            file_name=f"ADAPT_snapshot_{payload.get('strategy', 'UNKNOWN')}_{payload.get('latest_finalized_bar_date', 'na')}.json",
            mime="application/json",
            width="stretch",
        )

    snapshot_save_state = st.session_state.get(snapshot_save_state_key)
    if snapshot_save_state:
        st.success(f"Snapshot saved: {snapshot_save_state['json_name']}")
        with st.expander("Saved file details", expanded=False):
            st.caption(f"JSON: {snapshot_save_state['json_path']}")
            st.caption(f"Tickets CSV: {snapshot_save_state['tickets_path']}")


st.set_page_config(page_title="ADAPT Strategy Terminal", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #050d19 0%, #091223 100%); color: white; }
    .block-container { max-width: 1500px; padding-top: 2.8rem; }
    .hero { background: linear-gradient(135deg, #0f172a 0%, #111f3d 100%); border: 1px solid #334155; border-radius: 18px; padding: 26px 20px 14px 20px; margin-bottom: 12px; overflow: visible; }
    .metricbox { background: #0f172a; border: 1px solid #334155; border-radius: 14px; padding: 10px 12px; min-height: 100px; display:flex; flex-direction:column; justify-content:flex-start; }
    .metrichead { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .metriclabel { color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; }
    .metricvalue { color:white; font-size:1.15rem; font-weight:700; margin-top:4px; }
    .metricsub { color:#94a3b8; font-size:0.72rem; margin-top:4px; line-height:1.25; }
    .label-wrap { margin-bottom: 8px; position: relative; }
    .label-row { display:flex; align-items:center; gap:8px; }
    .label-title { font-weight:700; color:white; }
    .label-sub { display:block; color:#94a3b8; font-size:0.9rem; margin-top:4px; }
    .tooltip-wrap, .metric-tooltip-wrap { position:relative; display:inline-flex; }
    .tooltip-icon, .metric-tooltip-icon { display:inline-flex; align-items:center; justify-content:center; width:18px; height:18px; border-radius:50%; border:1px solid #64748b; font-size:11px; cursor:help; color:#cbd5e1; flex:0 0 auto; }
    .tooltip-box { visibility:hidden; opacity:0; position:absolute; top:24px; width:340px; background:#0f172a; border:1px solid #334155; color:#e2e8f0; border-radius:10px; padding:10px 12px; font-size:0.85rem; line-height:1.4; z-index:9999; transition:opacity 0.15s ease; box-shadow:0 10px 24px rgba(0,0,0,0.35); }
    .tooltip-left { left:0; }
    .tooltip-right { right:0; }
    .tooltip-wrap:hover .tooltip-box, .metric-tooltip-wrap:hover .tooltip-box { visibility:visible; opacity:1; }
    .side-note { background:#13233d; border:1px solid #334155; border-radius:14px; padding:14px; color:#dbe5f3; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div style="color:#fbbf24;font-size:0.66rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;">ADAPT Strategy Terminal</div>
      <h1 style="margin:4px 0 4px 0; font-size:1.7rem; line-height:1.08;">Institutional Strategy Terminal</h1>
      <div style="color:#94a3b8;font-size:0.78rem;">Institutional analytics for the ADAPT CORE + ALPHA system. Display layer only — preserves the validated ADAPT engine workflow and return stream calculations.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

bundle = get_strategy_bundle()
ranges = get_strategy_ranges(bundle)

if "last_run_inputs" not in st.session_state:
    st.session_state["last_run_inputs"] = None

with st.sidebar:
    hover_mode = st.radio("Hover Display", ["Performance", "Holdings %"], index=0)
    strategy = st.selectbox("Strategy", ["COMBINED", "CORE", "ALPHA"], index=0)
    start_capital = st.number_input("Starting Capital", min_value=1000.0, value=50000.0, step=1000.0)
    contribution_amount = st.number_input("Contribution Amount", min_value=0.0, value=0.0, step=100.0)
    contribution_frequency = st.selectbox("Contribution Frequency", ["None", "Monthly", "Yearly"], index=0)
    sub_target_threshold = st.number_input("Sub-Target CAGR Threshold (%)", min_value=0.0, value=5.0, step=1.0)
    sort_order = st.selectbox("Allocation Change Log Order", ["Newest to Oldest", "Oldest to Newest"], index=0)
    execution_mode = st.selectbox("Execution Mode", ["Fractional Shares", "Whole Shares"], index=0)
    commission_mode = st.selectbox("IBKR Commission Mode", ["Tiered", "Fixed"], index=0)
    estimated_slippage_bps = st.number_input("Estimated Slippage (bps)", min_value=0.0, value=5.0, step=0.5)
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

    run_clicked = st.button("Run Backtest", type="primary", width="stretch")

if run_clicked:
    st.session_state["last_run_inputs"] = {
        "hover_mode": hover_mode,
        "strategy": strategy,
        "start_capital": float(start_capital),
        "contribution_amount": float(contribution_amount),
        "contribution_frequency": contribution_frequency,
        "sub_target_threshold": float(sub_target_threshold),
        "sort_order": sort_order,
        "execution_mode": execution_mode,
        "commission_mode": commission_mode,
        "estimated_slippage_bps": float(estimated_slippage_bps),
        "benchmarks": list(benchmarks),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

run = st.session_state["last_run_inputs"] is not None

if run:
    active_inputs = st.session_state["last_run_inputs"]
    hover_mode = active_inputs["hover_mode"]
    strategy = active_inputs["strategy"]
    start_capital = float(active_inputs["start_capital"])
    contribution_amount = float(active_inputs["contribution_amount"])
    contribution_frequency = active_inputs["contribution_frequency"]
    sub_target_threshold = float(active_inputs["sub_target_threshold"])
    sort_order = active_inputs["sort_order"]
    execution_mode = active_inputs["execution_mode"]
    commission_mode = active_inputs.get("commission_mode", "Tiered")
    estimated_slippage_bps = float(active_inputs.get("estimated_slippage_bps", 5.0))
    benchmarks = list(active_inputs["benchmarks"])
    start_date = pd.Timestamp(active_inputs["start_date"]).date()
    end_date = pd.Timestamp(active_inputs["end_date"]).date()

    if end_date <= start_date:
        st.error("End date must be after start date.")
        st.stop()

    df = bundle[strategy].copy().loc[str(start_date):str(end_date)]
    if df.empty:
        st.error("No data for that date range.")
        st.stop()

    import datetime as dt
    latest_date = pd.to_datetime(df.index.max()).date()
    today = dt.date.today()
    if latest_date < today:
        st.warning(
            f"Latest data is {latest_date}. Today's market close may not be finalized yet."
        )

    bench_symbols = ["^NDX" if x == "NDX" else x for x in benchmarks]
    bench_px = get_price_history(bench_symbols, start=str(start_date))
    bench_px = bench_px.loc[str(start_date):str(end_date)].copy()
    if not bench_px.empty:
        bench_px = bench_px.rename(columns={"^NDX": "NDX"})
        bench_ret = bench_px.pct_change(fill_method=None).reindex(df.index).dropna(how="all")
    else:
        bench_ret = pd.DataFrame(index=df.index)

    counts, state_title, state_tooltip = get_state_distribution(strategy, df)
    fig_state = build_state_fig(counts)
    fig_eq = build_equity_fig(df, bench_ret, strategy, start_capital, contribution_amount, contribution_frequency, hover_mode)
    rolling_fig, rolling_3y, rolling_2y = build_rolling_fig(df["ret"])

    perf = perf_from_returns(df["ret"], start_capital)
    rebalance_events = count_rebalance_events(strategy, df)
    alloc_log = allocation_change_log(strategy, df, newest_first=(sort_order == "Newest to Oldest"))
    alloc_now = current_allocation_table(strategy, df)
    alloc_profile = state_allocation_profile(strategy, df)
    state_now = current_state_label(strategy, df)


    eq_series, dd_series, total_contrib = equity_with_contributions(df["ret"], start_capital, contribution_amount, contribution_frequency)
    trade_detail_df, trade_summary_df, final_positions_df = build_trade_reconstruction(strategy, df, eq_series, execution_mode)
    trade_detail_fmt, trade_summary_fmt, final_positions_fmt = format_trade_reconstruction(
        trade_detail_df, trade_summary_df, final_positions_df
    )
    trade_ticket_df = build_trade_tickets(trade_detail_df)
    daily_ticket_df, latest_trade_date_str, daily_trade_message = build_daily_trade_panel(trade_detail_df, df)
    live_execution_snapshot = build_live_execution_snapshot(
        strategy,
        df,
        daily_ticket_df,
        latest_trade_date_str,
        execution_mode,
        commission_mode,
        estimated_slippage_bps,
    )
    snapshot_payload = build_snapshot_payload(
        strategy,
        start_date,
        end_date,
        execution_mode,
        live_execution_snapshot,
        alloc_now,
        daily_ticket_df,
        final_positions_df,
    )
    median_rec, avg_rec, max_rec = recovery_stats(dd_series)
    ulcer = ulcer_index(dd_series)
    roll3_best = float(rolling_3y.max()) if not rolling_3y.empty else 0.0
    roll3_worst = float(rolling_3y.min()) if not rolling_3y.empty else 0.0
    roll3_med = float(rolling_3y.median()) if not rolling_3y.empty else 0.0
    roll2_med = float(rolling_2y.median()) if not rolling_2y.empty else 0.0
    prob_neg = float((rolling_2y < 0).mean()) if not rolling_2y.empty else 0.0
    sub_target = sub_target_threshold / 100.0
    prob_sub_target = float((rolling_2y < sub_target).mean()) if not rolling_2y.empty else 0.0

    render_live_execution_panel(live_execution_snapshot, daily_ticket_df)
    render_snapshot_logging_panel(snapshot_payload)

    st.markdown('<div style="font-size:1.45rem; font-weight:700; margin:16px 0 8px 0;">' + ("COMBINED Overview" if strategy == "COMBINED" else f"{strategy} Overview") + '</div>', unsafe_allow_html=True)

    under_tooltip = (
        "Probability that a rolling 2-year investment period produced a negative annualized return.<br><br>"
        f"For the selected period, the model found <strong>{prob_neg:.2%}</strong> of rolling 2-year windows below 0% CAGR."
    )
    sub_target_tooltip = (
        f"Probability that a rolling 2-year period produced less than <strong>{sub_target_threshold:.1f}%</strong> annualized return.<br><br>"
        f"For the selected period, the model found <strong>{prob_sub_target:.2%}</strong> of rolling 2-year windows below the chosen threshold."
    )

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        a1, a2 = st.columns(2)
        total_return = (eq_series.iloc[-1] / eq_series.iloc[0]) - 1
        a1.markdown(metric_card_html("CAGR", f"{perf['cagr']:.2%}", [f"{perf['days']:,} trading days", f"Total Return {total_return:.2%}"]), unsafe_allow_html=True)
        a2.markdown(metric_card_html("Max Drawdown", f"{perf['maxdd']:.2%}", [f"Ulcer Index {ulcer:.2%}", "Recovery metrics below"]), unsafe_allow_html=True)
    with r1c2:
        a3, a4 = st.columns(2)
        a3.markdown(metric_card_html("Sharpe", f"{perf['sharpe']:.2f}", [f"Calmar {perf['calmar']:.2f}", "Return-stream metric"]), unsafe_allow_html=True)
        a4.markdown(metric_card_html("Final Value", f"${eq_series.iloc[-1]:,.0f}", [f"Contributions ${total_contrib:,.0f}", "Equity includes contribution schedule"]), unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        b1, b2 = st.columns(2)
        b1.markdown(metric_card_html("Median Recovery", f"{0 if median_rec is None else median_rec:.0f} d", [f"Average {0 if avg_rec is None else avg_rec:.1f} d", f"Longest {0 if max_rec is None else max_rec:.0f} d"]), unsafe_allow_html=True)
        b2.markdown(metric_card_html("Rolling 3Y CAGR", f"{roll3_med:.2%}", [f"Best {roll3_best:.2%}", f"Worst {roll3_worst:.2%}"]), unsafe_allow_html=True)
    with r2c2:
        b3, b4 = st.columns(2)
        b3.markdown(metric_card_html("2Y Underperformance", f"{prob_neg:.2%}", ["Prob(2Y CAGR &lt; 0%)", f"Median 2Y CAGR {roll2_med:.2%}"], tooltip=under_tooltip, align="left"), unsafe_allow_html=True)
        b4.markdown(metric_card_html("Sub-Target Risk", f"{prob_sub_target:.2%}", [f"Prob(&lt;{sub_target_threshold:.1f}%) over rolling 2Y periods", f"Rebalance events {rebalance_events:,}"], tooltip=sub_target_tooltip, align="right"), unsafe_allow_html=True)

    st.plotly_chart(fig_eq, width="stretch")

    row2_left, row2_right = st.columns([1.15, 0.85])
    with row2_left:
        st.markdown(tooltip_label("Rolling CAGR Stability", "Rolling annualized returns over 2-year and 3-year windows.", align="left"), unsafe_allow_html=True)
        st.plotly_chart(rolling_fig, width="stretch")
    with row2_right:
        st.markdown(tooltip_label(state_title, state_tooltip, align="right"), unsafe_allow_html=True)
        st.plotly_chart(fig_state, width="stretch")

    st.markdown(tooltip_label("Current Allocation", "This table shows the current live allocation for the selected strategy and end date.<br><br><strong>Risk On</strong>: more aggressive exposure.<br><strong>Neutral</strong>: balanced intermediate stance.<br><strong>Risk Off</strong>: preservation-oriented stance.", f"Current state: {state_now}", align="left"), unsafe_allow_html=True)
    st.dataframe(dark_dataframe(alloc_now), width="stretch")

    st.markdown("#### State Allocation Profile")
    st.dataframe(dark_dataframe(alloc_profile), width="stretch")

    st.markdown("#### Allocation Change Log")
    st.dataframe(dark_dataframe(alloc_log.head(250)), width="stretch")

    st.markdown(tooltip_label(
        "Trade Reconstruction Panel",
        "Reconstructs each rebalance using the same-day official close for every traded ETF under the Market-on-Close assumption. "
        "Fractional mode allows exact target weights. Whole-share mode rounds share counts down to integers and carries the remainder as cash. "
        "This is an audit overlay for the displayed allocations and final value.",
        f"Mode: {execution_mode}",
        align="left"
    ), unsafe_allow_html=True)

    if not trade_summary_df.empty:
        reconstructed_final = float(final_positions_df["Market Value"].sum()) if not final_positions_df.empty else float("nan")
        backtest_final = float(eq_series.iloc[-1]) if not eq_series.empty else float("nan")
        diff = reconstructed_final - backtest_final if pd.notna(reconstructed_final) and pd.notna(backtest_final) else float("nan")

        drag_pct = (diff / backtest_final) if pd.notna(diff) and pd.notna(backtest_final) and backtest_final != 0 else float("nan")

        x1, x2, x3, x4 = st.columns(4)
        x1.markdown(metric_card_html("Holdings-Based Final Value", f"${reconstructed_final:,.0f}", ["From reconstructed share counts", f"Mode: {execution_mode}"]), unsafe_allow_html=True)
        x2.markdown(metric_card_html("Backtest Final Value", f"${backtest_final:,.0f}", ["From ADAPT return stream", "Current dashboard result"]), unsafe_allow_html=True)
        x3.markdown(metric_card_html("Reconstruction Difference", f"${diff:,.2f}", ["Reconstructed minus backtest", "Whole-share mode may differ due to rounding"]), unsafe_allow_html=True)
        x4.markdown(metric_card_html("Execution Drag vs Backtest", f"{drag_pct:.4%}", [f"Dollar drag ${diff:,.2f}", "Negative = reconstruction below backtest"]), unsafe_allow_html=True)

        st.markdown(tooltip_label(
            "What To Trade Today",
            "Shows only the trade instructions for the latest available rebalance date in the selected period. "
            "If the latest date has no rebalance, it will explicitly say there are no trades today. "
            "Use this as the operational panel for close-execution decisions.",
            f"Latest available date: {latest_trade_date_str or 'N/A'}",
            align="left"
        ), unsafe_allow_html=True)

        if not daily_ticket_df.empty:
            daily_file = f"ADAPT_daily_trade_tickets_{strategy}_{latest_trade_date_str}_{execution_mode.replace(' ', '_')}.csv"
            st.download_button(
                label="Download Today's Trade Tickets CSV",
                data=ticket_csv_bytes(daily_ticket_df),
                file_name=daily_file,
                mime="text/csv",
                width="stretch",
            )
            st.dataframe(dark_dataframe(daily_ticket_df), width="stretch")
        else:
            st.info(daily_trade_message)

        render_fill_tracking_panel(
            daily_ticket_df=daily_ticket_df,
            strategy_name=strategy,
            trade_date=latest_trade_date_str,
            execution_mode=execution_mode,
        )

        st.markdown(tooltip_label(
            "Trade Ticket Output",
            "Creates close-execution trade instructions from the reconstructed rebalance trades. "
            "BUY/SELL directions are derived from share-count changes. This is the most practical execution layer for manually verifying or placing trades.",
            f"Mode: {execution_mode}",
            align="left"
        ), unsafe_allow_html=True)

        ticket_file = f"ADAPT_trade_tickets_{strategy}_{start_date}_to_{end_date}_{execution_mode.replace(' ', '_')}.csv"
        st.download_button(
            label="Download Trade Tickets CSV",
            data=ticket_csv_bytes(trade_ticket_df),
            file_name=ticket_file,
            mime="text/csv",
            width="stretch",
        )
        st.dataframe(dark_dataframe(trade_ticket_df), width="stretch")

        st.markdown("#### Rebalance Summary")
        st.dataframe(dark_dataframe(trade_summary_fmt), width="stretch")

        st.markdown("#### Trade Detail")
        st.dataframe(dark_dataframe(trade_detail_fmt), width="stretch")

        st.markdown("#### Final Position Mark-to-Market")
        st.dataframe(dark_dataframe(final_positions_fmt), width="stretch")
    else:
        st.info("No trade reconstruction available for the selected period.")

    yearly_df = yearly_return_table(strategy, df["ret"], bench_ret)
    st.markdown("#### Yearly Return Heatmap")
    st.plotly_chart(build_yearly_heatmap(yearly_df), width="stretch")

    st.markdown(f"#### {strategy} vs Benchmarks")
    st.dataframe(
        dark_dataframe(benchmark_table(strategy, df["ret"], bench_ret, start_capital, contribution_amount, contribution_frequency)),
        width="stretch",
    )

    st.markdown("#### Yearly Returns (%)")
    st.dataframe(dark_dataframe(yearly_df, hide_index=False), width="stretch")
else:
    st.info("Run Backtest by clicking the arrow on top left corner to open & close the dashboard on left side of page. Here you can input your personal variables.")

st.markdown("---")
st.markdown(
    "**ADAPT — Asymmetric Dynamic Allocation with Protected Trend**",
)
st.markdown(
    "ADAPT is a rules-based trading strategy that automatically shifts between aggressive growth and capital protection "
    "depending on market conditions. When markets are trending well, it leans into leveraged growth. When signals turn "
    "bearish or a drawdown hits, it retreats into a defensive basket of uncorrelated assets. "
    "The goal is simple — capture the upside, protect the downside."
)
st.markdown(
    "<small>This Institutional Strategy for TQQQ was developed as a proprietary program by Airam Capital and Red Rock Fund. "
    "It is exclusively for personal use. Unauthorized use is prohibited without a license agreement.</small>",
    unsafe_allow_html=True,
)

with open("ADAPT_Whitepaper.pdf", "rb") as f:
    st.download_button(
        label="Download here for ADAPT White Paper",
        data=f,
        file_name="ADAPT_Whitepaper.pdf",
        mime="application/pdf",
    )
