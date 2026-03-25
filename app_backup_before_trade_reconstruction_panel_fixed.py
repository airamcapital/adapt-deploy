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


def build_trade_reconstruction(strategy_name: str, df: pd.DataFrame, start_value: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    for i, dt in enumerate(rebalance_dates):
        prices_row = px.loc[dt] if not px.empty else pd.Series(dtype=float)
        prev_values = {sym: shares.get(sym, 0.0) * float(prices_row.get(sym, np.nan)) for sym in traded_symbols}
        portfolio_before = float(sum(v for v in prev_values.values() if pd.notna(v)) + cash_balance)
        if i == 0 and portfolio_before == 0:
            portfolio_before = float(start_value)

        target_weights = alloc_df.loc[dt].fillna(0.0)
        num_trades = 0

        new_shares = {}
        new_values = {}
        cash_target = portfolio_before * float(target_weights.get("CASH", 0.0))

        for sym in traded_symbols:
            price = float(prices_row.get(sym, np.nan)) if sym in prices_row.index else np.nan
            prev_sh = float(shares.get(sym, 0.0))
            prev_val = float(prev_values.get(sym, 0.0))
            prev_w = (prev_val / portfolio_before) if portfolio_before > 0 else 0.0
            tgt_w = float(target_weights.get(sym, 0.0))
            tgt_val = portfolio_before * tgt_w

            if pd.isna(price) or price <= 0:
                ns = prev_sh
                nv = prev_val
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

        portfolio_after = float(sum(new_values.values()) + cash_target)
        summary_rows.append({
            "Date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
            "State": state_series.loc[dt],
            "Portfolio Value Before": portfolio_before,
            "Portfolio Value After": portfolio_after,
            "Cash Before": cash_balance,
            "Cash After": cash_target,
            "Number of Trades": num_trades,
        })

        shares = new_shares
        cash_balance = cash_target

    # final mark-to-market on end date
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
        "Portfolio Value Before": final_total,
        "Portfolio Value After": final_total,
        "Cash Before": cash_balance,
        "Cash After": cash_balance,
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
        for col in ["Portfolio Value Before", "Portfolio Value After", "Cash Before", "Cash After"]:
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

with st.sidebar:
    hover_mode = st.radio("Hover Display", ["Performance", "Holdings %"], index=0)
    strategy = st.selectbox("Strategy", ["COMBINED", "CORE", "ALPHA"], index=0)
    start_capital = st.number_input("Starting Capital", min_value=1000.0, value=50000.0, step=1000.0)
    contribution_amount = st.number_input("Contribution Amount", min_value=0.0, value=0.0, step=100.0)
    contribution_frequency = st.selectbox("Contribution Frequency", ["None", "Monthly", "Yearly"], index=0)
    sub_target_threshold = st.number_input("Sub-Target CAGR Threshold (%)", min_value=0.0, value=5.0, step=1.0)
    sort_order = st.selectbox("Allocation Change Log Order", ["Newest to Oldest", "Oldest to Newest"], index=0)
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

    trade_detail_df, trade_summary_df, final_positions_df = build_trade_reconstruction(strategy, df, start_capital)
    trade_detail_fmt, trade_summary_fmt, final_positions_fmt = format_trade_reconstruction(
        trade_detail_df, trade_summary_df, final_positions_df
    )

    eq_series, dd_series, total_contrib = equity_with_contributions(df["ret"], start_capital, contribution_amount, contribution_frequency)
    median_rec, avg_rec, max_rec = recovery_stats(dd_series)
    ulcer = ulcer_index(dd_series)
    roll3_best = float(rolling_3y.max()) if not rolling_3y.empty else 0.0
    roll3_worst = float(rolling_3y.min()) if not rolling_3y.empty else 0.0
    roll3_med = float(rolling_3y.median()) if not rolling_3y.empty else 0.0
    roll2_med = float(rolling_2y.median()) if not rolling_2y.empty else 0.0
    prob_neg = float((rolling_2y < 0).mean()) if not rolling_2y.empty else 0.0
    sub_target = sub_target_threshold / 100.0
    prob_sub_target = float((rolling_2y < sub_target).mean()) if not rolling_2y.empty else 0.0

    st.markdown('<div style="font-size:1.45rem; font-weight:700; margin:0 0 8px 0;">' + ("COMBINED Overview" if strategy == "COMBINED" else f"{strategy} Overview") + '</div>', unsafe_allow_html=True)

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
        a1.markdown(metric_card_html("CAGR", f"{perf['cagr']:.2%}", [f"{perf['days']:,} trading days", 'Calculated from df["ret"]']), unsafe_allow_html=True)
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
        "Shares are calculated as target dollars divided by that date's closing price. "
        "This is an audit overlay for the displayed allocations and final value.",
        "Uses actual close prices on rebalance dates",
        align="left"
    ), unsafe_allow_html=True)

    if not trade_summary_df.empty:
        reconstructed_final = float(final_positions_df["Market Value"].sum()) if not final_positions_df.empty else float("nan")
        backtest_final = float(eq_series.iloc[-1]) if not eq_series.empty else float("nan")
        diff = reconstructed_final - backtest_final if pd.notna(reconstructed_final) and pd.notna(backtest_final) else float("nan")

        x1, x2, x3 = st.columns(3)
        x1.markdown(metric_card_html("Reconstructed Final Value", f"${reconstructed_final:,.0f}", ["From reconstructed share counts", "Uses official close prices"]), unsafe_allow_html=True)
        x2.markdown(metric_card_html("Backtest Final Value", f"${backtest_final:,.0f}", ["From ADAPT return stream", "Current dashboard result"]), unsafe_allow_html=True)
        x3.markdown(metric_card_html("Reconstruction Difference", f"${diff:,.2f}", ["Reconstructed minus backtest", "Should generally be small"]), unsafe_allow_html=True)

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
    st.info("Run Backtest to load the polished dashboard.")
