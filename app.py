"""
B3 Market & Portfolio Pulse
---------------------------
A lightweight, modern Investment Analytics Dashboard for Brazilian B3 equities
and FIIs benchmarked against the Ibovespa (^BVSP), powered by FastAPI, SQLite,
yfinance, Pandas, and Plotly.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from backend.database import (
    B3_ASSETS,
    BENCHMARK_TICKER,
    DB_PATH,
    DEFAULT_TICKERS,
    TIMEFRAME_DAYS,
)
from backend.etl import (
    compute_portfolio_analytics as local_compute_portfolio_analytics,
    ensure_populated_db,
    generate_fallback_prices,
)


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="B3 Market & Portfolio Pulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL: str = os.getenv("B3_API_URL", "http://127.0.0.1:8000").rstrip("/")
API_TIMEOUT_SECONDS: float = 2.5

# Cohesive Fintech Dark Palette
COLOR_BG = "#0B0F19"
COLOR_CARD = "#111827"
COLOR_BORDER = "#1F2937"
COLOR_EMERALD = "#10B981"
COLOR_CYAN = "#06B6D4"
COLOR_ROSE = "#F43F5E"
COLOR_TEXT = "#F8FAFC"
COLOR_MUTED = "#94A3B8"


# ---------------------------------------------------------------------------
# Custom Dark Theme CSS
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Apply deep obsidian fintech styling and hide default Streamlit chrome."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

            /* Base canvas & typography */
            html, body, [class*="css"], .stApp {{
                background-color: {COLOR_BG};
                color: {COLOR_TEXT};
                font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            }}

            [data-testid="stAppViewContainer"] {{
                background-color: {COLOR_BG};
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            /* Hide default Streamlit menu, footer, and top accent bar */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            [data-testid="stDecoration"] {{display: none;}}

            /* Sidebar surface */
            section[data-testid="stSidebar"] {{
                background-color: #0E1422;
                border-right: 1px solid {COLOR_BORDER};
            }}

            section[data-testid="stSidebar"] .block-container {{
                padding-top: 1.75rem;
            }}

            /* Main container spacing */
            .main .block-container {{
                padding-top: 1.75rem;
                padding-bottom: 2.5rem;
                max-width: 1380px;
            }}

            /* Header block */
            .dashboard-header {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: flex-end;
                gap: 1rem;
                padding-bottom: 1.25rem;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid {COLOR_BORDER};
            }}

            .dashboard-title {{
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.025em;
                color: {COLOR_TEXT};
                margin: 0;
                line-height: 1.2;
            }}

            .dashboard-subtitle {{
                font-size: 0.92rem;
                color: {COLOR_MUTED};
                margin: 0.35rem 0 0 0;
            }}

            .timestamp-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: rgba(17, 24, 39, 0.85);
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 0.45rem 0.85rem;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.78rem;
                color: {COLOR_MUTED};
                white-space: nowrap;
            }}

            .timestamp-accent {{
                color: {COLOR_CYAN};
                font-weight: 600;
            }}

            /* Glassmorphic bordered KPI cards */
            [data-testid="stMetric"] {{
                background: linear-gradient(180deg, rgba(17, 24, 39, 0.92) 0%, rgba(15, 23, 42, 0.78) 100%);
                backdrop-filter: blur(10px);
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
                padding: 1.15rem 1.25rem;
                transition: border-color 0.2s ease, transform 0.2s ease;
            }}

            [data-testid="stMetric"]:hover {{
                border-color: #334155;
            }}

            [data-testid="stMetricLabel"] {{
                font-size: 0.80rem !important;
                font-weight: 500 !important;
                color: {COLOR_MUTED} !important;
                letter-spacing: 0.01em;
                margin-bottom: 0.35rem;
            }}

            [data-testid="stMetricValue"] {{
                font-family: 'JetBrains Mono', monospace !important;
                font-variant-numeric: tabular-nums;
                font-size: 1.65rem !important;
                font-weight: 600 !important;
                color: {COLOR_TEXT} !important;
                letter-spacing: -0.02em;
            }}

            [data-testid="stMetricDelta"] {{
                font-family: 'JetBrains Mono', monospace !important;
                font-variant-numeric: tabular-nums;
                font-size: 0.82rem !important;
                font-weight: 500 !important;
            }}

            /* Chart & Table Card Panels */
            .panel-header {{
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                margin-bottom: 0.35rem;
            }}

            .panel-title {{
                font-size: 1.02rem;
                font-weight: 600;
                color: {COLOR_TEXT};
                margin: 0;
            }}

            .panel-caption {{
                font-size: 0.80rem;
                color: {COLOR_MUTED};
                margin: 0.15rem 0 0.75rem 0;
            }}

            /* Styled radio buttons for Timeframe segmented selector */
            div[role="radiogroup"] {{
                display: flex;
                gap: 0.35rem;
                background: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 0.25rem;
            }}

            div[role="radiogroup"] label {{
                flex: 1;
                justify-content: center;
                background: transparent;
                border-radius: 7px;
                padding: 0.35rem 0.5rem !important;
                margin: 0 !important;
                transition: background-color 0.15s ease;
            }}

            div[role="radiogroup"] label:has(input:checked) {{
                background: rgba(6, 182, 212, 0.16);
                border: 1px solid rgba(6, 182, 212, 0.45);
            }}

            div[role="radiogroup"] label > div:first-child {{
                display: none;
            }}

            div[role="radiogroup"] label p {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.82rem;
                font-weight: 600;
                color: {COLOR_TEXT};
            }}

            /* Download CSV button */
            .stDownloadButton button {{
                background: rgba(16, 185, 129, 0.12) !important;
                color: {COLOR_EMERALD} !important;
                border: 1px solid rgba(16, 185, 129, 0.38) !important;
                border-radius: 9px !important;
                font-weight: 600 !important;
                font-size: 0.84rem !important;
                padding: 0.45rem 1rem !important;
                transition: all 0.2s ease !important;
            }}

            .stDownloadButton button:hover {{
                background: rgba(16, 185, 129, 0.22) !important;
                border-color: {COLOR_EMERALD} !important;
            }}

            /* Clean divider */
            .section-spacer {{
                height: 1.25rem;
            }}

            /* Dataframe container border */
            [data-testid="stDataFrame"] {{
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
                overflow: hidden;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data Layer (FastAPI Client -> Local SQLite/ETL Fallback -> Synthetic Fallback)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_b3_market_data() -> Tuple[pd.DataFrame, str, bool]:
    """
    Fetch 1 year of daily closing prices for B3 tickers and Ibovespa (^BVSP).

    Resolution order:
      1. FastAPI REST endpoint (`GET /api/v1/prices`)
      2. Local SQLite database (`data/market_data.db`), auto-running `backend.etl` if empty
      3. In-memory deterministic synthetic baseline (`generate_fallback_prices`)

    Returns:
        Tuple of (prices_df, updated_at_label, is_fallback)
    """
    # 1. Try FastAPI REST Service
    try:
        resp = requests.get(
            f"{API_BASE_URL}/api/v1/prices",
            timeout=API_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            payload = resp.json()
            prices_df = pd.DataFrame(
                payload["prices"],
                index=pd.to_datetime(payload["dates"]),
            )
            prices_df.index.name = "date"
            if not prices_df.empty and BENCHMARK_TICKER in prices_df.columns:
                return (
                    prices_df,
                    str(payload.get("updated_at", "")),
                    bool(payload.get("is_fallback", False)),
                )
    except Exception:
        pass

    # 2. Graceful Fallback to Local SQLite + ETL Pipeline
    try:
        prices_df, updated_at, is_fallback = ensure_populated_db(db_path=DB_PATH)
        if not prices_df.empty and BENCHMARK_TICKER in prices_df.columns:
            return prices_df, updated_at, is_fallback
    except Exception:
        pass

    # 3. Last-resort In-Memory Synthetic Fallback
    fallback_df = generate_fallback_prices()
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M BRT")
    return fallback_df, updated_at, True


def compute_portfolio_analytics(
    full_prices: pd.DataFrame,
    selected_tickers: List[str],
    timeframe: str,
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Compute normalized series, risk/return KPIs, and per-asset summary metrics.

    Attempts to delegate calculation to the FastAPI backend
    (`POST /api/v1/portfolio/analytics`) and falls back seamlessly to the shared
    local `backend.etl.compute_portfolio_analytics` engine.
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/portfolio/analytics",
            json={
                "selected_tickers": selected_tickers,
                "timeframe": timeframe,
                "initial_capital": float(initial_capital),
            },
            timeout=API_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            data = resp.json()
            dates_idx = pd.to_datetime(data["dates"])
            portfolio_norm = pd.Series(data["portfolio_norm"], index=dates_idx)
            ibov_norm = pd.Series(data["ibov_norm"], index=dates_idx)

            rows = [
                {
                    "Ticker": item["ticker"],
                    "Asset": item["asset"],
                    "Weight (%)": item["weight_pct"],
                    "Current Price (R$)": item["current_price"],
                    "Period Return (%)": item["period_return_pct"],
                    "20D Moving Avg (R$)": item["ma_20"],
                    "Volatility (%)": item["volatility_pct"],
                }
                for item in data["summary"]
            ]
            summary_df = pd.DataFrame(rows)

            return {
                "dates": dates_idx,
                "portfolio_norm": portfolio_norm,
                "ibov_norm": ibov_norm,
                "portfolio_return_pct": float(data["portfolio_return_pct"]),
                "ibov_return_pct": float(data["ibov_return_pct"]),
                "alpha_pct": float(data["alpha_pct"]),
                "final_value": float(data["final_value"]),
                "net_pnl": float(data["net_pnl"]),
                "port_vol_pct": float(data["port_vol_pct"]),
                "ibov_vol_pct": float(data["ibov_vol_pct"]),
                "summary_df": summary_df,
            }
    except Exception:
        pass

    return local_compute_portfolio_analytics(
        full_prices=full_prices,
        selected_tickers=selected_tickers,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )


# ---------------------------------------------------------------------------
# Plotly Visualizations
# ---------------------------------------------------------------------------

def build_performance_chart(
    dates: pd.DatetimeIndex,
    portfolio_norm: pd.Series,
    ibov_norm: pd.Series,
) -> go.Figure:
    """Create normalized Base-100 Portfolio vs. Ibovespa performance chart."""
    y_min = float(min(portfolio_norm.min(), ibov_norm.min(), 100.0))
    y_max = float(max(portfolio_norm.max(), ibov_norm.max(), 100.0))
    pad = max((y_max - y_min) * 0.14, 1.5)
    floor_val = y_min - pad

    fig = go.Figure()

    # Invisible baseline trace for clean bounded area fill
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[floor_val] * len(dates),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Portfolio Curve (Emerald with subtle translucent area fill)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_norm.round(2),
            name="Equal-Weight Portfolio",
            mode="lines",
            line=dict(color=COLOR_EMERALD, width=2.7, shape="spline", smoothing=0.55),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.11)",
            hovertemplate="<b>Portfolio</b>: %{y:.2f}<extra></extra>",
        )
    )

    # Benchmark Curve (Cyan)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ibov_norm.round(2),
            name="Ibovespa (^BVSP)",
            mode="lines",
            line=dict(color=COLOR_CYAN, width=2.1, shape="spline", smoothing=0.55),
            hovertemplate="<b>Ibovespa</b>: %{y:.2f}<extra></extra>",
        )
    )

    # Base 100 reference line
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="#334155",
        line_width=1.2,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=370,
        margin=dict(l=8, r=12, t=12, b=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=COLOR_CARD,
            bordercolor=COLOR_BORDER,
            font=dict(family="JetBrains Mono, monospace", size=12, color=COLOR_TEXT),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12, color=COLOR_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(size=11, color=COLOR_MUTED, family="JetBrains Mono, monospace"),
            linecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            range=[floor_val, y_max + pad * 0.7],
            gridcolor="rgba(31, 41, 55, 0.7)",
            zeroline=False,
            tickfont=dict(size=11, color=COLOR_MUTED, family="JetBrains Mono, monospace"),
            ticksuffix=" ",
        ),
    )
    return fig


def build_allocation_return_chart(summary_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart showing per-asset return and portfolio weight."""
    df_plot = summary_df.sort_values(by="Period Return (%)", ascending=True).copy()

    colors = [
        COLOR_EMERALD if val >= 0 else COLOR_ROSE
        for val in df_plot["Period Return (%)"]
    ]

    labels = [
        f"{ret:+.2f}%  ·  {wt:.0f}% wt"
        for ret, wt in zip(df_plot["Period Return (%)"], df_plot["Weight (%)"])
    ]

    fig = go.Figure(
        go.Bar(
            x=df_plot["Period Return (%)"],
            y=df_plot["Ticker"].str.replace(".SA", "", regex=False),
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=labels,
            textposition="auto",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color=COLOR_TEXT),
            customdata=np.stack([df_plot["Asset"], df_plot["Weight (%)"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b> (%{customdata[0]})<br>"
                "Period Return: %{x:+.2f}%<br>"
                "Allocation Weight: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
    )

    fig.add_vline(x=0, line_color="#334155", line_width=1.2)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=370,
        margin=dict(l=8, r=16, t=12, b=8),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=COLOR_CARD,
            bordercolor=COLOR_BORDER,
            font=dict(family="JetBrains Mono, monospace", size=12, color=COLOR_TEXT),
        ),
        xaxis=dict(
            gridcolor="rgba(31, 41, 55, 0.7)",
            zeroline=False,
            ticksuffix="%",
            tickfont=dict(size=11, color=COLOR_MUTED, family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=12, color=COLOR_TEXT, family="JetBrains Mono, monospace"),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Main Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    inject_custom_css()

    # Load cached B3 & Ibovespa data (FastAPI -> SQLite/ETL -> Synthetic Fallback)
    full_prices, updated_at, is_fallback = fetch_b3_market_data()
    available_tickers = [t for t in B3_ASSETS if t in full_prices.columns]
    default_selection = [t for t in DEFAULT_TICKERS if t in available_tickers] or available_tickers[:3]

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.markdown("### Portfolio Controls")
        st.caption("Configure your B3 basket, horizon, and capital simulation.")

        selected_tickers = st.multiselect(
            "Select B3 Assets",
            options=available_tickers,
            default=default_selection,
            format_func=lambda t: f"{t.replace('.SA', '')} · {B3_ASSETS.get(t, t)}",
            help="Equal-weight allocation is automatically distributed across selected assets.",
        )

        st.markdown("##### Timeframe")
        timeframe = st.radio(
            "Timeframe",
            options=list(TIMEFRAME_DAYS.keys()),
            index=3,  # Default to 1Y
            horizontal=True,
            label_visibility="collapsed",
        )

        initial_capital = st.number_input(
            "Initial Investment (R$)",
            min_value=100.0,
            max_value=100_000_000.0,
            value=10_000.0,
            step=1_000.0,
            format="%.2f",
        )

        st.divider()
        weight_info = f"{100.0 / len(selected_tickers):.1f}% per asset" if selected_tickers else "Select at least 1 asset"
        st.caption(f"Allocation Model: **Equal-Weight** ({weight_info})")
        st.caption("Benchmark: **Ibovespa (^BVSP)**")

    # ---------------- Header ----------------
    source_label = "Simulated Baseline" if is_fallback else "Yahoo Finance · B3"
    st.markdown(
        f"""
        <div class="dashboard-header">
            <div>
                <h1 class="dashboard-title">B3 Market & Portfolio Pulse</h1>
                <p class="dashboard-subtitle">
                    Brazilian equities & FII performance analytics benchmarked against Ibovespa (^BVSP)
                </p>
            </div>
            <div class="timestamp-badge">
                <span class="timestamp-accent">{source_label}</span>
                <span>·</span>
                <span>Updated {updated_at}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_fallback:
        st.warning(
            "Live Yahoo Finance data is temporarily unreachable. Displaying calibrated offline baseline data.",
            icon="⚠️",
        )

    if not selected_tickers:
        st.info("Select at least one B3 asset in the sidebar to render portfolio analytics.")
        return

    # ---------------- Analytics Computation ----------------
    metrics = compute_portfolio_analytics(
        full_prices=full_prices,
        selected_tickers=selected_tickers,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )

    # ---------------- Top Row: 4 Styled KPI Cards ----------------
    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        st.metric(
            label=f"Total Portfolio Return ({timeframe})",
            value=f"{metrics['portfolio_return_pct']:+.2f}%",
            delta=f"{metrics['net_pnl']:+,.2f} BRL",
        )

    with k2:
        st.metric(
            label="Simulated Final Value (R$)",
            value=f"R$ {metrics['final_value']:,.2f}",
            delta=f"From R$ {initial_capital:,.2f}",
            delta_color="off",
        )

    with k3:
        vol_spread = metrics["port_vol_pct"] - metrics["ibov_vol_pct"]
        st.metric(
            label="Annualized Volatility",
            value=f"{metrics['port_vol_pct']:.2f}%",
            delta=f"{vol_spread:+.2f}% vs IBOV",
            delta_color="off",
        )

    with k4:
        st.metric(
            label="Alpha vs. Ibovespa",
            value=f"{metrics['alpha_pct']:+.2f}%",
            delta=f"IBOV {metrics['ibov_return_pct']:+.2f}%",
            delta_color="normal",
        )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # ---------------- Middle Row: 2 Interactive Plotly Charts ----------------
    chart_left, chart_right = st.columns([1.65, 1], gap="medium")

    with chart_left:
        st.markdown(
            f"""
            <div class="panel-header">
                <h3 class="panel-title">Normalized Performance Comparison</h3>
            </div>
            <p class="panel-caption">Equal-weight portfolio vs. Ibovespa (^BVSP) rebased to 100 over {timeframe}</p>
            """,
            unsafe_allow_html=True,
        )
        perf_fig = build_performance_chart(
            dates=metrics["dates"],
            portfolio_norm=metrics["portfolio_norm"],
            ibov_norm=metrics["ibov_norm"],
        )
        st.plotly_chart(perf_fig, use_container_width=True, config={"displayModeBar": False})

    with chart_right:
        st.markdown(
            f"""
            <div class="panel-header">
                <h3 class="panel-title">Asset Allocation & Return Breakdown</h3>
            </div>
            <p class="panel-caption">Individual {timeframe} total return (%) and equal-weight share</p>
            """,
            unsafe_allow_html=True,
        )
        bar_fig = build_allocation_return_chart(metrics["summary_df"])
        st.plotly_chart(bar_fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    # ---------------- Bottom Row: Summary Table & CSV Export ----------------
    summary_df: pd.DataFrame = metrics["summary_df"]

    tbl_header_col, export_col = st.columns([4, 1], vertical_alignment="bottom")
    with tbl_header_col:
        st.markdown(
            f"""
            <h3 class="panel-title">Portfolio Asset Summary ({timeframe})</h3>
            <p class="panel-caption" style="margin-bottom: 0.5rem;">
                Latest closing prices, moving averages, period returns, and annualized volatility
            </p>
            """,
            unsafe_allow_html=True,
        )

    with export_col:
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"b3_portfolio_summary_{timeframe.lower()}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Asset": st.column_config.TextColumn("Asset Name", width="medium"),
            "Weight (%)": st.column_config.NumberColumn("Weight", format="%.1f%%"),
            "Current Price (R$)": st.column_config.NumberColumn("Current Price", format="R$ %.2f"),
            "Period Return (%)": st.column_config.NumberColumn(f"Return ({timeframe})", format="%+.2f%%"),
            "20D Moving Avg (R$)": st.column_config.NumberColumn("20D Moving Avg", format="R$ %.2f"),
            "Volatility (%)": st.column_config.NumberColumn("Annualized Vol", format="%.2f%%"),
        },
    )


if __name__ == "__main__":
    main()
