"""
Automated ETL Pipeline for B3 Equities, FIIs, and Ibovespa (^BVSP)
------------------------------------------------------------------
Implements the Extract -> Transform -> Load workflow:
  1. Extract: Fetch 1Y daily closing prices from Yahoo Finance (`yfinance`)
     with deterministic fallback synthesis for offline resilience.
  2. Transform: Clean time-series, compute daily returns, 20-day moving averages,
     rolling 20-day annualized volatility, and multi-horizon asset & portfolio KPIs.
  3. Load: Upsert normalized records into SQLite (`data/market_data.db`) and
     export flat CSV artifacts (`data/historical_prices.csv`, `data/asset_metrics.csv`).
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from backend.database import (
    ALL_TICKERS,
    B3_ASSETS,
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    DB_PATH,
    METRICS_CSV_PATH,
    PRICES_CSV_PATH,
    TIMEFRAME_DAYS,
    ensure_data_dir,
    load_prices_pivot,
    upsert_market_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extract Layer
# ---------------------------------------------------------------------------

def generate_fallback_prices(
    end_date: Optional[pd.Timestamp] = None,
    periods: int = 252,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate deterministic synthetic 1-year daily closing prices for offline
    resilience when Yahoo Finance is unreachable or rate-limited.

    Args:
        end_date: Final business date for the series (defaults to today).
        periods: Number of business days to synthesize (default 252 trading days).
        seed: Random number generator seed for reproducibility.

    Returns:
        Wide DataFrame indexed by business date with B3 tickers and ^BVSP as columns.
    """
    end_dt = (end_date or pd.Timestamp.today()).normalize()
    dates = pd.bdate_range(end=end_dt, periods=periods)
    rng = np.random.default_rng(seed=seed)

    specs: Dict[str, Tuple[float, float, float]] = {
        "PETR4.SA": (34.50, 0.0007, 0.018),
        "VALE3.SA": (63.20, 0.0002, 0.017),
        "ITUB4.SA": (31.40, 0.0006, 0.013),
        "WEGE3.SA": (38.90, 0.0009, 0.016),
        "MXRF11.SA": (10.15, 0.0003, 0.006),
        "^BVSP": (124500.0, 0.0004, 0.011),
    }

    series_map: Dict[str, np.ndarray] = {}
    for ticker, (start_price, drift, vol) in specs.items():
        shocks = rng.normal(loc=drift, scale=vol, size=len(dates))
        path = start_price * np.exp(np.cumsum(shocks))
        series_map[ticker] = np.round(path, 2)

    df = pd.DataFrame(series_map, index=dates)
    df.index.name = "date"
    return df


def extract_close_matrix(raw_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Normalize raw `yfinance.download` output (single-level or MultiIndex columns)
    into a clean Date x Ticker DataFrame of adjusted closing prices.

    Args:
        raw_df: Raw DataFrame returned by `yfinance.download`.
        tickers: Expected ticker symbols.

    Returns:
        Cleaned wide DataFrame of closing prices sorted chronologically.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    if isinstance(raw_df.columns, pd.MultiIndex):
        level_0 = raw_df.columns.get_level_values(0)
        if "Close" in level_0:
            close_df = raw_df["Close"].copy()
        elif "Adj Close" in level_0:
            close_df = raw_df["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        col_name = "Close" if "Close" in raw_df.columns else "Adj Close"
        if col_name not in raw_df.columns:
            return pd.DataFrame()
        close_df = raw_df[[col_name]].copy()
        close_df.columns = [tickers[0]]

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame()

    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
    close_df.index.name = "date"
    close_df = close_df.sort_index().ffill().dropna(how="all")
    return close_df


def extract_market_data(
    tickers: Optional[List[str]] = None,
    period: str = "1y",
    interval: str = "1d",
    force_fallback: bool = False,
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Extract 1 year of daily closing prices from Yahoo Finance with automatic
    fallback to deterministic synthetic prices if the upstream feed fails.

    Args:
        tickers: List of ticker symbols to fetch. Defaults to ALL_TICKERS.
        period: Lookback window passed to `yfinance.download` (default '1y').
        interval: Sampling interval (default '1d').
        force_fallback: If True, skip network call and generate synthetic baseline.

    Returns:
        Tuple of (wide_prices_df, updated_at_label, is_fallback).
    """
    target_tickers = tickers or ALL_TICKERS
    brt_now = datetime.now(timezone(timedelta(hours=-3)))
    updated_at = brt_now.strftime("%Y-%m-%d %H:%M BRT")

    if force_fallback:
        logger.info("Using deterministic synthetic fallback dataset (force_fallback=True).")
        return generate_fallback_prices(), updated_at, True

    try:
        logger.info("Downloading market data from Yahoo Finance for tickers: %s", target_tickers)
        raw = yf.download(
            tickers=target_tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        prices = extract_close_matrix(raw, target_tickers)

        valid_cols = [
            col
            for col in target_tickers
            if col in prices.columns and prices[col].dropna().shape[0] >= 30
        ]
        if BENCHMARK_TICKER not in valid_cols or len(valid_cols) < 3:
            raise ValueError("Incomplete market feed returned by yfinance.")

        prices = prices[valid_cols].ffill().bfill()
        prices.index.name = "date"
        return prices, updated_at, False

    except Exception as exc:
        logger.warning("Yahoo Finance extraction failed (%s). Falling back to synthetic dataset.", exc)
        fallback_df = generate_fallback_prices()
        return fallback_df, updated_at, True


# ---------------------------------------------------------------------------
# Transform Layer
# ---------------------------------------------------------------------------

def transform_prices_to_long(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a wide Date x Ticker price matrix into a normalized long-format
    DataFrame enriched with daily returns, 20-day moving average, and 20-day
    annualized volatility.

    Args:
        wide_prices: Wide DataFrame indexed by `DatetimeIndex` with ticker columns.

    Returns:
        Normalized DataFrame with columns:
        `[date, ticker, close_price, daily_return, ma_20, volatility_20d]`.
    """
    if wide_prices.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "close_price", "daily_return", "ma_20", "volatility_20d"]
        )

    cleaned = wide_prices.sort_index().ffill().bfill()
    records: List[pd.DataFrame] = []

    for ticker in cleaned.columns:
        series = cleaned[ticker].dropna()
        if series.empty:
            continue

        daily_ret = series.pct_change()
        ma_20 = series.rolling(window=20, min_periods=1).mean()
        vol_20d = daily_ret.rolling(window=20, min_periods=2).std() * np.sqrt(252) * 100.0

        ticker_df = pd.DataFrame(
            {
                "date": pd.to_datetime(series.index).strftime("%Y-%m-%d"),
                "ticker": ticker,
                "close_price": series.round(4).values,
                "daily_return": daily_ret.round(6).values,
                "ma_20": ma_20.round(4).values,
                "volatility_20d": vol_20d.round(4).values,
            }
        )
        records.append(ticker_df)

    if not records:
        return pd.DataFrame(
            columns=["date", "ticker", "close_price", "daily_return", "ma_20", "volatility_20d"]
        )

    long_df = pd.concat(records, ignore_index=True)
    return long_df.sort_values(by=["date", "ticker"]).reset_index(drop=True)


def transform_asset_metrics(wide_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-aggregated financial KPIs per ticker across all supported
    timeframes ('1M', '3M', '6M', '1Y').

    Args:
        wide_prices: Wide DataFrame indexed by `DatetimeIndex` with ticker columns.

    Returns:
        DataFrame with columns:
        `[ticker, timeframe, asset_name, current_price, period_return_pct,
          ma_20, annualized_volatility_pct, min_price, max_price, observations]`.
    """
    if wide_prices.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "timeframe",
                "asset_name",
                "current_price",
                "period_return_pct",
                "ma_20",
                "annualized_volatility_pct",
                "min_price",
                "max_price",
                "observations",
            ]
        )

    cleaned = wide_prices.sort_index().ffill().bfill()
    latest_date = pd.to_datetime(cleaned.index.max())
    ma20_latest = cleaned.rolling(window=20, min_periods=1).mean().iloc[-1]

    rows: List[Dict[str, Any]] = []
    for timeframe, days in TIMEFRAME_DAYS.items():
        cutoff_date = latest_date - pd.Timedelta(days=days)
        sliced = cleaned.loc[cleaned.index >= cutoff_date]
        if len(sliced) < 2:
            sliced = cleaned.tail(2)

        for ticker in cleaned.columns:
            series = sliced[ticker].dropna()
            if len(series) < 2:
                continue

            daily_ret = series.pct_change().dropna()
            period_ret_pct = float((series.iloc[-1] / series.iloc[0] - 1.0) * 100.0)
            ann_vol_pct = (
                float(daily_ret.std() * np.sqrt(252) * 100.0) if len(daily_ret) > 1 else 0.0
            )
            asset_name = (
                BENCHMARK_NAME
                if ticker == BENCHMARK_TICKER
                else B3_ASSETS.get(ticker, ticker)
            )

            rows.append(
                {
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "asset_name": asset_name,
                    "current_price": round(float(series.iloc[-1]), 2),
                    "period_return_pct": round(period_ret_pct, 2),
                    "ma_20": round(float(ma20_latest[ticker]), 2),
                    "annualized_volatility_pct": round(ann_vol_pct, 2),
                    "min_price": round(float(series.min()), 2),
                    "max_price": round(float(series.max()), 2),
                    "observations": int(len(series)),
                }
            )

    return pd.DataFrame(rows)


def compute_portfolio_analytics(
    full_prices: pd.DataFrame,
    selected_tickers: List[str],
    timeframe: str = "1Y",
    initial_capital: float = 10_000.0,
) -> Dict[str, Any]:
    """
    Compute equal-weight portfolio performance, benchmark comparison (Ibovespa),
    risk/return KPIs, and per-asset breakdown for the selected timeframe.

    Args:
        full_prices: 1-year wide DataFrame of closing prices including `^BVSP`.
        selected_tickers: List of B3 asset tickers included in the equal-weight basket.
        timeframe: Horizon key ('1M', '3M', '6M', or '1Y').
        initial_capital: Simulated starting investment in BRL (R$).

    Returns:
        Dictionary containing normalized time-series, portfolio KPIs, and `summary_df`.
    """
    if timeframe not in TIMEFRAME_DAYS:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from {list(TIMEFRAME_DAYS)}.")

    valid_tickers = [t for t in selected_tickers if t in full_prices.columns]
    if not valid_tickers:
        raise ValueError("None of the selected tickers are present in the price dataset.")

    if BENCHMARK_TICKER not in full_prices.columns:
        raise ValueError(f"Benchmark ticker '{BENCHMARK_TICKER}' is missing from price dataset.")

    days = TIMEFRAME_DAYS[timeframe]
    latest_date = pd.to_datetime(full_prices.index.max())
    cutoff_date = latest_date - pd.Timedelta(days=days)

    sliced = full_prices.loc[full_prices.index >= cutoff_date].copy()
    if len(sliced) < 2:
        sliced = full_prices.tail(2).copy()

    # 20-day moving average computed on full 1Y series so short windows stay accurate
    ma20_series = full_prices[valid_tickers].rolling(window=20, min_periods=1).mean().iloc[-1]

    asset_prices = sliced[valid_tickers]
    ibov_prices = sliced[BENCHMARK_TICKER]

    # Equal-weight normalized curves (Base = 100 at start of selected timeframe)
    norm_assets = (asset_prices / asset_prices.iloc[0]) * 100.0
    portfolio_norm = norm_assets.mean(axis=1)
    ibov_norm = (ibov_prices / ibov_prices.iloc[0]) * 100.0

    # Portfolio & benchmark period returns (%)
    portfolio_return_pct = float(portfolio_norm.iloc[-1] - 100.0)
    ibov_return_pct = float(ibov_norm.iloc[-1] - 100.0)
    alpha_pct = portfolio_return_pct - ibov_return_pct

    final_value = float(initial_capital * (1.0 + portfolio_return_pct / 100.0))
    net_pnl = float(final_value - initial_capital)

    # Annualized volatility (252 trading days)
    port_daily_ret = portfolio_norm.pct_change().dropna()
    ibov_daily_ret = ibov_norm.pct_change().dropna()

    port_vol_pct = (
        float(port_daily_ret.std() * np.sqrt(252) * 100.0) if len(port_daily_ret) > 1 else 0.0
    )
    ibov_vol_pct = (
        float(ibov_daily_ret.std() * np.sqrt(252) * 100.0) if len(ibov_daily_ret) > 1 else 0.0
    )

    # Asset-level breakdown table
    weight_pct = 100.0 / len(valid_tickers)
    rows: List[Dict[str, Any]] = []
    for ticker in valid_tickers:
        s = asset_prices[ticker]
        daily_ret = s.pct_change().dropna()
        asset_ret = float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0)
        asset_vol = float(daily_ret.std() * np.sqrt(252) * 100.0) if len(daily_ret) > 1 else 0.0

        rows.append(
            {
                "Ticker": ticker,
                "Asset": B3_ASSETS.get(ticker, ticker),
                "Weight (%)": round(weight_pct, 1),
                "Current Price (R$)": round(float(s.iloc[-1]), 2),
                "Period Return (%)": round(asset_ret, 2),
                "20D Moving Avg (R$)": round(float(ma20_series[ticker]), 2),
                "Volatility (%)": round(asset_vol, 2),
            }
        )

    summary_df = (
        pd.DataFrame(rows)
        .sort_values(by="Period Return (%)", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "dates": sliced.index,
        "portfolio_norm": portfolio_norm,
        "ibov_norm": ibov_norm,
        "portfolio_return_pct": portfolio_return_pct,
        "ibov_return_pct": ibov_return_pct,
        "alpha_pct": alpha_pct,
        "final_value": final_value,
        "net_pnl": net_pnl,
        "port_vol_pct": port_vol_pct,
        "ibov_vol_pct": ibov_vol_pct,
        "summary_df": summary_df,
    }


# ---------------------------------------------------------------------------
# Load Layer & Pipeline Orchestration
# ---------------------------------------------------------------------------

def load_to_storage(
    prices_long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    updated_at: str,
    is_fallback: bool,
    db_path: Path = DB_PATH,
    prices_csv_path: Optional[Path] = PRICES_CSV_PATH,
    metrics_csv_path: Optional[Path] = METRICS_CSV_PATH,
) -> None:
    """
    Persist transformed market data to SQLite and export CSV snapshots.

    Args:
        prices_long_df: Long-format daily price and technical indicator records.
        metrics_df: Multi-timeframe asset summary metrics.
        updated_at: Timestamp label of the ETL run.
        is_fallback: Flag indicating whether synthetic data was used.
        db_path: Target SQLite database file path.
        prices_csv_path: Optional CSV output path for historical prices.
        metrics_csv_path: Optional CSV output path for asset metrics.
    """
    ensure_data_dir(db_path)
    upsert_market_data(
        prices_long_df=prices_long_df,
        metrics_df=metrics_df,
        updated_at=updated_at,
        is_fallback=is_fallback,
        db_path=db_path,
    )

    if prices_csv_path is not None:
        prices_csv_path.parent.mkdir(parents=True, exist_ok=True)
        prices_long_df.to_csv(prices_csv_path, index=False)

    if metrics_csv_path is not None:
        metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_csv_path, index=False)


def run_etl_pipeline(
    db_path: Path = DB_PATH,
    prices_csv_path: Optional[Path] = PRICES_CSV_PATH,
    metrics_csv_path: Optional[Path] = METRICS_CSV_PATH,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """
    Execute the end-to-end Extract -> Transform -> Load pipeline.

    Args:
        db_path: Destination SQLite database file path.
        prices_csv_path: Destination CSV path for historical prices.
        metrics_csv_path: Destination CSV path for pre-computed asset metrics.
        force_fallback: If True, use deterministic offline synthetic prices.

    Returns:
        Summary dictionary describing the ETL run outcome.
    """
    wide_prices, updated_at, is_fallback = extract_market_data(force_fallback=force_fallback)
    prices_long_df = transform_prices_to_long(wide_prices)
    metrics_df = transform_asset_metrics(wide_prices)

    load_to_storage(
        prices_long_df=prices_long_df,
        metrics_df=metrics_df,
        updated_at=updated_at,
        is_fallback=is_fallback,
        db_path=db_path,
        prices_csv_path=prices_csv_path,
        metrics_csv_path=metrics_csv_path,
    )

    result = {
        "status": "success",
        "updated_at": updated_at,
        "is_fallback": is_fallback,
        "tickers_processed": list(wide_prices.columns),
        "price_rows": len(prices_long_df),
        "metric_rows": len(metrics_df),
        "db_path": str(db_path),
    }
    logger.info("ETL pipeline completed successfully: %s", result)
    return result


def ensure_populated_db(db_path: Path = DB_PATH) -> Tuple[pd.DataFrame, str, bool]:
    """
    Load the price matrix from SQLite, automatically triggering the ETL pipeline
    if the database does not exist yet or contains no records.
    """
    wide_df, updated_at, is_fallback = load_prices_pivot(db_path=db_path)
    if wide_df.empty or BENCHMARK_TICKER not in wide_df.columns:
        logger.info("SQLite database empty or missing; bootstrapping via ETL pipeline...")
        run_etl_pipeline(db_path=db_path)
        wide_df, updated_at, is_fallback = load_prices_pivot(db_path=db_path)
    return wide_df, updated_at, is_fallback


def main() -> None:
    """CLI entrypoint for running the B3 ETL pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run the B3 Market & Portfolio Pulse ETL pipeline."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use deterministic synthetic data without querying Yahoo Finance.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Custom path for the SQLite database file.",
    )
    args = parser.parse_args()

    summary = run_etl_pipeline(
        db_path=args.db_path,
        force_fallback=args.offline,
    )
    print(
        f"[ETL COMPLETE] Updated: {summary['updated_at']} | "
        f"Fallback: {summary['is_fallback']} | "
        f"Price Rows: {summary['price_rows']} | "
        f"Metric Rows: {summary['metric_rows']} | "
        f"DB: {summary['db_path']}"
    )


if __name__ == "__main__":
    main()
