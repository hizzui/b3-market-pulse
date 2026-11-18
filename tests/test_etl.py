"""
Unit & Integration Test Suite for B3 Market & Portfolio Pulse Backend
---------------------------------------------------------------------
Covers:
  1. Pandas ETL transformations (`transform_prices_to_long`, `transform_asset_metrics`,
     `compute_portfolio_analytics`, `extract_close_matrix`).
  2. SQLite database schema persistence & queries (`init_db`, `upsert_market_data`,
     `load_prices_pivot`, `query_asset_metrics`).
  3. FastAPI REST endpoints (`/api/v1/health`, `/api/v1/assets`, `/api/v1/prices`,
     `/api/v1/metrics`, `/api/v1/portfolio/analytics`, `/api/v1/etl/refresh`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.database import (
    ALL_TICKERS,
    B3_ASSETS,
    BENCHMARK_TICKER,
    DEFAULT_TICKERS,
    TIMEFRAME_DAYS,
    get_etl_status,
    init_db,
    load_prices_pivot,
    query_asset_metrics,
)
from backend.etl import (
    compute_portfolio_analytics,
    extract_close_matrix,
    generate_fallback_prices,
    run_etl_pipeline,
    transform_asset_metrics,
    transform_prices_to_long,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_wide_prices() -> pd.DataFrame:
    """Return a deterministic 252-day wide DataFrame of B3 and Ibovespa prices."""
    return generate_fallback_prices(
        end_date=pd.Timestamp("2026-11-18"),
        periods=252,
        seed=42,
    )


@pytest.fixture()
def populated_temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Create and populate an isolated SQLite database in `tmp_path` using the offline
    ETL pipeline, and configure `B3_DB_PATH` so FastAPI uses it.
    """
    db_file = tmp_path / "test_market_data.db"
    prices_csv = tmp_path / "historical_prices.csv"
    metrics_csv = tmp_path / "asset_metrics.csv"

    monkeypatch.setenv("B3_DB_PATH", str(db_file))
    run_etl_pipeline(
        db_path=db_file,
        prices_csv_path=prices_csv,
        metrics_csv_path=metrics_csv,
        force_fallback=True,
    )
    return db_file


# ---------------------------------------------------------------------------
# 1. Pandas Transformation Unit Tests
# ---------------------------------------------------------------------------

def test_generate_fallback_prices_shape_and_tickers(sample_wide_prices: pd.DataFrame) -> None:
    """Verify synthetic fallback generator produces 252 business days across all 6 tickers."""
    assert sample_wide_prices.shape == (252, len(ALL_TICKERS))
    for ticker in ALL_TICKERS:
        assert ticker in sample_wide_prices.columns
    assert not sample_wide_prices.isna().any().any()
    assert (sample_wide_prices > 0).all().all()


def test_extract_close_matrix_multiindex() -> None:
    """Verify MultiIndex DataFrame from yfinance is normalized into a clean Date x Ticker table."""
    dates = pd.bdate_range("2026-01-01", periods=5)
    arrays = [
        ["Close", "Close", "Volume", "Volume"],
        ["PETR4.SA", "VALE3.SA", "PETR4.SA", "VALE3.SA"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays)
    data = np.array(
        [
            [35.0, 62.0, 1000, 2000],
            [35.5, 62.5, 1100, 2100],
            [36.0, 61.8, 1200, 1900],
            [35.8, 63.0, 1050, 2050],
            [36.4, 63.5, 1300, 2200],
        ]
    )
    raw_df = pd.DataFrame(data, index=dates, columns=columns)

    close_df = extract_close_matrix(raw_df, ["PETR4.SA", "VALE3.SA"])
    assert list(close_df.columns) == ["PETR4.SA", "VALE3.SA"]
    assert len(close_df) == 5
    assert close_df.loc[dates[-1], "PETR4.SA"] == pytest.approx(36.4)


def test_transform_prices_to_long_indicators(sample_wide_prices: pd.DataFrame) -> None:
    """Verify long-format price transformation computes daily returns, 20D MA, and volatility."""
    long_df = transform_prices_to_long(sample_wide_prices)

    expected_cols = {
        "date",
        "ticker",
        "close_price",
        "daily_return",
        "ma_20",
        "volatility_20d",
    }
    assert set(long_df.columns) == expected_cols
    assert len(long_df) == 252 * len(ALL_TICKERS)

    petr = long_df[long_df["ticker"] == "PETR4.SA"].reset_index(drop=True)
    # First observation has NaN daily return, subsequent observations have valid returns
    assert pd.isna(petr.loc[0, "daily_return"])
    assert petr.loc[1:, "daily_return"].notna().all()
    assert petr["ma_20"].notna().all()

    # Check 20-day moving average calculation accuracy on row 19 (20th trading day)
    expected_ma20 = petr.loc[0:19, "close_price"].mean()
    assert petr.loc[19, "ma_20"] == pytest.approx(expected_ma20, rel=1e-3)


def test_transform_asset_metrics_all_timeframes(sample_wide_prices: pd.DataFrame) -> None:
    """Verify asset_metrics table contains all tickers across 1M, 3M, 6M, and 1Y."""
    metrics_df = transform_asset_metrics(sample_wide_prices)

    assert len(metrics_df) == len(ALL_TICKERS) * len(TIMEFRAME_DAYS)
    assert set(metrics_df["timeframe"].unique()) == set(TIMEFRAME_DAYS.keys())
    assert set(metrics_df["ticker"].unique()) == set(ALL_TICKERS)

    for _, row in metrics_df.iterrows():
        assert row["min_price"] <= row["current_price"] <= row["max_price"]
        assert row["annualized_volatility_pct"] >= 0.0
        assert row["observations"] >= 2


def test_compute_portfolio_analytics_math(sample_wide_prices: pd.DataFrame) -> None:
    """Verify equal-weight portfolio rebasing, Alpha vs Ibovespa, and capital simulation."""
    selected = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]
    initial_capital = 25_000.0

    result = compute_portfolio_analytics(
        full_prices=sample_wide_prices,
        selected_tickers=selected,
        timeframe="6M",
        initial_capital=initial_capital,
    )

    # Base 100 normalization on initial date
    assert result["portfolio_norm"].iloc[0] == pytest.approx(100.0)
    assert result["ibov_norm"].iloc[0] == pytest.approx(100.0)

    # Alpha = Portfolio Return - Ibovespa Return
    expected_alpha = result["portfolio_return_pct"] - result["ibov_return_pct"]
    assert result["alpha_pct"] == pytest.approx(expected_alpha, rel=1e-6)

    # Final Value & Net PnL consistency
    expected_final = initial_capital * (1.0 + result["portfolio_return_pct"] / 100.0)
    assert result["final_value"] == pytest.approx(expected_final, rel=1e-6)
    assert result["net_pnl"] == pytest.approx(expected_final - initial_capital, rel=1e-6)

    # Summary DataFrame equal weights (25.0% each for 4 assets)
    summary_df = result["summary_df"]
    assert len(summary_df) == 4
    assert (summary_df["Weight (%)"] == 25.0).all()


# ---------------------------------------------------------------------------
# 2. SQLite Database & ETL Persistence Tests
# ---------------------------------------------------------------------------

def test_sqlite_etl_persistence_and_queries(populated_temp_db: Path) -> None:
    """Verify SQLite database schema, ETL metadata, price matrix pivot, and metric queries."""
    assert populated_temp_db.exists()

    status = get_etl_status(db_path=populated_temp_db)
    assert status is not None
    assert status["is_fallback"] == 1
    assert status["row_count"] == 252 * len(ALL_TICKERS)

    wide_df, updated_at, is_fallback = load_prices_pivot(db_path=populated_temp_db)
    assert wide_df.shape == (252, len(ALL_TICKERS))
    assert is_fallback is True
    assert "BRT" in updated_at

    metrics_1y = query_asset_metrics(
        timeframe="1Y",
        tickers=list(B3_ASSETS.keys()),
        db_path=populated_temp_db,
    )
    assert len(metrics_1y) == len(B3_ASSETS)
    # Ordered by period_return_pct descending
    returns = [m["period_return_pct"] for m in metrics_1y]
    assert returns == sorted(returns, reverse=True)


# ---------------------------------------------------------------------------
# 3. FastAPI Endpoint Integration Tests
# ---------------------------------------------------------------------------

def test_api_health_endpoint(populated_temp_db: Path) -> None:
    """Verify GET /api/v1/health returns healthy status and row count."""
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["row_count"] == 252 * len(ALL_TICKERS)


def test_api_assets_catalog_endpoint(populated_temp_db: Path) -> None:
    """Verify GET /api/v1/assets lists B3 tickers, benchmark, and timeframes."""
    client = TestClient(app)
    response = client.get("/api/v1/assets")
    assert response.status_code == 200
    payload = response.json()
    assert payload["benchmark_ticker"] == BENCHMARK_TICKER
    assert payload["default_tickers"] == DEFAULT_TICKERS
    assert len(payload["assets"]) == len(B3_ASSETS)
    assert set(payload["timeframes"].keys()) == {"1M", "3M", "6M", "1Y"}


def test_api_prices_endpoint(populated_temp_db: Path) -> None:
    """Verify GET /api/v1/prices filters tickers and returns aligned date/price arrays."""
    client = TestClient(app)
    response = client.get("/api/v1/prices", params={"tickers": "PETR4.SA,VALE3.SA,^BVSP"})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["dates"]) == 252
    assert set(payload["tickers"]) == {"PETR4.SA", "VALE3.SA", "^BVSP"}
    assert len(payload["prices"]["PETR4.SA"]) == 252


def test_api_metrics_endpoint(populated_temp_db: Path) -> None:
    """Verify GET /api/v1/metrics returns pre-computed KPIs for requested timeframe."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics", params={"timeframe": "3M"})
    assert response.status_code == 200
    items = response.json()
    assert len(items) == len(ALL_TICKERS)
    assert all(item["timeframe"] == "3M" for item in items)


def test_api_portfolio_analytics_post_and_get(populated_temp_db: Path) -> None:
    """Verify POST and GET /api/v1/portfolio/analytics return identical simulation results."""
    client = TestClient(app)

    post_resp = client.post(
        "/api/v1/portfolio/analytics",
        json={
            "selected_tickers": ["PETR4.SA", "WEGE3.SA"],
            "timeframe": "1M",
            "initial_capital": 50000.0,
        },
    )
    assert post_resp.status_code == 200
    post_data = post_resp.json()

    get_resp = client.get(
        "/api/v1/portfolio/analytics",
        params={
            "tickers": "PETR4.SA,WEGE3.SA",
            "timeframe": "1M",
            "initial_capital": 50000.0,
        },
    )
    assert get_resp.status_code == 200
    get_data = get_resp.json()

    assert post_data["portfolio_return_pct"] == pytest.approx(get_data["portfolio_return_pct"])
    assert post_data["final_value"] == pytest.approx(get_data["final_value"])
    assert len(post_data["summary"]) == 2
    assert post_data["summary"][0]["weight_pct"] == 50.0


def test_api_invalid_timeframe_returns_400(populated_temp_db: Path) -> None:
    """Verify invalid timeframe request returns HTTP 400 Bad Request."""
    client = TestClient(app)
    response = client.get("/api/v1/metrics", params={"timeframe": "5Y"})
    assert response.status_code == 400
