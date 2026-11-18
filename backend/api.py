"""
FastAPI REST Service for B3 Market & Portfolio Pulse
----------------------------------------------------
Exposes REST endpoints serving historical B3 closing prices, pre-computed
asset KPIs from SQLite, and real-time equal-weight portfolio simulations
benchmarked against the Ibovespa (`^BVSP`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.database import (
    B3_ASSETS,
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    DB_PATH,
    DEFAULT_TICKERS,
    TIMEFRAME_DAYS,
    get_etl_status,
    infer_asset_type,
    query_asset_metrics,
)
from backend.etl import (
    compute_portfolio_analytics,
    ensure_populated_db,
    run_etl_pipeline,
)


def get_active_db_path() -> Path:
    """Resolve the active SQLite database path (supports override via B3_DB_PATH env var)."""
    env_path = os.getenv("B3_DB_PATH")
    return Path(env_path) if env_path else DB_PATH


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    database_path: str
    last_run_at: str
    source_label: str
    is_fallback: bool
    row_count: int


class AssetItem(BaseModel):
    ticker: str
    name: str
    asset_type: str


class CatalogResponse(BaseModel):
    assets: List[AssetItem]
    default_tickers: List[str]
    benchmark_ticker: str
    benchmark_name: str
    timeframes: Dict[str, int]


class PricesResponse(BaseModel):
    updated_at: str
    is_fallback: bool
    dates: List[str]
    tickers: List[str]
    prices: Dict[str, List[float]]


class StoredAssetMetric(BaseModel):
    ticker: str
    timeframe: str
    asset_name: str
    current_price: float
    period_return_pct: float
    ma_20: float
    annualized_volatility_pct: float
    min_price: float
    max_price: float
    observations: int
    is_fallback: bool
    updated_at: str


class PortfolioAnalyticsRequest(BaseModel):
    selected_tickers: List[str] = Field(
        default_factory=lambda: list(DEFAULT_TICKERS),
        min_length=1,
        description="List of B3 asset tickers included in the equal-weight portfolio.",
    )
    timeframe: str = Field(
        default="1Y",
        description="Lookback horizon: '1M', '3M', '6M', or '1Y'.",
    )
    initial_capital: float = Field(
        default=10_000.0,
        ge=100.0,
        le=100_000_000.0,
        description="Initial portfolio capital in BRL (R$).",
    )


class PortfolioAssetSummary(BaseModel):
    ticker: str
    asset: str
    weight_pct: float
    current_price: float
    period_return_pct: float
    ma_20: float
    volatility_pct: float


class PortfolioAnalyticsResponse(BaseModel):
    timeframe: str
    initial_capital: float
    updated_at: str
    is_fallback: bool
    dates: List[str]
    portfolio_norm: List[float]
    ibov_norm: List[float]
    portfolio_return_pct: float
    ibov_return_pct: float
    alpha_pct: float
    final_value: float
    net_pnl: float
    port_vol_pct: float
    ibov_vol_pct: float
    summary: List[PortfolioAssetSummary]


class ETLRefreshResponse(BaseModel):
    status: str
    updated_at: str
    is_fallback: bool
    tickers_processed: List[str]
    price_rows: int
    metric_rows: int
    db_path: str


# ---------------------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="B3 Market & Portfolio Pulse API",
    description=(
        "Production-style Data Engineering & Analytics REST API serving "
        "historical B3 equity/FII prices, technical indicators, and equal-weight "
        "portfolio simulations benchmarked against Ibovespa (^BVSP)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return service health and latest SQLite ETL audit status."""
    db_path = get_active_db_path()
    ensure_populated_db(db_path=db_path)
    meta = get_etl_status(db_path=db_path)
    if not meta:
        raise HTTPException(status_code=503, detail="Database is unavailable or uninitialized.")

    return HealthResponse(
        status="healthy",
        database_path=str(db_path),
        last_run_at=str(meta["last_run_at"]),
        source_label=str(meta["source_label"]),
        is_fallback=bool(meta["is_fallback"]),
        row_count=int(meta["row_count"]),
    )


@app.get("/api/v1/assets", response_model=CatalogResponse, tags=["Catalog"])
def list_assets() -> CatalogResponse:
    """Return the catalog of supported B3 tickers, benchmark, and timeframes."""
    assets = [
        AssetItem(ticker=t, name=name, asset_type=infer_asset_type(t))
        for t, name in B3_ASSETS.items()
    ]
    return CatalogResponse(
        assets=assets,
        default_tickers=DEFAULT_TICKERS,
        benchmark_ticker=BENCHMARK_TICKER,
        benchmark_name=BENCHMARK_NAME,
        timeframes=TIMEFRAME_DAYS,
    )


@app.get("/api/v1/prices", response_model=PricesResponse, tags=["Market Data"])
def get_prices(
    tickers: Optional[str] = Query(
        default=None,
        description="Optional comma-separated ticker symbols (e.g., 'PETR4.SA,VALE3.SA,^BVSP').",
    ),
) -> PricesResponse:
    """
    Retrieve the historical 1-year daily closing price matrix from SQLite.
    Automatically bootstraps the database via the ETL pipeline if empty.
    """
    db_path = get_active_db_path()
    wide_df, updated_at, is_fallback = ensure_populated_db(db_path=db_path)

    if tickers:
        requested = [t.strip() for t in tickers.split(",") if t.strip()]
        valid = [t for t in requested if t in wide_df.columns]
        if not valid:
            raise HTTPException(
                status_code=400,
                detail=f"None of the requested tickers {requested} exist in the database.",
            )
        wide_df = wide_df[valid]

    dates = [d.strftime("%Y-%m-%d") for d in wide_df.index]
    prices_dict: Dict[str, List[float]] = {
        col: [round(float(v), 4) for v in wide_df[col].tolist()]
        for col in wide_df.columns
    }

    return PricesResponse(
        updated_at=updated_at,
        is_fallback=is_fallback,
        dates=dates,
        tickers=list(wide_df.columns),
        prices=prices_dict,
    )


@app.get("/api/v1/metrics", response_model=List[StoredAssetMetric], tags=["Market Data"])
def get_asset_metrics(
    timeframe: str = Query(
        default="1Y",
        description="Timeframe horizon ('1M', '3M', '6M', or '1Y').",
    ),
    tickers: Optional[str] = Query(
        default=None,
        description="Optional comma-separated ticker filter.",
    ),
) -> List[StoredAssetMetric]:
    """Query pre-computed asset metrics for a given timeframe from SQLite."""
    if timeframe not in TIMEFRAME_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe '{timeframe}'. Allowed: {list(TIMEFRAME_DAYS.keys())}",
        )

    db_path = get_active_db_path()
    ensure_populated_db(db_path=db_path)

    ticker_list = (
        [t.strip() for t in tickers.split(",") if t.strip()] if tickers else None
    )
    rows = query_asset_metrics(timeframe=timeframe, tickers=ticker_list, db_path=db_path)

    return [
        StoredAssetMetric(
            ticker=r["ticker"],
            timeframe=r["timeframe"],
            asset_name=r["asset_name"],
            current_price=float(r["current_price"]),
            period_return_pct=float(r["period_return_pct"]),
            ma_20=float(r["ma_20"]),
            annualized_volatility_pct=float(r["annualized_volatility_pct"]),
            min_price=float(r["min_price"]),
            max_price=float(r["max_price"]),
            observations=int(r["observations"]),
            is_fallback=bool(r["is_fallback"]),
            updated_at=str(r["updated_at"]),
        )
        for r in rows
    ]


def _build_portfolio_response(
    selected_tickers: List[str],
    timeframe: str,
    initial_capital: float,
) -> PortfolioAnalyticsResponse:
    """Shared implementation for GET and POST portfolio analytics endpoints."""
    if timeframe not in TIMEFRAME_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe '{timeframe}'. Allowed: {list(TIMEFRAME_DAYS.keys())}",
        )

    db_path = get_active_db_path()
    wide_df, updated_at, is_fallback = ensure_populated_db(db_path=db_path)

    valid_tickers = [t for t in selected_tickers if t in wide_df.columns and t != BENCHMARK_TICKER]
    if not valid_tickers:
        raise HTTPException(
            status_code=400,
            detail="At least one valid B3 asset ticker must be selected.",
        )

    analytics = compute_portfolio_analytics(
        full_prices=wide_df,
        selected_tickers=valid_tickers,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )

    summary_items = [
        PortfolioAssetSummary(
            ticker=str(row["Ticker"]),
            asset=str(row["Asset"]),
            weight_pct=float(row["Weight (%)"]),
            current_price=float(row["Current Price (R$)"]),
            period_return_pct=float(row["Period Return (%)"]),
            ma_20=float(row["20D Moving Avg (R$)"]),
            volatility_pct=float(row["Volatility (%)"]),
        )
        for _, row in analytics["summary_df"].iterrows()
    ]

    return PortfolioAnalyticsResponse(
        timeframe=timeframe,
        initial_capital=initial_capital,
        updated_at=updated_at,
        is_fallback=is_fallback,
        dates=[d.strftime("%Y-%m-%d") for d in analytics["dates"]],
        portfolio_norm=[round(float(v), 4) for v in analytics["portfolio_norm"].tolist()],
        ibov_norm=[round(float(v), 4) for v in analytics["ibov_norm"].tolist()],
        portfolio_return_pct=round(float(analytics["portfolio_return_pct"]), 4),
        ibov_return_pct=round(float(analytics["ibov_return_pct"]), 4),
        alpha_pct=round(float(analytics["alpha_pct"]), 4),
        final_value=round(float(analytics["final_value"]), 2),
        net_pnl=round(float(analytics["net_pnl"]), 2),
        port_vol_pct=round(float(analytics["port_vol_pct"]), 4),
        ibov_vol_pct=round(float(analytics["ibov_vol_pct"]), 4),
        summary=summary_items,
    )


@app.post(
    "/api/v1/portfolio/analytics",
    response_model=PortfolioAnalyticsResponse,
    tags=["Portfolio Analytics"],
)
def calculate_portfolio_post(payload: PortfolioAnalyticsRequest) -> PortfolioAnalyticsResponse:
    """Compute equal-weight B3 portfolio performance and risk metrics via POST JSON body."""
    return _build_portfolio_response(
        selected_tickers=payload.selected_tickers,
        timeframe=payload.timeframe,
        initial_capital=payload.initial_capital,
    )


@app.get(
    "/api/v1/portfolio/analytics",
    response_model=PortfolioAnalyticsResponse,
    tags=["Portfolio Analytics"],
)
def calculate_portfolio_get(
    tickers: str = Query(
        default=",".join(DEFAULT_TICKERS),
        description="Comma-separated B3 tickers for equal-weight simulation.",
    ),
    timeframe: str = Query(default="1Y", description="Horizon: '1M', '3M', '6M', or '1Y'."),
    initial_capital: float = Query(
        default=10_000.0,
        ge=100.0,
        le=100_000_000.0,
        description="Initial investment capital in BRL (R$).",
    ),
) -> PortfolioAnalyticsResponse:
    """Compute equal-weight B3 portfolio performance and risk metrics via GET query parameters."""
    selected = [t.strip() for t in tickers.split(",") if t.strip()]
    return _build_portfolio_response(
        selected_tickers=selected,
        timeframe=timeframe,
        initial_capital=initial_capital,
    )


@app.post("/api/v1/etl/refresh", response_model=ETLRefreshResponse, tags=["ETL"])
def trigger_etl_refresh(
    offline: bool = Query(
        default=False,
        description="If true, use deterministic synthetic dataset instead of live Yahoo Finance.",
    ),
) -> ETLRefreshResponse:
    """Trigger an on-demand ETL execution and refresh the SQLite database."""
    db_path = get_active_db_path()
    result = run_etl_pipeline(db_path=db_path, force_fallback=offline)
    return ETLRefreshResponse(**result)
