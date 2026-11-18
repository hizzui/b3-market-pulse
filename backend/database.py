"""
SQLite Database Layer & Normalized Schema Definitions
-----------------------------------------------------
Manages SQLite connections, normalized relational tables (`assets`, `prices`,
`asset_metrics`, and `etl_metadata`), and query helpers for the B3 Market &
Portfolio Pulse backend.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Project Paths & Domain Constants
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
DB_PATH: Path = DATA_DIR / "market_data.db"
PRICES_CSV_PATH: Path = DATA_DIR / "historical_prices.csv"
METRICS_CSV_PATH: Path = DATA_DIR / "asset_metrics.csv"

B3_ASSETS: Dict[str, str] = {
    "PETR4.SA": "Petrobras PN",
    "VALE3.SA": "Vale ON",
    "ITUB4.SA": "Itaú Unibanco PN",
    "WEGE3.SA": "WEG ON",
    "MXRF11.SA": "Maxi Renda FII",
}

DEFAULT_TICKERS: List[str] = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA"]
BENCHMARK_TICKER: str = "^BVSP"
BENCHMARK_NAME: str = "Ibovespa Index"

ALL_TICKERS: List[str] = list(B3_ASSETS.keys()) + [BENCHMARK_TICKER]

TIMEFRAME_DAYS: Dict[str, int] = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
}


# ---------------------------------------------------------------------------
# Connection & Schema Management
# ---------------------------------------------------------------------------

def ensure_data_dir(db_path: Path = DB_PATH) -> None:
    """Ensure the parent directory for the SQLite database exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Context-managed SQLite connection configured with Row factory and foreign keys.

    Args:
        db_path: Target SQLite database file path.

    Yields:
        Active sqlite3.Connection object.
    """
    ensure_data_dir(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Initialize normalized SQLite database schema for B3 market prices and metrics.

    Tables created:
        - `assets`: Master catalog of tracked B3 equities, FIIs, and benchmark index.
        - `prices`: Normalized daily price time-series with rolling technical indicators.
        - `asset_metrics`: Pre-aggregated asset performance and risk KPIs by timeframe.
        - `etl_metadata`: Singleton audit record tracking the latest ETL execution.
    """
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS assets (
                ticker TEXT PRIMARY KEY,
                asset_name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                close_price REAL NOT NULL,
                daily_return REAL,
                ma_20 REAL,
                volatility_20d REAL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (date, ticker),
                FOREIGN KEY (ticker) REFERENCES assets (ticker) ON DELETE CASCADE
            );
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
            ON prices (ticker, date);
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_metrics (
                ticker TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                asset_name TEXT NOT NULL,
                current_price REAL NOT NULL,
                period_return_pct REAL NOT NULL,
                ma_20 REAL NOT NULL,
                annualized_volatility_pct REAL NOT NULL,
                min_price REAL NOT NULL,
                max_price REAL NOT NULL,
                observations INTEGER NOT NULL,
                is_fallback INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (ticker, timeframe),
                FOREIGN KEY (ticker) REFERENCES assets (ticker) ON DELETE CASCADE
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS etl_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_run_at TEXT NOT NULL,
                source_label TEXT NOT NULL,
                is_fallback INTEGER NOT NULL DEFAULT 0,
                row_count INTEGER NOT NULL DEFAULT 0
            );
            """
        )


# ---------------------------------------------------------------------------
# Persistence & Query Functions
# ---------------------------------------------------------------------------

def infer_asset_type(ticker: str) -> str:
    """Classify a B3 ticker into BENCHMARK, FII, or EQUITY."""
    if ticker == BENCHMARK_TICKER:
        return "BENCHMARK"
    if ticker.endswith("11.SA"):
        return "FII"
    return "EQUITY"


def upsert_market_data(
    prices_long_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    updated_at: str,
    is_fallback: bool,
    db_path: Path = DB_PATH,
) -> None:
    """
    Atomically persist transformed prices, per-timeframe asset metrics,
    and ETL execution metadata into the SQLite database.

    Args:
        prices_long_df: Normalized DataFrame with columns
            [date, ticker, close_price, daily_return, ma_20, volatility_20d].
        metrics_df: Aggregated metrics DataFrame with columns
            [ticker, timeframe, asset_name, current_price, period_return_pct,
             ma_20, annualized_volatility_pct, min_price, max_price, observations].
        updated_at: Formatted timestamp string (e.g., '2026-11-18 10:20 BRT').
        is_fallback: True if synthetic fallback data was used.
        db_path: Target SQLite database path.
    """
    init_db(db_path)
    fallback_flag = 1 if is_fallback else 0
    source_label = "Simulated Baseline" if is_fallback else "Yahoo Finance · B3"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Upsert Asset Catalog
        asset_records = []
        for ticker, name in B3_ASSETS.items():
            asset_records.append((ticker, name, infer_asset_type(ticker), updated_at))
        asset_records.append((BENCHMARK_TICKER, BENCHMARK_NAME, "BENCHMARK", updated_at))

        cursor.executemany(
            """
            INSERT INTO assets (ticker, asset_name, asset_type, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                asset_name = excluded.asset_name,
                asset_type = excluded.asset_type,
                updated_at = excluded.updated_at;
            """,
            asset_records,
        )

        # 2. Refresh Prices Table
        cursor.execute("DELETE FROM prices;")
        price_rows = [
            (
                str(row["date"]),
                str(row["ticker"]),
                float(row["close_price"]),
                None if pd.isna(row["daily_return"]) else float(row["daily_return"]),
                None if pd.isna(row["ma_20"]) else float(row["ma_20"]),
                None if pd.isna(row["volatility_20d"]) else float(row["volatility_20d"]),
                updated_at,
            )
            for _, row in prices_long_df.iterrows()
        ]
        cursor.executemany(
            """
            INSERT INTO prices (
                date, ticker, close_price, daily_return, ma_20, volatility_20d, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            price_rows,
        )

        # 3. Refresh Asset Metrics Table
        cursor.execute("DELETE FROM asset_metrics;")
        metric_rows = [
            (
                str(row["ticker"]),
                str(row["timeframe"]),
                str(row["asset_name"]),
                float(row["current_price"]),
                float(row["period_return_pct"]),
                float(row["ma_20"]),
                float(row["annualized_volatility_pct"]),
                float(row["min_price"]),
                float(row["max_price"]),
                int(row["observations"]),
                fallback_flag,
                updated_at,
            )
            for _, row in metrics_df.iterrows()
        ]
        cursor.executemany(
            """
            INSERT INTO asset_metrics (
                ticker, timeframe, asset_name, current_price, period_return_pct,
                ma_20, annualized_volatility_pct, min_price, max_price,
                observations, is_fallback, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            metric_rows,
        )

        # 4. Upsert Singleton ETL Metadata
        cursor.execute(
            """
            INSERT INTO etl_metadata (id, last_run_at, source_label, is_fallback, row_count)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                last_run_at = excluded.last_run_at,
                source_label = excluded.source_label,
                is_fallback = excluded.is_fallback,
                row_count = excluded.row_count;
            """,
            (updated_at, source_label, fallback_flag, len(price_rows)),
        )


def load_prices_pivot(
    tickers: Optional[List[str]] = None,
    db_path: Path = DB_PATH,
) -> Tuple[pd.DataFrame, str, bool]:
    """
    Read historical closing prices from SQLite and return a wide Date x Ticker DataFrame.

    Args:
        tickers: Optional list of tickers to filter. Defaults to all tracked tickers.
        db_path: Path to the SQLite database file.

    Returns:
        Tuple of (wide_prices_df, updated_at_str, is_fallback_bool).
        Returns an empty DataFrame if the database is missing or unpopulated.
    """
    if not db_path.exists():
        return pd.DataFrame(), "", False

    init_db(db_path)
    with get_connection(db_path) as conn:
        meta_row = conn.execute(
            "SELECT last_run_at, is_fallback FROM etl_metadata WHERE id = 1;"
        ).fetchone()

        if meta_row is None:
            return pd.DataFrame(), "", False

        updated_at = str(meta_row["last_run_at"])
        is_fallback = bool(meta_row["is_fallback"])

        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            query = f"""
                SELECT date, ticker, close_price
                FROM prices
                WHERE ticker IN ({placeholders})
                ORDER BY date ASC;
            """
            df_long = pd.read_sql_query(query, conn, params=tickers)
        else:
            query = """
                SELECT date, ticker, close_price
                FROM prices
                ORDER BY date ASC;
            """
            df_long = pd.read_sql_query(query, conn)

    if df_long.empty:
        return pd.DataFrame(), updated_at, is_fallback

    wide_df = df_long.pivot(index="date", columns="ticker", values="close_price")
    wide_df.index = pd.to_datetime(wide_df.index)
    wide_df = wide_df.sort_index().ffill().bfill()
    return wide_df, updated_at, is_fallback


def query_asset_metrics(
    timeframe: str = "1Y",
    tickers: Optional[List[str]] = None,
    db_path: Path = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Query pre-computed asset metrics for a specific timeframe from SQLite.

    Args:
        timeframe: Horizon filter ('1M', '3M', '6M', or '1Y').
        tickers: Optional subset of tickers to retrieve.
        db_path: Path to the SQLite database file.

    Returns:
        List of dictionaries representing asset metric rows sorted by return descending.
    """
    if not db_path.exists():
        return []

    init_db(db_path)
    with get_connection(db_path) as conn:
        if tickers:
            placeholders = ",".join("?" for _ in tickers)
            sql = f"""
                SELECT ticker, timeframe, asset_name, current_price, period_return_pct,
                       ma_20, annualized_volatility_pct, min_price, max_price,
                       observations, is_fallback, updated_at
                FROM asset_metrics
                WHERE timeframe = ? AND ticker IN ({placeholders})
                ORDER BY period_return_pct DESC;
            """
            params: List[Any] = [timeframe, *tickers]
        else:
            sql = """
                SELECT ticker, timeframe, asset_name, current_price, period_return_pct,
                       ma_20, annualized_volatility_pct, min_price, max_price,
                       observations, is_fallback, updated_at
                FROM asset_metrics
                WHERE timeframe = ?
                ORDER BY period_return_pct DESC;
            """
            params = [timeframe]

        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def get_etl_status(db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Return the latest ETL metadata record or None if no run has been recorded."""
    if not db_path.exists():
        return None

    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT last_run_at, source_label, is_fallback, row_count FROM etl_metadata WHERE id = 1;"
        ).fetchone()
        return dict(row) if row else None
