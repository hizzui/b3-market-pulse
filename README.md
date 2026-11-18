# B3 Market Pulse

Dashboard and data pipeline in Python for tracking Brazilian B3 stocks and FIIs (`PETR4.SA`, `VALE3.SA`, `ITUB4.SA`, `WEGE3.SA`, `MXRF11.SA`) against the Ibovespa (`^BVSP`) benchmark.

Historical daily closing prices are pulled from Yahoo Finance (`yfinance`), processed with Pandas/NumPy (daily returns, 20-day moving average, annualized volatility, and base-100 equal-weight portfolio curves across `1M`, `3M`, `6M`, and `1Y` windows), and stored in a local SQLite database (`data/market_data.db`).

A GitHub Actions workflow (`.github/workflows/daily_etl.yml`) runs on weekdays after market close to update the database automatically.

## Running locally

```bash
git clone https://github.com/hizzui/b3-market-pulse.git
cd b3-market-pulse
pip install -r requirements.txt

# Refresh local SQLite data
python -m backend.etl

# Run the Streamlit dashboard
streamlit run app.py
```

To run the FastAPI service (`http://127.0.0.1:8000/docs`) or the test suite:

```bash
uvicorn backend.api:app --reload --port 8000
pytest tests/
```
