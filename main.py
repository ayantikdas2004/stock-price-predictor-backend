from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta


app = FastAPI(title="Stock Price Predictor Backend - Phase 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only (in production: specify frontend domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------
# Utility: Download & normalize yfinance output (MultiIndex safe)
# -----------------------------------------------------------
def _download_data(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    """
    Downloads stock data using yfinance and returns a normalized dataframe
    with columns: Date, Close

    Handles MultiIndex columns like:
      ("Close","TCS.NS"), ("High","TCS.NS") ...
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            group_by="column",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"yfinance download failed: {str(e)}")

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")

    # If MultiIndex (your case): df columns are like ("Close","TCS.NS")
    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(1):
            available = list(set(df.columns.get_level_values(1)))
            raise HTTPException(
                status_code=500,
                detail=f"Ticker not found inside MultiIndex columns. Available: {available}"
            )
        df = df.xs(ticker, axis=1, level=1)

    if "Close" not in df.columns:
        raise HTTPException(status_code=500, detail=f"'Close' not found. Columns: {list(df.columns)}")

    # Make Date a column
    df = df.reset_index()

    if "Date" not in df.columns:
        raise HTTPException(status_code=500, detail=f"'Date' not found after reset_index. Columns: {list(df.columns)}")

    df = df[["Date", "Close"]].dropna()

    # Force correct types
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# -----------------------------------------------------------
# ML: Lag feature creation
# -----------------------------------------------------------
def _make_lag_dataset(close_prices: np.ndarray, lags: int = 10):
    """
    Create lag features:
      X[i] = [t-10, t-9, ..., t-1]
      y[i] = [t]
    """
    X, y = [], []
    for i in range(lags, len(close_prices)):
        X.append(close_prices[i - lags:i])
        y.append(close_prices[i])

    return np.array(X), np.array(y)


# -----------------------------------------------------------
# ML: Train model + predict future sequentially
# -----------------------------------------------------------
def _train_and_predict_future(df: pd.DataFrame, days: int = 30, lags: int = 10):
    close_prices = df["Close"].values.astype(float)

    if len(close_prices) < (lags + 60):
        raise HTTPException(status_code=400, detail="Not enough historical data to train model")

    X, y = _make_lag_dataset(close_prices, lags=lags)

    model = LinearRegression()
    model.fit(X, y)

    # Sequential future prediction
    last_window = list(close_prices[-lags:])
    future_prices = []

    for _ in range(days):
        x_input = np.array(last_window[-lags:]).reshape(1, -1)
        pred = float(model.predict(x_input)[0])
        future_prices.append(pred)
        last_window.append(pred)

    last_date = df["Date"].iloc[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

    prediction = []
    for d, p in zip(future_dates, future_prices):
        prediction.append({
            "date": pd.to_datetime(d).strftime("%Y-%m-%d"),
            "predicted_close": float(p)
        })

    return prediction


# -----------------------------------------------------------
# API ROUTES
# -----------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Stock Price Predictor Backend (Phase 1) is running"}


# ✅ History endpoint
@app.get("/history")
def get_history(ticker: str, period: str = "6mo"):
    """
    Example:
      /history?ticker=TCS.NS&period=6mo
    """
    df = _download_data(ticker=ticker, period=period)

    df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["close"] = df["Close"].astype(float)

    return {
        "ticker": ticker,
        "history": df[["date", "close"]].to_dict(orient="records")
    }


# ✅ Predict endpoint
@app.get("/predict")
def predict(ticker: str, days: int = 30, period: str = "1y", lags: int = 10):
    """
    Example:
      /predict?ticker=TCS.NS&days=30&period=1y&lags=10
    """
    df = _download_data(ticker=ticker, period=period)
    pred_records = _train_and_predict_future(df, days=days, lags=lags)

    return {
        "ticker": ticker,
        "days": days,
        "lags": lags,
        "prediction": pred_records
    }


# ✅ Metrics endpoint (Evaluation metrics)
@app.get("/metrics")
def metrics(ticker: str, period: str = "2y", lags: int = 10, test_size: float = 0.2):
    """
    Performance metrics for Phase 1 model.

    Example:
      /metrics?ticker=TCS.NS&period=2y&lags=10&test_size=0.2
    """
    df = _download_data(ticker=ticker, period=period)
    close_prices = df["Close"].values.astype(float)

    if len(close_prices) < (lags + 80):
        raise HTTPException(status_code=400, detail="Not enough data for evaluation")

    X, y = _make_lag_dataset(close_prices, lags=lags)

    # Time-series split (no shuffle)
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ✅ Regression metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # ✅ MAPE (%)
    mape = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100)

    # ✅ Baseline model (Naive prediction: next day = last day)
    y_pred_naive = X_test[:, -1]
    mae_naive = float(mean_absolute_error(y_test, y_pred_naive))
    rmse_naive = float(np.sqrt(mean_squared_error(y_test, y_pred_naive)))
    mape_naive = float(np.mean(np.abs((y_test - y_pred_naive) / (y_test + 1e-9))) * 100)

    return {
        "ticker": ticker,
        "period": period,
        "lags": lags,
        "test_size": test_size,
        "model": "LinearRegression with Lag Features (Phase 1)",
        "metrics": {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE_percent": mape,
            "R2_score": r2
        },
        "baseline_naive": {
            "MAE": mae_naive,
            "RMSE": rmse_naive,
            "MAPE_percent": mape_naive
        },
        "improvement_over_baseline": {
            "MAE_improvement_percent": float(((mae_naive - mae) / (mae_naive + 1e-9)) * 100),
            "RMSE_improvement_percent": float(((rmse_naive - rmse) / (rmse_naive + 1e-9)) * 100),
            "MAPE_improvement_percent": float(((mape_naive - mape) / (mape_naive + 1e-9)) * 100)
        }
    }


# ✅ Evaluation plot endpoint (Actual vs Predicted vs Naive series)
@app.get("/evaluate_plot_data")
def evaluate_plot_data(ticker: str, period: str = "2y", lags: int = 10, test_size: float = 0.2):
    """
    Provides plot data for evaluation chart.

    Example:
      /evaluate_plot_data?ticker=TCS.NS&period=2y
    """
    df = _download_data(ticker=ticker, period=period)

    close_prices = df["Close"].values.astype(float)
    dates = df["Date"].values

    if len(close_prices) < (lags + 80):
        raise HTTPException(status_code=400, detail="Not enough data for evaluation plot")

    X, y, y_dates = [], [], []
    for i in range(lags, len(close_prices)):
        X.append(close_prices[i - lags:i])
        y.append(close_prices[i])
        y_dates.append(dates[i])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates = y_dates[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # naive baseline
    y_pred_naive = X_test[:, -1]

    series = []
    for d, actual, pred, naive in zip(test_dates, y_test, y_pred, y_pred_naive):
        series.append({
            "date": pd.to_datetime(d).strftime("%Y-%m-%d"),
            "actual": float(actual),
            "predicted": float(pred),
            "naive": float(naive)
        })

    return {
        "ticker": ticker,
        "series": series
    }
