"""Module 1 – Data Architecture.

Fetch multi-asset historical daily data, handle NaNs, and compute daily log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EngineConfig


def fetch_prices(cfg: EngineConfig) -> pd.DataFrame:
    """Download adjusted-close prices from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    import yfinance as yf

    raw: pd.DataFrame = yf.download(
        tickers=cfg.tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=True,
        progress=False,
    )

    # yfinance may return MultiIndex columns; normalise to flat ticker columns.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices[cfg.tickers]  # enforce column order
    return prices


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Handle missing data in a price matrix.

    Strategy:
      1. Forward-fill short gaps (weekends / holidays).
      2. Backward-fill any leading NaNs (IPO alignment).
      3. Drop any column that is still >10 % NaN.
      4. Drop remaining rows with any NaN.
    """
    prices = prices.ffill().bfill()

    nan_pct = prices.isna().mean()
    bad_cols = nan_pct[nan_pct > 0.10].index.tolist()
    if bad_cols:
        import warnings

        warnings.warn(
            f"Dropping tickers with >10% missing data: {bad_cols}", stacklevel=2
        )
        prices = prices.drop(columns=bad_cols)

    prices = prices.dropna()
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute continuous (log) daily returns.  First row is dropped."""
    log_ret: pd.DataFrame = np.log(prices / prices.shift(1))
    return log_ret.iloc[1:]  # drop the NaN first row


def load_data(cfg: EngineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """High-level convenience: fetch → clean → returns.

    Returns
    -------
    prices : pd.DataFrame  – cleaned price matrix
    returns : pd.DataFrame – daily log-return matrix
    """
    prices = fetch_prices(cfg)
    prices = clean_prices(prices)
    returns = compute_log_returns(prices)
    return prices, returns
