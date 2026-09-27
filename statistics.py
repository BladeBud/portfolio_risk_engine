"""Module 2 – Baseline Statistics.

Annualized volatility, expected returns, and variance-covariance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import EngineConfig


@dataclass
class BaselineStats:
    """Container for baseline portfolio statistics."""

    # Per-asset
    ann_returns: pd.Series          # annualised expected returns
    ann_volatility: pd.Series       # annualised volatility
    cov_matrix: pd.DataFrame        # annualised covariance matrix
    corr_matrix: pd.DataFrame       # correlation matrix

    # Portfolio-level
    portfolio_return: float         # annualised
    portfolio_volatility: float     # annualised


def compute_baseline(
    returns: pd.DataFrame,
    cfg: EngineConfig,
) -> BaselineStats:
    """Compute all baseline statistics from daily log returns."""
    n = cfg.trading_days_per_year
    weights = cfg.resolved_weights(returns.shape[1])

    # --- Per-asset -----------------------------------------------------------
    daily_mean: pd.Series = returns.mean()
    daily_std: pd.Series = returns.std(ddof=1)

    ann_returns = daily_mean * n
    ann_volatility = daily_std * np.sqrt(n)

    cov_daily: pd.DataFrame = returns.cov()
    cov_annual: pd.DataFrame = cov_daily * n
    corr: pd.DataFrame = returns.corr()

    # --- Portfolio level -----------------------------------------------------
    port_ret: float = float(weights @ ann_returns.values)
    port_var: float = float(weights @ cov_annual.values @ weights)
    port_vol: float = float(np.sqrt(port_var))

    return BaselineStats(
        ann_returns=ann_returns,
        ann_volatility=ann_volatility,
        cov_matrix=cov_annual,
        corr_matrix=corr,
        portfolio_return=port_ret,
        portfolio_volatility=port_vol,
    )
