"""Module 4 – Monte Carlo Engine.

Correlated Geometric Brownian Motion via Cholesky decomposition.
Fully vectorised – zero Python loops over simulations or time steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from .config import EngineConfig
from .statistics import BaselineStats


@dataclass
class MonteCarloResult:
    """Raw output of the Monte Carlo simulation."""

    # (n_simulations, horizon+1, n_assets) – full price cube
    price_paths: np.ndarray

    # (n_simulations,) – terminal portfolio values
    terminal_values: np.ndarray

    # (n_simulations,) – portfolio returns over the horizon
    portfolio_returns: np.ndarray


def run_simulation(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    baseline: BaselineStats,
    cfg: EngineConfig,
) -> MonteCarloResult:
    """Generate correlated GBM price paths for every asset in the universe.

    The simulation is fully vectorised:
        Z  ~ N(0,I)   shape (n_sim, horizon, n_assets)
        ε  = Z @ L^T   (Cholesky-correlated shocks)
        S_{t+1} = S_t · exp[(μ - σ²/2)dt + σ√dt · ε]

    Parameters
    ----------
    prices : cleaned price DataFrame (used for S₀)
    returns : daily log-return DataFrame (used only indirectly via baseline)
    baseline : pre-computed BaselineStats
    cfg : EngineConfig
    """
    rng = np.random.default_rng(cfg.seed)
    n_assets = len(prices.columns)
    weights = cfg.resolved_weights(n_assets)

    # Annualised parameters → daily
    dt = 1.0 / cfg.trading_days_per_year
    mu_daily = baseline.ann_returns.values * dt                    # (n_assets,)
    sigma_daily = baseline.ann_volatility.values * np.sqrt(dt)     # (n_assets,)

    # Drift term: (μ - ½σ²)dt  – but mu_daily already contains μ·dt
    drift = mu_daily - 0.5 * sigma_daily ** 2                      # (n_assets,)

    # Cholesky factor of the correlation matrix (lower triangular)
    corr = baseline.corr_matrix.values
    L = cholesky(corr, lower=True)                                 # (n, n)

    # --- Vectorised shock generation -----------------------------------------
    # Z: independent standard normals
    Z = rng.standard_normal(
        (cfg.n_simulations, cfg.horizon_days, n_assets)
    )                                                              # (S, T, N)

    # Correlate: ε = Z @ Lᵀ  (last axis is assets)
    eps = Z @ L.T                                                  # (S, T, N)

    # Log-return increments
    log_increments = drift[np.newaxis, np.newaxis, :] + sigma_daily[np.newaxis, np.newaxis, :] * eps
    # shape: (S, T, N)

    # Cumulative sum → cumulative log-return from t=0
    cum_log_ret = np.cumsum(log_increments, axis=1)                # (S, T, N)

    # Prepend zeros for the starting price level
    zeros = np.zeros((cfg.n_simulations, 1, n_assets))
    cum_log_ret = np.concatenate([zeros, cum_log_ret], axis=1)     # (S, T+1, N)

    # Price paths: S_t = S_0 · exp(cumulative log-return)
    S0 = prices.iloc[-1].values[np.newaxis, np.newaxis, :]        # (1, 1, N)
    price_paths = S0 * np.exp(cum_log_ret)                         # (S, T+1, N)

    # --- Terminal portfolio values -------------------------------------------
    # Dollar allocation at t=0
    notional_per_asset = cfg.initial_portfolio_value * weights     # (N,)
    shares = notional_per_asset / prices.iloc[-1].values           # (N,)

    terminal_prices = price_paths[:, -1, :]                        # (S, N)
    terminal_values = terminal_prices @ shares                     # (S,)

    portfolio_returns = terminal_values / cfg.initial_portfolio_value - 1.0

    return MonteCarloResult(
        price_paths=price_paths,
        terminal_values=terminal_values,
        portfolio_returns=portfolio_returns,
    )
