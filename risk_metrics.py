"""Module 3 – Analytical Risk Metrics.

Historical VaR, Parametric VaR, and CVaR (Expected Shortfall)
at configurable confidence levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .config import EngineConfig
from .statistics import BaselineStats


@dataclass
class RiskResult:
    """VaR / CVaR result at a single confidence level."""

    confidence: float
    var: float       # Value at Risk (positive number ⇒ loss)
    cvar: float      # Conditional VaR / Expected Shortfall


@dataclass
class AnalyticalRisk:
    """Collection of all analytical risk outputs."""

    historical: Dict[float, RiskResult]
    parametric: Dict[float, RiskResult]


# ---------------------------------------------------------------------------
# Historical
# ---------------------------------------------------------------------------


def _historical_var_cvar(
    portfolio_returns: np.ndarray,
    alpha: float,
) -> RiskResult:
    """Non-parametric VaR and CVaR from the empirical return distribution."""
    sorted_ret = np.sort(portfolio_returns)
    cutoff_idx = int(np.floor((1 - alpha) * len(sorted_ret)))
    var = -sorted_ret[cutoff_idx]
    cvar = -sorted_ret[: cutoff_idx + 1].mean()
    return RiskResult(confidence=alpha, var=float(var), cvar=float(cvar))


# ---------------------------------------------------------------------------
# Parametric (variance-covariance / delta-normal)
# ---------------------------------------------------------------------------


def _parametric_var_cvar(
    port_return: float,
    port_vol: float,
    alpha: float,
    trading_days: int = 252,
) -> RiskResult:
    """Parametric VaR and CVaR assuming Gaussian returns (daily scale)."""
    daily_mu = port_return / trading_days
    daily_sigma = port_vol / np.sqrt(trading_days)

    # z_{1-α} = ppf(1-α) is negative (e.g. -1.645 for 95%)
    z = sp_stats.norm.ppf(1 - alpha)
    # Quantile of return distribution: q = μ + zσ  (a negative number for losses)
    # VaR = -q  (positive number representing loss)
    var = -(daily_mu + z * daily_sigma)

    # CVaR (Expected Shortfall) closed-form for Gaussian:
    # ES = -μ + σ · φ(z) / (1-α)   where φ is the standard normal pdf
    pdf_z = sp_stats.norm.pdf(z)
    cvar = -daily_mu + daily_sigma * pdf_z / (1 - alpha)

    return RiskResult(confidence=alpha, var=float(var), cvar=float(cvar))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_analytical_risk(
    returns: pd.DataFrame,
    baseline: BaselineStats,
    cfg: EngineConfig,
) -> AnalyticalRisk:
    """Compute historical and parametric VaR/CVaR at all confidence levels."""
    weights = cfg.resolved_weights(returns.shape[1])
    portfolio_returns: np.ndarray = returns.values @ weights

    hist: Dict[float, RiskResult] = {}
    para: Dict[float, RiskResult] = {}

    for alpha in cfg.confidence_levels:
        hist[alpha] = _historical_var_cvar(portfolio_returns, alpha)
        para[alpha] = _parametric_var_cvar(
            baseline.portfolio_return,
            baseline.portfolio_volatility,
            alpha,
            cfg.trading_days_per_year,
        )

    return AnalyticalRisk(historical=hist, parametric=para)
