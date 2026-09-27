"""Unit tests for the risk engine – run offline with synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk_engine.config import EngineConfig
from portfolio_risk_engine.monte_carlo import run_simulation
from portfolio_risk_engine.reporting import compute_mc_risk
from portfolio_risk_engine.risk_metrics import compute_analytical_risk
from portfolio_risk_engine.statistics import compute_baseline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_data(
    n_days: int = 504,
    n_assets: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic price and return data for deterministic tests."""
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i}" for i in range(n_assets)]
    dates = pd.bdate_range("2022-01-01", periods=n_days + 1)

    # Simulate correlated log-returns
    mu = rng.uniform(0.0002, 0.001, n_assets)
    cov = np.eye(n_assets) * 0.0004
    cov += 0.0001  # add some correlation
    L = np.linalg.cholesky(cov)

    log_ret = mu[np.newaxis, :] + rng.standard_normal((n_days, n_assets)) @ L.T
    cum_ret = np.cumsum(log_ret, axis=0)
    cum_ret = np.vstack([np.zeros((1, n_assets)), cum_ret])

    prices_arr = 100.0 * np.exp(cum_ret)
    prices = pd.DataFrame(prices_arr, index=dates, columns=tickers)
    returns = pd.DataFrame(log_ret, index=dates[1:], columns=tickers)
    return prices, returns


@pytest.fixture
def synth():
    return _synthetic_data()


@pytest.fixture
def cfg():
    return EngineConfig(
        tickers=["ASSET_0", "ASSET_1", "ASSET_2", "ASSET_3"],
        n_simulations=2_000,
        horizon_days=63,
        seed=99,
        initial_portfolio_value=1_000_000.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_shapes(self, synth, cfg):
        _, returns = synth
        b = compute_baseline(returns, cfg)
        assert b.ann_returns.shape == (4,)
        assert b.ann_volatility.shape == (4,)
        assert b.cov_matrix.shape == (4, 4)
        assert b.corr_matrix.shape == (4, 4)

    def test_positive_volatility(self, synth, cfg):
        _, returns = synth
        b = compute_baseline(returns, cfg)
        assert (b.ann_volatility > 0).all()

    def test_correlation_diagonal(self, synth, cfg):
        _, returns = synth
        b = compute_baseline(returns, cfg)
        np.testing.assert_allclose(np.diag(b.corr_matrix.values), 1.0)


class TestAnalyticalRisk:
    def test_var_positive(self, synth, cfg):
        prices, returns = synth
        b = compute_baseline(returns, cfg)
        ar = compute_analytical_risk(returns, b, cfg)
        for alpha in cfg.confidence_levels:
            assert ar.historical[alpha].var > 0
            assert ar.parametric[alpha].var > 0

    def test_cvar_ge_var(self, synth, cfg):
        """CVaR should always be ≥ VaR (it's the expected loss beyond VaR)."""
        prices, returns = synth
        b = compute_baseline(returns, cfg)
        ar = compute_analytical_risk(returns, b, cfg)
        for alpha in cfg.confidence_levels:
            assert ar.historical[alpha].cvar >= ar.historical[alpha].var - 1e-10
            assert ar.parametric[alpha].cvar >= ar.parametric[alpha].var - 1e-10

    def test_higher_confidence_higher_var(self, synth, cfg):
        """99% VaR should be larger than 95% VaR."""
        prices, returns = synth
        b = compute_baseline(returns, cfg)
        ar = compute_analytical_risk(returns, b, cfg)
        assert ar.historical[0.99].var > ar.historical[0.95].var
        assert ar.parametric[0.99].var > ar.parametric[0.95].var


class TestMonteCarlo:
    def test_price_path_shape(self, synth, cfg):
        prices, returns = synth
        b = compute_baseline(returns, cfg)
        mc = run_simulation(prices, returns, b, cfg)
        assert mc.price_paths.shape == (
            cfg.n_simulations,
            cfg.horizon_days + 1,
            4,
        )

    def test_terminal_values_positive(self, synth, cfg):
        prices, returns = synth
        b = compute_baseline(returns, cfg)
        mc = run_simulation(prices, returns, b, cfg)
        assert (mc.terminal_values > 0).all()

    def test_mc_risk_output(self, synth, cfg):
        prices, returns = synth
        b = compute_baseline(returns, cfg)
        mc = run_simulation(prices, returns, b, cfg)
        mc_risk = compute_mc_risk(mc, cfg)
        for alpha in cfg.confidence_levels:
            assert mc_risk[alpha].var_dollar > 0
            assert mc_risk[alpha].cvar_dollar >= mc_risk[alpha].var_dollar - 1.0


class TestConfig:
    def test_equal_weights(self):
        cfg = EngineConfig()
        w = cfg.resolved_weights(8)
        np.testing.assert_allclose(w.sum(), 1.0)
        np.testing.assert_allclose(w, 0.125)

    def test_custom_weights_validation(self):
        cfg = EngineConfig(weights=[0.5, 0.5])
        w = cfg.resolved_weights(2)
        np.testing.assert_allclose(w, [0.5, 0.5])

    def test_weight_mismatch_raises(self):
        cfg = EngineConfig(weights=[0.5, 0.5])
        with pytest.raises(ValueError):
            cfg.resolved_weights(3)
