"""Entry-point for the Algorithmic Portfolio Risk Engine.

Usage:
    python -m portfolio_risk_engine          # run with defaults
    python -m portfolio_risk_engine --help   # show CLI options
"""

from __future__ import annotations

import argparse
import sys
import time

from .config import EngineConfig
from .data import load_data
from .monte_carlo import run_simulation
from .reporting import compute_mc_risk, print_report
from .risk_metrics import compute_analytical_risk
from .statistics import compute_baseline


def _parse_args(argv: list[str] | None = None) -> EngineConfig:
    p = argparse.ArgumentParser(
        description="Algorithmic Portfolio Risk Engine",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Space-separated ticker symbols (default: diversified 8-asset universe)",
    )
    p.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--sims", type=int, default=10_000, help="Monte Carlo simulations")
    p.add_argument("--horizon", type=int, default=252, help="Simulation horizon (days)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--value",
        type=float,
        default=1_000_000.0,
        help="Initial portfolio value ($)",
    )
    args = p.parse_args(argv)

    kwargs: dict = dict(
        start_date=args.start,
        end_date=args.end,
        n_simulations=args.sims,
        horizon_days=args.horizon,
        seed=args.seed,
        initial_portfolio_value=args.value,
    )
    if args.tickers is not None:
        kwargs["tickers"] = args.tickers

    return EngineConfig(**kwargs)


def run(cfg: EngineConfig | None = None) -> None:
    """Execute the full pipeline."""
    if cfg is None:
        cfg = _parse_args()

    t0 = time.perf_counter()

    # Module 1 — Data
    print("▸ Fetching & cleaning data …")
    prices, returns = load_data(cfg)
    print(f"  {returns.shape[0]} trading days × {returns.shape[1]} assets")

    # Module 2 — Baseline Statistics
    print("▸ Computing baseline statistics …")
    baseline = compute_baseline(returns, cfg)

    # Module 3 — Analytical Risk
    print("▸ Computing analytical risk metrics …")
    analytical = compute_analytical_risk(returns, baseline, cfg)

    # Module 4 — Monte Carlo
    print(f"▸ Running Monte Carlo ({cfg.n_simulations:,} sims × {cfg.horizon_days}d) …")
    mc = run_simulation(prices, returns, baseline, cfg)

    # Module 5 — Reporting
    print("▸ Generating report …")
    mc_risk = compute_mc_risk(mc, cfg)
    print_report(baseline, analytical, mc_risk, mc, cfg)

    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.2f}s")


if __name__ == "__main__":
    run()
