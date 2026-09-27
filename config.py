"""Engine configuration and default parameters."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for the risk engine."""

    # Universe
    tickers: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "XOM", "JNJ"]
    )
    weights: List[float] | None = None  # None → equal-weight

    # Data
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    trading_days_per_year: int = 252

    # Risk
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])

    # Monte Carlo
    n_simulations: int = 10_000
    horizon_days: int = 252
    seed: int = 42

    # Portfolio
    initial_portfolio_value: float = 1_000_000.0

    def resolved_weights(self, n_assets: int):
        """Return explicit weight array, defaulting to equal-weight."""
        import numpy as np

        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            if w.shape[0] != n_assets:
                raise ValueError(
                    f"Weight vector length {w.shape[0]} != number of assets {n_assets}"
                )
            if not np.isclose(w.sum(), 1.0):
                raise ValueError(f"Weights must sum to 1.0, got {w.sum():.6f}")
            return w
        return np.full(n_assets, 1.0 / n_assets)
