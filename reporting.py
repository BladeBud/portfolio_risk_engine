"""Module 5 – Risk Reporting.

Extract Monte Carlo VaR / CVaR and produce a unified console report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import EngineConfig
from .monte_carlo import MonteCarloResult
from .risk_metrics import AnalyticalRisk, RiskResult
from .statistics import BaselineStats


# ---------------------------------------------------------------------------
# Monte Carlo VaR / CVaR
# ---------------------------------------------------------------------------


@dataclass
class MCRiskResult:
    """Monte Carlo risk at a single confidence level."""

    confidence: float
    var_pct: float       # VaR as a fraction of initial value
    cvar_pct: float      # CVaR as a fraction of initial value
    var_dollar: float    # VaR in dollar terms
    cvar_dollar: float   # CVaR in dollar terms


def _mc_var_cvar(
    portfolio_returns: np.ndarray,
    alpha: float,
    initial_value: float,
) -> MCRiskResult:
    sorted_ret = np.sort(portfolio_returns)
    cutoff = int(np.floor((1 - alpha) * len(sorted_ret)))
    var_pct = -sorted_ret[cutoff]
    cvar_pct = -sorted_ret[: cutoff + 1].mean()
    return MCRiskResult(
        confidence=alpha,
        var_pct=float(var_pct),
        cvar_pct=float(cvar_pct),
        var_dollar=float(var_pct * initial_value),
        cvar_dollar=float(cvar_pct * initial_value),
    )


def compute_mc_risk(
    mc: MonteCarloResult,
    cfg: EngineConfig,
) -> Dict[float, MCRiskResult]:
    results: Dict[float, MCRiskResult] = {}
    for alpha in cfg.confidence_levels:
        results[alpha] = _mc_var_cvar(
            mc.portfolio_returns, alpha, cfg.initial_portfolio_value
        )
    return results


# ---------------------------------------------------------------------------
# Formatted console report
# ---------------------------------------------------------------------------

_SEP = "─" * 72


def print_report(
    baseline: BaselineStats,
    analytical: AnalyticalRisk,
    mc_risk: Dict[float, MCRiskResult],
    mc: MonteCarloResult,
    cfg: EngineConfig,
) -> str:
    """Build and print the full risk report.  Returns the report string."""
    lines: list[str] = []

    def _h(title: str) -> None:
        lines.append("")
        lines.append(_SEP)
        lines.append(f"  {title}")
        lines.append(_SEP)

    # -- Header ---------------------------------------------------------------
    lines.append("")
    lines.append("═" * 72)
    lines.append("  ALGORITHMIC PORTFOLIO RISK ENGINE — REPORT")
    lines.append("═" * 72)

    # -- Baseline Statistics --------------------------------------------------
    _h("1 │ BASELINE STATISTICS")
    lines.append(f"  {'Asset':<10} {'Ann. Return':>14} {'Ann. Vol':>14}")
    lines.append(f"  {'─'*10} {'─'*14} {'─'*14}")
    for ticker in baseline.ann_returns.index:
        r = baseline.ann_returns[ticker]
        v = baseline.ann_volatility[ticker]
        lines.append(f"  {ticker:<10} {r:>+13.2%} {v:>13.2%}")

    lines.append("")
    lines.append(f"  Portfolio Ann. Return     : {baseline.portfolio_return:>+.4%}")
    lines.append(f"  Portfolio Ann. Volatility : {baseline.portfolio_volatility:>.4%}")

    # -- Correlation ----------------------------------------------------------
    _h("2 │ CORRELATION MATRIX")
    corr_str = baseline.corr_matrix.to_string(float_format=lambda x: f"{x:+.2f}")
    for row in corr_str.split("\n"):
        lines.append(f"  {row}")

    # -- Analytical VaR / CVaR ------------------------------------------------
    _h("3 │ ANALYTICAL RISK METRICS (daily)")

    lines.append(f"  {'Method':<14} {'Conf':>6} {'VaR':>12} {'CVaR':>12}")
    lines.append(f"  {'─'*14} {'─'*6} {'─'*12} {'─'*12}")

    for alpha in cfg.confidence_levels:
        h = analytical.historical[alpha]
        p = analytical.parametric[alpha]
        lines.append(
            f"  {'Historical':<14} {alpha:>5.0%} {h.var:>+11.4%} {h.cvar:>+11.4%}"
        )
        lines.append(
            f"  {'Parametric':<14} {alpha:>5.0%} {p.var:>+11.4%} {p.cvar:>+11.4%}"
        )

    # -- Monte Carlo ----------------------------------------------------------
    _h(f"4 │ MONTE CARLO RISK  ({cfg.n_simulations:,} sims × {cfg.horizon_days}d)")

    lines.append(
        f"  {'Conf':>6} {'VaR %':>10} {'CVaR %':>10} "
        f"{'VaR $':>14} {'CVaR $':>14}"
    )
    lines.append(
        f"  {'─'*6} {'─'*10} {'─'*10} {'─'*14} {'─'*14}"
    )
    for alpha in cfg.confidence_levels:
        m = mc_risk[alpha]
        lines.append(
            f"  {alpha:>5.0%} {m.var_pct:>+9.2%} {m.cvar_pct:>+9.2%} "
            f"  {m.var_dollar:>+13,.0f}   {m.cvar_dollar:>+13,.0f}"
        )

    # -- Terminal distribution summary ----------------------------------------
    _h("5 │ TERMINAL PORTFOLIO DISTRIBUTION")

    tv = mc.terminal_values
    pcts = np.percentile(tv, [1, 5, 25, 50, 75, 95, 99])
    labels = ["1st", "5th", "25th", "50th", "75th", "95th", "99th"]
    lines.append(f"  Initial Value : ${cfg.initial_portfolio_value:>14,.0f}")
    lines.append(f"  Mean Terminal : ${tv.mean():>14,.0f}")
    lines.append(f"  Std Dev       : ${tv.std():>14,.0f}")
    lines.append("")
    lines.append(f"  {'Percentile':>12} {'Value':>16}")
    lines.append(f"  {'─'*12} {'─'*16}")
    for lbl, val in zip(labels, pcts):
        lines.append(f"  {lbl:>12} ${val:>15,.0f}")

    lines.append("")
    lines.append("═" * 72)
    lines.append("  END OF REPORT")
    lines.append("═" * 72)
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report
