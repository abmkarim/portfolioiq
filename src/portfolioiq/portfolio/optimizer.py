"""Monthly mean-variance optimisation with CVXPY.

maximise  alpha'w - lambda w'Sigma w - kappa ||w - w_prev||_1
s.t.      sum(w) = 0                      (dollar neutral)
          sum_{i in s} w_i = 0  for all s  (sector neutral)
          ||w||_1 <= max_gross             (gross leverage)
          |w_i| <= max_weight
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from portfolioiq.utils import get_logger

log = get_logger(__name__)


def optimize_weights(
    alpha: np.ndarray, cov: np.ndarray, sectors: np.ndarray, w_prev: np.ndarray,
    risk_aversion: float, kappa: float, max_weight: float, max_gross: float, solver: str = "CLARABEL",
) -> np.ndarray:
    """Solve the optimisation problem; returns weights (zeros if every solver fails)."""
    n = len(alpha)
    if n == 0:
        return np.zeros(0)
    w = cp.Variable(n)
    # factorised risk term keeps the problem a well-conditioned SOCP
    evals, evecs = np.linalg.eigh((cov + cov.T) / 2)
    L = evecs * np.sqrt(np.clip(evals, 0.0, None))
    risk = cp.sum_squares(L.T @ w)
    obj = cp.Maximize(alpha @ w - risk_aversion * risk - kappa * cp.norm1(w - w_prev))
    cons = [cp.sum(w) == 0, cp.norm1(w) <= max_gross, cp.abs(w) <= max_weight]
    for s in np.unique(sectors):
        idx = np.where(sectors == s)[0]
        cons.append(cp.sum(w[idx]) == 0)
    prob = cp.Problem(obj, cons)
    for name in [solver, "CLARABEL", "ECOS", "SCS"]:
        try:
            prob.solve(solver=name)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return clean_weights(np.asarray(w.value).ravel(), sectors, max_weight, max_gross)
        except Exception as exc:  # noqa: BLE001
            log.warning("solver failed", solver=name, error=str(exc)[:160])
    log.warning("optimisation failed, flat portfolio", n=n)
    return np.zeros(n)


def clean_weights(w: np.ndarray, sectors: np.ndarray, max_weight: float, max_gross: float,
                  tol: float = 1e-6) -> np.ndarray:
    """Remove solver noise so constraints hold exactly (tiny weights -> 0, re-centre per sector)."""
    w = np.where(np.abs(w) < tol, 0.0, w)
    for _ in range(5):
        w = np.clip(w, -max_weight, max_weight)
        for s in np.unique(sectors):
            idx = sectors == s
            nz = idx & (w != 0)
            if nz.any():
                w[nz] -= w[idx].sum() / nz.sum()
    w = np.clip(w, -max_weight, max_weight)
    gross = np.abs(w).sum()
    if gross > max_gross:
        w *= max_gross / gross
    return w
