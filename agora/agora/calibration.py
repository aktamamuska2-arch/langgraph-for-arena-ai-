"""Structural-gravity calibration.

Why a calibrator and not a regression of raw flows
--------------------------------------------------
The naive regression

    ln(flow_ij) = c + a ln S_i + b ln D_j - g ln(d_ij + d0)

is biased for three separate reasons:

1. **Zeros.** Most ordered pairs never trade. Dropping them, or using ``ln(flow)``
   after adding a constant, throws away the extensive margin, which is precisely
   the margin the kernel governs.
2. **Multilateral resistance.** ``Pi_i`` and ``P_j`` enter the true model and are
   correlated with distance, so omitting them loads the distance coefficient with
   everything else (Anderson & van Wincoop, 2003).
3. **Spatial autocorrelation of comparative advantage.** Neighbouring settlements
   sit on similar land and therefore have *less* to trade with each other, which
   pushes the measured distance coefficient toward zero or even positive. This is
   the dominant effect in ASEOWS and is reported explicitly by
   ``distance_profile``.

The estimator implemented here is Poisson pseudo-maximum-likelihood (Santos Silva
& Tenreyro, 2006): it handles zeros natively, is consistent under
heteroscedasticity, and with exporter/importer fixed effects it is the PPML
structural-gravity estimator. Compensating the extensive margin with a kernel
that is normalised by multilateral resistance is what makes the estimator able to
recover the parameters that generated the link set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schemas import RESOURCES, EconomySpec

__all__ = ["GravityFit", "calibrate_gravity", "design_matrix", "distance_profile", "poisson_pml"]


@dataclass(frozen=True)
class GravityFit:
    """A fitted gravity equation."""

    alpha: float
    beta: float
    gamma: float
    tau: float
    se: dict[str, float]
    n: int
    zeros: int
    deviance: float
    converged: bool

    def as_dict(self) -> dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "tau": self.tau,
            "se_gamma": self.se.get("gamma", float("nan")),
            "n": float(self.n),
            "zeros": float(self.zeros),
            "deviance": self.deviance,
        }

    def z_score(self, name: str, value: float) -> float:
        se = self.se.get(name, 0.0)
        estimate = {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "tau": self.tau}[name]
        return 0.0 if se <= 0 else (estimate - value) / se


def poisson_pml(
    X: np.ndarray, y: np.ndarray, iters: int = 200, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit ``E[y] = exp(X b)`` by Newton-Raphson (Poisson PML).

    Returns ``(beta, standard_errors, deviance, converged)``.
    """

    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(float(y.mean()), 1e-9))
    ridge = 1e-10 * np.eye(X.shape[1])
    converged = False
    for _ in range(iters):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        weight = mu + 1e-9
        hessian = (X.T * weight) @ X + ridge
        step = np.linalg.solve(hessian, X.T @ (y - mu))
        beta = beta + step
        if float(np.abs(step).max()) < tol:
            converged = True
            break
    mu = np.exp(np.clip(X @ beta, -30.0, 30.0))
    weight = mu + 1e-9
    cov = np.linalg.inv((X.T * weight) @ X + ridge)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / np.maximum(mu, 1e-12)), 0.0)
    deviance = float(2.0 * np.sum(term - (y - mu)))
    return beta, se, deviance, converged


def design_matrix(
    surplus: np.ndarray,
    deficit: np.ndarray,
    distance: np.ndarray,
    transport_cost: np.ndarray | None,
    d0: float,
) -> np.ndarray:
    """Regressors: constant, exporter mass, importer mass, separation[, friction].

    The friction term is only included when it is not collinear with separation.
    On a lattice ``c_ij`` is very nearly proportional to ``d_ij``, so estimating
    ``gamma`` and ``tau`` together is an identification failure, not a finding.
    """

    columns = [
        np.ones(len(distance)),
        np.log(np.maximum(surplus, 1e-9)),
        np.log(np.maximum(deficit, 1e-9)),
        np.log(np.maximum(distance, 0.0) + d0),
    ]
    if transport_cost is not None:
        columns.append(np.maximum(transport_cost, 0.0))
    return np.column_stack(columns)


def _friction_is_identified(distance: np.ndarray, cost: np.ndarray, d0: float) -> bool:
    """A friction term is only estimable when it is not a distance proxy."""

    if cost.size < 4 or float(np.std(cost)) <= 1e-12:
        return False
    sep = np.log(np.maximum(distance, 0.0) + d0)
    corr = float(np.corrcoef(sep, cost)[0, 1])
    return abs(corr) < 0.98


def calibrate_gravity(
    model,
    econ: EconomySpec | None = None,
    resources: tuple[int, ...] | None = None,
    fit_friction: bool = False,
) -> GravityFit:
    """Estimate ``(alpha, beta, gamma, tau)`` from a realised world.

    Every ordered settlement pair inside the discovery radius contributes one
    observation (zeros included); ``y`` is the pair's delivered volume.
    """

    econ = econ or model.econ
    d0 = econ.gravity.d0
    resources = resources if resources is not None else tuple(range(len(RESOURCES)))
    alive = [s for s in model.settlements.values() if s.alive]

    rows: list[list[float]] = []
    flows: list[float] = []
    for seller in alive:
        surplus = seller.marketable(econ)
        for buyer in alive:
            if buyer.sid == seller.sid:
                continue
            distance = math.hypot(buyer.x - seller.x, buyer.y - seller.y)
            if distance > econ.discovery_radius:
                continue
            deficit = buyer.deficit(econ)
            for r in resources:
                if surplus[r] <= 1e-6 or deficit[r] <= 1e-6:
                    continue
                edge = model.edges.get((seller.sid, buyer.sid))
                volume = 0.0
                if edge is not None:
                    volume = float(edge.cum_flow) * float(edge.weights[r]) / max(float(edge.weights.sum()), 1e-9)
                cost = edge.cost if edge is not None else model.route_cost(seller.pos, buyer.pos)
                rows.append(
                    (
                        float(surplus[r]),
                        float(deficit[r]),
                        distance,
                        float(cost),
                    )
                )
                flows.append(volume)

    masses = np.array(rows, dtype=np.float64)
    y = np.array(flows, dtype=np.float64)
    # `tau` is held at its prior by default: on a lattice the cheapest route is
    # nearly proportional to separation, so estimating the two together is an
    # identification failure that produces a wildly inflated friction estimate.
    if fit_friction and not _friction_is_identified(masses[:, 2], masses[:, 3], d0):
        raise ValueError("transport cost is collinear with separation; fix tau instead of fitting it")
    X = design_matrix(masses[:, 0], masses[:, 1], masses[:, 2], masses[:, 3] if fit_friction else None, d0)
    beta, se, deviance, converged = poisson_pml(X, y)
    return GravityFit(
        alpha=float(beta[1]),
        beta=float(beta[2]),
        gamma=float(-beta[3]),
        tau=float(-beta[4]) if fit_friction else econ.gravity.tau,
        se={
            "alpha": float(se[1]),
            "beta": float(se[2]),
            "gamma": float(se[3]),
            "tau": float(se[4]) if fit_friction else float("nan"),
        },
        n=len(y),
        zeros=int((y <= 0).sum()),
        deviance=deviance,
        converged=converged,
    )


def distance_profile(model, econ: EconomySpec | None = None, bins: int = 4) -> dict[str, float]:
    """Link density and mean flow by distance band.

    The shape of this profile is what distinguishes genuine distance decay from
    endowment autocorrelation: a decaying *density* with a flat or rising *flow
    per link* means nearby partners are simply less complementary, not that
    distance is being penalised.
    """

    econ = econ or model.econ
    alive = [s for s in model.settlements.values() if s.alive]
    pairs: list[tuple[float, int, float]] = []
    for seller in alive:
        for buyer in alive:
            if buyer.sid == seller.sid:
                continue
            distance = math.hypot(buyer.x - seller.x, buyer.y - seller.y)
            if distance > econ.discovery_radius:
                continue
            edge = model.edges.get((seller.sid, buyer.sid))
            pairs.append((distance, 1 if edge is not None else 0, float(edge.total_flow) if edge else 0.0))
    if not pairs:
        return {}
    data = np.array(pairs)
    edges = np.linspace(data[:, 0].min(), data[:, 0].max() + 1e-6, bins + 1)
    out: dict[str, float] = {}
    for i in range(bins):
        mask = (data[:, 0] >= edges[i]) & (data[:, 0] < edges[i + 1])
        if not mask.any():
            continue
        out[f"band{i}_lo"] = float(edges[i])
        out[f"band{i}_density"] = float(data[mask, 1].mean())
        traded = data[mask & (data[:, 1] > 0)]
        out[f"band{i}_flow"] = float(traded[:, 2].mean()) if len(traded) else 0.0
    return out
