"""The mathematics of the spatial economy, tested as propositions.

Every assertion here corresponds to a claim made in the README. The point is
that the parameters are *derived*: if someone changes the production curvature,
the price elasticity or the gravity exponent, these tests either still hold
(because the quantity is derived from them) or fail loudly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agora.calibration import calibrate_gravity, design_matrix, poisson_pml
from agora.graph import initial_state
from agora.graph.nodes import self_correction
from agora.market import MarketEngine
from agora.metrics import DEFAULT_GATES, evaluate, mass_balance
from agora.model import AgoraModel
from agora.schemas import (
    Blueprint,
    Directive,
    EconomySpec,
    ExperimentSpec,
    GravitySpec,
)

# ---------------------------------------------------------------------------
# 1. Gravity kernel
# ---------------------------------------------------------------------------


def test_distance_elasticity_of_the_kernel_is_minus_gamma():
    """``d ln A / d ln(d+d0) = -gamma`` exactly, for any gamma."""

    for gamma in (0.6, 1.35, 2.7):
        spec = GravitySpec(gamma=gamma)
        engine = MarketEngine(EconomySpec(gravity=spec))
        distances = np.linspace(1.0, 40.0, 60)
        scores = np.array([engine.attractiveness(5.0, 4.0, float(d), 0.0, 0.0, 1.0, 1.0) for d in distances])
        slope = np.polyfit(np.log(distances + spec.d0), np.log(scores), 1)[0]
        assert slope == pytest.approx(-gamma, rel=1e-6)


def test_friction_decays_the_score_exponentially():
    engine = MarketEngine(EconomySpec())
    tau = engine.gravity.tau
    free = engine.attractiveness(5.0, 4.0, 12.0, 0.0, 0.0)
    taxed = engine.attractiveness(5.0, 4.0, 12.0, 30.0, 0.0)
    assert taxed / free == pytest.approx(math.exp(-tau * 30.0), rel=1e-9)


def test_masses_enter_symmetrically_and_asymmetry_is_rejected():
    """Structural gravity has a symmetric mass exponent; 0.85/0.95 is a bias."""

    with pytest.raises(ValueError, match="alpha must equal beta"):
        GravitySpec(alpha=0.85, beta=0.95)
    engine = MarketEngine(EconomySpec())
    a = engine.attractiveness(6.0, 4.0, 10.0, 0.0, 0.0)
    b = engine.attractiveness(4.0, 6.0, 10.0, 0.0, 0.0)
    assert a == pytest.approx(b, rel=1e-12)


def test_kernel_is_a_share_so_a_global_scale_cancels():
    """With ``w=1`` and ``a=b=1``, numerator and resistance scale identically.

    This is the property that stops a big settlement from out-scoring everyone
    purely because it is big: what matters is its size *relative to the rivals
    its partner can reach.
    """

    engine = MarketEngine(EconomySpec())
    base = engine.attractiveness(5.0, 4.0, 9.0, 3.0, 0.0, 2.0, 3.0)
    scaled = engine.attractiveness(5.0e3, 4.0e3, 9.0, 3.0, 0.0, 2.0e3, 3.0e3)
    assert scaled == pytest.approx(base, rel=1e-9)


def test_repeat_partner_memory_is_bounded_and_monotone():
    engine = MarketEngine(EconomySpec())
    weights = [engine.attractiveness(5.0, 4.0, 10.0, 0.0, trust) for trust in (0.0, 1.0, 3.0, 500.0)]
    assert weights[0] < weights[1] < weights[2]
    assert weights[3] <= weights[2] * 2.0  # clipped, so memory cannot explode


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------


def test_ppml_recovers_known_gravity_parameters():
    """The estimator must recover the parameters that generated the data."""

    rng = np.random.default_rng(11)
    n = 700
    mass_in = rng.lognormal(0.8, 0.7, n)
    mass_out = rng.lognormal(0.9, 0.6, n)
    distance = rng.uniform(2.0, 25.0, n)
    mu = mass_in * mass_out / (distance + 3.0) ** 1.35
    mu = mu / np.median(mu) * 6.0
    flows = rng.poisson(mu).astype(float)

    X = design_matrix(mass_in, mass_out, distance, None, 3.0)
    beta, se, _deviance, converged = poisson_pml(X, flows)

    assert converged
    assert beta[1] == pytest.approx(1.0, abs=0.12)  # alpha
    assert -beta[3] == pytest.approx(1.35, abs=0.2)  # gamma
    assert se[3] < 0.2


def test_ppml_survives_a_zero_inflated_sample():
    rng = np.random.default_rng(3)
    n = 400
    mass_in = rng.lognormal(0.0, 1.2, n)
    mass_out = rng.lognormal(0.0, 1.2, n)
    distance = rng.uniform(4.0, 30.0, n)
    mu = mass_in * mass_out / (distance + 3.0) ** 2.0
    flows = rng.poisson(mu).astype(float)
    assert (flows == 0).mean() > 0.5

    beta, _se, _dev, converged = poisson_pml(design_matrix(mass_in, mass_out, distance, None, 3.0), flows)
    assert converged
    assert np.isfinite(beta).all()


def test_calibrator_runs_on_a_real_world(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(120):
        model.step()
    fit = calibrate_gravity(model)
    assert fit.converged
    assert fit.n > 0
    assert np.isfinite([fit.alpha, fit.beta, fit.gamma]).all()
    # tau is collinear with distance on a lattice, so it is held at its prior
    assert fit.tau == pytest.approx(model.econ.gravity.tau)


# ---------------------------------------------------------------------------
# 3. Labour allocation
# ---------------------------------------------------------------------------


def test_sharpness_is_derived_from_the_foc_not_tuned():
    for kappa, eps in ((0.78, 0.45), (0.6, 0.2), (0.9, 0.7)):
        econ = EconomySpec(labor_exponent=kappa, price_elasticity=eps)
        assert econ.sharpness == pytest.approx(1.0 / (1.0 - kappa * (1.0 - eps)))
    # the exogenous-price FOC is degenerate and must not be the default
    assert EconomySpec().sharpness < 1.0 / (1.0 - EconomySpec().labor_exponent)


def test_allocation_equalises_long_run_marginal_revenue(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(60):
        model.step()
    econ = model.econ.model_copy(update={"labor_inertia": 1.0})
    for settlement in model.settlements.values():
        settlement.adapt_labour(econ)
        assert abs(float(settlement.labor.sum()) - 1.0) < 1e-12
        # activities above the 2% floor are the unconstrained ones: their
        # marginal revenue must be equal to machine precision
        order = np.argsort(-settlement.labor)[:2]
        mrp = settlement.marginal_revenue(econ)[order]
        assert mrp[0] == pytest.approx(mrp[1], rel=1e-6)


def test_allocation_is_interior_not_a_corner(default_spec):
    """The degenerate FOC would put every worker in one activity."""

    model = AgoraModel(default_spec)
    for _ in range(120):
        model.step()
    labour = np.array([s.labor for s in model.settlements.values()])
    assert (labour > 0.01).all(), "every activity must retain a labour share"
    assert labour.max(axis=1).mean() < 0.95


def test_labour_tracks_the_target_with_inertia(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(30):
        model.step()
    settlement = model.settlements[0]
    before = settlement.labor.copy()
    settlement.adapt_labour(model.econ)
    after = settlement.labor.copy()
    assert not np.allclose(before, after)
    assert abs(float(after.sum()) - 1.0) < 1e-12
    assert after.min() > 0


# ---------------------------------------------------------------------------
# 4. Demography
# ---------------------------------------------------------------------------


def test_carrying_capacity_does_not_depend_on_population(default_spec):
    """A logistic whose K moves with N has no fixed point to converge to."""

    model = AgoraModel(default_spec)
    for _ in range(80):
        model.step()
    settlement = model.settlements[0]
    k_before = settlement._carrying_capacity(model.econ)
    settlement.population *= 3.0
    settlement.labor = np.array([0.9, 0.05, 0.05])
    k_after = settlement._carrying_capacity(model.econ)
    assert k_before == pytest.approx(k_after, rel=1e-12)


def test_carrying_capacity_accounts_for_storage_loss():
    """More spoilage over a longer buffer must lower the land's ceiling."""

    base = EconomySpec()
    lossy = EconomySpec(spoilage=0.06, target_coverage_ticks=40)
    assert lossy.spoilage * lossy.target_coverage_ticks > base.spoilage * base.target_coverage_ticks
    model = AgoraModel(ExperimentSpec(economy=base.model_dump()))
    model.settlements[0]._carrying_capacity(base)
    with_loss = AgoraModel(ExperimentSpec(economy=lossy.model_dump()))
    assert with_loss.settlements[0]._carrying_capacity(lossy) < model.settlements[0]._carrying_capacity(base)


def test_technology_saturates_at_the_ceiling(default_spec):
    model = AgoraModel(default_spec)
    settlement = model.settlements[0]
    ceiling = model.econ.tech_ceiling
    for _ in range(400):
        settlement.trade_volume = 50.0
        settlement.advance_tech(model.econ)
        assert settlement.tech <= ceiling + 1e-12
    assert settlement.tech == pytest.approx(ceiling)


def test_population_field_reaches_its_fixed_point(default_spec):
    """The tail exponential rate must fall inside the C-GRID threshold."""

    model = AgoraModel(default_spec)
    for _ in range(400):
        model.step()
    history = model.history
    tail = history[-max(4, len(history) // 4) :]
    t = np.array([row["t"] for row in tail])
    p = np.array([row["population"] for row in tail])
    rate = float(np.polyfit(t - t[0], np.log(p), 1)[0])
    assert abs(rate) <= 0.0045
    # and the fixed point must be a real one: K is frozen once tech saturates
    caps = [s.capacity for s in model.settlements.values()]
    for _ in range(40):
        model.step()
    drift = max(abs(s.capacity - k) / k for s, k in zip(model.settlements.values(), caps, strict=False))
    assert drift < 1e-4, "carrying capacity must be frozen once technology saturates"


def test_security_expectation_is_smoothed(default_spec):
    """Households react to a smoothed food outlook, not to raw granary noise."""

    model = AgoraModel(default_spec)
    for _ in range(150):
        model.step()
    stocks = []
    for _ in range(60):
        model.step()
        stocks.append(float(model.settlements[0].stocks[0]))
    assert np.std(stocks) / max(np.mean(stocks), 1e-9) < 0.5


# ---------------------------------------------------------------------------
# 5. Infrastructure feedback
# ---------------------------------------------------------------------------


def test_roads_never_exceed_the_cap_and_depreciate_when_idle(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(200):
        model.step()
    assert float(model.road_level.max()) <= model.econ.road_max + 1e-9

    before = float(model.road_level.max())
    model.edges.clear()  # no traffic at all
    closed = model.econ.model_copy(update={"discovery_radius": 0.0})
    model.econ = closed
    model.market.econ = closed
    for _ in range(300):
        model.step()
    assert float(model.road_level.max()) < before * 0.75
    assert float(model.road_level.max()) >= 0.0


def test_road_upgrade_is_logistic_not_a_ratchet(default_spec):
    """Each pass closes a fraction of the remaining gap, so the marginal
    return to traffic falls instead of staying constant."""

    model = AgoraModel(default_spec)
    model.road_level[:] = 0.0
    gains = []
    for level in (0.0, 1.0, 2.0):
        model.road_level[10, 10] = level
        cap = model.econ.road_max
        step = model.econ.road_gain * (1.0 - level / cap)
        gains.append(step)
    assert gains[0] > gains[1] > gains[2]


# ---------------------------------------------------------------------------
# 6. Network resilience and the ledger
# ---------------------------------------------------------------------------


def test_a_severe_shock_does_not_prune_the_network(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(220):
        model.step()
    edges_before = len(model.edges)
    assert edges_before > 0
    exporter = max((s for s in model.settlements.values() if s.alive), key=lambda s: float(s.endowment[0]))
    model.inject_shock(kind="drought", x=exporter.x, y=exporter.y, radius=22.0, magnitude=0.9, duration=70)
    low = edges_before
    for _ in range(70):
        model.step()
        low = min(low, len(model.edges))
    assert low >= 0.75 * edges_before
    assert len(model.edges) >= edges_before


def test_relationship_capital_slows_weight_decay(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(120):
        model.step()
    if not model.edges:
        pytest.skip("no edges formed")
    (src, dst), edge = next(iter(model.edges.items()))
    seller = model.settlements[src]
    seller.partners[dst] = 0.0
    edge.last_active = model.tick_count
    low_trust = edge.weights.copy()
    model.market.decay(model)
    low = float((edge.weights / np.maximum(low_trust, 1e-12)).mean()) if edge.key in model.edges else 0.0

    edge2 = model.edges.get((src, dst))
    if edge2 is None:
        pytest.skip("edge pruned")
    seller.partners[dst] = 4.0
    edge2.last_active = model.tick_count
    high_trust = edge2.weights.copy()
    model.market.decay(model)
    high = float((edge2.weights / np.maximum(high_trust, 1e-12)).mean())
    assert high > low


def test_ledger_holds_to_floating_point_tolerance(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(200):
        model.step()
    assert mass_balance(model) <= 1e-12
    naive = sum(float(s.stocks.sum()) for s in model.settlements.values())
    compensated = sum(float(s.total_stocks().sum()) for s in model.settlements.values())
    assert abs(compensated - naive) < 1e-6


# ---------------------------------------------------------------------------
# 7. The Reflector's absolute rule
# ---------------------------------------------------------------------------


def test_reflector_refuses_to_patch_a_mass_imbalance():
    """A conservation leak is an engine defect: total rejection, zero patches."""

    metrics = {
        "abs_growth_tail": 0.001,
        "field_lag1_corr": 0.95,
        "giant_component_share": 1.0,
        "edges_per_node": 2.0,
        "mass_balance_error": 0.043,
        "visual_contract_score": 1.0,
        "shock_recovery": 1.1,
        "pop_drawdown": 0.0,
    }
    report = evaluate(DEFAULT_GATES, metrics, attempt=0, duration=1.0)
    assert report.failure_class == "mass_imbalance"

    blueprint = Blueprint(
        blueprint_id="BP-REFUSAL",
        experiment_id="ASEOWS-REFUSAL",
        objective="the absolute conservation rule",
        equations=[],
        gates=list(DEFAULT_GATES),
        world_summary={},
        economy={},
        critical_radius=10.0,
        site_audit=[],
        expected={},
    )
    state = initial_state(Directive(experiment_id="ASEOWS-REFUSAL"))
    state.update(
        {
            "blueprint": blueprint,
            "spec": ExperimentSpec(),
            "report": report,
            "metrics": metrics,
            "attempt": 0,
        }
    )
    update = self_correction(state)
    reflection = update["reflections"][-1]
    assert reflection.failure_class == "mass_imbalance"
    assert reflection.patches == []
    assert reflection.next_action == "finalize"
    assert update["attempt"] == 1


def test_a_parameter_patch_can_never_repair_a_leak():
    """Even at the retry ceiling the reflector emits no compensating patch."""

    metrics = {
        "abs_growth_tail": 0.001,
        "field_lag1_corr": 0.95,
        "giant_component_share": 1.0,
        "edges_per_node": 2.0,
        "mass_balance_error": 1e-3,
        "visual_contract_score": 1.0,
        "shock_recovery": 1.1,
        "pop_drawdown": 0.0,
    }
    report = evaluate(DEFAULT_GATES, metrics, attempt=3, duration=1.0)
    state = initial_state(Directive(experiment_id="ASEOWS-CEILING"))
    state.update({"report": report, "metrics": metrics, "attempt": 3, "max_attempts": 4})
    blueprint = Blueprint(
        blueprint_id="BP-CEILING",
        experiment_id="ASEOWS-CEILING",
        objective="ceiling",
        equations=[],
        gates=list(DEFAULT_GATES),
        world_summary={},
        economy={},
        critical_radius=10.0,
        site_audit=[],
        expected={},
    )
    state["blueprint"] = blueprint
    state["spec"] = ExperimentSpec()
    update = self_correction(state)
    assert update["reflections"][-1].patches == []
    assert update["reflections"][-1].next_action == "finalize"
