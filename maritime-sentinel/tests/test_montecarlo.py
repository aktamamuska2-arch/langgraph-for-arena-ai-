"""Tests for the Latin Hypercube ensemble driver."""

from __future__ import annotations

import numpy as np
import pytest

from maritime_sentinel.metrics import cvar, percentile_table, spearman
from maritime_sentinel.montecarlo import (
    apply_perturbation,
    baseline_spec,
    latin_hypercube,
    run_baseline,
    run_monte_carlo,
    scenario_key,
)


def test_latin_hypercube_is_stratified_in_every_dimension() -> None:
    rng = np.random.default_rng(0)
    n, d = 16, 5
    samples = latin_hypercube(n, d, rng)
    assert samples.shape == (n, d)
    assert np.all((samples >= 0) & (samples < 1))
    for j in range(d):
        strata = np.floor(samples[:, j] * n).astype(int)
        assert len(np.unique(strata)) == n, "each stratum must be sampled exactly once"


def test_latin_hypercube_respects_bounds() -> None:
    rng = np.random.default_rng(3)
    samples = latin_hypercube(8, 2, rng)
    scaled = 0.9 + samples * (1.2 - 0.9)
    assert scaled.min() >= 0.9 and scaled.max() < 1.2


def test_apply_perturbation_only_touches_the_requested_field(scenario) -> None:  # noqa: ANN001
    updated = apply_perturbation(scenario, "weather_intensity", 2.0)
    assert updated.stress.weather_intensity == 2.0
    assert updated.stress.geo_intensity == scenario.stress.geo_intensity
    assert updated.network == scenario.network
    with pytest.raises(KeyError):
        apply_perturbation(scenario, "nonsense", 1.0)


def test_baseline_strips_every_stress_source(scenario) -> None:  # noqa: ANN001
    base = baseline_spec(scenario)
    assert base.stress.geo_intensity == 0.0
    assert base.stress.shock is None
    assert base.stress.weather_intensity < scenario.stress.weather_intensity


def test_baseline_throughput_is_reported(short_scenario) -> None:  # noqa: ANN001
    reference = run_baseline(short_scenario, replications=2)
    assert reference.throughput_teu_day > 0
    assert reference.effective_throughput_teu_day > 0
    assert len(reference.rows) == 2


def test_monte_carlo_ensemble_shape_and_aggregates(short_scenario) -> None:  # noqa: ANN001
    mc = run_monte_carlo(short_scenario, n_replications=4, seed=77)
    assert len(mc.outcomes) == 4
    assert len(mc.samples) == 4
    assert mc.aggregate["replications"] == 4
    assert mc.aggregate["baseline_throughput"] > 0
    assert 0.0 <= mc.aggregate["p_instability"] <= 1.0
    assert not mc.fan_chart.empty
    assert set(mc.fan_chart.columns) >= {"t_day", "p05", "p50", "p95"}
    assert not mc.port_table.empty
    assert (mc.outcomes["resilience_index"] >= 0).all()


def test_monte_carlo_is_reproducible(short_scenario) -> None:  # noqa: ANN001
    a = run_monte_carlo(short_scenario, n_replications=3, seed=5)
    b = run_monte_carlo(short_scenario, n_replications=3, seed=5)
    assert a.outcomes["throughput_teu_day"].tolist() == pytest.approx(
        b.outcomes["throughput_teu_day"].tolist()
    )


def test_tornado_is_ordered_by_absolute_effect(short_scenario) -> None:  # noqa: ANN001
    mc = run_monte_carlo(short_scenario, n_replications=8, seed=13)
    if mc.tornado.empty:
        pytest.skip("no perturbations configured")
    magnitudes = mc.tornado["rho_resilience"].abs().tolist()
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert mc.tornado["rho_resilience"].between(-1, 1).all()
    assert mc.tornado["rho_p95_sojourn"].between(-1, 1).all()


def test_risk_helpers() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert percentile_table(values, [50.0])["p50"] == pytest.approx(3.0)
    low = np.linspace(0, 1, 101)
    assert cvar(low, 95.0) < float(np.median(low))
    assert spearman(np.arange(10), np.arange(10)) == pytest.approx(1.0)
    assert spearman(np.arange(10), -np.arange(10)) == pytest.approx(-1.0)
    assert spearman(np.ones(10), np.arange(10)) == 0.0


def test_scenario_key_is_sensitive_to_stress(scenario) -> None:  # noqa: ANN001
    a = scenario_key(scenario)
    b = scenario_key(apply_perturbation(scenario, "demand_multiplier", 1.1))
    assert a != b and len(a) == 16
