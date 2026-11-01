"""Visual contract: the picture must actually carry the simulation's signal."""

from __future__ import annotations

import numpy as np

from agora.metrics import visual_contract
from agora.model import AgoraModel
from agora.render import render_frame


def test_frame_size_tracks_scale(default_spec):
    model = AgoraModel(default_spec)
    image = render_frame(model, scale=5)
    assert image.size == (model.width * 5, model.height * 5)
    assert image.mode == "RGB"


def test_frame_is_not_blank(warm_model):
    array = np.asarray(render_frame(warm_model, scale=6, hud=False), dtype=float)
    assert array.std() > 10
    colours = np.unique(array.reshape(-1, 3), axis=0)
    assert colours.shape[0] > 200, "the frame should be rich, not a flat wash"


def test_population_shading_grows_over_time(default_spec):
    model = AgoraModel(default_spec)
    early = model.pop_field()[0].astype(float)
    for _ in range(300):
        model.step()
    late = model.pop_field()[0].astype(float)
    assert late.mean() > early.mean()
    assert late.max() > early.max()


def test_hatching_tiers_are_present(warm_model):
    _, hatch = warm_model.pop_field()
    assert hatch.max() == 3
    assert (hatch == 1).sum() > 0
    assert (hatch >= 2).sum() > 0


def test_flows_land_anchored(warm_model):
    for edge in warm_model.edges.values():
        x, y = edge.path[-1]
        assert warm_model.world.terrain[y, x] != 0


def test_visual_contract_scores_full_mark(default_spec):
    model = AgoraModel(default_spec)
    before_pop, before_hatch = model.pop_field()

    class _Snapshot:
        def __init__(self, pop, hatch):
            self._v = (pop, hatch)

        def pop_field(self):
            return self._v

    snapshot = _Snapshot(before_pop, before_hatch)
    for _ in range(320):
        model.step()
    score, detail = visual_contract(snapshot, model)
    assert detail["checks"]["not_blank"]
    assert detail["checks"]["terrain_variety"]
    assert detail["checks"]["hatching_present"]
    assert detail["checks"]["flows_land_anchored"]
    assert score >= 0.99, detail["checks"]


def test_pulses_appear_during_activity(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(200):
        model.step()
    assert len(model.pulses) > 0
    frame = model.frame()
    assert frame["pulses"]
    assert all(0 <= p["age"] <= 13 for p in frame["pulses"])


def test_frame_payload_is_bounded(default_spec):
    model = AgoraModel(default_spec)
    for _ in range(260):
        model.step()
    frame = model.frame()
    assert len(frame["flows"]) <= 90
    assert len(frame["pulses"]) <= 480
    assert len(frame["history"]) <= 160
    for flow in frame["flows"]:
        assert 2 <= len(flow["p"]) <= 26


def test_scarcity_pulses_fire_when_granaries_run_dry(default_spec):
    """Consumption-side events must be visible, not just production."""

    model = AgoraModel(default_spec)
    for _ in range(60):
        model.step()
    # a food-poor settlement cannot feed itself, so scarcity is real rather than
    # a one-tick dip that its own fields instantly repair
    victim = min(model.settlements.values(), key=lambda s: float(s.endowment[0]))
    victim.endowment[0] = 0.0
    model.pulses.clear()
    for _ in range(14):
        victim.stocks[0] = 0.0  # keep it starving across the staggered window
        model.step()
        victim.endowment[0] = 0.0
    kinds = {p["kind"] for p in model.pulses}
    assert "scarcity" in kinds, kinds
    frame = model.frame()
    assert any(p["k"] == "scarcity" for p in frame["pulses"])
