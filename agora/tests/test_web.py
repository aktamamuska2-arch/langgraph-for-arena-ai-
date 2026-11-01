"""HTTP contract of the interactive console."""

from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

from web.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/api/control", json={"action": "pause"})
        yield test_client
        test_client.post("/api/control", json={"action": "pause"})


def test_index_serves_the_console(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "ASEOWS" in body
    assert "<canvas" in body
    assert "/api/state" in body


def test_health(client):
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert payload["tick"] >= 0


def test_meta_payload(client):
    meta = client.get("/api/meta").json()
    assert meta["width"] == 96 and meta["height"] == 64
    assert len(meta["terrain"]) == 96 * 64
    assert len(meta["settlements"]) == 16
    assert meta["resources"] == ["FOOD", "WOOD", "ORE"]
    assert set(meta["terrain"]) <= {0, 1, 2, 3, 4, 5, 6}
    assert meta["spec"]["economy"]["gravity"]["tau"] == pytest.approx(0.012)


def test_state_payload_shape(client):
    state = client.get("/api/state").json()
    for key in ("tick", "pop", "hatch", "roads", "settlements", "flows", "pulses", "network", "history"):
        assert key in state
    assert len(base64.b64decode(state["pop"])) == 96 * 64
    assert len(base64.b64decode(state["hatch"])) == 96 * 64
    assert len(state["settlements"]) == 16
    settlement = state["settlements"][0]
    for key in ("id", "name", "x", "y", "population", "stocks", "prices", "specialization", "degree"):
        assert key in settlement
    assert state["network"]["nodes"] > 0


def test_transport_controls(client):
    paused = client.post("/api/control", json={"action": "pause"}).json()
    assert paused["running"] is False
    before = client.get("/api/state").json()["tick"]
    stepped = client.post("/api/control", json={"action": "step"}).json()
    assert stepped["tick"] == before + 1
    client.post("/api/control", json={"action": "speed", "speed": 24})
    played = client.post("/api/control", json={"action": "play"}).json()
    assert played["running"] is True
    client.post("/api/control", json={"action": "pause"})


def test_seek_runs_many_ticks(client):
    client.post("/api/control", json={"action": "pause"})
    start = client.get("/api/state").json()["tick"]
    result = client.post("/api/control", json={"action": "seek", "ticks": 25}).json()
    assert result["tick"] == start + 25


def test_shock_injection(client):
    response = client.post(
        "/api/shock",
        json={"kind": "drought", "x": 40, "y": 24, "radius": 12, "magnitude": 0.8, "duration": 30},
    )
    assert response.status_code == 200
    record = response.json()
    assert record["kind"] == "drought"
    assert record["magnitude"] == pytest.approx(0.8)
    state = client.get("/api/state").json()
    assert any(s["kind"] == "drought" for s in state["shocks"])


def test_global_shock(client):
    record = client.post(
        "/api/shock", json={"kind": "demand_boom", "x": None, "y": None, "magnitude": 0.5, "duration": 20}
    ).json()
    assert record["x"] == -1 and record["y"] == -1


def test_snapshot_endpoint_returns_png(client):
    response = client.get("/api/snapshot.png", params={"scale": 6})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(response.content) > 5000


def test_reset_with_seed_is_reproducible(client):
    client.post("/api/control", json={"action": "pause"})
    first = client.post("/api/control", json={"action": "reset", "seed": 4242}).json()
    assert first["tick"] == 0
    meta_a = client.get("/api/meta").json()
    client.post("/api/control", json={"action": "reset", "seed": 4242})
    meta_b = client.get("/api/meta").json()
    assert meta_a["terrain"] == meta_b["terrain"]
    assert meta_a["spec"]["world"]["seed"] == 4242
    client.post("/api/control", json={"action": "reset", "seed": 20261101})


def test_invalid_shock_kind_is_rejected(client):
    response = client.post("/api/shock", json={"kind": "meteor", "x": 1, "y": 1})
    assert response.status_code == 422


def test_state_is_json_serialisable(client):
    payload = client.get("/api/state").json()
    assert json.dumps(payload)
    assert "nan" not in json.dumps(payload).lower()
