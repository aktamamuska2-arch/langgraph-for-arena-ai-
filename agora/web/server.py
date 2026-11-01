"""FastAPI service backing the interactive simulation console.

The model runs in a daemon thread so the browser can poll cheaply; a single
lock guards the simulation state. Every endpoint is a thin adapter over
``AgoraModel`` - no economic logic lives here.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from agora.model import AgoraModel
from agora.render import render_frame
from agora.schemas import ExperimentSpec

HERE = Path(__file__).resolve().parent
INDEX = HERE / "index.html"


class ControlRequest(BaseModel):
    action: Literal["play", "pause", "step", "reset", "speed", "seek"]
    speed: float | None = Field(default=None, ge=0.5, le=120.0)
    seed: int | None = None
    ticks: int | None = Field(default=None, ge=1, le=2000)


class ShockRequest(BaseModel):
    kind: Literal["drought", "blight", "ore_discovery", "demand_boom", "plague", "embargo", "road_collapse"]
    x: int | None = None
    y: int | None = None
    radius: float = Field(default=14.0, ge=1.0, le=64.0)
    magnitude: float = Field(default=0.6, ge=0.05, le=1.0)
    duration: int = Field(default=45, ge=5, le=600)


class Engine:
    """Owns the model plus the playback thread."""

    def __init__(self, spec: ExperimentSpec | None = None, warmup: int = 220) -> None:
        self.spec = spec or ExperimentSpec()
        self.lock = threading.RLock()
        self.model = AgoraModel(self.spec)
        # the world is already old when you arrive - warm it up so the market
        # structure is visible instead of an empty lattice of villages
        for _ in range(warmup):
            self.model.step()
        self.running = True
        self.speed = 8.0
        self._frame: dict[str, Any] | None = None
        self._frame_tick = -1
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # -- playback ------------------------------------------------------
    def _loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._stopped.is_set():
            if not self.running:
                time.sleep(0.05)
                next_tick = time.perf_counter()
                continue
            with self.lock:
                self.model.step()
                speed = self.speed
            next_tick += 1.0 / max(speed, 0.5)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(min(delay, 0.25))
            else:
                next_tick = time.perf_counter()

    def frame(self) -> dict[str, Any]:
        with self.lock:
            if self._frame is None or self._frame_tick != self.model.tick_count:
                self._frame = self.model.frame()
                self._frame["running"] = self.running
                self._frame["speed"] = self.speed
                self._frame_tick = self.model.tick_count
            return self._frame

    def state(self) -> dict[str, Any]:
        return self.frame()

    def control(self, request: ControlRequest) -> dict[str, Any]:
        with self.lock:
            if request.action == "play":
                self.running = True
            elif request.action == "pause":
                self.running = False
            elif request.action == "step":
                self.running = False
                self.model.step()
                self._frame = None
            elif request.action == "speed" and request.speed is not None:
                self.speed = float(request.speed)
            elif request.action == "seek" and request.ticks is not None:
                self.running = False
                for _ in range(int(request.ticks)):
                    self.model.step()
                self._frame = None
            elif request.action == "reset":
                spec = self.spec
                if request.seed is not None:
                    spec = spec.model_copy(update={"world": spec.world.model_copy(update={"seed": request.seed})})
                self.spec = spec
                self.model = AgoraModel(spec)
                self._frame = None
                self._frame_tick = -1
                self.running = False
        return {"ok": True, "tick": self.model.tick_count, "running": self.running, "speed": self.speed}

    def shock(self, request: ShockRequest) -> dict[str, Any]:
        with self.lock:
            record = self.model.inject_shock(
                kind=request.kind,
                x=request.x,
                y=request.y,
                radius=request.radius,
                magnitude=request.magnitude,
                duration=request.duration,
            )
            self._frame = None
        return record


ENGINE = Engine()
app = FastAPI(title="ASEOWS Console", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    if not INDEX.exists():  # pragma: no cover
        raise HTTPException(status_code=500, detail="index.html missing")
    return FileResponse(INDEX)


@app.get("/api/meta")
def meta() -> JSONResponse:
    with ENGINE.lock:
        payload = ENGINE.model.meta()
        payload["spec"] = ENGINE.model.spec.model_dump(mode="json")
    return JSONResponse(payload)


@app.get("/api/state")
def state() -> JSONResponse:
    return JSONResponse(ENGINE.state())


@app.post("/api/control")
def control(request: ControlRequest) -> JSONResponse:
    return JSONResponse(ENGINE.control(request))


@app.post("/api/shock")
def shock(request: ShockRequest) -> JSONResponse:
    return JSONResponse(ENGINE.shock(request))


@app.get("/api/snapshot.png")
def snapshot(scale: int = 9) -> Response:
    with ENGINE.lock:
        image = render_frame(ENGINE.model, scale=max(3, min(scale, 16)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "tick": ENGINE.model.tick_count, "running": ENGINE.running}
