"""Headless cartographic renderer.

The same visual contract the browser canvas honours is rendered here with PIL so
the Sandbox Visual Tester can audit the picture without a display: terrain base,
population shading/hatching tiers, road network, trade flow curves, event pulses
and settlement glyphs.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from .schemas import RESOURCES
from .world import TERRAIN_NAMES  # noqa: F401  (kept for downstream labels)

if TYPE_CHECKING:  # pragma: no cover
    from .model import AgoraModel

TERRAIN_RGB = {
    0: (8, 19, 31),
    1: (21, 45, 58),
    2: (37, 55, 38),
    3: (25, 43, 31),
    4: (60, 56, 43),
    5: (89, 87, 94),
    6: (89, 75, 47),
}
OCEAN_DEEP = np.array([6, 15, 26], dtype=np.float32)
OCEAN_SHALLOW = np.array([16, 44, 60], dtype=np.float32)

POP_RGB = np.array([255, 178, 74], dtype=np.float32)
FLOW_RGB = {
    "FOOD": (255, 196, 92),
    "WOOD": (104, 214, 148),
    "ORE": (186, 140, 255),
}
ACCENT = (46, 230, 197)


def _base_image(model: AgoraModel) -> np.ndarray:
    h, w = model.height, model.width
    terrain = model.world.terrain
    depth = model.world.depth

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for code, colour in TERRAIN_RGB.items():
        rgb[terrain == code] = colour

    ocean = terrain == 0
    if ocean.any():
        fade = np.clip(depth / 9.0, 0.0, 1.0)[..., None]
        rgb[ocean] = OCEAN_SHALLOW * (1 - fade[ocean]) + OCEAN_DEEP * fade[ocean]

    # gentle elevation tint: lowlands sit back, highlands lift out of the frame
    elevation = model.world.elevation
    land = ~ocean
    relief = (0.82 + 0.46 * elevation)[..., None] * land[..., None] + (~land)[..., None]
    rgb = np.clip(rgb * relief, 0, 255)

    road = model.road_level / max(model.econ.road_max, 1)
    if road.any():
        warm = np.array([126, 112, 82], dtype=np.float32)
        mix = np.clip(road, 0, 1)[..., None] * 0.55
        rgb = rgb * (1 - mix) + warm * mix
    return rgb


def _population_overlay(rgb: np.ndarray, model: AgoraModel) -> np.ndarray:
    pop, hatch = model.pop_field()
    field = (pop.astype(np.float32) / 255.0) ** 1.35
    field = field[..., None]
    blended = rgb * (1 - 0.62 * field) + POP_RGB * (0.62 * field)
    return np.clip(blended, 0, 255), hatch


def _hatch_mask(hatch: np.ndarray, h: int, w: int, scale: int) -> np.ndarray:
    """Sub-cell hatch pattern: tier 1 = none, 2 = single diagonal, 3 = cross."""

    mask = np.zeros((h * scale, w * scale), dtype=bool)
    ys, xs = np.mgrid[0 : h * scale, 0 : w * scale]
    diagonal = ((xs + ys) // max(scale // 2, 2)) % 2 == 0
    counter = ((xs - ys) // max(scale // 2, 2)) % 2 == 0
    tier = np.repeat(np.repeat(hatch, scale, axis=0), scale, axis=1)
    mask |= (tier >= 2) & diagonal
    mask |= (tier >= 3) & counter
    return mask


def _curve(points: list[tuple[float, float]], kind: str) -> list[tuple[float, float]]:
    """Bow the polyline so overlapping routes stay legible."""

    if len(points) < 3:
        return points
    (x0, y0), (x1, y1) = points[0], points[-1]
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    bow = 0.10 * length * (1.0 if kind == "ORE" else -1.0)
    out = []
    for i, (x, y) in enumerate(points):
        t = i / (len(points) - 1)
        amp = bow * math.sin(math.pi * t)
        out.append((x + nx * amp, y + ny * amp))
    return out


def render_frame(model: AgoraModel, scale: int = 9, hud: bool = True) -> Image.Image:
    """Render the current model state to an RGB image."""

    h, w = model.height, model.width
    rgb = _base_image(model)
    rgb, hatch = _population_overlay(rgb, model)
    image = Image.fromarray(rgb.astype(np.uint8)).resize((w * scale, h * scale), Image.NEAREST)

    hatch_layer = Image.fromarray(
        np.where(_hatch_mask(hatch, h, w, scale)[..., None], np.array([236, 158, 66], np.uint8), 0).astype(np.uint8)
    )
    hatch_alpha = Image.fromarray((np.where(_hatch_mask(hatch, h, w, scale), 38, 0)).astype(np.uint8), mode="L")
    image = Image.composite(hatch_layer, image, hatch_alpha)

    draw = ImageDraw.Draw(image, "RGBA")
    _draw_pulses(draw, model, scale)
    _draw_flows(draw, model, scale)
    _draw_settlements(draw, model, scale)
    if hud:
        _draw_hud(draw, model, image.size)
    return image


def _draw_flows(draw: ImageDraw.ImageDraw, model: AgoraModel, scale: int) -> None:
    edges = sorted(model.edges.values(), key=lambda e: e.weight)
    for edge in edges:
        if not edge.path or edge.weight <= 0:
            continue
        name = RESOURCES[int(edge.dominant)]
        colour = FLOW_RGB[name]
        points = _curve([(x * scale, y * scale) for x, y in edge.path], name)
        width = max(2, round(2.0 + 2.4 * min(edge.weight, 4.0) / 4.0))
        alpha = int(120 + 110 * min(1.0, edge.weight / 2.0))
        draw.line(points, fill=(*colour, alpha), width=width, joint="curve")
        tip = points[-1]
        draw.ellipse([tip[0] - 2.5, tip[1] - 2.5, tip[0] + 2.5, tip[1] + 2.5], fill=(*colour, 230))


def _draw_pulses(draw: ImageDraw.ImageDraw, model: AgoraModel, scale: int) -> None:
    for pulse in model.pulses:
        age = model.tick_count - pulse["t0"]
        if age < 0 or age > 13:
            continue
        t = age / 13.0
        radius = (pulse["r0"] + 2.6 * t) * scale
        alpha = int(150 * (1.0 - t) ** 1.6)
        if alpha <= 3:
            continue
        colour = {
            "production": (255, 208, 130),
            "trade": (108, 226, 255),
            "road": (255, 236, 170),
            "shock": (255, 96, 96),
            "discovery": (186, 140, 255),
        }.get(pulse["kind"], (255, 255, 255))
        cx, cy = pulse["x"] * scale + scale / 2, pulse["y"] * scale + scale / 2
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(*colour, alpha), width=1)


def _draw_settlements(draw: ImageDraw.ImageDraw, model: AgoraModel, scale: int) -> None:
    alive = [s for s in model.settlements.values() if s.alive]
    peak = max((s.population for s in alive), default=1.0)
    degree: dict[int, int] = dict.fromkeys(model.settlements, 0)
    for src, dst in model.edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1
    for settlement in model.settlements.values():
        cx, cy = settlement.x * scale + scale / 2, settlement.y * scale + scale / 2
        if not settlement.alive:
            draw.line([cx - 4, cy - 4, cx + 4, cy + 4], fill=(120, 120, 130, 150), width=2)
            continue
        level = settlement.population / max(peak, 1e-6)
        radius = 3.0 + 5.0 * level
        draw.ellipse(
            [cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2],
            outline=(*ACCENT, 60),
        )
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(*ACCENT, 235))
        draw.ellipse(
            [cx - radius * 0.45, cy - radius * 0.45, cx + radius * 0.45, cy + radius * 0.45],
            fill=(9, 26, 33, 235),
        )
        if degree.get(settlement.sid, 0) >= 6:
            draw.ellipse(
                [cx - radius - 6, cy - radius - 6, cx + radius + 6, cy + radius + 6],
                outline=(255, 196, 92, 130),
                width=1,
            )


def _draw_hud(draw: ImageDraw.ImageDraw, model: AgoraModel, size: tuple[int, int]) -> None:
    summary = model.network_summary()
    population = sum(s.population for s in model.settlements.values() if s.alive)
    lines = [
        f"TICK {model.tick_count:>4d}",
        f"POP {population:8.1f}",
        f"LINKS {int(summary['nodes']) and int(summary['edges']):>3d}",
        f"TRADE {summary['total_flow']:8.1f}",
        f"ROADS {int(summary['roads_upgraded']):>3d}",
    ]
    pad = 10
    width = 128
    draw.rectangle([pad, pad, pad + width, pad + 18 * len(lines) + 12], fill=(6, 14, 20, 190))
    for i, line in enumerate(lines):
        draw.text((pad + 10, pad + 8 + 18 * i), line, fill=(196, 224, 232, 235))
    draw.text(
        (pad + 10, size[1] - 22),
        f"{model.spec.world.seed} / {len(model.settlements)} settlements",
        fill=(120, 150, 160, 200),
    )
