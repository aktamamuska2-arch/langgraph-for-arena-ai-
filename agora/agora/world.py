"""Procedural world generation: terrain, resource endowments, settlement siting.

Everything here is deterministic given ``WorldSpec.seed``: the noise fields are
built from a seeded ``numpy.random.default_rng`` and the settlement siting is a
greedy farthest-point rule, so the same seed always produces the same map.

Coordinate convention: ``grid[y, x]`` with ``x`` in ``[0, width)`` and ``y`` in
``[0, height)``. Row 0 is the north edge.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schemas import (
    COAST,
    DESERT,
    FOREST,
    HILLS,
    MOUNTAIN,
    OCEAN,
    PLAINS,
    RESOURCES,
    WorldSpec,
)

#: movement cost per cell, i.e. how many "units" a caravan spends crossing it.
#: ``inf`` marks impassable water.
TERRAIN_COST: dict[int, float] = {
    OCEAN: float("inf"),
    COAST: 1.25,
    PLAINS: 1.0,
    FOREST: 1.7,
    HILLS: 1.9,
    MOUNTAIN: 4.2,
    DESERT: 2.3,
}

#: base yield multiplier of a cell for each resource, indexed [resource, terrain]
YIELD_TABLE: np.ndarray = np.array(
    [
        # ocean coast plains forest hills mountain desert
        # NOTE: the zeros are deliberate. A settlement whose territory holds no
        # hills cannot mine at any labour allocation, so it *must* import ore.
        # Structural scarcity - not a productivity gap - is what makes trade
        # necessary rather than optional.
        [0.00, 0.62, 1.00, 0.18, 0.12, 0.00, 0.05],  # FOOD
        [0.00, 0.04, 0.08, 1.00, 0.22, 0.05, 0.00],  # WOOD
        [0.00, 0.02, 0.00, 0.00, 0.60, 1.00, 0.22],  # ORE
    ],
    dtype=np.float32,
)

TERRAIN_NAMES: dict[int, str] = {
    OCEAN: "ocean",
    COAST: "coast",
    PLAINS: "plains",
    FOREST: "forest",
    HILLS: "hills",
    MOUNTAIN: "mountain",
    DESERT: "desert",
}


def value_noise(rng: np.random.Generator, width: int, height: int, octaves: int = 4, base: int = 4) -> np.ndarray:
    """Fractal value noise in ``[0, 1]`` built from bilinearly upsampled lattices."""

    field = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total = 0.0
    cells = base
    for _ in range(octaves):
        lattice = rng.random((cells + 1, cells + 1)).astype(np.float32)
        ys = np.linspace(0, cells, height, endpoint=False)
        xs = np.linspace(0, cells, width, endpoint=False)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        fy = (ys - y0)[:, None]
        fx = (xs - x0)[None, :]
        # smoothstep interpolation kills the lattice grid artefacts
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        v00 = lattice[np.ix_(y0, x0)]
        v01 = lattice[np.ix_(y0, x0 + 1)]
        v10 = lattice[np.ix_(y0 + 1, x0)]
        v11 = lattice[np.ix_(y0 + 1, x0 + 1)]
        top = v00 * (1 - fx) + v01 * fx
        bottom = v10 * (1 - fx) + v11 * fx
        field += amplitude * (top * (1 - fy) + bottom * fy)
        total += amplitude
        amplitude *= 0.5
        cells *= 2
    field /= total
    return (field - field.min()) / max(1e-9, field.max() - field.min())


def box_blur(field: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur (edge-padded) used for local-yield scoring."""

    if radius <= 0:
        return field.astype(np.float32)
    pad = np.pad(field.astype(np.float32), radius, mode="edge")
    csum = np.cumsum(pad, axis=0)
    csum = np.vstack([np.zeros((1, csum.shape[1]), dtype=np.float32), csum])
    h = field.shape[0]
    rows = (csum[2 * radius + 1 : 2 * radius + 1 + h] - csum[:h]) / (2 * radius + 1)
    csum = np.cumsum(rows, axis=1)
    csum = np.hstack([np.zeros((csum.shape[0], 1), dtype=np.float32), csum])
    w = field.shape[1]
    return (csum[:, 2 * radius + 1 : 2 * radius + 1 + w] - csum[:, :w]) / (2 * radius + 1)


def classify_terrain(elevation: np.ndarray, moisture: np.ndarray, sea_level: float) -> np.ndarray:
    """Map continuous elevation/moisture to terrain codes.

    Biome thresholds for mountains, hills, forest and desert are taken as
    *quantiles of the land distribution* rather than absolute values. That keeps
    the terrain mix stable across seeds while still mapping the noise field
    monotonically, so a map always contains highlands, woodland and drylands.
    """

    terrain = np.full(elevation.shape, PLAINS, dtype=np.int8)
    land = elevation >= sea_level
    terrain[land & (elevation < sea_level + 0.045)] = COAST

    inner = land & (terrain == PLAINS)
    if inner.any():
        elev_land = elevation[inner]
        moist_land = moisture[inner]
        hi_elev = float(np.quantile(elev_land, 0.90))
        peak_elev = float(np.quantile(elev_land, 0.975))
        dry = float(np.quantile(moist_land, 0.30))
        wet = float(np.quantile(moist_land, 0.58))

        terrain[inner & (elevation > peak_elev)] = MOUNTAIN
        terrain[inner & (elevation > hi_elev) & (elevation <= peak_elev)] = HILLS
        plains_now = inner & (terrain == PLAINS)
        terrain[plains_now & (moisture > wet)] = FOREST
        dry_now = plains_now & (moisture < dry) & (elevation > sea_level + 0.06)
        terrain[dry_now] = DESERT

    terrain[~land] = OCEAN
    return terrain


@dataclass(frozen=True)
class WorldData:
    """Immutable snapshot of the generated geography."""

    width: int
    height: int
    seed: int
    terrain: np.ndarray  # int8 (H, W)
    deposits: np.ndarray  # float32 (3, H, W) richness multipliers around 1.0
    yields: np.ndarray  # float32 (3, H, W) effective per-cell yield
    cost: np.ndarray  # float32 (H, W) base movement cost, inf on water
    sites: tuple[tuple[int, int], ...]  # settlement origins
    depth: np.ndarray  # float32 (H, W) distance from land in cells (0 on land)
    elevation: np.ndarray  # float32 (H, W) normalised height, used for relief shading

    def as_arrays(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "terrain": self.terrain.tolist(),
            "deposits": self.deposits.astype(np.float32).tolist(),
            "sites": [list(s) for s in self.sites],
        }

    def summary(self) -> dict[str, Any]:
        counts = {
            TERRAIN_NAMES[int(k)]: int(v) for k, v in zip(*np.unique(self.terrain, return_counts=True), strict=False)
        }
        land = float(np.mean(self.terrain != OCEAN))
        return {
            "size": [self.width, self.height],
            "terrain_share": {k: round(v / self.terrain.size, 4) for k, v in counts.items()},
            "land_fraction": round(land, 4),
            "sites": len(self.sites),
            "mean_deposit": [round(float(self.deposits[r].mean()), 4) for r in range(len(RESOURCES))],
        }


def _smooth_candidates(terrain: np.ndarray, valid: np.ndarray, passes: int = 1) -> np.ndarray:
    """Keep only cells whose 3x3 neighbourhood is also valid (habitable micro-sites)."""

    mask = valid.copy()
    for _ in range(passes):
        shifted = np.zeros_like(mask)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        mask = mask & shifted
    return mask & (terrain != OCEAN)


def generate_world(spec: WorldSpec) -> WorldData:
    rng = np.random.default_rng(spec.seed)
    w, h = spec.width, spec.height

    elevation = value_noise(rng, w, h, octaves=5, base=3)
    moisture = value_noise(rng, w, h, octaves=4, base=5)
    # radial falloff pushes the coast to the map edge so the landmass is an island
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    radius = np.sqrt(((xx - cx) / (w * 0.50)) ** 2 + ((yy - cy) / (h * 0.50)) ** 2)
    falloff = np.clip(1.0 - radius**2.4, 0.0, 1.0) ** 1.15
    elevation = elevation * falloff
    elevation = (elevation - elevation.min()) / max(1e-9, elevation.max() - elevation.min())

    terrain = classify_terrain(elevation, moisture, spec.sea_level)

    deposits = np.ones((len(RESOURCES), h, w), dtype=np.float32)
    for r, resource in enumerate(RESOURCES):
        field = value_noise(rng, w, h, octaves=3, base=6)
        rich = 0.35 + 1.9 * field**2
        deposits[r] = rich * spec.deposit_richness[resource]

    # resource provinces: a handful of localised concentrations. These are what
    # give a few settlements a *structural* export advantage - an ore vein makes
    # its holders miners and everyone else a permanent importer.
    provinces = {
        "ORE": (np.array([HILLS, MOUNTAIN, DESERT]), 5, 2.4),
        "WOOD": (np.array([FOREST]), 4, 1.7),
        "FOOD": (np.array([PLAINS, COAST]), 3, 1.3),
    }
    yy2, xx2 = np.mgrid[0:h, 0:w]
    for r, resource in enumerate(RESOURCES):
        allowed, count, amount = provinces[resource]
        candidates = np.argwhere(np.isin(terrain, allowed))
        if len(candidates) == 0:
            continue
        picks = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        blob = np.zeros((h, w), dtype=np.float32)
        for idx in picks:
            cy2, cx2 = candidates[int(idx)]
            sigma = 5.0 + 4.0 * float(rng.random())
            blob += amount * np.exp(-(((xx2 - cx2) ** 2 + (yy2 - cy2) ** 2) / (2 * sigma**2)))
        deposits[r] = deposits[r] * (1.0 + blob)

    yields = np.zeros((len(RESOURCES), h, w), dtype=np.float32)
    for r in range(len(RESOURCES)):
        flat = YIELD_TABLE[r][terrain]
        yields[r] = (flat * deposits[r]).astype(np.float32)

    cost = np.vectorize(TERRAIN_COST.get, otypes=[np.float32])(terrain.astype(int))
    cost = cost.astype(np.float32)

    sites = _choose_sites(terrain, yields, cost, spec)
    return WorldData(
        width=w,
        height=h,
        seed=spec.seed,
        terrain=terrain,
        deposits=deposits,
        yields=yields,
        cost=cost,
        sites=tuple(sites),
        depth=_land_distance(terrain),
        elevation=elevation.astype(np.float32),
    )


def _land_distance(terrain: np.ndarray) -> np.ndarray:
    """Cheap chamfer distance-to-land transform used for ocean depth shading."""

    h, w = terrain.shape
    big = float(h + w)
    dist = np.where(terrain == OCEAN, big, 0.0).astype(np.float32)
    for _ in range(2):
        for y in range(h):
            for x in range(w):
                best = dist[y, x]
                if y > 0:
                    best = min(best, dist[y - 1, x] + 1.0)
                if x > 0:
                    best = min(best, dist[y, x - 1] + 1.0)
                if y > 0 and x > 0:
                    best = min(best, dist[y - 1, x - 1] + 1.4142)
                if y > 0 and x < w - 1:
                    best = min(best, dist[y - 1, x + 1] + 1.4142)
                dist[y, x] = best
        for y in range(h - 1, -1, -1):
            for x in range(w - 1, -1, -1):
                best = dist[y, x]
                if y < h - 1:
                    best = min(best, dist[y + 1, x] + 1.0)
                if x < w - 1:
                    best = min(best, dist[y, x + 1] + 1.0)
                if y < h - 1 and x < w - 1:
                    best = min(best, dist[y + 1, x + 1] + 1.4142)
                if y < h - 1 and x > 0:
                    best = min(best, dist[y + 1, x - 1] + 1.4142)
                dist[y, x] = best
    return dist


def _site_score(yields: np.ndarray, terrain: np.ndarray, spec: WorldSpec) -> np.ndarray:
    """Habitable cells, scored by a blend of local food/wood/ore security."""

    food, wood, ore = yields[0], yields[1], yields[2]
    rr = max(2, spec.territory_radius)
    local_food = box_blur(food, rr)
    local_wood = box_blur(wood, rr)
    local_ore = box_blur(ore, rr)
    h, w = terrain.shape
    margin = np.ones((h, w), dtype=bool)
    edge = 3
    margin[:edge, :] = margin[-edge:, :] = False
    margin[:, :edge] = margin[:, -edge:] = False
    habitable = margin & (terrain != OCEAN) & (terrain != MOUNTAIN) & (food > 0.02)
    score = habitable.astype(np.float32) * (
        0.60 * local_food + 0.22 * local_wood + 0.18 * local_ore + 0.06 * (terrain == COAST)
    )
    return score


def _choose_sites(
    terrain: np.ndarray, yields: np.ndarray, cost: np.ndarray, spec: WorldSpec
) -> list[tuple[int, int]]:
    """Greedy farthest-point siting subject to a minimum separation."""

    score = _site_score(yields, terrain, spec)
    if not np.isfinite(score).all():  # pragma: no cover - defensive
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    smooth = box_blur(score, 3)

    min_sep = spec.min_site_separation
    candidates = np.argwhere(smooth > np.percentile(smooth[smooth > 0], 55)) if (smooth > 0).any() else []
    candidates = sorted(
        ((int(y), int(x)) for y, x in candidates),
        key=lambda c: (-float(smooth[c[0], c[1]]), c[0], c[1]),
    )
    sites: list[tuple[int, int]] = []
    for y, x in candidates:
        if len(sites) >= spec.settlement_count:
            break
        if all((y - sy) ** 2 + (x - sx) ** 2 >= min_sep**2 for sy, sx in sites):
            sites.append((y, x))
    # if separation was too aggressive, relax it rather than returning a short list
    if len(sites) < spec.settlement_count:
        for y, x in candidates:
            if len(sites) >= spec.settlement_count:
                break
            if (y, x) not in sites:
                sites.append((y, x))
    return sites[: spec.settlement_count]


def territorial_yield(world: WorldData, centre: tuple[int, int], radius: int) -> np.ndarray:
    """Total resource yield inside a settlement's territory (Chebyshev radius)."""

    y, x = centre
    h, w = world.terrain.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    territory = world.yields[:, y0:y1, x0:x1]
    dist = np.zeros(territory.shape[1:], dtype=np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32)
    weight = np.clip(1.0 - dist / max(radius + 1, 1), 0.0, 1.0)
    return (territory * weight[None, :, :]).sum(axis=(1, 2)).astype(np.float32)


def astar_path(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Cheapest 8-connected route between two cells (A*, ``None`` if blocked)."""

    import heapq

    h, w = cost.shape
    sy, sx = start
    gy, gx = goal
    if not np.isfinite(cost[sy, sx]) or not np.isfinite(cost[gy, gx]):
        return []
    if (sy, sx) == (gy, gx):
        return [start]

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dy, dx = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dy + dx) + (np.sqrt(2) - 2) * min(dy, dx)

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start, goal), start)]
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        if current in closed:
            continue
        closed.add(current)
        cy, cx = current
        for dy, dx in steps:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if not np.isfinite(cost[ny, nx]):
                continue
            step = float(cost[ny, nx]) * (1.4142 if dy and dx else 1.0)
            tentative = g_score[current] + step
            if tentative < g_score.get((ny, nx), float("inf")):
                g_score[(ny, nx)] = tentative
                came[(ny, nx)] = current
                heapq.heappush(open_heap, (tentative + heuristic((ny, nx), goal), (ny, nx)))
    return []


def path_cost(cost: np.ndarray, path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (y0, x0), (y1, x1) in itertools.pairwise(path):
        diag = 1.4142 if (y0 != y1 and x0 != x1) else 1.0
        total += float(cost[y1, x1]) * diag
    return total
