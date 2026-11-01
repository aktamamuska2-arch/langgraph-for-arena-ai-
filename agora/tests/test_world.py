"""World generation: terrain, deposits, siting and pathfinding."""

from __future__ import annotations

import numpy as np
import pytest

from agora.schemas import OCEAN, WorldSpec
from agora.world import (
    TERRAIN_COST,
    astar_path,
    generate_world,
    path_cost,
    territorial_yield,
)


def test_terrain_codes_are_valid(default_spec):
    world = generate_world(default_spec.world)
    assert world.terrain.shape == (default_spec.world.height, default_spec.world.width)
    assert set(np.unique(world.terrain)).issubset({0, 1, 2, 3, 4, 5, 6})


def test_world_is_a_plausible_island():
    world = generate_world(WorldSpec(seed=20261101))
    land = float((world.terrain != OCEAN).mean())
    assert 0.30 <= land <= 0.55, f"land fraction {land} outside the design envelope"
    summary = world.summary()
    assert summary["terrain_share"]["ocean"] > 0.4
    # every resource must have at least one viable source somewhere
    for r in range(3):
        assert world.yields[r].max() > 0.5


def test_sites_are_on_land_and_separated():
    spec = WorldSpec()
    world = generate_world(spec)
    assert len(world.sites) == spec.settlement_count
    coords = []
    for y, x in world.sites:
        assert world.terrain[y, x] != OCEAN
        # never on the map edge: territories must fit inside the frame
        assert 2 <= x < world.width - 2 and 2 <= y < world.height - 2
        coords.append((x, y))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            assert d >= spec.min_site_separation * 0.75


def test_generation_is_deterministic():
    a = generate_world(WorldSpec(seed=7))
    b = generate_world(WorldSpec(seed=7))
    assert np.array_equal(a.terrain, b.terrain)
    assert np.array_equal(a.deposits, b.deposits)
    assert a.sites == b.sites


def test_different_seeds_differ():
    a = generate_world(WorldSpec(seed=7))
    b = generate_world(WorldSpec(seed=99))
    assert not np.array_equal(a.terrain, b.terrain)


def test_astar_route_avoids_water():
    world = generate_world(WorldSpec())
    (y0, x0), (y1, x1) = world.sites[0], world.sites[-1]
    path = astar_path(world.cost, (y0, x0), (y1, x1))
    assert path[0] == (y0, x0) and path[-1] == (y1, x1)
    assert all(world.terrain[y, x] != OCEAN for y, x in path)
    assert path_cost(world.cost, path) > 0


def test_territorial_yield_is_local():
    world = generate_world(WorldSpec())
    y, x = world.sites[0]
    near = territorial_yield(world, (y, x), 4)
    far = territorial_yield(world, (0, 0), 4)  # corner of the map, open water
    assert near.sum() > far.sum()
    assert (near >= 0).all()


def test_terrain_costs_are_ordered():
    assert TERRAIN_COST[2] < TERRAIN_COST[3] < TERRAIN_COST[4]
    assert np.isinf(TERRAIN_COST[0])


@pytest.mark.parametrize("seed", [7, 99])
def test_other_seeds_still_place_every_settlement(seed):
    spec = WorldSpec(seed=seed)
    world = generate_world(spec)
    assert len(world.sites) == spec.settlement_count
    assert len(set(world.sites)) == spec.settlement_count
