# ASEOWS — Emergent Spatial‑Economic Open‑World Simulator

A self‑contained 2D agent‑based world where **nobody plans the economy**.
Sixteen autonomous settlements sit on a procedurally generated island, each
endowed with whatever its terrain gives it. Every few ticks a settlement looks
around its own horizon for a neighbour that is short of something it has spare.
Links are scored with a **spatial gravity kernel**, traded only when the price
premium beats the freight bill, and paved into roads once enough cargo has worn
the route in. Nothing in the loop assigns partners, sets prices, or routes
goods — those are all outputs.

The project is also a **build pipeline**: a LangGraph state machine
(`Planner → Spatial‑Economic Model Synthesizer → Sandbox Visual Tester →
Self‑Correction`) compiles the world, measures it against eight gates, and
repairs its own mis‑tuning before signing off. The loop starts from a
deliberately parsimonious hypothesis, discovers it is too tight, and widens it:

```
[planner]      world 96x64 seed 20261101 -> 16 sites, land 42.55%, critical radius 9.85
[synthesizer]  attempt 0: compiled spec (radius=9.85, partners=8, growth=0.0160)
[tester]       attempt 0: FAIL (7/8 gates) failing=['N-COMP']
[reflector]    attempt 0 failed as network_fragmentation -> 3 patches
[synthesizer]  attempt 1: compiled spec (radius=17.73, partners=10)
[tester]       attempt 1: PASS (8/8 gates)
[finalizer]    verdict PASS_WITH_RETRIES after 2 attempt(s)
```

![Atlas: the trade network emerging over 460 ticks](docs/atlas.png)

---

## Contents

1. [What you are looking at](#what-you-are-looking-at)
2. [Quickstart](#quickstart)
3. [The interactive console](#the-interactive-console)
4. [The model, mathematically](#the-model-mathematically)
5. [The build pipeline](#the-build-pipeline)
6. [Gates: what "correct" means here](#gates-what-correct-means-here)
7. [Self‑correction and the fault matrix](#self-correction-and-the-fault-matrix)
8. [HTTP API](#http-api)
9. [Tests](#tests)
10. [Repository layout](#repository-layout)
11. [Design notes and limitations](#design-notes-and-limitations)

---

## What you are looking at

| Mechanic | Where it lives | What it does |
|---|---|---|
| Procedural island | `agora/world.py` | fractal value noise → elevation/moisture → 7 biomes, resource provinces, greedy farthest‑point siting |
| Territorial endowments | `world.territorial_yield` | each settlement's production capacity, from the terrain under its radius |
| Cobb‑Douglas production | `settlement.produce` | output = endowment × labour^κ × knowledge × road bonus × shock |
| Decentralised market discovery | `market.discover` | every settlement scans its own radius and scores candidate links with the gravity kernel |
| Weighted trade edges | `market.TradeEdge` | per‑resource weights that decay every tick and must be re‑earned by trade |
| Spatial arbitrage | `market.execute` | a link only moves goods when `p_j − p_i > τ·c_ij` |
| Road self‑organisation | `model.on_trade` | sustained cargo flow lowers the transport cost on the route it uses |
| Choropleth + hatching | `model.pop_field`, `web/index.html` | population renders as shading intensity with density hatching bands |
| Pulses | `model._pulse` | production, trade, road, shock and discovery events ring the map |
| Flow lines | `frame.flows` + canvas | animated dashes follow the actual A\* path, coloured by the dominant resource |
| Shocks | `model.inject_shock` | seven kinds, injectable live from the browser |

---

## Quickstart

Python 3.11+.

```bash
cd agora
pip install -r requirements.txt          # numpy, networkx, mesa, fastapi, uvicorn, pillow, pydantic

python scripts/run_workflow.py           # compile the world, run the gates, write artifacts/
python scripts/serve.py                  # interactive console on http://localhost:8010
python -m pytest tests -q                # 69 tests
```

`run_workflow.py` prints the build log, every gate and the verdict, and writes
`artifacts/final_report.json`, `artifacts/blueprint.json`, per‑attempt frame
renders and gate metrics.

The console binds `0.0.0.0:8010`, so it also works behind a container proxy:

```bash
python scripts/serve.py --host 0.0.0.0 --port 8010
```

---

## The interactive console

A single‑page canvas application served by FastAPI. The world is already ~220
ticks old when you open it, so there is a live market to look at immediately.

**Transport** — `play`, `pause`, `step`, `reset` (random seed), and a speed
slider from 1 to 48 ticks/second. `space` toggles play/pause.

**The map** — terrain with an ocean depth ramp, population as an amber
choropleth whose intensity is relative to the settlement's carrying capacity,
hatching bands (single diagonal → cross hatch) over the dense cores, roads as a
pale overlay, and trade flows as animated dashed curves bowing to the side so
overlapping routes stay readable. Hover a settlement for a summary; click it to
pin the inspector.

**The inspector** — population against capacity, wealth, partner count, stock
bars against the settlement's target buffer, local prices, and its labour split.
This is the whole local view: a settlement can only see this much.

**Inject shock** — pick one of seven shocks, then click the map to place it (or
tick *world‑wide*). `esc` cancels. The shock lands on the simulation
immediately and you can watch the network reorganise:

| Shock | Effect |
|---|---|
| `drought` | food production cut by up to 85 %, eased with radial falloff |
| `blight` | wood production cut |
| `ore_discovery` | permanent endowment boost — the only *positive* shock |
| `demand_boom` | consumption multiplied |
| `plague` | added mortality |
| `embargo` | freight on affected routes multiplied by up to 4× |
| `road_collapse` | wipes the road network inside the radius |

**Live KPIs** — population, trade links, cumulative volume moved, road upgrades,
connected core share, mean degree, a population/betweenness sparkline and an
event log.

---

## The model, mathematically

Notation: $i,j$ index settlements, $r \in \{\text{FOOD},\text{WOOD},\text{ORE}\}$.

### 1. Territory and endowments

A settlement at cell $c_i$ owns the Chebyshev neighbourhood of radius 4, weighted
by a linear taper $\omega(d) = \max(0, 1 - d/5)$:

$$
E_i^r \;=\; \frac{9}{|\mathcal{T}_i|}\sum_{c \in \mathcal{T}_i} \omega(d(c,c_i))\; y^r_{\text{terrain}(c)}\; \delta^r_c,
$$

where $y$ is the per‑biome yield table, $\delta$ the deposit multiplier (noise ×
resource provinces), and $|\mathcal{T}_i|$ the territory area. The scaling keeps
endowments on the same order as per‑capita consumption.

> **The zeros in the yield table are the whole point.** A territory with no
> hills cannot mine ore at *any* labour allocation, so it must import. Trade is
> not a productivity gap — it is structural scarcity.

### 2. Production (Cobb–Douglas)

$$
Q_i^r(t) \;=\; E_i^r\,\big(\ell_i^r\big)^{\kappa}\,(1+\theta_i)\;R_i\;\cdot\;\sigma_i^r(t),
\qquad \kappa = 0.78
$$

$\ell$ is the labour share, $\theta$ accumulated knowledge from trade contact,
$R$ the road multiplier at the settlement cell, $\sigma$ the shock modifier.

### 3. Labour reallocation (decentralised)

$$
\ell_i^r \;\leftarrow\; (1-\lambda)\,\ell_i^r + \lambda\,
\frac{\big(p_i^r E_i^r\big)^{\nu}}{\sum_s \big(p_i^s E_i^s\big)^{\nu}},
\qquad \lambda = 0.25,\; \nu = 2.2
$$

Labour drifts toward the locally most valuable activity, with inertia. Only
*local* prices enter — this is why specialisation is emergent rather than
assigned.

### 4. Scarcity pricing

$$
p_i^r \;=\; \mathrm{clip}\!\left[
p_0^r\left(\frac{\bar S_i^r}{S_i^r}\right)^{\epsilon},\;
\phi\,p_0^r,\; \Phi\,p_0^r \right],
\qquad \bar S_i^r = 26 \cdot C_i^r,\; \epsilon = 0.45
$$

$\bar S$ is a 26‑tick consumption buffer, $\phi = 0.25$, $\Phi = 6$. Prices are
pinned at the floor when a granary is full and at the ceiling when it is empty —
which is exactly what the ore market looks like in a world with only three
mining settlements.

### 5. Gravity routing kernel

This is the core of the trade model. For a candidate directed link $i \to j$ in
resource $r$:

$$
\boxed{\;
A_{ij}^{r} \;=\; G\;
\frac{\big(S_i^r\big)^{\alpha}\,\big(D_j^r\big)^{\beta}}
     {\big(d_{ij}+d_0\big)^{\gamma}}
\;\cdot\; e^{-\tau\,c_{ij}}
\;\cdot\;\big(1+\mu\,m_{ij}\big)
\;}
$$

| Symbol | Value | Meaning |
|---|---|---|
| $G$ | 1.0 | kernel scale |
| $S_i^r$ | ≥ 0 | origin **marketable surplus** = stock − target buffer |
| $D_j^r$ | ≥ 0 | destination **deficit** = target buffer − stock |
| $d_{ij}$ | cells | Euclidean separation |
| $d_0$ | 3.0 | softening offset, keeps nearby pairs finite |
| $\alpha,\beta$ | 0.85, 0.95 | mass elasticities (destination scarcity bites slightly harder) |
| $\gamma$ | 1.35 | distance decay — super‑linear, the classic gravity result |
| $c_{ij}$ | cost | terrain‑weighted cost of the cheapest A\* route (water impassable, hills 3.8×, mountains 7.5×) |
| $\tau$ | 0.012 | freight friction per unit of $c$ |
| $m_{ij}$ | ≥ 0 | repeat‑partner memory, +0.05 per successful trade, capped at 6 |
| $\mu$ | 0.45 | memory weight — trust is worth up to 3.7× raw gravity |

The two exponents mean a link is attractive when the *pair* is complementary:
a huge surplus is worth little if the destination does not need it.

### 6. Link selection is *competitive*, not thresholded

An absolute threshold would either connect everything or nothing, depending on
the world's scale. Instead each settlement ranks its candidates **relative to
its own best partner for that resource**, and keeps a partner only if

$$
A_{ij}^{r} \;\ge\; \max\!\left(A_{\min},\; \kappa \cdot \max_{k} A_{ik}^{r}\right),
\qquad A_{\min}=0.02,\; \kappa = 0.32,
$$

capped at `max_partners` partners per settlement. This is what makes the network
sparse and structured instead of a complete graph, and it is entirely local: the
normaliser is the settlement's own best option.

### 7. Edge weights

$$
w_{ij}^{r}(t+1) \;=\; \max\!\left(w_{ij}^{r}(t)\,e^{-\delta},\;
\min\!\left(4,\; 2\,\frac{A_{ij}^{r}}{\max_k A_{ik}^{r}}\right)\right),
\qquad \delta = 0.035 \;\text{(i.e. } e^{-\delta}=0.965\text{)}
$$

Weights decay every tick and are only refreshed by a discovery pass. A link
whose weight drops below `edge_min_weight = 0.06`, or that stays idle longer
than `edge_max_idle = 24` ticks, is pruned. **The network has to be maintained
by real exchange** — it is not a static proximity graph.

### 8. Spatial arbitrage condition (the actual trade rule)

A link trades resource $r$ in a tick only if the destination premium covers the
freight bill:

$$
p_j^r - p_i^r \;>\; \tau\,c_{ij}\,\mu_{\text{embargo}}
\quad\Longrightarrow\quad
q_{ij}^r \;=\; \min\!\big(\text{available}_i^r,\ \text{needed}_j^r\big)\;\eta\;\min(1, w_{ij}^r)
$$

with $\eta = 0.45$ and $\text{available}_i^r = S_i^r - 0.35\bar S_i^r$ (a
subsistence reserve is never sold). Delivered goods are
$q(1-\ell_{\text{transit}})$, $\ell = 0.04$; freight $\tau c q$ is burned. The
margin $p_j - p_i$ therefore funds the transport sector, and a route only
survives as long as the price gap it exploits does.

### 9. Infrastructure feedback

$$
\sum_{t} q_{ij}(t) > \Theta \;\Longrightarrow\;
\rho_{c} \leftarrow \min(3, \rho_c + 0.34)\ \ \forall c \in \text{path}_{ij},
\qquad
c_{xy} \;=\; \frac{c^0_{xy}}{1 + 0.34\,\rho_{xy}}
$$

$\Theta = 55$ units of cumulative flow. Paving a route lowers its cost, which
raises its gravity score, which attracts more flow — the positive feedback that
turns a footpath into a road. `roads_upgraded` counts these events.

### 10. Demography

$$
\dot N_i = N_i\Big[\rho\,\sigma(\text{food})\big(1 - N_i/K_i\big) - \delta_m\Big],
\qquad
K_i = K_0 + 9 E_i^{\text{food}}(1+\theta_i) + 14 R_i
$$

$$
\sigma(\text{food}) = \mathrm{clip}\!\left(\frac{S_i^{\text{food}}/\bar S_i^{\text{food}} - 0.15}{0.85},\,0,\,1.25\right)
$$

Food security gates fecundity; capacity tracks food endowment, knowledge and
road access. A settlement below 1.2 people is abandoned.

### 11. Conservation (the invariant everything else rests on)

Every unit that enters or leaves a stock is booked in a ledger, so the whole
lattice must satisfy

$$
S_{\text{end}} \;=\; S_{\text{start}} + Q - C + M_{\text{in}} - M_{\text{out}} - \Sigma
\;\pm\; 10^{-13}.
$$

This holds to ~6 × 10⁻¹⁴ relative error over 460 ticks and 16 settlements, and
is enforced as the `N-MASS` gate. Shocks that create matter (`ore_discovery`)
must book it as production — a test caught exactly that bug.

---

## The build pipeline

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
   START ──► planner ──► synthesizer ──► sandbox_tester ──┬──► finalizer ──► END
                    ▲                                     │
                    │                                     ▼
                    └──────────── self_correction ◄────────┘
```

A `langgraph` `StateGraph` over a typed `PipelineState`. Five named nodes, two
conditional routing functions, one cycle.

| Node | Responsibility |
|---|---|
| **planner** | generates the world, audits every site's territory yield, derives the **critical radius** (longest edge of the Euclidean MST over settlement positions), writes the `Blueprint` with 9 equations, 8 gates and risk notes |
| **synthesizer** | compiles a frozen `ExperimentSpec`, applying every patch accumulated so far through dotted JSON pointers (`economy.discovery_radius`, `horizon`, …) |
| **sandbox_tester** | runs the full lattice, injects a representative drought on the dominant food exporter, re‑runs for recovery, renders a PNG, evaluates all gates into a `ValidationReport` |
| **self_correction** | classifies the failure, writes a diagnosis and typed `Patch` objects, decides `synthesize` / `finalize` |
| **finalizer** | decides `PASS` / `PASS_WITH_RETRIES` / `FAILED_NEEDS_REVIEW`, hashes artifacts, writes `final_report.json` |

**The retry budget is hard**: `max_attempts = 1 + max_retries` (default 4 → one
build plus three repairs). The router checks the budget *before* routing back,
so the loop cannot spin.

All state crossing node boundaries is Pydantic v2 (`Blueprint`, `Gate`,
`ValidationReport`, `Patch`, `Reflection`, `ArtifactRecord`, `FinalReport`,
`Directive`, …), all frozen with `extra="forbid"`.

### Why the first attempt is *supposed* to fail

The planner deliberately starts at exactly the geometric critical radius. That
is the tightest radius at which a connected graph is even *possible*. But
geometry is not economics: a link requires a settlement with a **complementary
endowment** inside the horizon, not merely a neighbour. At radius 9.85 the
realised giant component is 0.44 → `N-COMP` fails → the reflector widens to
17.73 and the network connects (0.94). The parsimonious hypothesis is measured,
rejected and corrected — which is the entire point of the loop.

---

## Gates: what "correct" means here

Eight gates, evaluated on every attempt. Results from the committed run
(`artifacts/final_report.json`):

| Gate | Metric | Observed | Criterion | Result |
|---|---|---|---|---|
| `C-GRID` | `abs_growth_tail` | 0.00291 | ≤ 0.0045 | PASS |
| `C-COHER` | `field_lag1_corr` | 0.97710 | ≥ 0.55 | PASS |
| `N-COMP` | `giant_component_share` | 0.93750 | ≥ 0.85 | PASS |
| `N-EDGE` | `edges_per_node` | 1.43750 | ≥ 0.90 | PASS |
| `N-MASS` | `mass_balance_error` | 5.84e‑14 | ≤ 1e‑09 | PASS |
| `V-CONTRACT` | `visual_contract_score` | 1.00000 | ≥ 0.85 | PASS |
| `R-SHOCK` | `shock_recovery` | 1.09206 | ≥ 0.60 | PASS |
| `C-STABLE` | `pop_drawdown` | 0.00000 | ≤ 0.35 | PASS |

**Grid convergence (`C-GRID`, `C-COHER`, `C-STABLE`).** The lattice must reach a
stationary field. We least‑squares fit $\ln N(t)$ on the final 25 % of the
history and require $|r| \le 0.0045$ per tick — the population of the whole
island is flat to a third of a percent. The spatial field must stay coherent:
lag‑1 autocorrelation of the population choropleth ≥ 0.55 (observed 0.98),
proving the shading is a smooth density surface and not per‑cell noise. A 2×2
block‑averaging of the field must conserve its total exactly
(`block_mass_error = 0`), which catches any normalisation bug.

**Network integrity (`N-COMP`, `N-EDGE`, `N-MASS`).** At least 85 % of living
settlements in one trade component, at least one link per node, and the ledger
identity above. Orphans are counted and reported (`orphan_fraction = 0.0625` —
one settlement genuinely has no partner at the final tick).

**Visual contract (`V-CONTRACT`).** The rendered frame must not be blank, must
show at least five terrain classes, must have *grown* its population shading
since the study began, must contain hatch bands, must have every flow anchored
on land, and must carry live pulses. Seven binary checks; the score is the pass
fraction.

**Resilience (`R-SHOCK`).** The tester injects a magnitude‑0.85 drought on the
dominant food exporter for 55 ticks and measures trade volume in 40‑tick
windows before/during/after. Recovery must reach 60 % of baseline. In the
committed run the dip is 1.48× and the recovery 1.09× — a supply shock *raises*
trade volume, because scarcity widens price spreads and arbitrage becomes more
profitable. That is the model behaving like a market, not a bug.

---

## Self‑correction and the fault matrix

`scripts/fault_matrix.py` injects four faults through the same patch mechanism
the reflector uses, and checks the loop's response:

| Fault | Injected | Detected | Attempts | Outcome |
|---|---|---|---|---|
| `F1-too-local` | discovery radius pinned at 6 | `network_fragmentation` | 2 | repaired |
| `F2-short-horizon` | horizon truncated to 120 | `non_convergence` | 2 | repaired |
| `F3-double-fault` | radius **and** horizon both wrong | `non_convergence` | 3 | repaired (sequentially) |
| `F4-broken-ledger` | a 4.3 % mass leak | `mass_imbalance` | — | **refused** |

Repair rules (each derived from the metric that failed, not from a lookup table
of test names):

- **fragmentation** → widen the radius to `max(r·1.6, 1.8·critical_radius)` capped
  at `2.8·critical_radius`; if links per node are also short, relax the
  competitor cut by 15 % and add two partner slots.
- **non‑convergence** → extend the horizon by `1.8·H + 120` ticks (a logistic
  needs time to plateau), damp fecundity by 18 %, deepen the stock buffer by 10 %,
  and pull carrying capacity down 25 % when the tail rate is still above 0.006.
- **instability** → cut transit loss by 20 % and extend `edge_max_idle` by 8 ticks
  so the network can rebuild partners after a shock.
- **visual contract** → raise carrying capacity 15 % to give the choropleth range.
- **mass imbalance** → **no patch.** Conservation is an engine property, not a
  tuning knob. The reflector emits zero patches, sets `next_action = finalize`,
  and the run ends `FAILED_NEEDS_REVIEW`. A loop that can "fix" a conservation
  bug by turning a parameter is a loop that lies.

---

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | the console (single‑file canvas app) |
| `/api/meta` | GET | static: terrain codes, ocean depth, settlement roster, resolved spec |
| `/api/state` | GET | one frame: tick, base64 pop/hatch/road layers, settlement snapshots, flows, pulses, network summary, history, events |
| `/api/control` | POST | `{"action": "play"\|"pause"\|"step"\|"seek"\|"speed"\|"reset", ...}` |
| `/api/shock` | POST | `{"kind": "drought", "x": 40, "y": 24, "radius": 12, "magnitude": 0.8, "duration": 30}` |
| `/api/snapshot.png` | GET | server‑side cartographic render of the current tick |
| `/api/health` | GET | liveness + current tick |

The model runs in a daemon thread behind a single lock; the browser polls
`/api/state` at ~8 Hz and animates the flow dashes locally at 60 fps, so motion
stays smooth regardless of poll rate.

---

## Tests

```bash
python -m pytest tests -q      # 69 passed
```

| File | What it proves |
|---|---|
| `test_world.py` | terrain codes, land fraction in envelope, no edge‑touching sites, determinism across processes, A\* never crosses water |
| `test_economy.py` | price clamps, monotone gravity kernel, labour normalisation, **exact** mass balance (also under every shock kind), population convergence, bounded granaries |
| `test_network.py` | links only form after a discovery pass, arbitrage invariant, decay/pruning, roads lower cost, embargo suppresses trade, reproducibility |
| `test_pipeline.py` | graph topology and cycle, both routers, retry ceiling, blueprint completeness, the run actually self‑heals, artifacts are hashed, reflector refuses a conservation defect |
| `test_render.py` | frame size, non‑blank, shading grows, hatch tiers present, flows land‑anchored, payload bounds |
| `test_web.py` | full HTTP contract incl. transport controls, seek, shocks, PNG snapshot, seed reproducibility, NaN‑free JSON |

Two of those tests exist because they caught real bugs during development: the
`ore_discovery` shock was creating matter outside the ledger, and the population
choropleth was being renormalised by whichever settlement happened to lead,
erasing visible growth.

---

## Repository layout

```
agora/
├── agora/
│   ├── schemas.py        # every Pydantic v2 contract (specs, artifacts, telemetry)
│   ├── world.py          # noise, biomes, provinces, siting, A*, territorial yield
│   ├── settlement.py     # the Settlement agent (Mesa)
│   ├── market.py         # gravity kernel, discovery, arbitrage, decay
│   ├── model.py          # the ticking lattice, roads, shocks, frame serialisation
│   ├── metrics.py        # convergence/network/visual metrics + gate evaluation
│   ├── render.py         # headless cartographic renderer (PIL)
│   └── graph/
│       ├── state.py      # PipelineState + pointer helpers
│       ├── nodes.py      # planner / synthesizer / tester / reflector / finalizer
│       └── build.py      # StateGraph wiring + run_pipeline()
├── web/
│   ├── server.py         # FastAPI service, playback thread
│   └── index.html        # the entire console: canvas renderer + panels
├── scripts/
│   ├── run_workflow.py   # the end‑to‑end build
│   ├── fault_matrix.py   # injects faults, proves the loop repairs them
│   ├── render_atlas.py   # the README filmstrip
│   └── serve.py          # console launcher
├── tests/                # 69 tests
├── artifacts/            # frames, metrics, blueprint, final report
└── docs/atlas.png
```

---

## Design notes and limitations

**Why Mesa.** The settlement is a real `mesa.space.Agent` on a `MultiGrid`, and
the model is a `mesa.Model` with a seeded RNG. Mesa's scheduler is not used for
the tick order: production and consumption must happen before the market round,
so the pipeline is explicit and deterministic. That is a deliberate trade of
scheduler generality for reproducibility — two runs with the same spec produce
bit‑identical populations and stocks.

**Why not NetworkX for pathfinding.** `networkx.astar_path` over a 6 144‑node
grid rebuilds adjacency on every call; the local A\* in `world.py` runs on
pre‑computed float32 cost arrays and caches per road epoch. NetworkX is used
where a graph library genuinely helps: MST for the critical radius, connected
components and density for the network gates.

**Known limitations.**

- **Ore is structurally scarce.** Only a handful of territories overlap a
  mountain or hill province, so most settlements sit at the ore price ceiling
  ($6\times$ base) permanently. That is a real feature of an island economy with
  three mines, but it means the ore corridor carries far less volume than food.
- **Trade is seller‑initiated.** Discovery runs from the surplus side. A
  settlement that needs something but has nothing to sell is never the one that
  opens the link. Barter asymmetry is realistic but not modelled.
- **No money stock.** Wealth is bookkeeping; it does not buy anything. There is
  arbitrage profit and freight cost, but no capital accumulation channel.
- **Shocks are exogenous.** The seven kinds modify parameters over a radius and
  a window. Nothing in the world endogenously causes a drought.
- **Grid convergence is a field criterion**, not a mesh‑refinement study: we
  measure that the lattice reaches a stationary, spatially coherent state, not
  that the answer is invariant under a finer lattice. Running the same seed at
  192×128 would change the world, not refine it.

**Determinism.** Everything is seeded from `WorldSpec.seed` through
`numpy.random.default_rng`; no builtin `hash()` is used anywhere in the seeding
path (Python salts string hashing per process, which silently breaks
reproducibility across restarts).

---

## License

Part of the `langgraph` examples workspace. Same license as the parent
repository.
