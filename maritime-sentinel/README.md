# ⚓ MARITIME SENTINEL

**Autonomous Macro-Dynamic Stress-Testing Engine Synthesizer (AM-SES v2.0)**

A self-contained, self-healing simulation lab for **systemic disruption of the regional maritime
logistics chain** under extreme-weather anomalies and geopolitical bottlenecks at primary port
nodes.

The lab compiles itself. A cyclic [LangGraph](https://github.com/langchain-ai/langgraph) state
machine decomposes the macro-scenario, **generates** an executable simulation artifact, **runs it in
a sandbox**, and — if any acceptance gate fails — **reflects on the failure and patches the build**
before retrying. The result is a validated, hashed, reproducible build directory.

> The engine is a *model*, not a control system. Nothing in this repository connects to a vessel, a
> terminal operating system, or a live AIS feed. Every number in this repository is produced by
> the discrete-event simulation described below.

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [Quick start](#2-quick-start)
3. [Repository layout](#3-repository-layout)
4. [The state graph (AM-SES core)](#4-the-state-graph-am-ses-core)
5. [The simulation model](#5-the-simulation-model)
6. [Monte Carlo perturbation & resilience](#6-monte-carlo-perturbation--resilience)
7. [Acceptance gates](#7-acceptance-gates)
8. [Reference results](#8-reference-results)
9. [Configuration & extension](#9-configuration--extension)
10. [Reproducibility & determinism](#10-reproducibility--determinism)
11. [Dependencies](#11-dependencies)
12. [Limitations](#12-limitations)

---

## 1. What you get

| Deliverable | Location | Notes |
| --- | --- | --- |
| Agentic build loop (`StateGraph`, conditional edges, retry cap) | `maritime_sentinel/graph/` | planner → synthesizer → sandbox tester → reflector ↺ finalize |
| Discrete-event simulation engine | `maritime_sentinel/engine.py` | SimPy 4, dynamic berth/strait pools, live re-routing |
| Weather model | `maritime_sentinel/weather.py` | Seasonal continuous-time Markov chains per sea area |
| Geopolitical model | `maritime_sentinel/geopolitics.py` | Marked non-homogeneous Poisson incidents + scheduled shock |
| Topology & criticality | `maritime_sentinel/topology.py` | NetworkX lane graph, betweenness, articulation points |
| Monte Carlo driver | `maritime_sentinel/montecarlo.py` | Latin Hypercube, resilience index, CVaR, tornado |
| Code synthesis + static scan | `maritime_sentinel/codegen.py` | Emits manifest, runner, design memo, acceptance file |
| Sandboxed executor | `maritime_sentinel/sandbox.py` | `python -I`, rlimits, scrubbed env, wall-clock kill |
| CLI — build graph | `scripts/run_workflow.py` | `make build` |
| CLI — ensemble | `scripts/run_montecarlo.py` | `make mc` |
| Tests | `tests/` | 59 tests, ~17 s |

---

## 2. Quick start

```bash
cd maritime-sentinel

# 1. dependencies (Python >= 3.10)
python -m pip install -r requirements.txt

# 2. compile + sandbox-test the artifact (the real deliverable)
python scripts/run_workflow.py

# 3. Monte Carlo sweep to disk
python scripts/run_montecarlo.py --reps 32 --out artifacts/mc

# 4. verification
python -m pytest tests -q
```

Everything is driven through `make` as well:

```bash
make build      # AM-SES build graph
make mc         # headless ensemble
make test       # pytest
make lint       # ruff check
```

### Fault injection (proving the loop is real)

The self-healing loop is exercised on every run because the *first draft deliberately under-sizes
the fleet*. To prove the reflector handles other failure classes, inject a synthetic defect:

```bash
python scripts/run_workflow.py --fault-mode schema       # invalid probability vector
python scripts/run_workflow.py --fault-mode runtime      # uncaught KeyError in the build
python scripts/run_workflow.py --fault-mode divergence   # capacity starved into queue divergence
python scripts/run_workflow.py --fault-mode timeout --fault-persist   # exhausts the retry budget
```

`--fault-persist` re-injects on every attempt, which drives the engine to its hard stop
(`FAILED_NEEDS_REVIEW` after 1 initial attempt + 3 retries).

---

## 3. Repository layout

```text
maritime-sentinel/
├── maritime_sentinel/
│   ├── schemas.py        Pydantic v2 contracts for every state boundary
│   ├── config.py         the canonical Indonesian network + stress envelope
│   ├── topology.py       NetworkX graph, port loads, criticality
│   ├── weather.py        seasonal CTMC weather field
│   ├── geopolitics.py    marked Poisson incident field + scheduled shock
│   ├── engine.py         SimPy discrete-event core
│   ├── metrics.py        KPIs, convergence diagnostics, risk bands
│   ├── montecarlo.py     Latin Hypercube ensemble
│   ├── codegen.py        artifact synthesis + Erlang-C sizing math
│   ├── sandbox.py        isolated execution harness
│   └── graph/
│       ├── state.py      State TypedDict + RunContext
│       ├── nodes.py      planner / synthesizer / tester / reflector / finalizer
│       └── build.py      StateGraph wiring + retry governor
├── scripts/{run_workflow,run_montecarlo}.py
├── tests/
└── artifacts/            (git-ignored) every build, hash and report
```

---

## 4. The state graph (AM-SES core)

### 4.1 State

`maritime_sentinel/graph/state.py` declares the mandated schema as a `TypedDict`:

| Field | Type | Lifetime |
| --- | --- | --- |
| `scenario_params` | `dict` (JSON form of `ScenarioSpec`) | rewritten by the reflector |
| `architectural_blueprint` | `dict` (JSON form of `ArchitecturalBlueprint`) | rewritten by the reflector |
| `code_artifacts` | `dict` (build dir, hashes, entrypoint) | rewritten every synthesis |
| `execution_logs` | `Annotated[list, operator.add]` | **accumulates** across nodes |
| `retry_count` | `int` | incremented by the reflector |
| `test_report`, `reflection`, `reflection_trail`, `verdict`, `final_report` | derived | per attempt |

Immutability is a contract, not a convention: every Pydantic model inherits from a `Frozen` base
(`model_config = ConfigDict(frozen=True, extra="forbid")`), nodes never mutate the dictionaries they
receive, and `execution_logs` is the only field with a reducer (`operator.add`), so any super-step
checkpoint is a valid snapshot of the build.

### 4.2 Nodes

| Node | Reads | Writes | Responsibility |
| --- | --- | --- | --- |
| `node_planner` | `scenario_params` | `architectural_blueprint` | Build the lane graph, compute betweenness/articulation points, solve stationary weather laws, derive Erlang-C capacity requirements, emit equations + acceptance gates |
| `node_synthesizer` | blueprint, scenario | `code_artifacts` | Derive a `DeploymentPlan`, render `generated_simulation.py`, write the manifest, memo and acceptance file, static-scan and hash everything |
| `node_sandbox_tester` | artifacts, blueprint | `test_report` | Execute the artifact in an isolated interpreter, parse the machine-readable result line, evaluate every gate, classify the failure |
| `node_reflector` | test report | scenario, blueprint, `retry_count` | Diagnose the failure class and emit **machine-applicable patches** (`pointer`, `previous`, `updated`, `rationale`) |
| `node_finalizer` | whole state | `final_report`, `verdict` | Write `final_report.json`, `execution_log.json`, `state_snapshot.json` |

### 4.3 Edges and the retry governor

```text
START ──▶ planner ──(conditional)──▶ synthesizer ──(conditional)──▶ sandbox_tester
                                                                        │
                                                        (conditional)   │
                                          ┌──────────── passed ─────────┤
                                          │                             │
                                          ▼                             ▼
                                       finalize ◀──(budget spent)── reflector
                                          │                             │
                                          ▼                   (retries left)
                                         END                            │
                                                                        ▼
                                                                   synthesizer
```

Three `add_conditional_edges` calls carry the control flow:

* `route_after_test` → `"finalize"` if the report passed **or** `retry_count >= max_retries`, else `"reflector"`.
* `route_after_reflect` → `"synthesizer"` while budget remains, else `"finalize"`.
* `route_after_synthesis` → `"sandbox_tester"` when an entrypoint exists, else `"reflector"`.

The budget is **1 initial attempt + 3 self-healing retries = 4 executions**, hard-capped. On
exhaustion the artifact is stamped `FAILED_NEEDS_REVIEW` — it is never retried indefinitely, and it
is never silently reported as a success.

### 4.4 What the reflector actually does

It is not a "try again" loop; each failure class maps to a *specific mathematical repair*:

| Failure class | Diagnosis | Patch |
| --- | --- | --- |
| `instability` (ρ > gate) | queue saturation under stressed productivity | `stress.capacity_scale ← f(ρ_observed/ρ_gate)` **and** `sizing_strategy ← tail_risk` (adopt the planner's Erlang-C requirement); second consecutive failure widens `resilience_margin += 0.08` |
| `divergence` (backlog slope) | non-stationary queue | same capacity patch family |
| `underperformance` (SL / throughput) | service collapse | same capacity patch family |
| `schema_error` | probability vector off the simplex | project every transition row onto the simplex, emit one patch per repaired row |
| `runtime_error` | uncaught exception | if Pydantic → schema repair; else damp `service_cv`
| `timeout` | exceeded wall-clock budget | shrink `horizon_days` by 40 % (floor 60 d) |

Each patch is recorded with its previous value, updated value and rationale, so the build is fully
auditable after the fact.

---

## 5. The simulation model

### 5.1 Network

A directed-in-practice, undirected-in-representation graph `G = (V, E)`:

* **Ports** `P` — 7 terminals (Tanjung Priok, Tanjung Perak, Belawan, Batu Ampar, Makassar New Port,
  Kariangau, Tanjung Emas) with berth count, TEU/berth/day productivity and a weather zone.
* **Chokepoints** `C` — Malacca, Karimata, Sunda, Lombok and Makassar straits with a finite number
  of concurrent transit slots and a Poisson incident rate.
* **Lanes** `E` — 15 sea legs with distance (nm), capacity (TEU/day), cost (USD/TEU) and the zone
  they traverse. One lane is a *contingency* bypass (Belawan–Batam open sea, +26 h, +165 USD/TEU)
  that only opens when Malacca's capacity factor drops below 0.55.
* **Channels** — 14 origin–destination Poisson streams (18.2 vessel calls/day offering
  18 900 TEU/day) each with its own TEU distribution and contractual SLA.

### 5.2 Weather: seasonal continuous-time Markov chains

For each sea zone `z` the sea state is a CTMC `W_z(t)` on `{CALM, ADVERSE, SEVERE, EXTREME}`:

```
Q_z[i,j] = r_z[i] · P_z[i,j]   (i ≠ j),      Q_z[i,i] = -r_z[i],      r_z[i] = 1 / dwell_z[i]
```

`P_z` is re-weighted by monsoon season and by the scenario's `weather_intensity`, then projected back
onto the simplex:

```
P'[i,j] = P[i,j] · (bias)          for escalations  j > i
P'[i,j] = P[i,j] / bias^(1/4)      for recoveries   j < i
bias    = season_bias(z, season) · weather_intensity
```

The stationary law solves `π Q = 0`, `Σπ = 1` via a bordered linear system. It is used twice: to
warm-start the simulation (so day 1 is not artificially calm) and to size capacity against the
*expected* stressed productivity rather than the nominal one.

Operational coupling:

| State | Sailing speed | Berth productivity |
| --- | --- | --- |
| CALM | 1.00 | 1.00 |
| ADVERSE | 0.86 | 0.93 |
| SEVERE | 0.62 | 0.76 |
| EXTREME | 0.34 | 0.45 |

Extreme sea states additionally trigger terminal closures with hazard
`p_close · Δt` per 6-hour control interval (duration ~ LogNormal, capped at 36 h).

### 5.3 Geopolitics: marked Poisson incidents

Chokepoint incidents arrive as a non-homogeneous Poisson process:

```
λ_c(t) = (g_c / 365) · geo_intensity · (1 + 0.35·sin(2π(t + φ_c)/365))
S     ~ min(max_severity_c, Beta(2,5) · max_severity_c)
D     ~ min(21, max(0.25, LogNormal(ln 2.2, 0.75)))
```

Overlapping incidents combine through the instantaneous capacity factor

```
κ_c(t) = 1 − max_k [ S_k · (1 − 0.55·(t − t_k)/D_k) ]      for incidents active at t
```

i.e. an incident hits hard on impact and decays to 45 % of its severity before clearing. Terminals
receive operational incidents (crane failure, cyber-attack, congestion surge) at 1.4/year scaled by
`geo_intensity`. A scheduled `ShockSpec` (default: Malacca blockade, day 55, 18 days, severity 0.72)
is superimposed deterministically so the resilience/recovery measurement has an anchor.

### 5.4 Queueing at terminals

Each terminal is a FIFO multi-server queue whose number of *online* servers changes at runtime:

```
online_p(t) = 0                                   if the terminal is closed
            = round(B_p · κ_p(t))                 otherwise
μ_b(t)      = μ_b0 · productivity(W_{z(p)}(t))
```

SimPy's `Resource` cannot resize, so `CapacityPool` implements an explicit server count with a
deficit mode: when `online < busy`, the excess is absorbed as vessels finish service. Berth demand
per port is derived from the O-D channels — a call consumes a berth slot at **both** endpoints:

```
berth_days_p = Σ_{ch ∋ p} (λ_ch · TEU_ch) / μ_p
ρ_p          = berth_days_p / c_p
```

### 5.5 Sizing: Erlang C with transient headroom

The planner solves for the smallest berth count that meets a wait target, using the productivity
averaged over the *stationary stressed weather law*:

```
a  = λ_p / μ_eff,          μ_eff = μ_p0 · E[productivity] · availability
C(c,a) = (a^c / (c!(1−ρ))) / ( Σ_{k=0}^{c−1} a^k/k! + a^c / (c!(1−ρ)) )
W_q    = C(c,a) / (c·μ_eff − λ_p)
c*     = min { c : W_q(c) ≤ W_target }
```

Stationary Erlang C alone under-sizes the fleet, because the service rate is *time-varying* rather
than i.i.d. The blueprint therefore also requires a headroom term
`c ≥ ⌈λ / (μ_eff · 0.80)⌉` and multiplies by a resilience margin
`m = clip(1 + 0.06·geo_intensity + 0.05·max(0, weather_intensity − 1), 1, 1.5)`.

The residual gap between the naive *proportional* deployment and this *tail-risk* deployment is
exactly what the self-healing loop discovers and closes.

### 5.6 Voyages and routing

Each vessel draws a TEU quantity `~ LogNormal(ln TEU_mean, cv)`, occupies a berth at the origin,
sails the current shortest path, occupies a berth at the destination, and is measured
door-to-door. Path weights are dynamic:

```
w(u,v) = distance_nm / (v_0 · speed_factor(zone)) + Σ_{chokepoint ∈ {u,v}} transit_hours·(1 + q/slots)
```

with `v_0 = 15.5 kn`. Under `adaptive` routing the path is recomputed at departure from the live
queue lengths and capacity factors; under `static` routing vessels follow the free-flow plan.
Contingency lanes are reachable only when the chokepoint they bypass is degraded below threshold.

### 5.7 KPIs

| KPI | Definition |
| --- | --- |
| Throughput | TEU delivered post-warm-up / measured days |
| Service-constrained throughput | Σ TEU · `exp(−max(0, sojourn − SLA)/SLA)` / measured days |
| Service level | share of vessels inside their door-to-door SLA |
| Berth utilisation | Σ service-hours ∩ measured window / (berths × measured days) |
| Backlog slope | OLS slope of total queued vessels vs time |
| Convergence RSE | batch-means relative standard error (k = 10 batches) |
| Resilience index | service-constrained throughput ÷ benign-forcing baseline |
| Demurrage | `900 USD/h × max(0, sojourn − SLA)` |

A run **diverges** (and is aborted, not silently completed) when total berth queue > 320 vessels or
any vessel sojourn exceeds 45 days. Divergence is itself a measurable outcome, reported as
`P(instability)`.

---

## 6. Monte Carlo perturbation & resilience

### 6.1 Latin Hypercube

Five uncertain inputs are sampled over a box:

| Key | Range | Meaning |
| --- | --- | --- |
| `demand_multiplier` | 0.92 – 1.18 | freight demand surge |
| `weather_intensity` | 0.85 – 1.65 | monsoon severity multiplier |
| `geo_intensity` | 0.60 – 1.90 | incident arrival multiplier |
| `service_cv` | 0.30 – 0.62 | handling-time variability |
| `capacity_scale` | 0.97 – 1.03 | crane/berth availability noise |

Each of the `N` replicates receives one stratified sample per dimension — every marginal is covered
uniformly, and the joint sample remains uniform over the box. Replicate `i` also gets its own seed
(`master + 1000·(i+1)`), so the whole ensemble is reproducible from a single integer.

### 6.2 Resilience index

```
R = Θ_SLA(stress) / Θ_SLA(baseline)
Θ_SLA = Σ TEU_k · exp(−max(0, sojourn_k − SLA_k) / SLA_k) / measured_days
```

The baseline is the same network under benign forcing (`weather_intensity = 0.55`, `geo_intensity =
0`, no shock), median of 3 replications.

Raw throughput is deliberately **not** used as the numerator: late cargo is still delivered before
the horizon ends, so raw throughput saturates at offered demand and would report a badly congested
network as healthy. The exponential lateness penalty keeps the metric continuous and sensitive
without collapsing to a binary on-time share.

Lower-tail risk is reported as `CVaR₉₅(X) = E[X | X ≤ F_X⁻¹(0.05)]`.

### 6.3 Sensitivity

Tornado loadings are Spearman rank correlations between each perturbation and the *service*
KPIs (resilience index, mean wait, p95 sojourn). With `n = 24` the standard error of a Spearman
coefficient is ≈ 0.21, so loadings below ~0.4 should not be interpreted; use `--reps 64` or more
for stable sensitivity statements.

---

## 7. Acceptance gates

Evaluated inside the sandbox on the generated artifact, not on the source tree:

| ID | Metric | Gate | Meaning |
| --- | --- | --- | --- |
| `C-UTIL` | `max_utilization` | ≤ 0.92 | stability of the worst terminal |
| `C-SL` | `service_level` | ≥ 0.55 | share of vessels inside SLA |
| `C-RSE` | `convergence_rse` | ≤ 0.06 | steady-state estimator converged |
| `C-SLOPE` | `backlog_slope_per_day` | ≤ 2.5 | no non-stationary queue |
| `C-RUNTIME` | `runtime_seconds` | ≤ 90 | executable inside a realistic budget |
| `C-THRU` | `throughput_ratio` | ≥ 0.55 | capacity actually clears the offered demand |

A gate is only "passed" if the artifact produced a machine-readable result *and* every criterion
holds. Missing metrics fail closed.

The sandbox that evaluates them:

* `python -I -B` → no user site-packages, no `PYTHONPATH`, no bytecode writes
* `RLIMIT_CPU`, `RLIMIT_AS` (3 GB), `RLIMIT_NOFILE`, `RLIMIT_CORE = 0`
* scrubbed environment (no proxies, no tokens — asserted by a test)
* wall-clock timeout with hard kill
* static pre-scan of the generated source for `subprocess`, `eval(`, `exec(`, `socket.`, `ctypes`, …

---

## 8. Reference results

**Build graph** — `python scripts/run_workflow.py`, default scenario (seed 20410):

```text
planner          blueprint compiled
synthesizer      build emitted
sandbox_tester   sandbox run failed      <- gates failed: max_utilization, service_level
reflector        reflection emitted
                 -> stress.capacity_scale: 1.0 => 1.2209
                    observed rho=0.988 exceeded the 0.92 stability gate
                 -> sizing_strategy: proportional => tail_risk
                    proportional sizing ignores stationary productivity loss under stress
synthesizer      build emitted
sandbox_tester   sandbox run passed      <- all gates satisfied
finalizer        artifact finalized: PASS_WITH_RETRIES
```

| Gate | Observed | Threshold | |
| --- | --- | --- | --- |
| `C-UTIL` | 0.739 | ≤ 0.92 | PASS |
| `C-SL` | 0.867 | ≥ 0.55 | PASS |
| `C-RSE` | 0.0247 | ≤ 0.06 | PASS |
| `C-SLOPE` | −0.0090 | ≤ 2.5 | PASS |
| `C-RUNTIME` | 0.35 s | ≤ 90 s | PASS |
| `C-THRU` | 1.052 | ≥ 0.55 | PASS |

Re-running the same command reproduces these figures byte-for-byte (verified across separate
interpreters with different `PYTHONHASHSEED`; see `test_replication_is_stable_across_processes`).

Per-terminal sizing adopted by the healed build (base → required → deployed):

| Port | Base berths | Required (Erlang-C + headroom) | Deployed (margin 1.09, floored by the capacity scale) |
| --- | --- | --- | --- |
| Tanjung Priok | 8 | 9 | 10 |
| Tanjung Perak | 7 | 8 | 9 |
| Belawan | 3 | 3 | 4 |
| Batu Ampar | 9 | 10 | 11 |
| Makassar New Port | 5 | 6 | 7 |
| Kariangau | 4 | 4 | 5 |
| Tanjung Emas | 3 | 3 | 4 |

**Ensemble** — `python scripts/run_montecarlo.py --reps 24` (values are seed-dependent):

```text
baseline throughput      : 19,143 TEU/day
baseline on-time (SLA)   : 18,015 TEU/day  (service level 77.2%)
stressed throughput      : p05=17,949  p50=19,345  p95=21,186
stressed on-time (SLA)   : p05= 7,108  p50=12,534  p95=17,837
resilience index         : p05=0.395  p50=0.696  p95=0.990
resilience CVaR95        : 0.330
P(instability)           : 16.67%
P(service level < 0.60)  : 83.33%
```

Reading: the **as-built** network (baseline berth provisioning) keeps raw throughput at ~19 k
TEU/day but only ~70 % of service-constrained throughput, and one replication in six diverges.
Capacity, not demand, is the binding constraint — which is precisely the finding the build graph
acts on by prescribing +21 % berth provision.

---

## 9. Configuration & extension

The whole scenario is one validated object. To model a different corridor, copy `config.py` and edit:

```python
from maritime_sentinel.schemas import ScenarioSpec
from maritime_sentinel.graph import run_workflow

spec = ScenarioSpec.model_validate_json(open("my_scenario.json").read())
state, context = run_workflow(spec)
print(state["verdict"], context.run_dir())
```

```bash
python scripts/run_workflow.py --scenario my_scenario.json
```

Pydantic rejects, with a specific message, any of: a transition row that is not a probability
vector, operational factors that increase with severity, a lane referencing an unknown node, a
demand channel between a port and itself, a warm-up longer than half the horizon, or an unknown
field.

To add a new failure mode, extend `failure_class` in `TestReport`, add a branch in
`node_reflector`, and it is immediately routable — the graph topology does not change.

---

## 10. Reproducibility & determinism

* One integer seed per replication: `np.random.SeedSequence(seed).spawn(6)` gives six independent
  streams (weather, geopolitics, demand, service, ops, auxiliary). Same seed ⇒ bit-identical KPIs
  (asserted by `test_simulation_is_deterministic_given_the_seed`).
* Ensemble replicate `i` uses `master + 1000·(i+1)`; baseline `j` uses `master + 7000 + j`.
* Every build writes `scenario_manifest.json` (the exact scenario + deployment used) and the SHA-256
  of every artifact into `final_report.json`, so a reported verdict can always be tied to bytes.
* `state_snapshot.json` lets a finished build be re-opened and audited without re-running the sandbox.

```text
artifacts/<run-id>/
├── attempt-00/{scenario_manifest.json, generated_simulation.py, blueprint.md, acceptance.json}
├── attempt-01/{...}
├── final_report.json      verdict, gates, artifacts + hashes, reflection trail
├── execution_log.json     every node transition
└── state_snapshot.json    full graph state
```

---

## 11. Dependencies

| Package | Version | Role |
| --- | --- | --- |
| `langgraph` | ≥ 1.0, < 2 | state graph runtime, conditional routing |
| `pydantic` | ≥ 2.7.4, < 3 | strict schemas at every node boundary |
| `simpy` | ≥ 4.1, < 5 | discrete-event kernel |
| `networkx` | ≥ 3.2, < 4 | lane graph, shortest paths, centrality |
| `numpy` | ≥ 1.26, < 3 | RNG streams, linear algebra, statistics |
| `pandas` | ≥ 2.1, < 4 | telemetry frames |
| `pytest` | ≥ 8.0 | verification only |

Python **3.10+** (developed and verified on 3.11).

```bash
pip install -r requirements.txt   # or: pip install -e ".[dev]"
```

---

## 12. Limitations

Stated plainly, because a stress-test that hides its assumptions is worse than none:

1. **Reduced network.** Seven terminals and five straits are a structural abstraction of the
   Indonesian grid, not a full AIS-derived lane graph. Absolute TEU figures are illustrative; the
   *relative* dynamics under stress are the object of study.
2. **Berth-level, not yard-level.** The model resolves berth occupancy and sailing, not yard
   stacking, trucking appointment systems or empty-container repositioning. Terminal productivity is
   a scalar per berth modulated by weather.
3. **M/M/c sizing vs lognormal service.** Erlang C assumes exponential service; the simulator uses
   lognormal service (CV ≈ 0.42) plus time-varying productivity. This is why the planner adds an
   explicit headroom term instead of trusting the analytic wait in isolation.
4. **Geopolitical incidents are exogenous.** Severity and duration are sampled, not derived from an
   escalation game or an intelligence feed.
5. **Throughput saturates by construction.** Because every generated vessel is eventually served
   before the horizon ends, raw throughput tracks offered demand. Service-constrained throughput and
   tail sojourn are the discriminating KPIs — not headline TEU/day.
6. **Not an operational system.** No live feeds, no actuation, no optimisation of a real fleet. The
   deliverable is the compiled, self-healing *software artifact* and its telemetry.

---

Built with LangGraph, SimPy, NetworkX, Pydantic v2 and Streamlit.
