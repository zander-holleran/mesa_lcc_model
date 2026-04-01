# Persistent Array Vehicle Kernel: Architecture Reference

This document is the architecture reference for the persistent array kernel refactor. It is intended for human review. The implementation spec (step-by-step coding instructions) is separate.

---

## Context

The goal is to replace the per-object vehicle update loop with a persistent numpy array kernel where the arrays are the source of truth, not the agent objects. This eliminates per-step extraction and writeback loops entirely.

---

## What the Repo Is Doing Now

### Key files
| File | Role |
|------|------|
| `traffic/model/traffic_model.py` | `TrafficModel`: model init, `step()` orchestration, `update_next_agents()` |
| `traffic/agents/vehicle_agent.py` | `VehicleAgent`: `adjust_speed()`, `move_along_path()`, `adjust_status()`, `calculate_time_lost()` |
| `traffic/agents/car_agent.py` | `CarAgent`: samples behavioral parameters from model distributions |
| `traffic/agents/bus_agent.py` | `BusAgent`: fixed params, overrides passenger charge logic |
| `traffic/agents/blocker_agent.py` | `BlockerAgent`: countdown timer, triggers next_agent pointer reset on death |
| `traffic/agents/road_segment_agent.py` | Static data container (never stepped, no dynamic state) |
| `traffic/model/generate.py` | Spawning: persons, cars, buses, crashes, canyon closures |
| `traffic/model/init_helpers.py` | Builds road arrays: `rs_pos`, `rs_distance`, `rs_speed_limit`, `rs_curvature` |
| `traffic/model/hybrid_collector.py` | 4-tier data collection with pre-allocated numpy arrays |

### Simulation entry points
- `notebooks/single_day_season.ipynb` → `SeasonOrchestrator.run_season()` → `TrafficModel` per day

### Current step order (one tick of `TrafficModel.step()`)
1. `update_tolls()` — read signal, apply transform, update `model.current_toll_car`
2. `gen.generate_person()` — Bernoulli arrival, spawn `TrafficPersonAgent` + car or queue for bus
3. `gen.generate_new_bus()` — if interval elapsed, spawn bus and board waiting persons
4. **`update_next_agents()`** — sort `vehicles_list + blockers_list` by `distance_traveled`, set `next_agent` pointers and `gap` on each object
5. `do_adjust_status()` — each vehicle sets own `status`
6. Filter to driving/slowing vehicles
7. **`do_adjust_speed()`** — each vehicle computes new speed: gap control → speed limit → accelerate → prevent-pass clamp
8. **`do_move_along_path()`** — each vehicle advances `speed` meters along path, updates segment index
9. `do_calculate_time_lost()` — per-vehicle delay metric vs implicit speed limit
10. Crash and canyon closure generation
11. `do_tick()` — decrement blocker timers, remove expired
12. `datacollector.collect()` — Tier 1 + Tier 2
13. Termination checks

### Where state currently lives
- **Dynamic, hot:** Python vehicle agent objects — `speed`, `break_cooldown`, `status`, `distance_traveled`, `path_index`, `_s`, `gap`, `ideal_gap`
- **Static per-vehicle (numpy, built at spawn):** `path_xy`, `_seg_speed`, `_seg_curve`, `_seg_vec`, `_seg_len`, `_seg_dir`, `_weights` — redundant copies of model-level road data
- **Static model-level (numpy):** `rs_pos`, `rs_distance`, `rs_speed_limit`, `rs_curvature`, `rs_road_section`
- **Per-vehicle behavioral traits:** `acceptable_over`, `ideal_distance_multiplier`, `curve_responce`, `performance` — immutable after spawn

### Where the expensive logic lives
- **`update_next_agents()`** — O(n log n) sort + O(n) pointer loop, every step
- **`do_adjust_speed()`** — O(n) per-agent branching + per-vehicle speed limit lookup
- **`do_move_along_path()`** — O(n) per-agent while loop across segment boundaries
- **Data collection Tier 1** — pure Python loops over `vehicles_list`

---

## Scope: What Is Being Replaced

The target is **CarAgent and BusAgent** (both via their shared `VehicleAgent` base). These are the only entities that are both stepped every tick and hold dynamic numeric state.

Everything else stays as objects:
- `BlockerAgent` — small count, complex teardown side effects, keep as object
- `TrafficPersonAgent` — not stepped per tick, passive after mode decision, keep as object
- `RoadSegmentAgent` — already vestigial; real road data lives in `model.rs_*` arrays
- `SeasonPerson` — updated once per day, not per step, keep as dataclass

---

## Target Architecture: Persistent Arrays + Swap-With-Tail

### Core idea

Model-level numpy arrays are the source of truth for all vehicle dynamic state. They are pre-allocated once at model init and persist across steps. The kernel operates on them in-place — no extraction loop, no writeback loop.

### Array sizing

Pre-allocate to **peak concurrent vehicle count** — vehicles actually on the road at the same time, not cumulative trips. LCC peak concurrent is approximately 500–1000 vehicles.

`max_concurrent_vehicles` is a **model parameter** passed to `TrafficModel.__init__()` (default `1500`). It is stored as `model.max_concurrent_vehicles` and passed to `VehicleStore`. This makes it easy to tune for different scenario scales or to later expose as a scenario config option.

Key point: after 3000 total trips, at most ~1000 rows are ever active. The arrays never grow beyond the ceiling. This scales with concurrent load, not trip volume.

### Slot management: swap-with-tail

Track `vs.n_active` — the count of currently active vehicles. All kernel ops run on `arrays[:n_active]` only.

**On spawn:** write into slot `n_active`, increment counter, store `vid → slot` mapping.

**On arrival/removal:** copy last active slot into the vacated slot, decrement counter, update the mapping for the moved vehicle. O(1).

The `vid_to_slot` dict (max ~1000 entries) is the only lookup structure needed. `slot_to_vid` is a small integer array for reverse lookup during swaps.

### Persistent array schema

All arrays owned by `VehicleStore`, pre-allocated to `max_concurrent_vehicles`:

**Dynamic state (updated every step by kernel):**
| Array | dtype | Notes |
|-------|-------|-------|
| `dist` | float64 | cumulative distance from road start — ordering key |
| `path_idx` | int32 | current road segment index |
| `seg_s` | float64 | float offset within current segment (meters) |
| `speed` | float64 | m/s |
| `break_cd` | int8 | countdown 5→0 |
| `status` | int8 | 0=driving, 1=slowing, 2=crash, 3=canyon_closure |
| `gap` | float64 | distance to leader (also read by collector) |
| `ideal_gap` | float64 | `max(speed * idm, 5)` |
| `implicit_sl` | float64 | weighted lookahead speed limit, m/s |
| `speed_delta` | float64 | mph over implicit limit (for time-lost) |
| `cumtime_lost` | float64 | cumulative seconds lost (accumulates in-place) |
| `car_interactions` | int32 | cumulative gap-braking events |
| `steps_taken` | int32 | steps since spawn |

**Immutable traits (written at spawn, never changed):**
| Array | dtype | Notes |
|-------|-------|-------|
| `veh_type` | int8 | 0=car, 1=bus |
| `acceptable_ov` | float32 | sampled at spawn |
| `ideal_dm` | float32 | ideal distance multiplier |
| `curve_resp` | float32 | curve sensitivity |
| `performance` | float32 | 0–1 percentile |
| `route_end_seg` | int32 | segment index at route end |
| `route_end_dist` | float64 | `rs_cumulative_s[route_end_seg]` — enables single float arrival check |
| `created_step` | int32 | for trip log |
| `toll_paid` | float32 | baked in at spawn |

### Road lookup (static, all on `model.rs_*`)

| Array | Shape | Status | Notes |
|-------|-------|--------|-------|
| `rs_speed_limit` | `(N,)` | existing | m/s per segment |
| `rs_curvature` | `(N,)` | existing | degrees per segment |
| `rs_distance` | `(N,)` | existing | cumulative distance per segment node |
| `rs_pos` | `(N, 2)` | **upgrade** | currently a Python list of tuples — convert to `np.array`; first point replaced with `model.start_point` to match per-vehicle behavior |
| `rs_seg_len` | `(N-1,)` | **move to model** | currently only on per-vehicle `_seg_len` |
| `rs_seg_dir` | `(N-1, 2)` | **move to model** | currently only on per-vehicle `_seg_dir` |
| `rs_cumulative_s` | `(N-1,)` | **new** | `np.cumsum(rs_seg_len)` — key to vectorized movement |

---

## Vehicle Objects: Thin Shells

After this change, `VehicleAgent` holds only:
- `passengers` — list of `TrafficPersonAgent` references
- `unique_id` — for `vid_to_slot` mapping and trip logging
- `end_of_road()` — reads final values from `model.vs` at the vehicle's slot, logs to `finished_agents`, calls `vehicle_to_tp_info_pass()`
- `vehicle_to_tp_info_pass()` — unchanged

Everything else is deleted.

### Distribution sampling: how it works after the change

**In Phase 1**, sampling still occurs in `CarAgent.__init__()` and `BusAgent.__init__()` exactly as today. Immediately after construction, `generate.py` reads sampled values off the object and writes them into `VehicleStore` slots. The instance attributes become vestigial.

**In Phase 2 (optional)**, sampling moves directly into `generate.py`, bypassing the class attributes entirely.

---

## Vectorization Map

| Method/Function | Category |
|-----------------|----------|
| `update_next_agents()` sort+link | vectorizable after sort — `np.argsort`; leader = sort_pos+1; gap = dist diff |
| `adjust_status()` | vectorizable with backward pass for blocker cascade |
| `get_speed_limit()` | vectorized transform — advanced index `rs_speed_limit[path_idx + offsets]` |
| `less_smooth_brake()` | vectorized transform |
| `speed_limit_brake()` | vectorized transform |
| `accel_curve(speed_mph)` | grouped operation — group by `(performance*100).astype(int)` |
| `adjust_speed()` priority branches | vectorized with masks |
| `prevent_pass` clamp | short ordered backward pass |
| `move_along_path()` while loop | vectorized via `searchsorted(rs_cumulative_s, dist + speed)` |
| `calculate_time_lost()` | vectorized transform |
| `do_*` dispatch loops | deleted |
| `end_of_road()` | keep as object logic — rare, reads from store at departure |
| `BlockerAgent.tick()` | keep as object logic |

---

## vehicles_list

Kept in sync as a parallel list of shell objects. Every spawn appends the shell; every departure removes it (in `end_of_road()`). `VolumeSignal`, crash generation, and termination checks continue to work unchanged. `model.vs.n_active` is also available for count-only uses.

---

## Continuous-Space Movement

Vehicles occupy continuous positions between road nodes. `vs.dist` is a float (meters from road start). Position is always linearly interpolated:

```python
new_dist = vs.dist[:n] + vs.speed[:n]
new_path_idx = np.searchsorted(model.rs_cumulative_s, new_dist, side='right')
new_path_idx = np.minimum(new_path_idx, len(model.rs_cumulative_s) - 1)
prev_cum = np.where(new_path_idx > 0, model.rs_cumulative_s[new_path_idx - 1], 0.0)
new_seg_s = new_dist - prev_cum
new_pos_xy = model.rs_pos[new_path_idx] + model.rs_seg_dir[new_path_idx] * new_seg_s[:, np.newaxis]
arrived_mask = new_dist >= vs.route_end_dist[:n]
```

---

## Implementation Stack

**numpy arrays only.** All kernel operations are direct numpy in-place ops. No pandas, no pyarrow in the hot path.

---

## Baseline Protocol

### Before any changes
```bash
python tests/performance_free_flow.py --save-baseline --runs 3
python tests/performance_congestion.py --save-baseline --runs 3
python tests/optimization_check.py --save-baseline
```

Also add temporary per-phase timing to `traffic_model.step()` before changes, to get concrete phase-by-phase breakdown.

### After changes
1. `python tests/optimization_check.py --verify` — trip counts and crash counts must match; per-step trajectory differences acceptable due to batch RNG draws
2. `python tests/performance_free_flow.py --verify --runs 3`
3. `python tests/performance_congestion.py --verify --runs 3`
4. Compare `avg_steps_per_sec` against baseline

---

## Phase 1: Core Kernel

Files modified:
| File | Change |
|------|--------|
| `traffic/model/vehicle_store.py` | **New** — `VehicleStore` class |
| `traffic/model/vehicle_kernel.py` | **New** — in-place numpy kernel |
| `traffic/model/traffic_model.py` | Remove 5 methods; add `VehicleStore`, kernel call, `vid_to_vehicle` dict |
| `traffic/model/generate.py` | Spawn/remove writes to `vs`; also maintains `vehicles_list` |
| `traffic/model/init_helpers.py` | Upgrade `rs_pos`; add `rs_seg_len`, `rs_seg_dir`, `rs_cumulative_s` |
| `traffic/agents/vehicle_agent.py` | Strip to thin shell |

Files unchanged: `car_agent.py`, `bus_agent.py`, `blocker_agent.py`, `traffic_person_agent.py`, `road_segment_agent.py`, `tolling.py`, `hybrid_collector.py`, all season/ files, all tests.

---

## Phase 2: Cleanup

- Remove redundant per-vehicle road arrays from `VehicleAgent.__init__()`
- Vectorize Tier 1 histogram collection using `vs.speed[:n_active]`
- Optionally move distribution sampling from agent classes into `generate.py`
- Optionally drop Mesa Agent subclassing for vehicle shells if registry overhead is measurable
- Optionally remove `RoadSegmentAgent` if registry overhead is measurable

---

## Risk and Payoff

| Dimension | Rating |
|-----------|--------|
| Fit for array kernel | High |
| Semantic risk | Low-Medium |
| Implementation complexity | Medium |
| Expected speedup (Phase 1) | ~3x end-to-end; 4–8x on kernel phases |
