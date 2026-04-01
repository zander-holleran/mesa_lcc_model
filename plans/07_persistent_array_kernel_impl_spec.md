# Phase 1 Implementation Spec: Persistent Array Vehicle Kernel

**Architecture reference:** `plans/06_persistent_array_kernel_architecture.md`

This spec is written for a coding agent. Execute steps in order. Do not skip ahead. Verify after each step before continuing.

---

## Decisions locked in

| Decision | Answer |
|----------|--------|
| Array source of truth | `VehicleStore` (persistent, in-place) |
| vehicles_list | Keep in sync as parallel list of shells |
| Mesa Agent subclass | Keep (Phase 2 drops it) |
| Mesa space | **Drop entirely.** Add `pos_x`/`pos_y` to store. Update Tier 2 collector to read from store. |
| Blocker integration | Inject as sentinel rows before sort |
| RNG determinism | Batch draws accepted — determinism check relaxed to aggregate outputs only |
| max_concurrent_vehicles | Model parameter, default 1500 |

---

## Pre-work: Save baselines (before any code changes)

```bash
python tests/performance_free_flow.py --save-baseline --runs 3
python tests/performance_congestion.py --save-baseline --runs 3
python tests/optimization_check.py --save-baseline
```

Then add the following temporary timing instrumentation to `traffic_model.step()`. Run `python tests/optimization_check.py --verify` once to record the phase breakdown, then **remove the timing code before proceeding**. The timing output is for human reference only.

```python
import time
# At the top of step():
_t0 = time.perf_counter()
# After update_next_agents():
self._t_sort = self._t_sort + (time.perf_counter() - _t0) if hasattr(self, '_t_sort') else 0.0
# ... etc for each phase
```

---

## Step 1: Extend `traffic/model/init_helpers.py`

**Read first:** `traffic/model/init_helpers.py` (all lines), `traffic/agents/vehicle_agent.py` lines 75–98.

**Goal:** Build four model-level numpy arrays that the kernel will use. These currently exist only as per-vehicle copies inside `VehicleAgent.__init__()`.

**Exact changes to `init_road_segments(model, road_gdf)`:**

After the existing six lines that build `model.rs_pos`, `model.rs_distance`, etc., add:

```python
# Upgrade rs_pos from list-of-tuples to numpy array, then apply start_point override
model.rs_pos = np.array([(pt.x, pt.y) for pt in road_gdf.geometry], dtype=np.float64)  # shape (N, 2)
model.rs_pos[0] = np.array(model.start_point, dtype=np.float64)   # match per-vehicle path_xy[0] = p_start

# Segment geometry (N-1 segments for N points)
seg_vecs = model.rs_pos[1:] - model.rs_pos[:-1]                   # shape (N-1, 2)
rs_seg_len = np.hypot(seg_vecs[:, 0], seg_vecs[:, 1])             # shape (N-1,)
rs_seg_len[rs_seg_len == 0.0] = 1e-12                              # guard zero-length segments
model.rs_seg_len = rs_seg_len                                      # shape (N-1,)
model.rs_seg_dir = seg_vecs / rs_seg_len[:, np.newaxis]            # shape (N-1, 2)
model.rs_cumulative_s = np.cumsum(rs_seg_len)                      # shape (N-1,), road total = rs_cumulative_s[-1]
```

Add `import numpy as np` at the top of `init_helpers.py` if not already present.

**Verify:** Run `python tests/optimization_check.py --verify`. Must pass (model behavior unchanged).

---

## Step 2: Create `traffic/model/vehicle_store.py`

**Read first:** Nothing — this is a new file. Cross-reference the schema in `plans/06_persistent_array_kernel_architecture.md`.

**Create the file with this exact interface:**

```python
import numpy as np


class VehicleStore:
    """
    Persistent numpy arrays for all active vehicle dynamic state.
    Arrays are pre-allocated to max_vehicles. Active vehicles occupy slots [:n_active].
    Slot management uses swap-with-tail for O(1) removal.
    """

    def __init__(self, max_vehicles: int = 1500):
        M = max_vehicles
        self.MAX = M
        self.n_active = 0

        # Slot ↔ vid mappings
        self.vid_to_slot: dict[int, int] = {}
        self.slot_to_vid = np.zeros(M, dtype=np.int64)

        # --- Dynamic state (updated every step by kernel) ---
        self.dist             = np.zeros(M, dtype=np.float64)   # meters from road start
        self.path_idx         = np.zeros(M, dtype=np.int32)     # current segment index
        self.seg_s            = np.zeros(M, dtype=np.float64)   # meters into current segment
        self.speed            = np.zeros(M, dtype=np.float64)   # m/s
        self.break_cd         = np.zeros(M, dtype=np.int8)      # cooldown counter 5→0
        self.status           = np.zeros(M, dtype=np.int8)      # 0=driving 1=slowing 2=crash 3=cc
        self.gap              = np.full(M, np.inf, dtype=np.float64)
        self.ideal_gap        = np.zeros(M, dtype=np.float64)
        self.implicit_sl      = np.zeros(M, dtype=np.float64)   # m/s
        self.speed_delta      = np.zeros(M, dtype=np.float64)   # mph over implicit limit
        self.cumtime_lost     = np.zeros(M, dtype=np.float64)   # cumulative seconds lost
        self.car_interactions = np.zeros(M, dtype=np.int32)
        self.steps_taken      = np.zeros(M, dtype=np.int32)
        self.driving_action   = np.zeros(M, dtype=np.int8)      # encoded, see ACTION_MAP in kernel
        self.pos_x            = np.zeros(M, dtype=np.float64)   # 2D position (replaces Mesa space)
        self.pos_y            = np.zeros(M, dtype=np.float64)

        # --- Immutable traits (written at spawn, never changed) ---
        self.veh_type         = np.zeros(M, dtype=np.int8)      # 0=car 1=bus
        self.acceptable_ov    = np.zeros(M, dtype=np.float32)
        self.ideal_dm         = np.zeros(M, dtype=np.float32)
        self.curve_resp       = np.zeros(M, dtype=np.float32)
        self.performance      = np.zeros(M, dtype=np.float32)
        self.route_end_dist   = np.zeros(M, dtype=np.float64)   # rs_cumulative_s[-1] for all vehicles
        self.created_step     = np.zeros(M, dtype=np.int32)
        self.toll_paid        = np.zeros(M, dtype=np.float32)

    def add(self, vid: int, veh_type: int, pos_x: float, pos_y: float,
            speed: float, route_end_dist: float,
            acceptable_ov: float, ideal_dm: float, curve_resp: float,
            performance: float, created_step: int, toll_paid: float) -> int:
        """
        Write spawn-time values into the next free slot.
        Dynamic fields (dist, path_idx, seg_s, gap, etc.) start at their zero-initialized values.
        Returns the slot index.
        """
        if self.n_active >= self.MAX:
            raise RuntimeError(
                f"VehicleStore full ({self.MAX}). Increase max_concurrent_vehicles."
            )
        slot = self.n_active
        self.slot_to_vid[slot] = vid
        self.vid_to_slot[vid] = slot

        self.veh_type[slot]      = veh_type
        self.pos_x[slot]         = pos_x
        self.pos_y[slot]         = pos_y
        self.speed[slot]         = speed
        self.route_end_dist[slot]= route_end_dist
        self.acceptable_ov[slot] = acceptable_ov
        self.ideal_dm[slot]      = ideal_dm
        self.curve_resp[slot]    = curve_resp
        self.performance[slot]   = performance
        self.created_step[slot]  = created_step
        self.toll_paid[slot]     = toll_paid
        self.gap[slot]           = np.inf
        self.status[slot]        = 0  # driving
        self.break_cd[slot]      = 0
        self.dist[slot]          = 0.0
        self.path_idx[slot]      = 0
        self.seg_s[slot]         = 0.0
        self.cumtime_lost[slot]  = 0.0
        self.car_interactions[slot] = 0
        self.steps_taken[slot]   = 0

        self.n_active += 1
        return slot

    def remove(self, vid: int) -> None:
        """
        Swap-with-tail removal. The last active slot is copied into the freed slot.
        Updates vid_to_slot and slot_to_vid mappings.
        """
        slot = self.vid_to_slot.pop(vid)
        last = self.n_active - 1

        if slot != last:
            # Copy last slot into freed slot for every array
            moved_vid = int(self.slot_to_vid[last])
            for arr in self._all_dynamic + self._all_traits:
                arr[slot] = arr[last]
            self.slot_to_vid[slot] = moved_vid
            self.vid_to_slot[moved_vid] = slot

        self.n_active -= 1

    @property
    def _all_dynamic(self):
        return [
            self.dist, self.path_idx, self.seg_s, self.speed, self.break_cd,
            self.status, self.gap, self.ideal_gap, self.implicit_sl, self.speed_delta,
            self.cumtime_lost, self.car_interactions, self.steps_taken,
            self.driving_action, self.pos_x, self.pos_y,
        ]

    @property
    def _all_traits(self):
        return [
            self.veh_type, self.acceptable_ov, self.ideal_dm, self.curve_resp,
            self.performance, self.route_end_dist, self.created_step, self.toll_paid,
        ]
```

**Note on `slot_to_vid`:** `slot_to_vid` is a numpy int64 array, so `int(self.slot_to_vid[last])` cast is needed before using it as a dict key.

**Verify:** Import the file from a Python shell and instantiate `VehicleStore()`. No errors.

---

## Step 3: Create `traffic/model/vehicle_kernel.py`

**Read first:** `traffic/agents/vehicle_agent.py` lines 131–277 (all speed methods), lines 279–317 (`move_along_path`). `traffic/agents/blocker_agent.py` (all lines).

**Goal:** Single public function `step(model)` that operates in-place on `model.vs` and returns a list of vehicle shell objects that arrived this step.

**Status and action encoding (define at module level):**

```python
import numpy as np

STATUS_DRIVING  = np.int8(0)
STATUS_SLOWING  = np.int8(1)
STATUS_CRASH    = np.int8(2)
STATUS_CC       = np.int8(3)

STATUS_MAP = {'driving': STATUS_DRIVING, 'slowing': STATUS_SLOWING,
              'crash': STATUS_CRASH, 'canyon_closure': STATUS_CC}

ACTION_ACCELERATE    = np.int8(0)
ACTION_COAST         = np.int8(1)
ACTION_SLOW_ACCEL    = np.int8(2)
ACTION_SL_BRAKE      = np.int8(3)
ACTION_SMOOTH_BRAKE  = np.int8(4)
ACTION_PREVENT_PASS  = np.int8(5)

MPS_TO_MPH = 2.23694
MPH_TO_MPS = 1.0 / MPS_TO_MPH
```

**Full kernel algorithm — implement in this exact order:**

```python
def step(model) -> list:
    vs = model.vs
    n = vs.n_active
    if n == 0:
        return []

    # ── 1. Build combined vehicle + blocker arrays for ordering ──────────────
    blockers = model.blockers_list
    n_b = len(blockers)

    veh_dist   = vs.dist[:n].copy()    # copy so sort doesn't affect store until writeback
    veh_speed  = vs.speed[:n].copy()
    veh_status = vs.status[:n].copy()

    if n_b > 0:
        blk_dist   = np.array([b.distance_traveled for b in blockers], dtype=np.float64)
        blk_status = np.array([STATUS_MAP[b.status] for b in blockers], dtype=np.int8)
        blk_speed  = np.zeros(n_b, dtype=np.float64)
        comb_dist   = np.concatenate([veh_dist,   blk_dist])
        comb_speed  = np.concatenate([veh_speed,  blk_speed])
        comb_status = np.concatenate([veh_status, blk_status])
        is_vehicle  = np.array([True] * n + [False] * n_b)
    else:
        comb_dist   = veh_dist
        comb_speed  = veh_speed
        comb_status = veh_status
        is_vehicle  = np.ones(n, dtype=bool)

    n_comb = n + n_b

    # ── 2. Sort combined by distance ─────────────────────────────────────────
    comb_ord = np.argsort(comb_dist, kind='stable')
    sorted_dist   = comb_dist[comb_ord]
    sorted_status = comb_status[comb_ord]
    sorted_speed  = comb_speed[comb_ord]
    sorted_is_veh = is_vehicle[comb_ord]

    # Inverse map: original index → rank in sorted order
    inv_ord = np.empty(n_comb, dtype=np.int32)
    inv_ord[comb_ord] = np.arange(n_comb, dtype=np.int32)
    veh_ranks = inv_ord[:n]   # rank of each vehicle slot in sorted combined order

    # ── 3. Gap to next entity (vehicle or blocker) ───────────────────────────
    leader_rank = np.minimum(veh_ranks + 1, n_comb - 1)
    gap = sorted_dist[leader_rank] - vs.dist[:n]
    gap = np.maximum(gap, 0.0)
    vs.gap[:n] = gap

    leader_status = sorted_status[leader_rank]
    leader_speed  = sorted_speed[leader_rank]

    # ── 4. Adjust status ─────────────────────────────────────────────────────
    ideal_gap = vs.ideal_gap[:n]
    # slowing: gap <= ideal_gap AND leader is not freely driving
    slowing_mask = (gap <= ideal_gap) & (leader_status != STATUS_DRIVING)
    new_status = np.where(slowing_mask, STATUS_SLOWING, STATUS_DRIVING)

    # Backward cascade: inherit crash/cc status from a stopped blocker when very slow
    # Process in sorted vehicle order (front to back = high rank to low rank)
    # If a vehicle is nearly stopped and its leader is crash/cc, adopt that status
    nearly_stopped = vs.speed[:n] < 1.0   # m/s
    inherit_mask = nearly_stopped & (
        (leader_status == STATUS_CRASH) | (leader_status == STATUS_CC)
    )
    new_status[inherit_mask] = leader_status[inherit_mask]
    vs.status[:n] = new_status

    # ── 5. Filter to active movers ───────────────────────────────────────────
    active = (new_status == STATUS_DRIVING) | (new_status == STATUS_SLOWING)

    # ── 6. Speed limit lookup (weighted 5-segment lookahead) ─────────────────
    N_ahead = 5
    n_segs = len(model.rs_speed_limit)
    path_idx = vs.path_idx[:n]

    # Build (n, N_ahead) index array clamped to valid segment range
    offsets = np.arange(N_ahead, dtype=np.int32)
    lookahead_idx = np.minimum(path_idx[:, np.newaxis] + offsets, n_segs - 1)  # (n, N_ahead)

    sl_ahead   = model.rs_speed_limit[lookahead_idx]   # (n, N_ahead), m/s → convert to mph below
    curv_ahead = model.rs_curvature[lookahead_idx]     # (n, N_ahead)

    # Weights: [1, 0.5, 0.333, 0.25, 0.2], normalized (same as VehicleAgent._weights)
    raw_w = np.array([1.0 / (1 + i) for i in range(N_ahead)], dtype=np.float64)
    w = raw_w / raw_w.sum()

    avg_sl_mps  = sl_ahead  @ w   # (n,)
    avg_curv    = curv_ahead @ w  # (n,)

    avg_sl_mph = avg_sl_mps * MPS_TO_MPH
    acceptable_over = vs.acceptable_ov[:n].astype(np.float64)
    curve_resp = vs.curve_resp[:n].astype(np.float64)

    # Curve adjustment (mirrors VehicleAgent.get_speed_limit)
    curve_effect = np.clip(avg_curv / 90.0, 0.0, 1.0)
    speed_effect = np.clip((avg_sl_mph - 10.0) / (60.0 - 10.0), 0.0, 1.0)
    speed_effect[avg_sl_mph <= 15.0] = 0.0
    curve_sl_mph = avg_sl_mph * (1.0 - curve_resp * curve_effect * speed_effect)
    implicit_sl_mph = curve_sl_mph + acceptable_over
    implicit_sl_mps = implicit_sl_mph * MPH_TO_MPS

    vs.implicit_sl[:n] = implicit_sl_mps

    # ── 7. Speed update (priority branches via masks) ─────────────────────────
    speed = vs.speed[:n].copy()
    break_cd = vs.break_cd[:n].copy()
    action = vs.driving_action[:n].copy()
    idm = vs.ideal_dm[:n].astype(np.float64)

    # Masks (mutually exclusive priority):
    gap_brake_mask  = active & (gap < np.maximum(speed * idm, 5.0))
    sl_brake_mask   = active & ~gap_brake_mask & (speed > implicit_sl_mps + 1e-6)
    accel_mask      = active & ~gap_brake_mask & ~sl_brake_mask

    # --- Gap brake ---
    force = np.clip((np.maximum(speed * idm, 5.0) - gap) / np.maximum(np.maximum(speed * idm, 5.0), 1e-6), 0.0, 1.0)
    noise = model.rng.normal(0.0, 0.1, n)
    decel = force ** 2 * 8.0 + noise
    speed[gap_brake_mask] -= decel[gap_brake_mask]
    break_cd[gap_brake_mask] = 5
    action[gap_brake_mask] = ACTION_SMOOTH_BRAKE

    # --- Speed limit brake ---
    over_mph = (speed - implicit_sl_mps) * MPS_TO_MPH
    # mirrors speed_limit_brake: piecewise decel based on how far over
    sl_decel_mps = np.where(
        over_mph < 5.0,
        over_mph * MPH_TO_MPS * 0.3,
        over_mph * MPH_TO_MPS * 0.6,
    )
    speed[sl_brake_mask] -= sl_decel_mps[sl_brake_mask]
    break_cd[sl_brake_mask] = 3
    action[sl_brake_mask] = ACTION_SL_BRAKE

    # --- Accelerate (with cooldown ramp) ---
    # Acceleration magnitude from accel_curve per performance group
    speed_mph = speed * MPS_TO_MPH
    perf_group = (vs.performance[:n] * 100).astype(np.int32)
    perf_group = np.clip(perf_group, 0, 100)

    accel_delta = np.zeros(n, dtype=np.float64)
    for g in np.unique(perf_group[accel_mask]):
        grp_mask = accel_mask & (perf_group == g)
        if grp_mask.any():
            accel_fn = model.accel_curve_cache[g]
            accel_delta[grp_mask] = accel_fn(speed_mph[grp_mask])

    coast_mask      = accel_mask & (break_cd >= 4)
    slow_accel_mask = accel_mask & (break_cd >= 1) & (break_cd < 4)
    full_accel_mask = accel_mask & (break_cd == 0)

    action[coast_mask]      = ACTION_COAST
    action[slow_accel_mask] = ACTION_SLOW_ACCEL
    action[full_accel_mask] = ACTION_ACCELERATE

    ramp = np.where(slow_accel_mask, (4 - break_cd) / 4.0, 1.0)
    speed[slow_accel_mask] += accel_delta[slow_accel_mask] * ramp[slow_accel_mask]
    speed[full_accel_mask] += accel_delta[full_accel_mask]

    break_cd[coast_mask]      = np.maximum(break_cd[coast_mask] - 1, 0)
    break_cd[slow_accel_mask] = np.maximum(break_cd[slow_accel_mask] - 1, 0)

    speed = np.maximum(speed, 0.0)

    # --- Prevent-pass backward clamp ---
    # Must process in front-to-back order (descending dist).
    # Work in sorted vehicle order; clamp only vehicle-to-vehicle (skip blockers).
    sorted_veh_mask = sorted_is_veh   # True where sorted position is a vehicle
    sorted_veh_slots = comb_ord[:n_comb][sorted_veh_mask]  # original slot indices, front to back

    for rank_idx in range(len(sorted_veh_slots) - 1, -1, -1):
        # NOTE: this loop is O(n) but n is small (~500–1000) and runs once per step
        slot_i = sorted_veh_slots[rank_idx]
        if not active[slot_i]:
            continue
        leader_combined_rank = veh_ranks[slot_i] + 1
        if leader_combined_rank >= n_comb:
            continue
        if not sorted_is_veh[leader_combined_rank]:
            continue   # leader is a blocker; no prevent-pass
        leader_slot = comb_ord[leader_combined_rank]
        if leader_slot >= n:
            continue
        would_pass = (speed[slot_i] - speed[leader_slot]) > gap[slot_i]
        if would_pass:
            speed[slot_i] = speed[leader_slot] - 1.0
            break_cd[slot_i] = 5
            action[slot_i] = ACTION_PREVENT_PASS

    # Write speed/cooldown/action back to store
    vs.speed[:n]          = speed
    vs.break_cd[:n]       = break_cd
    vs.driving_action[:n] = action

    # Update ideal_gap post speed update
    vs.ideal_gap[:n] = np.maximum(speed * idm, 5.0)

    # ── 8. Movement ──────────────────────────────────────────────────────────
    new_dist = vs.dist[:n] + speed   # continuous float, meters from road start

    rs_cs = model.rs_cumulative_s    # shape (N-1,)
    n_segs_1 = len(rs_cs)            # N-1

    new_path_idx = np.searchsorted(rs_cs, new_dist, side='right')
    new_path_idx = np.minimum(new_path_idx, n_segs_1 - 1)

    prev_cum = np.where(new_path_idx > 0, rs_cs[new_path_idx - 1], 0.0)
    new_seg_s = new_dist - prev_cum

    new_pos_xy = (
        model.rs_pos[new_path_idx]
        + model.rs_seg_dir[new_path_idx] * new_seg_s[:, np.newaxis]
    )

    arrived_mask = new_dist >= vs.route_end_dist[:n]
    # Clamp arrived vehicles to road end
    new_pos_xy[arrived_mask] = model.rs_pos[-1]
    new_path_idx[arrived_mask] = n_segs_1 - 1
    new_seg_s[arrived_mask] = rs_cs[-1] - (rs_cs[-2] if n_segs_1 > 1 else 0.0)

    vs.dist[:n]     = new_dist
    vs.path_idx[:n] = new_path_idx
    vs.seg_s[:n]    = new_seg_s
    vs.pos_x[:n]    = new_pos_xy[:, 0]
    vs.pos_y[:n]    = new_pos_xy[:, 1]
    vs.steps_taken[:n] += 1

    # ── 9. Time lost ─────────────────────────────────────────────────────────
    speed_mph_final = speed * MPS_TO_MPH
    implicit_sl_mph_final = implicit_sl_mps * MPS_TO_MPH
    speed_delta = speed_mph_final - implicit_sl_mph_final
    time_lost = np.maximum(-speed_delta / np.maximum(implicit_sl_mph_final, 1e-6), 0.0)
    vs.speed_delta[:n]   = speed_delta
    vs.cumtime_lost[:n] += time_lost

    # ── 10. Return arrived vehicle shells for end_of_road() ──────────────────
    arrived_vids = vs.slot_to_vid[:n][arrived_mask]
    return [model.vid_to_vehicle[int(vid)] for vid in arrived_vids]
```

**Important implementation note on the prevent-pass loop:** The Python loop over sorted vehicles is intentional. The loop body is O(1) and n is ~500–1000 max, so total cost is ~1–10μs. A fully vectorized backward pass is possible but adds complexity for minimal gain at this scale. Do not replace with a vectorized version unless profiling shows it's a bottleneck.

**Verify:** Import the module. No import errors.

---

## Step 4: Update `traffic/model/traffic_model.py`

**Read first:** `traffic/model/traffic_model.py` lines 1–300 (full file). Identify every call to: `update_next_agents`, `do_adjust_status`, `do_adjust_speed`, `do_move_along_path`, `do_calculate_time_lost`.

**Changes to `__init__` signature:** Add one parameter after `hybrid_collector_config`:

```python
max_concurrent_vehicles: int = 1500,
```

**Changes to `__init__` body:**

After `init_road_segments(self, road_gdf)` call, add:

```python
from traffic.model.vehicle_store import VehicleStore
from traffic.model import vehicle_kernel  # noqa: F401 (imported for use in step)
self.vs = VehicleStore(max_concurrent_vehicles)
self.max_concurrent_vehicles = max_concurrent_vehicles
self.vid_to_vehicle: dict[int, object] = {}  # vid → VehicleAgent shell
```

**Remove these five methods entirely from `TrafficModel`:**
- `update_next_agents(self)`
- `do_adjust_status(self, vehicles)`
- `do_adjust_speed(self, vehicles)`
- `do_move_along_path(self, vehicles)`
- `do_calculate_time_lost(self, vehicles)`

**Changes to `step(self)`:** Replace the block that calls the five removed methods with:

```python
from traffic.model import vehicle_kernel
arrived = vehicle_kernel.step(self)
for v in arrived:
    v.end_of_road()
```

Remove the lines that filter driving vehicles (e.g. `driving_vehicles = [v for v in ... if v.status in ...]`). The kernel handles filtering internally.

**Verify:** `python -c "from traffic.model.traffic_model import TrafficModel; print('ok')"` — no import errors.

---

## Step 5: Update `traffic/model/generate.py`

**Read first:** `traffic/model/generate.py` (all lines). Focus on `generate_person()`, `generate_new_bus()`, and `too_close()`.

**Goal:** After each vehicle creation, write spawn values into `model.vs` and register in `model.vid_to_vehicle`. Keep all existing Mesa agent creation calls unchanged.

**Pattern to add after every `model.vehicles_list.append(new_car)` or `model.vehicles_list.append(new_bus)`:**

```python
# Register in VehicleStore
model.vs.add(
    vid=new_car.unique_id,
    veh_type=0,                                        # 0=car
    pos_x=float(model.start_point[0]),
    pos_y=float(model.start_point[1]),
    speed=0.0,
    route_end_dist=float(model.rs_cumulative_s[-1]),
    acceptable_ov=float(new_car.acceptable_over),
    ideal_dm=float(new_car.ideal_distance_multiplier),
    curve_resp=float(new_car.curve_responce),
    performance=float(new_car.performance),
    created_step=int(model.steps),
    toll_paid=float(new_car.toll_paid),
)
model.vid_to_vehicle[new_car.unique_id] = new_car
```

For buses, use `veh_type=1` and replace `new_car` with `new_bus`.

**For empty cars** (no TrafficPerson): same pattern with `veh_type=0`.

**Verify:** Model can be instantiated and runs at least 100 steps without error. Run `python tests/optimization_check.py --verify` — expect failure on exact trajectory (RNG change) but NOT on trip count or crash count.

---

## Step 6: Update `traffic/agents/vehicle_agent.py`

**Read first:** `traffic/agents/vehicle_agent.py` (all lines). `traffic/agents/car_agent.py` (all lines). `traffic/agents/bus_agent.py` (all lines).

**Goal:** Strip `VehicleAgent` to a thin shell. Keep Mesa Agent subclass.

**Keep exactly:**
- Class definition: `class VehicleAgent(Agent):`
- `__init__`: Reduce to only:
  ```python
  def __init__(self, model):
      super().__init__(model)
      self.passengers: list = []
  ```
- `vehicle_to_tp_info_pass(self)` — unchanged
- `end_of_road(self)` — rewrite (see below)

**Delete entirely:**
- All attributes set in old `__init__`: `path_xy`, `_seg_speed`, `_seg_curve`, `_weights`, `_seg_vec`, `_seg_len`, `_seg_dir`, `path_index`, `_last_sl_index`, `_s`, `distance_traveled`, `speed`, `break_cooldown`, `status`, `gap`, `ideal_gap`, `next_agent`, `driving_action`, `acceptable_over`, `ideal_distance_multiplier`, `curve_responce`, `performance`, `accel_curve`, `toll_paid`, `created_at_step`, `steps_taken`, `car_interactions`, `cumtime_lost_sec`, `time_lost_sec`, `speed_delta`, `posted_speed_limit`, `implicit_speed_limit`, `implicit_speed_limit_mps`
- Methods: `adjust_status`, `adjust_speed`, `move_along_path`, `calculate_time_lost`, `get_speed_limit`, `less_smooth_brake`, `speed_limit_brake`, `reset_next_agent`
- All helper methods absorbed by the kernel

**Rewrite `end_of_road(self)`:**

```python
def end_of_road(self):
    m = self.model
    vs = m.vs
    slot = vs.vid_to_slot.get(self.unique_id)
    if slot is None:
        return   # already removed (guard against double-call)

    import traffic.utils.unit_conversion_utils as uc

    dist         = float(vs.dist[slot])
    steps        = int(vs.steps_taken[slot])
    interactions = int(vs.car_interactions[slot])
    cumtime      = float(vs.cumtime_lost[slot])
    toll         = float(vs.toll_paid[slot])
    performance  = float(vs.performance[slot])
    curve_resp   = float(vs.curve_resp[slot])
    acceptable_ov= float(vs.acceptable_ov[slot])
    ideal_dm     = float(vs.ideal_dm[slot])
    created      = int(vs.created_step[slot])
    veh_type     = int(vs.veh_type[slot])

    m.finished_agents.append({
        "AgentID":                 self.unique_id,
        "AgentType":               "CarAgent" if veh_type == 0 else "BusAgent",
        "created_at_step":         created,
        "steps_taken":             steps,
        "car_interactions":        interactions,
        "distance_traveled":       dist,
        "approx_average_mph":      uc.meters_to_miles(dist) / (steps / 3600) if steps > 0 else 0.0,
        "performance":             performance,
        "curve_responce":          curve_resp,
        "acceptable_over":         uc.get_mph(acceptable_ov),
        "ideal_distance_multiplier": ideal_dm,
        "cumtime_lost":            cumtime,
        "toll_paid":               toll,
    })

    # Notify passengers
    self.vehicle_to_tp_info_pass()

    # Slot cleanup
    vs.remove(self.unique_id)
    del m.vid_to_vehicle[self.unique_id]
    m.vehicles_list.remove(self)
    self.remove()   # Mesa Agent removal
```

**`CarAgent` and `BusAgent`:** Keep `__init__` sampling logic unchanged (it still runs and sets attributes that `generate.py` reads for `vs.add()`). No other changes needed in Phase 1.

**Verify:** `python tests/optimization_check.py --verify`. Trip counts and crash counts must match. Exact trajectory mismatch is expected and acceptable.

---

## Step 7: Update `traffic/model/hybrid_collector.py` — drop Mesa space, update Tier 2

**Read first:** `traffic/model/hybrid_collector.py` lines 275–320 (Tier 2 collector).

**Goal:** Tier 2 currently reads `v.pos[0]`, `v.pos[1]`, `v.distance_traveled`, `v.gap`, `v.ideal_gap`, `v.speed`, `v.status`, `v.driving_action` from agent objects. Replace with reads from `model.vs` arrays. Remove all `model.space` calls.

**In `Tier2Collector.collect(model)`**, replace the loop over `model.vehicles_list` that reads agent attributes with:

```python
vs = model.vs
n = vs.n_active
if n == 0:
    return

# Read all data from store arrays (vectorized, no per-agent loop)
vids      = vs.slot_to_vid[:n]
xs        = vs.pos_x[:n]
ys        = vs.pos_y[:n]
dists     = vs.dist[:n]
gaps      = vs.gap[:n]
ideal_gaps= vs.ideal_gap[:n]
speeds    = vs.speed[:n]
statuses  = vs.status[:n]
actions   = vs.driving_action[:n]
# ... write to pre-allocated Tier 2 buffer using same format as before
```

Preserve the existing Tier 2 buffer format and all existing columns — only the data source changes.

**Remove all calls to `model.space`:** Search the entire codebase for `space.place_agent`, `space.move_agent`, `space.remove_agent`. Remove every such call. `model.space` can remain as an attribute (initialized but unused) or be removed from `traffic_model.py` — either is fine.

**In `VehicleAgent.__init__`** (already stripped in Step 6): ensure `self.model.space.place_agent(self, ...)` is not present (it was in the old `__init__`, which is now gone).

**Verify:** Run `python tests/optimization_check.py --verify`. Run a full season notebook to confirm animations still render using Tier 2 data.

---

## Step 8: Final verification

```bash
# Correctness (trip count and crash count must match; trajectory diff acceptable)
python tests/optimization_check.py --verify

# Performance comparison against pre-change baselines
python tests/performance_free_flow.py --verify --runs 3
python tests/performance_congestion.py --verify --runs 3
```

Record `avg_steps_per_sec` from both scenarios and compare against the saved `pre` baseline. Report the speedup ratio.

If performance is worse than baseline, profile to identify the bottleneck before continuing to Phase 2.

---

## Common failure modes to watch for

| Symptom | Likely cause |
|---------|-------------|
| `KeyError` in `vs.remove()` | `end_of_road()` called twice for the same vehicle |
| `RuntimeError: VehicleStore full` | `max_concurrent_vehicles` too small for the scenario |
| Vehicles stuck at position 0 | `rs_cumulative_s` not built, or `vs.add()` not called on spawn |
| All vehicles arrive immediately | `route_end_dist` set to 0 instead of `rs_cumulative_s[-1]` |
| Wrong positions in Tier 2 | `pos_x`/`pos_y` not written by kernel before collector runs |
| `IndexError` in speed limit lookup | `path_idx` exceeds `n_segs - 1`; add clamp in kernel |
| Prevent-pass loop wrong slot indices | `comb_ord` indexing off-by-one; re-read the inverse mapping logic |

---

## What is explicitly NOT changed in Phase 1

- `car_agent.py`, `bus_agent.py` — sampling logic unchanged
- `blocker_agent.py` — unchanged; kernel reads blocker attributes as-is
- `traffic_person_agent.py` — unchanged
- `road_segment_agent.py` — unchanged
- `tolling.py` — unchanged; `VolumeSignal` still uses `len(model.vehicles_list)`
- `season/` — all files unchanged
- `tests/` — all test files unchanged (verification commands use them as-is)
