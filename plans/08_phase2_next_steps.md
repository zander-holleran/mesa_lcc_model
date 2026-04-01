# Phase 2 Next Steps: Post-Kernel Cleanup

These are follow-on improvements after Phase 1 is validated. Each is independent — they can be done in any order. None are required for correctness; all improve performance, memory, or code clarity.

---

## 2a. Remove redundant per-vehicle road arrays from spawn

**What:** `CarAgent.__init__()` and `BusAgent.__init__()` still build per-vehicle numpy arrays (`_seg_speed`, `_seg_curve`, `_seg_vec`, `_seg_len`, `_seg_dir`, `_weights`, `path_xy`) because they haven't been removed yet. After Phase 1, these are written to instance attributes but never read again.

**Change:** Strip `VehicleAgent.__init__()` further — remove the `road_segments`, `_rs`, `path_xy`, `_seg_speed`, `_seg_curve`, `_weights`, `_seg_vec`, `_seg_len`, `_seg_dir`, `N_ahead` lines (vehicle_agent.py lines 75–98 in the original). These were the expensive part of spawn.

**Impact:** Eliminates 6–7 numpy array allocations and one `model.agents.select()` call per spawned vehicle. At 3000 vehicles per season this is meaningful spawn-time savings.

**Also move distribution sampling:** `CarAgent.__init__()` currently samples from `model.dist_acceptable_over` etc. and sets attributes that `generate.py` immediately reads for `vs.add()`. Move the sampling directly into `generate.py` instead. `CarAgent.__init__()` becomes:
```python
def __init__(self, model):
    super().__init__(model)
    # no sampling, no arrays — generate.py handles everything
```

---

## 2b. Vectorize Tier 1 histogram collection

**What:** `hybrid_collector.py` Tier 1 still loops `model.vehicles_list` in Python to compute speed histograms. After Phase 1, `model.vs.speed[:n_active]` and `model.vs.speed_delta[:n_active]` are already available as compact numpy slices.

**Change:** Replace per-agent Python loops in `Tier1Collector.collect()` with:
```python
n = model.vs.n_active
speeds_mph = model.vs.speed[:n] * 2.23694
delta = model.vs.speed_delta[:n]
speed_counts = np.bincount(np.searchsorted(speed_bins, speeds_mph), minlength=len(speed_bins))
delta_counts = np.bincount(np.searchsorted(delta_bins, delta), minlength=len(delta_bins))
```

**Impact:** Two Python loops per collection step → two numpy one-liners.

---

## 2c. Drop Mesa Agent subclassing for vehicle shells

**What:** After Phase 1, `VehicleAgent` holds only `passengers`, `unique_id`, and `end_of_road()`. It still subclasses `Agent` and stays in `model.agents` registry. Mesa's agent registry bookkeeping touches these objects on every `model.agents.select()` call.

**Change:** Convert `VehicleAgent` to a plain Python class:
```python
class VehicleShell:
    __slots__ = ('unique_id', 'passengers', 'model')
    def __init__(self, model):
        self.unique_id = model._next_id()   # or equivalent
        self.passengers = []
        self.model = model
```

Update `CarAgent` and `BusAgent` to subclass `VehicleShell`. Update `generate.py` to use direct construction instead of `model.agent_cls['car'].create_agents()`. Remove `model.agents.add(new_car)` and `self.remove()` calls.

**Prerequisite:** Confirm nothing else in the codebase calls `model.agents.select(agent_type=VehicleAgent)` or similar after Phase 1 cleanup. The old `update_next_agents()` and blocker `self_distruct()` both did this — check if any other callers remain.

**Impact:** Removes Mesa registry overhead for vehicles. Smaller memory footprint per shell.

---

## 2d. Remove RoadSegmentAgent

**What:** `RoadSegmentAgent` objects are created at init and immediately ignored. All their data is now in `model.rs_*` arrays. They sit in `model.agents` consuming registry space.

**Change:** Remove `init_road_segments()` agent creation loop. Remove `RoadSegmentAgent` class or leave it but stop instantiating it. Remove `model.agent_cls['road']` registration.

**Prerequisite:** Confirm `model.agents.select(agent_type=RoadSegmentAgent)` is called nowhere active after Phase 1. The old `VehicleAgent.__init__()` called it — that's gone after 2a above.

**Impact:** Faster model init. Smaller Mesa registry.

---

## 2e. Vectorize the prevent-pass backward pass (optional)

**What:** The prevent-pass loop in `vehicle_kernel.py` is a Python `for` loop over sorted vehicles. It's O(n) and intentionally left as a loop in Phase 1 for clarity.

**Change:** Replace with a vectorized backward cumulative minimum pass:
```python
# In sorted order (front to back), leader speed is sorted_speed[rank+1]
# Clamp: vehicle speed cannot exceed leader_speed - 1 if it would pass
# This is a backward cumulative minimum on (speed + gap) relationships
```

The fully vectorized version requires careful index mapping (sorted combined space → slot space). Only worth doing if profiling shows the loop is measurable.

---

## Priority order

| Item | Effort | Impact |
|------|--------|--------|
| 2a (remove spawn arrays + move sampling) | Low | Medium — spawn-time savings |
| 2b (vectorize Tier 1 histograms) | Very low | Low — collection is rarely the bottleneck |
| 2c (drop Mesa Agent subclass) | Medium | Low-Medium — registry overhead at scale |
| 2d (remove RoadSegmentAgent) | Low | Low — minor init savings |
| 2e (vectorize prevent-pass) | Medium | Negligible — loop is ~1–10μs |

Do 2a first. Do 2b anytime. Do 2c and 2d together if Mesa registry overhead shows up in profiling.
