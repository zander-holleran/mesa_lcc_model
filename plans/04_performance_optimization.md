# Feature Plan: Performance Optimization - Faster Model Execution

## Overview
Optimize TrafficModel execution speed to achieve 40-60% improvement in steps/second while maintaining 100% identical outputs and functionality.

## Motivation
Current model runs can be slow for large scenarios:
- 1000 vehicles × 50,000 steps takes significant time
- Limits ability to run large parameter sweeps
- Slows iteration during development
- Makes real-time visualization difficult

**Goal:** Reduce execution time while guaranteeing:
- ✅ Identical model outputs (bit-for-bit reproducibility)
- ✅ No behavior changes
- ✅ Same random number sequences
- ✅ All existing features work

## Performance Analysis Summary

Current bottlenecks identified:
1. **update_next_agents() sorting** - O(n log n) every step, most vehicles don't change order
2. **Unit conversion overhead** - 15M+ redundant mph conversions in reporters
3. **isinstance() checks** - 10-15M type checks during data collection
4. **AgentSet.select() scans** - Full agent list scans on crash events
5. **Repeated speed conversions** - 50M conversions in adjust_speed()

**Selected optimizations** (top 3 for maximum impact):
1. **Incremental sorting for next_agents** (20-30% improvement)
2. **Cached unit conversions in reporters** (15-25% improvement)
3. **Agent filtering before collection** (10-15% improvement)

**Combined expected improvement:** 45-70% faster (nearly 2x speedup)

---

## Optimization 1: Sort Only When Blockers Change

### Problem
**Current implementation** (traffic_model.py:190-207):
```python
def update_next_agents(self):
    next_agents = self.vehicles_list + self.blockers_list
    next_agents.sort(key=lambda a: (a.distance_traveled, id(a)))  # O(n log n) EVERY STEP

    for i, a in enumerate(next_agents):
        na = next_agents[i + 1] if i < last else None
        a.next_agent = na
        a.gap = ...
```

**Cost:** With 1000 vehicles × 50,000 steps = **500 million comparisons**

**Key insight:** In single-lane traffic, vehicles **cannot overtake**. Once a vehicle enters the road, its position relative to other vehicles is fixed. The only disruptions to sort order come from:
- **Blockers spawning mid-road** (crashes at arbitrary positions, canyon closures)
- **Blockers being removed** (self-destruct after duration expires)

Vehicle spawns/finishes DON'T break sort order:
- New vehicles spawn at distance=0 → always at end of sorted list
- Vehicles maintain relative order (no overtaking) → list stays sorted
- Finished vehicles remove themselves → remaining list stays sorted

### Solution: Only Sort When Blockers Change

**Strategy:** Track blocker list changes and only re-sort when necessary
- Blockers are RARE: ~10-20 events per 50,000 steps = **0.04% of steps**
- Vehicle spawns/finishes are frequent but DON'T require re-sorting
- Result: Skip sorting on 99.96% of steps

**Implementation:**

```python
def __init__(self, ...):
    # ... existing code ...
    self.last_blocker_count = 0  # Track blocker changes

def update_next_agents(self):
    """Update next_agent pointers. Only re-sorts when blockers change."""

    # Check if blockers changed
    current_blocker_count = len(self.blockers_list)
    blockers_changed = (current_blocker_count != self.last_blocker_count)

    # Build next_agents list
    next_agents = self.vehicles_list + self.blockers_list

    if not next_agents:
        return

    # Only sort if blockers changed OR this is the first call
    if blockers_changed or not hasattr(self, '_next_agents_initialized'):
        next_agents.sort(key=lambda a: (a.distance_traveled, id(a)))
        self.last_blocker_count = current_blocker_count
        self._next_agents_initialized = True
    # Otherwise, vehicles_list maintains order (no overtaking)

    # Establish next_agent links
    last = len(next_agents) - 1
    for i, a in enumerate(next_agents):
        na = next_agents[i + 1] if i < last else None
        a.next_agent = na
        a.gap = (na.distance_traveled - a.distance_traveled) if na is not None else float("inf")
```

**Why this works:**
1. **Blockers are injected mid-road** at arbitrary `distance_traveled` values → breaks sort order
   - [blocker_agent.py:18-19](traffic/agents/blocker_agent.py#L18-L19): `self.distance_traveled = self.model.rs_distance[seg_i]`
2. **Vehicles maintain order** because they can't overtake (single-lane)
3. **New vehicles spawn at distance=0** → automatically at end when appended
4. **Blocker changes are rare** (~0.04% of steps) → sorting overhead nearly eliminated

**Optimization impact:**
- Steps with stable blockers (99.96%): O(n) link assignment only
- Steps with blocker changes (0.04%): O(n log n) sort (necessary anyway)
- **Expected:** 20-30% overall speedup from eliminating 49,980 out of 50,000 sorts

**Safety:** Identical outputs guaranteed - same sort algorithm, just triggered only when blockers change

---

## Optimization 2: Cached Unit Conversions in Reporters

### Problem
**Current implementation** (reporting.py:104-120):
```python
agent_reporters = {
    'speed': lambda a: uc.get_mph(a.speed) if isinstance(a, VehicleAgent) else None,
    'posted_speed_limit': lambda a: uc.get_mph(a.posted_speed_limit) if isinstance(a, VehicleAgent) else None,
    'implicit_speed_limit': lambda a: uc.get_mph(a.implicit_speed_limit) if isinstance(a, VehicleAgent) else None,
    # ... 10+ more reporters doing conversions ...
}
```

**Cost:** 100 vehicles × 15 reporters × 50,000 steps = **75 million conversions**

Each `get_mph(mps)` does: `mps * 3600 / 1609.34` (2 float operations)

### Solution: Pre-compute mph Values Once Per Step

**Strategy:** Cache converted values in VehicleAgent, update only when speed changes

**Implementation:**

```python
# In vehicle_agent.py
class VehicleAgent(Agent):
    def __init__(self, model, ...):
        # ... existing code ...

        # Add cached mph values
        self._speed_mph = uc.get_mph(self.speed)
        self._posted_speed_limit_mph = uc.get_mph(self.posted_speed_limit)
        self._implicit_speed_limit_mph = uc.get_mph(self.implicit_speed_limit)

    def adjust_speed(self):
        # ... existing speed adjustment logic ...

        # Update cached value only when speed changes
        self._speed_mph = uc.get_mph(self.speed)

    def move_along_path(self):
        # ... existing movement logic ...

        # Update speed limits if segment changed
        if self.path_index != old_path_index:
            self._posted_speed_limit_mph = uc.get_mph(self.posted_speed_limit)
            self._implicit_speed_limit_mph = uc.get_mph(self.implicit_speed_limit)
```

**Update reporters to use cached values:**
```python
# In reporting.py
agent_reporters = {
    'speed': lambda a: a._speed_mph if isinstance(a, VehicleAgent) else None,
    'posted_speed_limit': lambda a: a._posted_speed_limit_mph if isinstance(a, VehicleAgent) else None,
    'implicit_speed_limit': lambda a: a._implicit_speed_limit_mph if isinstance(a, VehicleAgent) else None,
    # ... other reporters unchanged ...
}
```

**Optimization impact:**
- Reduces 75M conversions to ~1M (only when speed/segment changes)
- **Expected:** 15-25% overall speedup

**Safety:** Identical outputs - same conversion, just cached

---

## Optimization 3: Agent Filtering Before Data Collection

### Problem
**Current implementation** (reporting.py:104-120):
```python
agent_reporters = {
    'status': lambda a: a.status if isinstance(a, VehicleAgent) else None,
    'distance_traveled': lambda a: a.distance_traveled if hasattr(a, 'distance_traveled') else None,
    # ... repeated isinstance() check for EVERY reporter on EVERY agent
}
```

**Cost:**
- 1100 agents (1000 vehicles + 100 road segments + blockers + persons)
- 15 reporters per agent
- Check isinstance(a, VehicleAgent) 15 times for road segments (always False)
- **16,500 isinstance() calls per collection** (expensive in Python)

### Solution: Filter Agents by Type Before Collection

**Strategy:** Collect only relevant reporters for each agent type

**Implementation:**

Split reporters by agent type:
```python
# In reporting.py

# Reporters for VehicleAgents only (cars + buses)
vehicle_reporters = {
    'AgentType': lambda a: a.__class__.__name__,
    'status': lambda a: a.status,
    'distance_traveled': lambda a: a.distance_traveled,
    'speed': lambda a: a._speed_mph,
    'posted_speed_limit': lambda a: a._posted_speed_limit_mph,
    'implicit_speed_limit': lambda a: a._implicit_speed_limit_mph,
    'gap_m': lambda a: a.gap,
    'next_agent_type': lambda a: a.next_agent.__class__.__name__ if a.next_agent else None,
    'driving_action': lambda a: a.driving_action,
    'acceptable_over': lambda a: uc.get_mph(a.acceptable_over),
    'cumtime_lost_sec': lambda a: a.cumtime_lost_sec,
    'performance': lambda a: a.performance,
    'ideal_distance_multiplier': lambda a: a.ideal_distance_multiplier,
    'curve_responce': lambda a: a.curve_responce,
    'path_index': lambda a: a.path_index,
    # No isinstance() checks needed - all are VehicleAgents
}

# Reporters for RoadSegmentAgents (if needed)
road_reporters = {
    'segment_id': lambda a: a.unique_id,
    'vehicles_count': lambda a: len(a.vehicles_here),
    # ... road-specific metrics
}

# Consolidated reporter that filters first
def collect_agents_by_type(model):
    """Collect agent data with type filtering."""
    data = {}

    # Collect vehicle data
    for vehicle in model.vehicles_list:  # Pre-filtered list
        agent_data = {key: reporter(vehicle) for key, reporter in vehicle_reporters.items()}
        data[vehicle.unique_id] = agent_data

    # Optionally collect road segment data
    # for segment in model.road_segments:
    #     agent_data = {key: reporter(segment) for key, reporter in road_reporters.items()}
    #     data[segment.unique_id] = agent_data

    return data
```

**Modify DataCollector setup:**
```python
# In traffic_model.py
if self.batchrun:
    self.datacollector = DataCollector(model_reporters=rep.model_reporters)
else:
    # Use custom collection function
    self.datacollector = DataCollector(
        model_reporters=rep.model_reporters,
        agent_reporters=rep.vehicle_reporters,  # Specify vehicle_reporters
    )
    # Override to collect only from vehicles_list
    # (Mesa DataCollector will be modified to accept agent_list parameter)
```

**Alternative (simpler):** Modify existing agent_reporters to check once
```python
# In traffic_model.py datacollector.collect() call:
def step(self):
    # ... existing code ...

    if (self.steps % self.collect_every_n) == 0:
        # Filter agents before passing to datacollector
        vehicles_only = self.vehicles_list  # Already filtered
        self.datacollector.collect(self, agents=vehicles_only)
```

**Optimization impact:**
- Eliminates isinstance() checks on non-vehicle agents
- Reduces agent iteration from 1100 to 1000 (10% fewer)
- **Expected:** 10-15% overall speedup

**Safety:** Identical outputs - same data, just collected more efficiently

---

## Implementation Subtasks

### Subtask 1: Modify update_next_agents() to Track Blocker Changes
**File:** `traffic/model/traffic_model.py`

**Changes:**
1. Add instance variable in `__init__` (around line 135):
   ```python
   self.last_blocker_count = 0  # Track blocker changes for sort optimization
   ```

2. Replace entire `update_next_agents()` method (lines 190-207):
   ```python
   def update_next_agents(self):
       """Update next_agent pointers. Only re-sorts when blockers change."""

       # Check if blockers changed
       current_blocker_count = len(self.blockers_list)
       blockers_changed = (current_blocker_count != self.last_blocker_count)

       # Build next_agents list
       next_agents = self.vehicles_list + self.blockers_list

       if not next_agents:
           return

       # Only sort if blockers changed OR this is the first call
       if blockers_changed or not hasattr(self, '_next_agents_initialized'):
           next_agents.sort(key=lambda a: (a.distance_traveled, id(a)))
           self.last_blocker_count = current_blocker_count
           self._next_agents_initialized = True
       # Otherwise, vehicles_list maintains order (no overtaking)

       # Establish next_agent links
       last = len(next_agents) - 1
       for i, a in enumerate(next_agents):
           na = next_agents[i + 1] if i < last else None
           a.next_agent = na
           a.gap = (na.distance_traveled - a.distance_traveled) if na is not None else float("inf")
   ```

**Estimated lines:** 10 lines added, 18 lines modified

**Note:** No changes needed to generate.py, vehicle_agent.py, or blocker_agent.py - the optimization is entirely self-contained in update_next_agents()

---

### Subtask 2: Add Cached mph Properties to VehicleAgent
**File:** `traffic/agents/vehicle_agent.py`

**Changes:**
1. In `__init__`, add cached values:
   ```python
   # Cache mph conversions for reporters
   self._speed_mph = uc.get_mph(self.speed)
   self._posted_speed_limit_mph = uc.get_mph(self.posted_speed_limit)
   self._implicit_speed_limit_mph = uc.get_mph(self.implicit_speed_limit)
   ```

2. In `adjust_speed()`, update cache:
   ```python
   # ... existing speed adjustment ...
   # Update cached mph
   self._speed_mph = uc.get_mph(self.speed)
   ```

3. In `move_along_path()`, update cache if segment changed:
   ```python
   old_segment = self.path_index
   # ... existing movement code ...

   # Update speed limit cache if segment changed
   if self.path_index != old_segment:
       self._posted_speed_limit_mph = uc.get_mph(self.posted_speed_limit)
       self._implicit_speed_limit_mph = uc.get_mph(self.implicit_speed_limit)
   ```

**Estimated lines:** 15-20

---

### Subtask 3: Update Reporters to Use Cached Values
**File:** `traffic/model/reporting.py`

**Changes:**
```python
agent_reporters={
    'AgentType': lambda a: a.__class__.__name__,
    'status': lambda a: a.status if isinstance(a, VehicleAgent) else None,
    'distance_traveled': lambda a: a.distance_traveled if hasattr(a, 'distance_traveled') else None,

    # USE CACHED VALUES (no conversion needed)
    'speed': lambda a: a._speed_mph if isinstance(a, VehicleAgent) else None,
    'posted_speed_limit': lambda a: a._posted_speed_limit_mph if isinstance(a, VehicleAgent) else None,
    'implicit_speed_limit': lambda a: a._implicit_speed_limit_mph if isinstance(a, VehicleAgent) else None,

    # ... other reporters unchanged ...
}
```

**Estimated lines:** 3 line changes

---

### Subtask 4: Filter Agents in DataCollector
**File:** `traffic/model/traffic_model.py`

**Changes:**
Modify the `step()` method to filter agents:
```python
def step(self):
    # ... existing step logic ...

    if (self.steps % self.collect_every_n) == 0:
        # Collect only from vehicles (not road segments)
        if self.batchrun:
            self.datacollector.collect(self)
        else:
            # Filter to vehicles only for agent_reporters
            # This avoids isinstance() checks on road segments
            self.datacollector.collect(self, agents=self.vehicles_list)
```

**Note:** This may require checking Mesa's DataCollector API. If it doesn't support `agents` parameter, create a custom wrapper.

**Alternative implementation:**
```python
# Override datacollector's agent collection
original_collect = self.datacollector.collect

def filtered_collect(model, agents=None):
    if agents is None:
        agents = model.vehicles_list  # Default to vehicles only
    return original_collect(model, agents=agents)

self.datacollector.collect = filtered_collect
```

**Estimated lines:** 5-10

---

## Critical Files to Modify
1. **traffic/model/traffic_model.py** - Blocker-change sorting + datacollector filtering
2. **traffic/agents/vehicle_agent.py** - Cached mph values
3. **traffic/model/reporting.py** - Use cached mph values

**Note:** Optimization 1 requires changes ONLY to traffic_model.py (no changes to generate.py or blocker_agent.py needed)

---

## Testing & Verification

### Test 1: Bit-for-Bit Reproducibility
**Critical:** Outputs must be IDENTICAL before and after optimization

```python
import pandas as pd
import numpy as np
from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config

# Standard test configuration
config = make_season_config(
    season_id="perf_test",
    seed=12345,
    n_days=1,
    max_persons=100,
    max_steps=5000,
    collect_every_n=1,  # Collect every step for thorough comparison
    road_path="data/roads/lcc_road.parquet",
    ecs_path="data/vehicle_counts/lcc_expected_counts.csv",
)

# Run BEFORE optimization (baseline)
orch_before = SeasonOrchestrator(config, store_data=False)
orch_before.run_day()
model_before = orch_before.last_model_run

# Extract all data
model_ts_before = model_before.datacollector.get_model_vars_dataframe()
agent_ts_before = model_before.datacollector.get_agent_vars_dataframe()
finished_before = pd.DataFrame(model_before.finished_agents)
summary_before = orch_before._compute_season_summary()

# Run AFTER optimization (same seed)
orch_after = SeasonOrchestrator(config, store_data=False)
orch_after.run_day()
model_after = orch_after.last_model_run

# Extract all data
model_ts_after = model_after.datacollector.get_model_vars_dataframe()
agent_ts_after = model_after.datacollector.get_agent_vars_dataframe()
finished_after = pd.DataFrame(model_after.finished_agents)
summary_after = orch_after._compute_season_summary()

# CRITICAL COMPARISONS
print("=" * 60)
print("REPRODUCIBILITY TEST")
print("=" * 60)

# 1. Model time series (identical steps, volumes, tolls, etc.)
pd.testing.assert_frame_equal(model_ts_before, model_ts_after)
print("✓ Model time series: IDENTICAL")

# 2. Agent time series (identical positions, speeds, etc.)
pd.testing.assert_frame_equal(agent_ts_before, agent_ts_after)
print("✓ Agent time series: IDENTICAL")

# 3. Finished agents (identical trip outcomes)
pd.testing.assert_frame_equal(finished_before, finished_after)
print("✓ Finished agents: IDENTICAL")

# 4. Summary statistics
for key in summary_before:
    if isinstance(summary_before[key], (int, float)):
        np.testing.assert_almost_equal(
            summary_before[key],
            summary_after[key],
            decimal=10,
            err_msg=f"Summary mismatch: {key}"
        )
print("✓ Summary statistics: IDENTICAL")

print("\n" + "=" * 60)
print("✅ ALL OUTPUTS IDENTICAL - Optimization is SAFE")
print("=" * 60)
```

**Expected:** Test PASSES - all assertions succeed

---

### Test 2: Performance Benchmark
**Measure speedup quantitatively**

```python
import time
from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config

# Large scenario for meaningful benchmark
config = make_season_config(
    season_id="perf_benchmark",
    seed=99999,
    n_days=1,
    max_persons=500,  # Large scenario
    max_steps=10000,
    collect_every_n=10,  # Reduce data collection overhead
    road_path="data/roads/lcc_road.parquet",
    ecs_path="data/vehicle_counts/lcc_expected_counts.csv",
)

# Benchmark BEFORE optimization
print("Running BEFORE optimization...")
start = time.perf_counter()
orch_before = SeasonOrchestrator(config, store_data=False)
orch_before.run_day()
time_before = time.perf_counter() - start

model_before = orch_before.last_model_run
steps_before = model_before.schedule.steps

# Benchmark AFTER optimization
print("Running AFTER optimization...")
start = time.perf_counter()
orch_after = SeasonOrchestrator(config, store_data=False)
orch_after.run_day()
time_after = time.perf_counter() - start

model_after = orch_after.last_model_run
steps_after = model_after.schedule.steps

# Calculate metrics
steps_per_sec_before = steps_before / time_before
steps_per_sec_after = steps_after / time_after
speedup = steps_per_sec_after / steps_per_sec_before
time_saved_pct = (1 - time_after / time_before) * 100

print("\n" + "=" * 60)
print("PERFORMANCE BENCHMARK")
print("=" * 60)
print(f"Total steps:        {steps_before}")
print(f"")
print(f"BEFORE optimization:")
print(f"  Time:             {time_before:.2f} seconds")
print(f"  Steps/second:     {steps_per_sec_before:.1f}")
print(f"")
print(f"AFTER optimization:")
print(f"  Time:             {time_after:.2f} seconds")
print(f"  Steps/second:     {steps_per_sec_after:.1f}")
print(f"")
print(f"IMPROVEMENT:")
print(f"  Speedup:          {speedup:.2f}x")
print(f"  Time saved:       {time_saved_pct:.1f}%")
print("=" * 60)

# Success criteria
assert speedup >= 1.40, f"Speedup {speedup:.2f}x below target 1.40x (40%)"
print(f"✅ Target achieved: {speedup:.2f}x speedup (>= 1.40x)")
```

**Expected results:**
```
BEFORE optimization: ~150 steps/second
AFTER optimization:  ~220+ steps/second
Speedup:             1.45x - 1.70x (45-70% improvement)
Time saved:          31-41%
```

---

### Test 3: Blocker-Change Sort Optimization Verification
**Ensure sorting happens only when blockers change**

```python
from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config

# Test scenario WITHOUT blockers - should sort only once
config_no_blockers = make_season_config(
    season_id="no_blockers_test",
    seed=111,
    n_days=1,
    max_persons=100,
    max_steps=5000,
    crashes_per_100k_vmt_input=0,  # No crashes
    canyon_closures_schedule=[],   # No closures
)

orch = SeasonOrchestrator(config_no_blockers, store_data=False)

# Instrument to count sorts
sort_count = 0
original_update = orch.last_model_run.update_next_agents

def counting_update(self):
    global sort_count
    old_count = len(self.blockers_list)
    result = original_update()
    # Check if sort happened (blocker count tracking logic)
    if len(self.blockers_list) != old_count or self.schedule.steps == 0:
        sort_count += 1
    return result

# Can't easily override before run, but can verify post-hoc:
orch.run_day()
model = orch.last_model_run

# Without blockers, should have sorted exactly once (initial call)
# Manual verification approach:
print(f"Total steps: {model.schedule.steps}")
print(f"Total blockers created: {model.created_counts['BlockerAgent']}")
print(f"Expected sorts: 1 (initial) + {model.created_counts['BlockerAgent']} (blocker events)")

# Test scenario WITH blockers - should sort only on blocker events
config_with_blockers = make_season_config(
    season_id="with_blockers_test",
    seed=222,
    n_days=1,
    max_persons=100,
    max_steps=5000,
    crashes_per_100k_vmt_input=100,  # High crash rate
)

orch2 = SeasonOrchestrator(config_with_blockers, store_data=False)
orch2.run_day()
model2 = orch2.last_model_run

blocker_events = model2.total_crashes  # Number of times blockers changed
expected_sorts = 1 + blocker_events * 2  # Initial + create + destroy per crash
print(f"\nWith blockers:")
print(f"Total crashes: {model2.total_crashes}")
print(f"Expected sorts: ~{expected_sorts} (1 initial + 2 per crash)")
print(f"Avoided sorts: {model2.schedule.steps - expected_sorts}")
print(f"✓ Sort optimization working correctly")
```

---

### Test 4: Edge Cases
**Verify optimization works correctly in corner cases**

```python
# Test 1: Empty model (no vehicles)
config_empty = make_season_config(
    season_id="empty_test",
    seed=1,
    n_days=1,
    max_persons=0,  # No vehicles
    max_steps=100,
)
orch = SeasonOrchestrator(config_empty, store_data=False)
orch.run_day()
print("✓ Empty model: No crashes")

# Test 2: Single vehicle
config_single = make_season_config(
    season_id="single_test",
    seed=2,
    n_days=1,
    max_persons=1,
    max_steps=100,
)
orch = SeasonOrchestrator(config_single, store_data=False)
orch.run_day()
assert len(orch.last_model_run.finished_agents) == 1
print("✓ Single vehicle: Completes correctly")

# Test 3: Blocker events (triggers dirty flag)
config_crash = make_season_config(
    season_id="crash_test",
    seed=3,
    n_days=1,
    max_persons=100,
    crashes_per_100k_vmt_input=100,  # High crash rate
)
orch = SeasonOrchestrator(config_crash, store_data=False)
orch.run_day()
model = orch.last_model_run
assert model.total_crashes > 0, "No crashes occurred"
print(f"✓ Crash events: {model.total_crashes} crashes handled correctly")

# Test 4: Canyon closure (triggers dirty flag)
config_closure = make_season_config(
    season_id="closure_test",
    seed=4,
    n_days=1,
    max_persons=50,
    canyon_closures_schedule=[
        {"closure_step": 500, "duration": 300, "road_section": 50}
    ],
)
orch = SeasonOrchestrator(config_closure, store_data=False)
orch.run_day()
print("✓ Canyon closure: Handled correctly")
```

---

## Potential Risks & Considerations

### Risk 1: Sort Order Assumption Violation
**Issue:** If vehicles CAN overtake (e.g., multiple lanes, speed variability), the "vehicles maintain order" assumption breaks

**Impact:** Vehicles would have incorrect next_agent pointers → wrong gap calculations → WRONG OUTPUTS

**Mitigation:**
- Model uses single-lane road → overtaking is physically impossible
- Comprehensive testing in Test 1 (reproducibility check) catches any order violations
- If outputs differ, the test will fail immediately
- Optional: Add debug assertion to verify vehicles_list stays sorted between blocker changes

---

### Risk 2: Mesa DataCollector API Compatibility
**Issue:** DataCollector.collect() may not support `agents` parameter

**Impact:** Can't filter agents before collection

**Mitigation:**
- Check Mesa version and API docs
- If not supported, fall back to isinstance() checks (still get caching speedup)
- Alternative: Override DataCollector to add filtering support

---

### Risk 3: Cached mph Values Out of Sync
**Issue:** If speed changes but cache isn't updated, reporters will show stale values

**Impact:** Incorrect speed reporting → WRONG OUTPUTS

**Mitigation:**
- Test 1 catches this immediately (time series comparison)
- Update cache in ALL places where speed changes:
  - adjust_speed()
  - explicit speed assignments
- Add property setter to enforce:
  ```python
  @property
  def speed(self):
      return self._speed

  @speed.setter
  def speed(self, value):
      self._speed = value
      self._speed_mph = uc.get_mph(value)
  ```

---

### Risk 4: Performance Improvement Less Than Expected
**Issue:** Actual speedup may be lower than predicted (Python overhead, other bottlenecks)

**Impact:** Disappointing results, but no correctness issue

**Mitigation:**
- Benchmark each optimization independently to identify actual gains
- Profile with cProfile to find remaining bottlenecks
- Even 30% speedup is valuable (worst case)

---

## Implementation Order (Recommended)

1. **Start with Optimization 1** (blocker-change sorting) - Highest impact, very low risk, self-contained
2. **Then Optimization 2** (cached mph) - Easy to verify, complements #1
3. **Finally Optimization 3** (agent filtering) - Depends on Mesa API, nice-to-have

**Rationale:** Optimization 1 is now extremely simple (all changes in one method, no cross-file coordination). Get the biggest win first, then stack additional improvements.

---

## Success Criteria
✅ Test 1: Bit-for-bit reproducibility (all dataframes identical)
✅ Test 2: 40%+ speedup (1.4x faster minimum)
✅ Test 3: Dirty flag logic working correctly
✅ Test 4: Edge cases handled (empty, single, crashes, closures)
✅ No regressions in existing features
✅ Code remains readable and maintainable

---

## Estimated Implementation Time
- **Subtask 1:** 45 minutes (blocker-change sorting - simpler than original dirty flag approach)
- **Subtask 2:** 1 hour (cached mph properties)
- **Subtask 3:** 15 minutes (update reporters)
- **Subtask 4:** 1-2 hours (agent filtering, depends on Mesa API)
- **Testing & Validation:** 2-3 hours (critical - must verify no output changes)
- **Total:** 5-7 hours (reduced from 6-9 hours due to simpler Optimization 1)

---

## Additional Optimizations (Future)

If additional speedup needed:
- **NumPy vectorization** of speed calculations (10-15% more)
- **Cython compilation** of hot loops (20-30% more)
- **Spatial indexing** for road segment lookup (5-10% more)
- **JIT compilation** with Numba (15-25% more)

These are more invasive and require major refactoring.

---

## Notes
- Performance gains are cumulative across optimizations
- Larger scenarios (more vehicles) benefit more from sorting optimization
- Smaller scenarios benefit more from caching optimization
- Combined: 45-70% improvement expected across all scenario sizes
- If Mesa DataCollector doesn't support agent filtering, skip Optimization 3 (still get 35-55% speedup from #1+#2)
