# Hybrid Data Collection System for Mesa Traffic Simulation

## Overview

Implement a 4-tier hybrid data collection system that minimizes runtime overhead while providing flexible data access for analysis and animations.

**Current Issues:**
- Mesa DataCollector collects ALL agent data at every interval (15 reporters per agent)
- No tiered approach - full data or nothing
- Event logging uses print statements (unstructured)
- Animation data collected at full resolution then downsampled post-hoc

**Goals:**
- Pre-allocated numpy arrays for performance-critical data
- Configurable sampling intervals per tier
- Structured event logging (crashes, canyon closures)
- Preserve exact data formats for existing animations
- Extensible "bookshelf" pattern: metrics defined in code, selection configurable per-run

---

## Tier Specifications

### Tier 1: Aggregate Metrics (Every Step)

**Purpose:** System-level time series for analysis and tolling feedback.

#### Core Metrics (Pre-allocated numpy arrays)

| Metric | Dtype | Description |
|--------|-------|-------------|
| `step` | int32 | Step number |
| `current_toll` | float32 | Toll for this step |
| `vehicle_count` | int16 | Total active vehicles |
| `active_cars` | int16 | Active car agents |
| `active_buses` | int16 | Active bus agents |
| `bus_riders_waiting` | int16 | People at bus stop |
| `bus_mode_share_recent` | float32 | % choosing bus in recent window |
| `total_finished` | int32 | Cumulative completed trips |
| `recent_travel_time_avg` | float32 | Mean travel time of trips completed in last 5 min |

#### Distribution Metrics: Histogram Bins (O(n) - No Sorting)

Instead of computing percentiles (which require O(n log n) sorting), we use **fixed histogram bins** for O(n) collection:

**Implicit Speed Delta Histogram** (`implicit_sl_delta_bins`):
- Bins: `[-inf, -30, -20, -10, -5, 0, +5, +inf]` mph
- Stores: int16 count per bin (7 values)
- Interpretation: bin[0] = severely congested (<-30), bin[6] = speeding (>+5)

**Speed Histogram** (`speed_mps_bins`):
- Bins: `[0, 5, 10, 15, 20, 25, 30, inf]` m/s (~0-67 mph)
- Stores: int16 count per bin (7 values)
- Interpretation: full speed distribution captured

**Why histogram bins are fastest:**
- Mean/std: O(n) single pass
- Percentiles: O(n log n) requires sorting
- Histogram bins: O(n) single pass, just increment bin counters

**Memory:** ~1MB for 50k steps (13 scalars + 14 histogram bins = 27 values × 4 bytes × 50k)

---

### Tier 2: Sampled Spatial Data (Every N Steps, Default N=10)

**Purpose:** Animation-compatible agent positions/states at configurable intervals.

**Pre-allocated numpy arrays** with int8 encoding for categorical fields:

| Field | Dtype | Description |
|-------|-------|-------------|
| `step` | int32 | Step number |
| `agent_id` | int32 | Unique agent ID |
| `pos_x` | float32 | X coordinate |
| `pos_y` | float32 | Y coordinate |
| `status` | int8 | Encoded: 0=driving, 1=slowing, 2=crash, 3=canyon_closure |
| `distance_traveled` | float32 | Meters along road |
| `gap_m` | float32 | Distance to next vehicle |
| `ideal_gap_m` | float32 | Desired following distance |
| `driving_action` | int8 | Encoded: 0=coast, 1=accelerate, 2=slow_accelerate, 3=smooth_break, 4=speed_limit_break, 5=prevent_pass |
| `speed_mps` | float32 | Current speed |

**Configuration:** `tier2_sample_interval` parameter (default=10) controls collection frequency.

---

### Tier 3: Event Logging (On Events Only)

**Purpose:** Structured logging of discrete events for summary exports.

**Events tracked:**
- **CRASH**: step, segment_index, distance_m, pos_x, pos_y, duration_sec, vehicles_on_road
- **CANYON_CLOSURE**: step, segment_index, duration_sec

**NOT tracked as events:** Toll changes (already in Tier 1 per-step data)

**Data structure:** List of TrafficEvent dataclasses, converted to DataFrame for export.

---

### Tier 4: Full Snapshots (Optional, Disabled by Default)

**Purpose:** Complete agent state at key moments for debugging.

**Triggers:** Configurable (every N steps, on crash, manual).

---

## Extensibility: Bookshelf Pattern

Metrics are **defined in code** (stable, tested) but **selected per-run via config**.

### Tier 1 Metric Registry (in code)

```python
# traffic/model/hybrid_collector.py

# === Scalar Metrics ===
TIER1_SCALARS = {
    # Core (always collected)
    'step': {'dtype': np.int32, 'fn': lambda m: m.steps},
    'current_toll': {'dtype': np.float32, 'fn': lambda m: m.current_toll_car},
    'vehicle_count': {'dtype': np.int16, 'fn': lambda m: len(m.vehicles_list)},
    'active_cars': {'dtype': np.int16, 'fn': lambda m: sum(1 for v in m.vehicles_list if v.__class__.__name__ == 'CarAgent')},
    'active_buses': {'dtype': np.int16, 'fn': lambda m: sum(1 for v in m.vehicles_list if v.__class__.__name__ == 'BusAgent')},
    'bus_riders_waiting': {'dtype': np.int16, 'fn': lambda m: len(m.at_bus_stop)},
    'bus_mode_share_recent': {'dtype': np.float32, 'fn': compute_bus_mode_share},
    'total_finished': {'dtype': np.int32, 'fn': lambda m: len(m.finished_agents)},
    'recent_travel_time_avg': {'dtype': np.float32, 'fn': compute_recent_travel_time, 'window': 300},
}

# === Histogram Metrics (fixed bins, O(n) collection) ===
TIER1_HISTOGRAMS = {
    'implicit_sl_delta': {
        'bins': [-np.inf, -30, -20, -10, -5, 0, 5, np.inf],
        'dtype': np.int16,
        'fn': lambda v: get_mph(v.speed) - v.implicit_speed_limit,  # per vehicle
    },
    'speed_mps': {
        'bins': [0, 5, 10, 15, 20, 25, 30, np.inf],
        'dtype': np.int16,
        'fn': lambda v: v.speed,  # per vehicle
    },
}
```

### Per-Run Configuration

```python
config = HybridCollectorConfig(
    # Select which scalars to collect (subset of TIER1_SCALARS keys)
    tier1_scalars=['step', 'current_toll', 'vehicle_count', 'bus_mode_share_recent'],

    # Select which histograms to collect (subset of TIER1_HISTOGRAMS keys)
    tier1_histograms=['implicit_sl_delta'],  # or [] to disable

    tier2_sample_interval=10,
    tier2_enabled=True,
    tier3_enabled=True,
)
```

### Adding New Metrics

**To add a new scalar metric:**
1. Add entry to `TIER1_SCALARS` dict with `dtype` and `fn`
2. `fn` takes model, returns single value
3. Select in config when running

**To add a new histogram metric:**
1. Add entry to `TIER1_HISTOGRAMS` dict with `bins`, `dtype`, and `fn`
2. `fn` takes vehicle agent, returns value to bin
3. Select in config when running

**No class modifications needed** - just add to registry and select in config.

---

## Implementation Plan

### Branch Strategy

Create a new branch `hybrid-collector` that **completely replaces** the old DataCollector system:
- No side-by-side operation
- Old DataCollector code removed entirely
- Speed test by comparing `main` branch vs `hybrid-collector` branch

```bash
git checkout -b hybrid-collector
# Implement all changes
# Test against main branch for speed comparison
```

---

### Phase 1: Create HybridDataCollector Module

**File:** `traffic/model/hybrid_collector.py` (NEW, ~400 lines)

```
HybridCollectorConfig (dataclass)
├── max_steps: int = 50000
├── tier1_scalars: List[str] = [...]
├── tier1_histograms: List[str] = [...]
├── tier2_sample_interval: int = 10
├── tier2_enabled: bool = True
├── tier3_enabled: bool = True
├── tier4_enabled: bool = False

HybridDataCollector
├── tier1: Tier1Collector
├── tier2: Tier2Collector
├── tier3: Tier3Collector
├── tier4: Tier4Collector (optional)
├── collect(model) → calls tier1 + tier2 (checks interval internally)
├── log_crash(model, segment_index, duration)
├── log_canyon_closure(model, segment_index, duration)
├── get_tier1_dataframe() → pd.DataFrame
├── get_tier2_dataframe() → pd.DataFrame (animation-compatible)
├── get_events_dataframe() → pd.DataFrame
└── get_model_vars_dataframe() → compatibility wrapper
└── get_agent_vars_dataframe() → compatibility wrapper
```

---

### Phase 2: Replace DataCollector in TrafficModel

**File:** `traffic/model/traffic_model.py`

**Changes:**
1. **Remove** old DataCollector import and replace with HybridDataCollector:
   ```python
   # REMOVE: from mesa.datacollection import DataCollector
   from traffic.model.hybrid_collector import HybridDataCollector, HybridCollectorConfig
   ```

2. **Replace** DataCollector initialization (lines 129-133):
   ```python
   # REMOVE old DataCollector code
   # REPLACE with:
   hybrid_config = HybridCollectorConfig(
       max_steps=max_steps,
       tier2_sample_interval=collect_every_n,
   )
   self.datacollector = HybridDataCollector(hybrid_config)  # Keep same attribute name for compatibility
   ```

3. **Replace** collection in `step()` (lines 286-287):
   ```python
   # REMOVE old: if (self.steps % self.collect_every_n) == 0:
   #                 self.datacollector.collect(self)
   # REPLACE with:
   self.datacollector.collect(self)  # Interval checking is internal
   ```

---

### Phase 3: Update Event Logging

**File:** `traffic/model/generate.py`

**Line 124** - Replace print with structured logging:
```python
def generate_blocker(model, blocker_type, self_distruct_timer, seg_i):
    # Structured event logging (replaces print statement)
    if blocker_type == "crash":
        model.datacollector.log_crash(model, seg_i, self_distruct_timer)
    elif blocker_type == "canyon_closure":
        model.datacollector.log_canyon_closure(model, seg_i, self_distruct_timer)

    # Rest unchanged...
```

---

### Phase 4: Update SeasonOrchestrator

**File:** `season/season_orchestrator.py`

**Replace** `_save_datacollector_outputs()` (around line 155) to use new collector:
```python
def _save_datacollector_outputs(self, tm, day_index):
    # Tier 1: Aggregate metrics
    tier1_df = tm.datacollector.get_tier1_dataframe()
    tier1_df.to_parquet(self.output_dir / f"day_{day_index}_model_ts.parquet")

    # Tier 2: Spatial data for animations
    spatial_df = tm.datacollector.get_tier2_dataframe()
    spatial_df.to_parquet(self.output_dir / f"day_{day_index}_spatial.parquet")

    # Tier 3: Events
    events_df = tm.datacollector.get_events_dataframe()
    events_df.to_parquet(self.output_dir / f"day_{day_index}_events.parquet")
```

---

### Phase 5: Update analysis_utils

**File:** `traffic/utils/analysis_utils.py`

**Update** functions to use new collector interface:

```python
def vehicle_agent_data_time_series(model, plots=True):
    # Use new Tier 2 data
    vehicles_full = model.datacollector.get_tier2_dataframe()
    # ... rest of function
```

```python
def model_data_time_series(model):
    # Use new Tier 1 data
    model_df = model.datacollector.get_tier1_dataframe()
    return model_df
```

---

### Phase 6: Remove Old Reporting Code

**File:** `traffic/model/reporting.py`

**DELETE** this file entirely - no longer needed. The metric definitions now live in `hybrid_collector.py`.

---

## Files to Modify

| File | Action | Description |
|------|--------|-------------|
| `traffic/model/hybrid_collector.py` | CREATE | New file (~400 lines) |
| `traffic/model/traffic_model.py` | MODIFY | Replace DataCollector with HybridDataCollector |
| `traffic/model/generate.py` | MODIFY | Line 124 - structured event logging |
| `traffic/model/reporting.py` | DELETE | No longer needed |
| `season/season_orchestrator.py` | MODIFY | Update `_save_datacollector_outputs()` |
| `traffic/utils/analysis_utils.py` | MODIFY | Update to use new collector interface |

---

## Animation Compatibility

**Critical:** Tier 2 `to_dataframe()` must produce exact format:

```python
# Required columns for animate_traffic():
['Step', 'AgentID', 'pos', 'status']
# pos must be tuple (x, y)
# status must be string: 'driving', 'crash', 'slowing', 'canyon_closure'

# Required columns for animate_relative_distance():
['Step', 'AgentID', 'distance_traveled', 'gap_m', 'ideal_gap_m', 'status', 'driving_action']
# driving_action must be string: 'coast', 'accelerate', 'smooth_break', etc.
```

---

## Expected Performance Improvements

| Metric | Current | Hybrid | Improvement |
|--------|---------|--------|-------------|
| Memory (25M records) | ~2GB | ~50MB | 40x |
| Collection overhead | 15 reporters/agent | 1 array write | ~10x |
| Animation compile | Full data + downsample | Pre-sampled | 10x |
| Event logging | Print (unstructured) | Structured DataFrame | Queryable |

---

## Verification Plan

### 1. Speed Comparison: Branch vs Branch

Compare `main` branch (old DataCollector) vs `hybrid-collector` branch (new system):

```bash
# Step 1: Run benchmark on main branch
git checkout main
python tests/benchmark_collectors.py > benchmark_main.txt

# Step 2: Run benchmark on new branch
git checkout hybrid-collector
python tests/benchmark_collectors.py > benchmark_hybrid.txt

# Step 3: Compare results
diff benchmark_main.txt benchmark_hybrid.txt
```

**Benchmark script** (`tests/benchmark_collectors.py`):
```python
import time
import tracemalloc
import geopandas as gpd
import pandas as pd
from traffic.model.traffic_model import TrafficModel

# Load test data
road_gdf = gpd.read_parquet('data/roads/hw210_sl_and_curvs.parquet')
ecs_df = pd.read_csv('data/expected_counts_seconds.csv')

def benchmark_simulation(n_steps: int = 5000, n_runs: int = 3):
    """Benchmark simulation runs."""
    results = []
    for i in range(n_runs):
        tracemalloc.start()
        start_time = time.perf_counter()

        model = TrafficModel(road_gdf, ecs_df, max_steps=n_steps, batchrun=False)
        model.run_model()

        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'run': i,
            'elapsed_sec': elapsed,
            'peak_memory_mb': peak / 1024 / 1024,
            'steps_per_sec': n_steps / elapsed,
        })

    # Average across runs
    avg = {
        'elapsed_sec': sum(r['elapsed_sec'] for r in results) / n_runs,
        'peak_memory_mb': sum(r['peak_memory_mb'] for r in results) / n_runs,
        'steps_per_sec': sum(r['steps_per_sec'] for r in results) / n_runs,
    }

    print(f"Steps/sec: {avg['steps_per_sec']:.1f}")
    print(f"Peak memory: {avg['peak_memory_mb']:.1f} MB")
    print(f"Elapsed: {avg['elapsed_sec']:.1f} sec")

if __name__ == "__main__":
    benchmark_simulation()
```

**Expected Results:**
- 5-10x faster collection (fewer lambda calls, direct array writes)
- 20-40x less memory (pre-allocated vs DataFrame growth)

---

### 2. Unit Tests

```python
# tests/test_hybrid_collector.py

def test_tier1_preallocates_arrays():
    """Verify arrays are pre-allocated to max_steps."""

def test_tier1_collects_all_metrics():
    """Verify all configured metrics are collected each step."""

def test_tier1_histogram_bins_correct():
    """Verify histogram bin counts are accurate."""

def test_tier2_samples_at_correct_interval():
    """Verify tier2 only collects every N steps."""

def test_tier2_encodes_status_correctly():
    """Verify status strings map to correct int8 values."""

def test_tier2_dataframe_has_animation_columns():
    """Verify to_dataframe() produces required columns for animate_traffic()."""

def test_tier3_logs_crash_events():
    """Verify crash events are logged with correct fields."""

def test_tier3_logs_canyon_closure_events():
    """Verify canyon closure events are logged correctly."""

def test_extensibility_add_new_scalar():
    """Verify new scalar metrics can be added via registry."""

def test_extensibility_add_new_histogram():
    """Verify new histogram metrics can be added via registry."""
```

---

### 3. Notebook Validation

**Run `notebooks/season_run.ipynb`** with the new hybrid collector:

1. Execute full notebook end-to-end
2. Verify all cells complete without errors
3. Verify animations render correctly:
   - `animate_traffic()` works with Tier 2 data
   - `animate_relative_distance()` works with Tier 2 data
4. Verify all analysis plots render correctly
5. Check output parquet files are created with correct structure

---

## Next Steps (Not Covered by This Plan)

### 1. Cumulative End-of-Run Metrics

This plan does not include recording **cumulative metrics at the end of each model run**. These are values that only need to be recorded once when the simulation completes:

- `total_cars` - Total cars generated during the run
- `total_buses` - Total buses generated during the run
- `total_people` - Total people (TrafficPersonAgents) processed
- `total_crashes` - Total crash events
- Other summary statistics

**Future implementation:** Add a `get_run_summary()` method to HybridDataCollector that returns a dict of end-of-run metrics, called once after `run_model()` completes.

### 2. Population/Season-Level Data Collection

This plan focuses on **traffic model data collection** (within a single day). It does not address recording **population changes across days** at the season level:

- SeasonPerson belief evolution over days
- Mode choice distributions over time
- Learning/adaptation metrics
- Population-level summary statistics per day

**Future implementation:** This would require a separate `SeasonDataCollector` that operates at the SeasonOrchestrator level, tracking how the SeasonPerson population evolves across multiple daily simulations. This is architecturally distinct from the traffic-level HybridDataCollector.
