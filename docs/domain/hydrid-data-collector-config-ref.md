## HybridCollectorConfig Reference

```python
from traffic.model.hybrid_collector import HybridCollectorConfig
```

---

### Global
| Field | Default | Notes |
|---|---|---|
| `max_steps` | `50000` | Pre-allocation size — match model `max_steps` |

---

### Tier 1 — Aggregate Time Series
Collected every `tier1_interval` steps into pre-allocated numpy arrays. One row per interval.

| Field | Default | Notes |
|---|---|---|
| `tier1_enabled` | `True` | |
| `tier1_interval` | `1` | Set to 10–100 to reduce overhead |
| `tier1_scalars` | see below | List of scalar keys to collect |
| `tier1_window_scalars` | see below | Rolling-window metrics (lookback = `tier1_window_seconds`) |
| `tier1_histograms` | `['implicit_sl_delta', 'speed_mps']` | Binned distributions per step |
| `tier1_window_seconds` | `300` | Lookback window in seconds for window scalars |

**Available scalars:**
```python
# Core
'step', 'current_toll', 'vehicle_count', 'active_cars', 'active_buses',
'persons_at_bus_stop', 'bus_mode_share_recent',
'persons_finished',        # cumulative completed trips
'persons_pool_remaining',  # persons not yet assigned a trip today
'persons_in_transit',      # persons currently active (in car, on bus, or at stop)
'p_generate'

# Window-based (rolling over tier1_window_seconds)
'recent_travel_time_avg',            # mean trip time (min) of recently finished trips
'rolling_count_vehicles_generated',  # vehicle spawns in window
'rolling_count_persons_generated'    # person spawns in window

# Histograms
'implicit_sl_delta'  # speed delta vs implicit speed limit (mph), bins: (-∞,-30,-20,-10,0,∞)
'speed_mps'          # speed distribution (m/s), bins: (0,10,20,30,40,∞)
```

**Access:** `model.datacollector.get_tier1_dataframe()`  
**Saved as:** `day_{N}_model_ts.parquet`

**Use for:**
- Toll / congestion time series
- Mode share trends, p_generate evolution
- Speed & speed-limit-compliance distributions
- Rolling throughput analysis

---

### Tier 2 — Spatial Snapshots (Vehicle-Level)
Samples all active vehicles every `tier2_sample_interval` steps. One row per vehicle per sample.

| Field | Default | Notes |
|---|---|---|
| `tier2_enabled` | `True` | |
| `tier2_sample_interval` | `10` | Steps between snapshots (10 steps = 10 sec) |
| `tier2_max_samples` | `5000` | Max number of collection events (pre-alloc) |
| `tier2_max_agents_per_sample` | `200` | Excess vehicles silently dropped |

**Columns:** `Step`, `AgentID`, `AgentType`, `pos`, `status`, `distance_traveled`, `gap_m`, `ideal_gap_m`, `driving_action`, `speed` (mph), `speed_mps`

**Access:** `model.datacollector.get_tier2_dataframe()`  
**Saved as:** `day_{N}_spatial.parquet`

**Use for:**
- Animations / trajectory visualization
- Fundamental diagram (speed vs density)
- Gap distribution & car-following behavior
- Per-vehicle speed profiles

---

### Tier 3 — Event Log (Crashes & Closures)
Logged on-demand when events occur. One row per event.

| Field | Default | Notes |
|---|---|---|
| `tier3_enabled` | `True` | |

**Columns:** `event_type` (`crash`/`canyon_closure`), `step`, `segment_index`, `distance_m`, `pos_x`, `pos_y`, `duration_sec`, `vehicles_on_road`

**Access:** `model.datacollector.get_events_dataframe()`  
**Saved as:** `day_{N}_events.parquet`

**Use for:**
- Incident impact analysis (flow before/after crash)
- Closure duration tracking
- Spatial crash patterns

---

### Tier 4 — Full Snapshots (Debug Only)
Complete per-agent state captures. Stored in memory as list of dicts.

| Field | Default | Notes |
|---|---|---|
| `tier4_enabled` | `False` | Off by default — memory-heavy |
| `tier4_snapshot_interval` | `0` | Every N steps; 0 = disabled |
| `tier4_snapshot_on_crash` | `False` | Auto-snapshot on each crash event |
| `tier4_max_snapshots` | `100` | Hard cap to prevent OOM |

**Access:** `model.datacollector.get_snapshots()` → `List[Dict]`

**Use for:**
- Step-level debugging / replay
- Diagnosing crashes or stalls at exact timestep
- **Not for production runs**

---

### Preset Configs (from `tests/perf_utils.py`)
```python
from tests.perf_utils import get_collector_config
cfg = get_collector_config("lean")  # returns HybridCollectorConfig or None
```

| Preset | T1 interval | T2 | T3 | T4 | Use when |
|---|---|---|---|---|---|
| `"lean"` | 300 | off | off | off | batch sweeps, minimal data |
| `"small"` | 10 | off | off | off | perf benchmarks |
| `"matched"` | 100 | off | off | off | matches old DataCollector output |
| `"medium"` | 1 | on (Δ=10) | on | off | standard analysis run |
| `"large"` | 1 | on (Δ=5) | on | on (Δ=500) | full detail, high memory |
| `None` | default | default | default | default | notebook/exploration |
