# Data Collection

The `HybridDataCollector` is a 4-tier data collection system that replaces Mesa's built-in `DataCollector` for better performance and flexibility. Defined in `traffic/model/hybrid_collector.py`.

---

## Why HybridDataCollector

Mesa's built-in `DataCollector` creates Python objects each step, which becomes expensive in simulations that can run 50,000+ steps with hundreds of active agents. The `HybridDataCollector` uses **pre-allocated numpy arrays** for Tier 1 and structured append lists for other tiers, minimizing per-step allocation overhead.

---

## Tier 1: Aggregate Metrics

**Purpose:** Per-step scalar values and histograms for time-series analysis.

**Cadence:** Every `tier1_interval` steps (default: 1 = every step).

**Scalar metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `step` | int32 | Current simulation step |
| `current_toll` | float32 | Active car toll ($) |
| `vehicle_count` | int16 | Total vehicles on road |
| `active_cars` | int16 | Cars currently on road |
| `active_buses` | int16 | Buses currently on road |
| `bus_riders_waiting` | int16 | Persons in bus stop queue |
| `bus_mode_share_recent` | float32 | % of recent persons who chose bus |
| `total_finished` | int32 | Cumulative completed trips |
| `recent_travel_time_avg` | float32 | Mean travel time of recently completed trips |

**Histogram metrics:**

| Metric | Bins | Description |
|--------|------|-------------|
| `implicit_sl_delta` | [-inf, -30, -20, -10, 0, +inf] | Speed vs. implicit speed limit (mph) |
| `speed_mps` | [0, 10, 20, 30, 40, +inf] | Vehicle speed distribution (m/s) |

**Implementation:** Pre-allocated numpy arrays sized to `max_steps`. A write index advances each collection. Converted to DataFrame via `to_dataframe()`.

---

## Tier 2: Sampled Spatial Data

**Purpose:** Agent position and state snapshots for generating animations.

**Cadence:** Every `tier2_sample_interval` steps (default: 10).

**Per-agent data captured:**

| Field | Description |
|-------|-------------|
| `step` | Simulation step |
| `agent_id` | Unique agent ID |
| `agent_type` | `CarAgent` or `BusAgent` |
| `pos` | (x, y) position |
| `status` | Encoded: driving=0, slowing=1, crash=2, canyon_closure=3, arrived=4 |
| `distance_traveled` | Meters from start |
| `gap_m` | Gap to next entity (meters) |
| `ideal_gap_m` | Desired following distance (meters) |
| `driving_action` | Encoded: accelerate=0, coast=1, slow_accel=2, sl_brake=3, smooth_brake=4, prevent_pass=5 |
| `speed` | Current speed (m/s) |

**Limits:** Up to `tier2_max_agents_per_sample` agents per sample, up to `tier2_max_samples` total samples.

**Disabled in batch mode** by default (`tier2_enabled = not batchrun`) since animations aren't needed for parameter sweeps.

---

## Tier 3: Event Log

**Purpose:** Discrete event records for crashes and canyon closures.

**Trigger:** Logged when `generate_blocker()` is called (not on a cadence).

**Fields:**

| Field | Description |
|-------|-------------|
| `event_type` | `"crash"` or `"canyon_closure"` |
| `step` | When the event occurred |
| `segment_index` | Road segment where the blocker was placed |
| `distance_m` | Distance along road (meters) |
| `pos_x`, `pos_y` | Spatial coordinates |
| `duration_sec` | How long the blocker lasts |
| `vehicles_on_road` | Vehicle count at time of event |

Always lightweight -- only records when events actually happen.

---

## Tier 4: Full Snapshots

**Purpose:** Complete model state dumps for debugging and deep analysis.

**Trigger:** At `tier4_snapshot_interval` step intervals and/or on crash events (`tier4_snapshot_on_crash`).

**Disabled by default** (`tier4_enabled=False`) to prevent memory bloat.

**Limits:** `tier4_max_snapshots` caps the total number of snapshots stored.

---

## Configuration

All tiers are controlled via `HybridCollectorConfig`:

```python
HybridCollectorConfig(
    max_steps=100000,

    # Tier 1
    tier1_enabled=True,
    tier1_interval=1,
    tier1_scalars=['step', 'current_toll', 'vehicle_count', ...],
    tier1_histograms=['speed_mps'],

    # Tier 2
    tier2_enabled=True,
    tier2_sample_interval=30,
    tier2_max_samples=3000,
    tier2_max_agents_per_sample=150,

    # Tier 3
    tier3_enabled=True,

    # Tier 4
    tier4_enabled=False,
    tier4_snapshot_on_crash=True,
    tier4_max_snapshots=20,

    # Metric parameters
    recent_travel_time_window=180,
)
```

---

## Output Files

When `SeasonOrchestrator` runs with `store_data=True`, each day's collector outputs are saved:

| File | Tier | Contents |
|------|------|----------|
| `day_N_model_ts.parquet` | 1 | Time-series scalars and histograms |
| `day_N_spatial.parquet` | 2 | Spatial snapshots (if non-empty) |
| `day_N_events.parquet` | 3 | Crash/closure events (if non-empty) |
