# Plan: BusCostConfig Framework

**Supersedes:** `plans/01_bus_service_cost_reporting.md` (that plan added per-step cost tracking inside the simulation loop; this approach is purely post-hoc with zero simulation overhead).

## Goal

Add a `BusCostConfig` dataclass and `compute_bus_costs()` function that estimate the cost of operating the bus service based on TE-standard formulas, using only data already captured by the simulation. **Zero impact on model compute performance.**

## Data Sources (all already exist)

| What we need | Where it lives | Access path from orchestrator |
|---|---|---|
| Avg one-way bus travel time (min) | Trip log `realized_tt` for `mode == "bus"` | Already computed as `bus_df["realized_tt"].mean()` in `_aggregate_trips()` |
| Bus headway (min) | `DayParams.bus_interval` / `TrafficModel.bus_interval` | `self.last_model_run.bus_interval` |
| Model duration (steps) | `TrafficModel.steps` | `self.last_model_run.steps` |
| Bus trip count | Trip log | Already computed as `len(bus_df)` in `_aggregate_trips()` |

No new attributes, counters, or per-step hooks needed.

---

## New File: `traffic/model/bus_system_cost.py`

Mirrors `traffic/model/tolling.py` — self-contained module with dataclass + pure functions.

### BusCostConfig Dataclass

```python
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Optional

@dataclass
class BusCostConfig:
    """Bus service cost estimation parameters.
    
    All calculation is post-hoc — zero simulation overhead.
    """
    # Fleet sizing
    layover_recovery_pct: float = 0.10      # 10% of running time (TCRP Report 30)
    return_trip_factor: float = 1.0         # return leg as multiplier on one-way time
    spare_ratio: float = 0.20              # 20% spare fleet (FTA guidance)

    # Capital
    bus_purchase_price: float = 550_000.0   # midpoint 40-ft heavy-duty (FTA Capital Cost DB)
    useful_life_years: float = 14.0         # FTA Default ULB for heavy-duty bus

    # Labor
    driver_hourly_rate: float = 35.0        # fully loaded: BLS median $24 + ~45% fringe
    relief_factor: float = 1.17            # breaks, sick, vacation (TCRP scheduling)

    # Operations (non-labor: fuel + maintenance)
    ops_cost_per_hour: float = 40.0         # NTD non-labor portion of cost/revenue-hr

    # Overhead
    overhead_multiplier: float = 1.10       # 10% admin/insurance/facility

    # Time scaling
    service_hours_per_day: float = 12.0     # 6am-6pm standard ski bus window
    service_days_per_year: float = 150.0    # ~5 month ski season

    @staticmethod
    def default() -> "BusCostConfig":
        """Convenience: all research-backed defaults, no args needed."""
        return BusCostConfig()
```

### compute_bus_costs() Function

```python
def compute_bus_costs(
    config: BusCostConfig,
    avg_one_way_tt_min: float,
    headway_min: int,
    model_steps: int,
) -> dict:
    """Compute bus service cost metrics from post-run data.
    
    Returns dict with fleet sizing, hourly rates, and three cost metrics.
    Returns None if headway_min <= 0 (no bus service).
    """
    if headway_min <= 0:
        return None

    # Step 1: Fleet sizing
    cycle_time = avg_one_way_tt_min * (1 + config.return_trip_factor) * (1 + config.layover_recovery_pct)
    active_buses = ceil(cycle_time / headway_min)
    total_fleet = ceil(active_buses * (1 + config.spare_ratio))

    # Step 2: Hourly cost rates
    labor_per_hour = active_buses * config.relief_factor * config.driver_hourly_rate
    ops_per_hour = active_buses * config.ops_cost_per_hour
    capital_per_hour = (config.bus_purchase_price * total_fleet) / (
        config.useful_life_years * config.service_days_per_year * config.service_hours_per_day
    )
    total_per_hour = (labor_per_hour + ops_per_hour + capital_per_hour) * config.overhead_multiplier

    # Step 3: Three output metrics
    model_hours = model_steps / 3600.0
    model_run_cost = total_per_hour * model_hours
    daily_cost = total_per_hour * config.service_hours_per_day
    annual_cost = daily_cost * config.service_days_per_year

    return {
        # fleet
        "bus_cost_cycle_time_min": round(cycle_time, 1),
        "bus_cost_active_buses": active_buses,
        "bus_cost_total_fleet": total_fleet,
        # rates
        "bus_cost_per_hour": round(total_per_hour, 2),
        "bus_cost_labor_per_hour": round(labor_per_hour, 2),
        "bus_cost_ops_per_hour": round(ops_per_hour, 2),
        "bus_cost_capital_per_hour": round(capital_per_hour, 2),
        # three metrics
        "bus_cost_model_run": round(model_run_cost, 2),
        "bus_cost_daily": round(daily_cost, 2),
        "bus_cost_annual": round(annual_cost, 2),
    }
```

---

## Integration Points (4 source files + 1 doc, ~60 lines total)

### 1. `season/configs.py`

```python
# At top: add import
from traffic.model.bus_system_cost import BusCostConfig

# In SeasonConfig dataclass, after toll_config line:
bus_cost_config: Optional[BusCostConfig] = None

# In make_season_config() signature, after toll kwarg:
bus_cost_config: BusCostConfig = None,

# In make_season_config() return, in SeasonConfig constructor:
bus_cost_config=bus_cost_config,

# In summarize_config(), after toll summary block:
if config.bus_cost_config is not None:
    bc = config.bus_cost_config
    lines.append(f"  {b('Bus cost:')} enabled (${bc.bus_purchase_price:,.0f}/bus, "
                 f"${bc.driver_hourly_rate}/hr driver, {bc.service_days_per_year:.0f} days/yr)")
else:
    lines.append(f"  {b('Bus cost:')} not configured")
```

### 2. `season/season_orchestrator.py` — `_compute_day_summary()`

After computing `metrics` via `_aggregate_trips()`, before appending to `self.day_summaries`:

```python
# Bus cost calculation (post-hoc, zero sim overhead)
bc_config = self.config.bus_cost_config
tm = self.last_model_run
if bc_config is not None and tm.bus_interval > 0 and metrics.get("bus_trips", 0) > 0:
    from traffic.model.bus_system_cost import compute_bus_costs
    bus_costs = compute_bus_costs(
        config=bc_config,
        avg_one_way_tt_min=metrics["avg_tt_bus"],
        headway_min=tm.bus_interval,
        model_steps=tm.steps,
    )
    if bus_costs is not None:
        metrics.update(bus_costs)
```

Add bus cost to the existing print statement:
```python
# append to print string if bus cost was computed
f"bus_cost_daily:${summary.get('bus_cost_daily', 0):.0f}"
```

### 3. `season/season_orchestrator.py` — `_compute_season_summary()`

**Understanding time in this model:** Each day doesn't simulate a fixed time window. It simulates a fixed number of persons (calibrated to ~8 hrs but variable). `model.steps` is the actual wall-clock seconds simulated. `total_rev` is toll revenue collected over that variable period. **The bus cost metric must handle time identically for an apples-to-apples comparison.**

**Season-level bus cost metrics — three numbers, two purposes:**

| Metric | Definition | Comparable to | Purpose |
|---|---|---|---|
| `bus_cost_season_total` | `sum(day_summary["bus_cost_model_run"])` across all simulated days | **`total_rev`** | Direct revenue-vs-cost comparison. Both cover the exact same simulated hours. "Did toll revenue cover bus operations for the time we actually simulated?" |
| `bus_cost_avg_daily` | `avg(bus_cost_daily)` across simulated days | Per-day benchmarks | What a full 12-hr service day costs on average (extrapolated) |
| `bus_cost_annual` | `avg(bus_cost_daily) × service_days_per_year` | NTD/DOT annual operating expense reports | Extrapolated annual cost for comparison with published transit agency budgets |

**Why `bus_cost_season_total` uses `bus_cost_model_run` (not `bus_cost_daily`):**
`total_rev` is toll revenue from the actual simulated period — if the model ran 6 hours, it's 6 hours of tolls. Comparing that to 12 hours of bus cost would be meaningless. `bus_cost_model_run` uses `model.steps / 3600` as its time basis, so both `total_rev` and `bus_cost_season_total` cover exactly the same time window for each day. This is the only honest comparison.

`bus_cost_daily` and `bus_cost_annual` serve a different purpose: they extrapolate to standard time windows (12-hr day, 150-day season) for DOT/NTD comparability. These are planning metrics, not direct model-output comparisons.

**Implementation:**
```python
# In _compute_season_summary(), after metrics = _aggregate_trips(df, median_vot):
bc_config = self.config.bus_cost_config
if bc_config is not None and self.day_summaries:
    # model_run costs — same time basis as total_rev
    run_costs = [d["bus_cost_model_run"] for d in self.day_summaries if d.get("bus_cost_model_run") is not None]
    # daily costs — extrapolated to full service day
    daily_costs = [d["bus_cost_daily"] for d in self.day_summaries if d.get("bus_cost_daily") is not None]
    if run_costs:
        metrics["bus_cost_season_total"] = round(sum(run_costs), 2)
    if daily_costs:
        avg_daily = sum(daily_costs) / len(daily_costs)
        metrics["bus_cost_avg_daily"] = round(avg_daily, 2)
        metrics["bus_cost_annual"] = round(avg_daily * bc_config.service_days_per_year, 2)
```

Print addition:
```python
f"Bus cost (sim period): ${summary.get('bus_cost_season_total', 0):,.0f}, "
f"Bus cost (annual est): ${summary.get('bus_cost_annual', 0):,.0f}"
```

### 4. `season/season_orchestrator.py` — `get_day_summary_df()`

Add to `col_order` list after the existing bus metrics block:

```python
# bus cost (day level)
"bus_cost_active_buses", "bus_cost_total_fleet", "bus_cost_cycle_time_min",
"bus_cost_per_hour", "bus_cost_model_run", "bus_cost_daily", "bus_cost_annual",
```

Season summary adds: `"bus_cost_season_total"` (comparable to `total_rev`), `"bus_cost_avg_daily"`, `"bus_cost_annual"`

### 5. `docs/domain/bus-cost-calculation.md` (NEW)

Domain documentation with formula walkthrough and three worked examples (see Documentation section below).

---

---

## New Documentation: `docs/domain/bus-cost-calculation.md`

A domain doc (matching the style of existing `docs/domain/*.md` files) explaining the cost model with three worked examples showing how config and headway changes affect outcomes.

### Contents

1. **Overview** — what the cost model does, that it's purely post-hoc, TE methodology references
2. **Formula walkthrough** — cycle time → fleet sizing → hourly rates → three metrics
3. **Three worked examples:**

**Example 1: Default config, 15-min headway, 35-min one-way travel time**
```
cycle_time = 35 × (1 + 1.0) × (1 + 0.10) = 77.0 min
active_buses = ceil(77.0 / 15) = 6
total_fleet = ceil(6 × 1.20) = 8
labor/hr  = 6 × 1.17 × $35 = $245.70
ops/hr    = 6 × $40 = $240.00
capital/hr = ($550,000 × 8) / (14 × 150 × 12) = $174.60
total/hr  = ($245.70 + $240.00 + $174.60) × 1.10 = $726.33
daily     = $726.33 × 12 = $8,715.96
annual    = $8,715.96 × 150 = $1,307,394
```
Shows the baseline — what it costs to run 15-min service with defaults.

**Example 2: Same config, 30-min headway (halved frequency)**
```
cycle_time = 77.0 min (unchanged — same route)
active_buses = ceil(77.0 / 30) = 3
total_fleet = ceil(3 × 1.20) = 4
labor/hr  = 3 × 1.17 × $35 = $122.85
ops/hr    = 3 × $40 = $120.00
capital/hr = ($550,000 × 4) / (14 × 150 × 12) = $87.30
total/hr  = ($122.85 + $120.00 + $87.30) × 1.10 = $363.17
daily     = $363.17 × 12 = $4,358.00
annual    = $4,358.00 × 150 = $653,697
```
Demonstrates: doubling headway roughly halves cost (3 buses vs 6). This is the fundamental TE principle — "double the frequency, double the cost."

**Example 3: Electric bus fleet (modified config), 15-min headway**
```python
BusCostConfig(
    bus_purchase_price=1_100_000,  # electric bus (2024 median)
    useful_life_years=14,
    ops_cost_per_hour=25.0,       # lower fuel cost (electricity vs diesel)
    # all other defaults unchanged
)
```
```
cycle_time = 77.0 min
active_buses = 6, total_fleet = 8
labor/hr  = $245.70 (unchanged)
ops/hr    = 6 × $25 = $150.00 (lower)
capital/hr = ($1,100,000 × 8) / (14 × 150 × 12) = $349.21 (2× higher)
total/hr  = ($245.70 + $150.00 + $349.21) × 1.10 = $819.40
daily     = $819.40 × 12 = $9,832.80
annual    = $9,832.80 × 150 = $1,474,920
```
Shows: electric buses have lower ops but higher capital cost. Net effect is ~13% more expensive at these parameters. Demonstrates how BusCostConfig parameterization enables fleet technology comparisons.

4. **Parameter reference table** — all BusCostConfig fields with defaults, units, and sources

---

## What This Does NOT Touch

- `TrafficModel` internals — no changes
- `BusAgent` — no changes
- `VehicleAgent` — no changes
- Any step loop, scheduler, or per-agent logic
- `HybridDataCollector` — no new tiers or collection hooks
- Determinism — pure post-hoc math on existing outputs

## Edge Cases

| Case | Handling |
|---|---|
| `bus_interval == 0` (no bus service) | `compute_bus_costs()` returns `None`; summary has no `bus_cost_*` keys |
| `bus_cost_config is None` | Skip entirely — no cost columns in output |
| No bus trips completed (all in-transit at model end) | `avg_tt_bus` is NaN → skip cost calc |
| Variable bus_interval across season days | Each day uses its own interval; season summary uses mean |

## Old Plan Disposition

`plans/01_bus_service_cost_reporting.md` should be renamed to `plans/01_bus_service_cost_reporting(superseded).md` to match the convention of `plans/03_..._(done).md` and `plans/05_..._(done).md`.
