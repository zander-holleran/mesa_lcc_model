# Bus Service Cost Calculation

The simulation estimates the cost of operating the bus service using standard transportation engineering (TE) formulas. All calculation is **purely post-hoc** — it reads existing model outputs (trip travel times, headway, step count) and adds zero overhead to the simulation loop.

---

## Formula

### Step 1: Fleet Sizing

```
cycle_time = avg_one_way_tt × (1 + return_trip_factor) × (1 + layover_recovery_pct)
active_buses = ceil(cycle_time / headway)
total_fleet = ceil(active_buses × (1 + spare_ratio))
```

- **cycle_time**: round-trip time including layover/recovery. The model is one-way; `return_trip_factor` (default 1.0) scales the return leg relative to the uphill trip. Layover/recovery adds 10% for driver breaks and schedule padding (TCRP Report 30).
- **active_buses**: how many buses must be in simultaneous operation to maintain the headway.
- **total_fleet**: includes spare buses for maintenance rotation (FTA recommends 20%).

### Step 2: Hourly Cost Rates

```
labor/hr   = active_buses × relief_factor × driver_hourly_rate
ops/hr     = active_buses × ops_cost_per_hour
capital/hr = (bus_purchase_price × total_fleet) / (useful_life_years × service_days_per_year × service_hours_per_day)
total/hr   = (labor/hr + ops/hr + capital/hr) × overhead_multiplier
```

- **relief_factor** (1.17): accounts for the fact that you need more drivers than buses to cover breaks, sick days, and vacation.
- **capital/hr**: straight-line amortization of the entire fleet purchase spread across all service hours over the fleet's useful life.

### Step 3: Three Output Metrics

```
model_run_cost = total/hr × (model_steps / 3600)
daily_cost     = total/hr × service_hours_per_day
annual_cost    = daily_cost × service_days_per_year
```

| Metric | Time basis | Compare against |
|---|---|---|
| `bus_cost_model_run` | Actual simulated hours (`model.steps / 3600`) | `total_rev` — both cover the exact same simulated period |
| `bus_cost_daily` | Standard service day (default 12 hrs) | Planning estimates, per-day benchmarks |
| `bus_cost_annual` | Full season (default 150 days) | NTD/DOT annual operating expense reports |

**Why `model_run_cost` matches `total_rev`:** The model runs a variable number of steps per day (driven by person count, typically ~8 hrs). `total_rev` is toll revenue collected over those actual steps. `model_run_cost` uses the same step count as its time basis, so both metrics cover the identical time window.

---

## Worked Examples

All examples assume a one-way bus travel time of **35 minutes** (typical LCC uphill run).

### Example 1: Default Config, 15-min Headway

Using `BusCostConfig.default()` with `bus_interval=15`:

```
cycle_time    = 35 × (1 + 1.0) × (1 + 0.10) = 77.0 min
active_buses  = ceil(77.0 / 15) = 6
total_fleet   = ceil(6 × 1.20) = 8

labor/hr      = 6 × 1.17 × $35.00   = $245.70
ops/hr        = 6 × $40.00           = $240.00
capital/hr    = ($550,000 × 8) / (14 × 150 × 12) = $174.60
total/hr      = ($245.70 + $240.00 + $174.60) × 1.10 = $726.33

model_run (8 hr sim) = $726.33 × 8     = $5,810.64
daily (12 hr)        = $726.33 × 12    = $8,715.96
annual (150 days)    = $8,715.96 × 150 = $1,307,394
```

This is the baseline: 15-minute headway requires 6 active buses, 8 total fleet.

### Example 2: Same Config, 30-min Headway

Doubling the headway to `bus_interval=30`:

```
cycle_time    = 77.0 min (unchanged — same route)
active_buses  = ceil(77.0 / 30) = 3
total_fleet   = ceil(3 × 1.20) = 4

labor/hr      = 3 × 1.17 × $35.00   = $122.85
ops/hr        = 3 × $40.00           = $120.00
capital/hr    = ($550,000 × 4) / (14 × 150 × 12) = $87.30
total/hr      = ($122.85 + $120.00 + $87.30) × 1.10 = $363.17

model_run (8 hr sim) = $363.17 × 8     = $2,905.33
daily (12 hr)        = $363.17 × 12    = $4,358.00
annual (150 days)    = $4,358.00 × 150 = $653,697
```

Doubling the headway halves the fleet (3 vs 6 active buses) and roughly halves the cost. This is the fundamental TE principle: **double the frequency, double the cost**.

### Example 3: Electric Bus Fleet, 15-min Headway

Using a modified config for electric buses:

```python
BusCostConfig(
    bus_purchase_price=1_100_000,  # electric bus (2024 median)
    ops_cost_per_hour=25.0,       # lower fuel cost (electricity vs diesel)
    # all other defaults unchanged
)
```

```
cycle_time    = 77.0 min
active_buses  = 6, total_fleet = 8

labor/hr      = 6 × 1.17 × $35.00        = $245.70   (unchanged)
ops/hr        = 6 × $25.00               = $150.00   (lower — electricity cheaper)
capital/hr    = ($1,100,000 × 8) / (14 × 150 × 12) = $349.21   (2× higher)
total/hr      = ($245.70 + $150.00 + $349.21) × 1.10 = $819.40

model_run (8 hr sim) = $819.40 × 8     = $6,555.20
daily (12 hr)        = $819.40 × 12    = $9,832.80
annual (150 days)    = $9,832.80 × 150 = $1,474,920
```

Electric buses lower operating costs (cheaper fuel) but double the capital cost. Net effect: **~13% more expensive** at these parameters. The `BusCostConfig` parameterization enables direct fleet technology comparison without changing the simulation.

---

## Parameter Reference

| Parameter | Default | Units | Source |
|---|---|---|---|
| `layover_recovery_pct` | 0.10 | fraction | TCRP Report 30 |
| `return_trip_factor` | 1.0 | multiplier on one-way time | Conservative (return = uphill); set to 0.8 for faster downhill |
| `spare_ratio` | 0.20 | fraction | FTA guidance |
| `bus_purchase_price` | 550,000 | $ | FTA Capital Cost DB, midpoint 40-ft heavy-duty (2024) |
| `useful_life_years` | 14 | years | FTA Default ULB for heavy-duty bus |
| `driver_hourly_rate` | 35.0 | $/hr (fully loaded) | BLS OES 53-3052 median $24 + ~45% fringe |
| `relief_factor` | 1.17 | multiplier | TCRP scheduling guidance |
| `ops_cost_per_hour` | 40.0 | $/hr | NTD non-labor portion of operating cost |
| `overhead_multiplier` | 1.10 | multiplier | Admin, insurance, facility |
| `service_hours_per_day` | 12.0 | hours | 6am-6pm ski bus window |
| `service_days_per_year` | 150.0 | days | ~5 month ski season (Nov-Apr) |
