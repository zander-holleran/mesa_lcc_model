# Feature Plan: Bus Service Cost Reporting

## Overview
Add comprehensive bus service cost tracking and reporting to enable policy analysis of public transit subsidies and cost-effectiveness comparisons across scenarios.

## Motivation
Currently, the model tracks bus ridership, travel times, and tolls, but lacks a cost model for bus operations. This makes it impossible to:
- Calculate the subsidy required to operate bus service
- Compare cost per passenger across different bus intervals
- Evaluate whether toll revenue from cars can cover bus operating costs
- Assess the financial efficiency of transit interventions

## Design Decisions

### 1. Cost Structure: **Per-Step (Time-Based)**
**Choice:** Buses accumulate costs based on time operating (steps_taken × hourly_rate / 3600)

**Implications:**
- **Pros:**
  - Simpler implementation - no distance tracking modifications needed (distance_traveled already exists)
  - Directly represents driver labor costs (main operating cost)
  - Matches real-world transit budgeting (driver hours)
  - Easy to understand and communicate

- **Cons:**
  - Doesn't capture fuel/maintenance costs proportional to distance
  - Less accurate for wear-and-tear expenses
  - Could underestimate costs for longer routes

**Parameters needed:**
- `bus_fixed_cost_per_dispatch`: Capital cost when bus is created (default: $50)
- `bus_hourly_operating_cost`: Variable cost per hour of operation (default: $75/hr)

### 2. Revenue Model: **Costs Only (Track Tolls Separately)**
**Choice:** Only track operating costs. Toll revenue already captured via `current_toll_bus`.

**Implications:**
- **Pros:**
  - Clean separation of costs and revenue
  - Allows flexible policy analysis (toll revenue can subsidize buses)
  - User can compute cost recovery ratios post-hoc: `total_toll_car / total_bus_cost`
  - Fare-free buses supported (no fare collection logic needed)

- **Cons:**
  - Cannot model fare revenue if user wants to add bus fares later
  - Requires post-processing to evaluate subsidy needs

**Future extension:** If fares are added later, implement `bus_fare_per_passenger` parameter and sum `len(bus.passengers) × fare` at trip end.

### 3. Reporting Granularity: **Daily Aggregate**
**Choice:** Report total bus costs per day in day_summary, alongside existing metrics like `avg_tt_car`, `total_toll_car`, etc.

**Implications:**
- **Pros:**
  - Consistent with existing reporting structure
  - Minimal data storage overhead
  - Directly supports policy scenario comparisons
  - Easy to visualize trends across days

- **Cons:**
  - Cannot analyze individual bus utilization variance
  - Cannot identify which buses were empty vs. full
  - No time-series of cumulative costs during the day

**Metrics to report:**
- `total_bus_fixed_cost`: Sum of fixed costs for all buses dispatched
- `total_bus_operating_cost`: Sum of time-based costs for all buses
- `total_bus_cost`: Fixed + operating
- `cost_per_bus_rider`: `total_bus_cost / bus_riders` (if bus_riders > 0)
- `avg_bus_utilization`: `bus_riders / (bus_counter × bus_capacity)` (percentage)

## Implementation Subtasks

### Subtask 1: Add Cost Parameters to Model
**File:** `traffic/model/traffic_model.py`
**Changes:**
1. Add to `__init__` parameters:
   ```python
   bus_fixed_cost_per_dispatch=50.0,
   bus_hourly_operating_cost=75.0,
   ```
2. Store as instance variables:
   ```python
   self.bus_fixed_cost_per_dispatch = bus_fixed_cost_per_dispatch
   self.bus_hourly_operating_cost = bus_hourly_operating_cost
   ```
3. Initialize cost tracking counters:
   ```python
   self.total_bus_fixed_cost = 0.0
   self.total_bus_operating_cost = 0.0
   ```

**Estimated lines changed:** 5-10

---

### Subtask 2: Track Costs in BusAgent
**File:** `traffic/agents/bus_agent.py`
**Changes:**
1. Add to `__init__`:
   ```python
   self.fixed_cost = model.bus_fixed_cost_per_dispatch
   self.operating_cost = 0.0  # Accumulated during trip
   ```
2. Increment model's fixed cost counter at creation:
   ```python
   model.total_bus_fixed_cost += self.fixed_cost
   ```

**Estimated lines changed:** 3-5

---

### Subtask 3: Accumulate Operating Costs During Travel
**File:** `traffic/agents/vehicle_agent.py`
**Changes:**
1. In `end_of_road()` method (called when bus finishes trip):
   ```python
   # Calculate operating cost for buses
   if isinstance(self, BusAgent):
       hours_operated = self.steps_taken / 3600.0
       self.operating_cost = hours_operated * self.model.bus_hourly_operating_cost
       self.model.total_bus_operating_cost += self.operating_cost
   ```
2. Add cost fields to `finished_agents` dict:
   ```python
   "fixed_cost": getattr(self, "fixed_cost", 0.0),
   "operating_cost": getattr(self, "operating_cost", 0.0),
   ```

**Estimated lines changed:** 5-10

---

### Subtask 4: Add Cost Metrics to Day Summary
**File:** `season/season_orchestrator.py`
**Changes:**
1. In `_compute_day_summary()` method (around line 202-281):
   ```python
   # Bus cost metrics
   tm = self.last_model_run
   total_bus_fixed = tm.total_bus_fixed_cost
   total_bus_operating = tm.total_bus_operating_cost
   total_bus_cost = total_bus_fixed + total_bus_operating

   bus_riders = len(bus_df) if not bus_df.empty else 0
   cost_per_rider = total_bus_cost / bus_riders if bus_riders > 0 else 0.0

   bus_capacity_offered = tm.bus_counter * tm.bus_capacity
   avg_utilization = bus_riders / bus_capacity_offered if bus_capacity_offered > 0 else 0.0
   ```
2. Add to `summary` dict:
   ```python
   "total_bus_fixed_cost": total_bus_fixed,
   "total_bus_operating_cost": total_bus_operating,
   "total_bus_cost": total_bus_cost,
   "cost_per_bus_rider": cost_per_rider,
   "avg_bus_utilization": avg_utilization,
   ```
3. Add to print statement:
   ```python
   f"Total bus cost: ${summary['total_bus_cost']:.2f}, "
   f"Cost/rider: ${summary['cost_per_bus_rider']:.2f}, "
   ```

**Estimated lines changed:** 15-20

---

### Subtask 5: Propagate Parameters Through Configuration
**File:** `season/configs.py`
**Changes:**
1. Add to `SeasonConfig` dataclass (around line 118-146):
   ```python
   bus_fixed_cost_per_dispatch: float = 50.0
   bus_hourly_operating_cost: float = 75.0
   ```
2. Add to `make_season_config()` function parameters (around line 150-180):
   ```python
   bus_fixed_cost_per_dispatch: float = 50.0,
   bus_hourly_operating_cost: float = 75.0,
   ```
3. Pass to `SeasonConfig` constructor:
   ```python
   bus_fixed_cost_per_dispatch=bus_fixed_cost_per_dispatch,
   bus_hourly_operating_cost=bus_hourly_operating_cost,
   ```

**File:** `season/season_orchestrator.py`
**Changes:**
1. In `_build_model()` method (around line 117-148):
   ```python
   bus_fixed_cost_per_dispatch=self.config.bus_fixed_cost_per_dispatch,
   bus_hourly_operating_cost=self.config.bus_hourly_operating_cost,
   ```

**Estimated lines changed:** 10-15 across both files

---

### Subtask 6: Update Season Summary
**File:** `season/season_orchestrator.py`
**Changes:**
1. In `_compute_season_summary()` method (around line 320-374):
   ```python
   # Bus cost aggregates across all days
   total_bus_cost_all = sum(summary['total_bus_cost'] for summary in self.day_summaries)
   avg_cost_per_rider_all = total_bus_cost_all / total_trips if total_trips > 0 else 0.0

   # Add to summary dict:
   "total_bus_cost_season": total_bus_cost_all,
   "avg_cost_per_rider_season": avg_cost_per_rider_all,
   ```
2. Add to print statement for season summary

**Estimated lines changed:** 8-12

---

## Critical Files to Modify
1. **traffic/model/traffic_model.py** - Add parameters and counters
2. **traffic/agents/bus_agent.py** - Track fixed cost at creation
3. **traffic/agents/vehicle_agent.py** - Calculate operating cost at trip end
4. **season/season_orchestrator.py** - Compute and report cost metrics
5. **season/configs.py** - Propagate cost parameters through config system

## Testing & Verification

### Unit Tests
1. Create test bus with known `steps_taken` and verify cost calculation:
   ```python
   # Bus operating 10 minutes (600 steps)
   # Hourly rate: $75/hr
   # Expected: 600/3600 × 75 = $12.50
   ```

2. Verify fixed cost is added only once per bus at creation

3. Check that empty buses still accumulate operating costs

### Integration Tests
1. Run single-day season with known bus parameters:
   - `bus_interval = 30` (2 buses expected in ~1 hour)
   - `bus_fixed_cost_per_dispatch = 50`
   - `bus_hourly_operating_cost = 75`
   - Verify: `total_bus_cost ≈ 2×50 + 2×(trip_time/3600)×75`

2. Compare day summaries across scenarios:
   - Scenario A: `bus_interval = 15` (frequent service, higher cost)
   - Scenario B: `bus_interval = 60` (infrequent service, lower cost)
   - Verify: Scenario A has higher `total_bus_cost` but lower `cost_per_rider`

### Output Validation
1. Check `day_summary.parquet` contains new columns:
   - `total_bus_fixed_cost`
   - `total_bus_operating_cost`
   - `total_bus_cost`
   - `cost_per_bus_rider`
   - `avg_bus_utilization`

2. Verify season summary JSON includes aggregated bus cost metrics

3. Confirm CSV log (`data/season_summary_log.csv`) expands to include bus cost columns

## Potential Risks & Considerations

### Risk 1: Cost Calculation Timing
**Issue:** Operating cost is calculated at `end_of_road()`. If a bus doesn't finish before max_steps, its cost won't be recorded.

**Mitigation:**
- Document this limitation (negligible for most scenarios)
- Alternative: Calculate costs incrementally each step (adds overhead)

### Risk 2: Empty Buses
**Issue:** Buses are dispatched even if no passengers waiting. These incur costs but no riders.

**Impact:** `cost_per_rider` could be misleading if many empty buses are dispatched.

**Mitigation:** Also report `avg_bus_utilization` to show capacity usage.

### Risk 3: Parameter Sensitivity
**Issue:** Default cost parameters ($50 fixed, $75/hr) are placeholders. Real costs vary.

**Mitigation:**
- Document that parameters should be calibrated to real-world data
- Provide references for typical transit operating costs
- Enable easy parameter sweeps to test sensitivity

### Risk 4: Multi-Day Accumulation
**Issue:** `total_bus_cost` resets each day. Season-level total requires summing day summaries.

**Verification:** Confirm season summary correctly aggregates `total_bus_cost_season`.

## Future Extensions

1. **Distance-Based Costs:** Add per-kilometer cost option by tracking `distance_traveled`
2. **Fare Revenue:** Add `bus_fare_per_passenger` parameter and revenue tracking
3. **Per-Bus Reporting:** Store cost data in `finished_agents` for detailed analysis
4. **Cost Recovery Analysis:** Add computed field `cost_recovery_ratio = total_toll_car / total_bus_cost`
5. **Time-Series Costs:** Add cumulative bus cost to `model_reporters` for intra-day tracking

## Estimated Implementation Time
- **Subtask 1-3:** 1-2 hours (core cost tracking)
- **Subtask 4-6:** 1-2 hours (reporting integration)
- **Testing & Validation:** 1-2 hours
- **Total:** 3-6 hours (depending on familiarity with codebase)

## Success Criteria
✅ Bus operating costs accumulate based on time in operation
✅ Fixed dispatch costs recorded at bus creation
✅ Day summaries include 5 new bus cost metrics
✅ Season summaries aggregate costs across all days
✅ Cost parameters configurable via `make_season_config()`
✅ No impact on existing model outputs (travel times, mode shares, tolls)
✅ Documentation updated with cost parameter definitions
