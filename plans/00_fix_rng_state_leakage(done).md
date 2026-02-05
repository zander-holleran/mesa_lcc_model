# Critical Fix: RNG State Leakage in VehicleAgent

## Priority
🚨 **BLOCKING ISSUE** - Must be fixed BEFORE implementing parallel processing

## Problem Description

### The Issue
There is ONE instance of improper random number generation in the codebase that breaks reproducibility:

**File:** `traffic/agents/vehicle_agent.py`, **Line 193**
```python
noise = np.random.normal(0, .1)  # ❌ WRONG - uses global RNG state
```

This line is in the `less_smooth_brake()` method, which adds "human-like noise" to braking behavior.

### Why This Breaks Parallel Processing

**Global RNG state is shared across processes:**
- When using `np.random.normal()` without an RNG object, it uses NumPy's global random state
- In parallel execution, this global state can become desynchronized
- Different processes may draw from the same RNG sequence in unpredictable order
- **Result:** Non-reproducible outputs even with the same seed

**Example failure scenario:**
```python
# Sequential run with seed=123
model1 = TrafficModel(seed=123, ...)
model1.run_model()
result1 = model1.datacollector.get_model_vars_dataframe()

# Parallel run with seed=123
model2 = TrafficModel(seed=123, ...)  # Run in parallel with other models
model2.run_model()
result2 = model2.datacollector.get_model_vars_dataframe()

# These should be identical, but won't be!
assert result1.equals(result2)  # ❌ FAILS due to RNG state leakage
```

### Current Good Practices in Codebase

The model ALREADY uses proper RNG in most places:

**Model initialization (correct):**
```python
# traffic/model/traffic_model.py, line 42
self.rng = np.random.default_rng(seed)
```

**Proper usage examples:**
```python
# traffic/model/traffic_model.py, line 183
prob = self.rng.random()  # ✓ CORRECT

# traffic/agents/car_agent.py, lines 9-11
self.acceptable_over = model.rng.normal(...)  # ✓ CORRECT
self.ideal_distance_multiplier = model.rng.normal(...)  # ✓ CORRECT
self.curve_responce = model.rng.normal(...)  # ✓ CORRECT
```

The VehicleAgent line 193 is the ONLY exception.

---

## The Fix

### Single Line Change Required

**File:** `traffic/agents/vehicle_agent.py`, **Line 193**

**Before:**
```python
noise = np.random.normal(0, .1)
```

**After:**
```python
noise = self.model.rng.normal(0, .1)
```

That's it. One line change.

---

## Implementation

### Step 1: Make the Fix
**File:** `traffic/agents/vehicle_agent.py`
**Method:** `less_smooth_brake()` (around line 175-197)

**Full context:**
```python
def less_smooth_brake(self, gap, ideal_gap):
    """
    Calculate braking deceleration based on gap and ideal gap.
    Returns deceleration in m/s^2.
    """
    if ideal_gap <= 0 or np.isnan(ideal_gap):
        return 0
    force = max((ideal_gap - gap) / ideal_gap, 0)
    # Squared to overreact when too close
    base = force ** 2

    # Add some human-like noise
    noise = self.model.rng.normal(0, .1)  # ✓ FIXED
    break_pct = self._clip01(base + noise)

    deceleration = break_pct * 8  # <- this is acting as max decel in mps
    return deceleration
```

**Lines changed:** 1
**Estimated time:** 2 minutes

---

### Step 2: Verify No Other Issues

Run grep to confirm no other problematic patterns exist:
```bash
grep -r "np\.random\." traffic/ --include="*.py" | grep -v "rng" | grep -v "default_rng" | grep -v "Generator"
```

**Expected output:** Only the line you just fixed (or nothing if already fixed)

---

## Testing & Verification

### Test 1: Single-Run Reproducibility
```python
from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config

# Create identical configs
config = make_season_config(
    season_id="rng_test",
    seed=12345,
    n_days=1,
    max_persons=50,
    road_path="data/roads/lcc_road.parquet",
    ecs_path="data/vehicle_counts/lcc_expected_counts.csv",
)

# Run 1
orch1 = SeasonOrchestrator(config, store_data=False)
orch1.run_day()
result1 = orch1.last_model_run.datacollector.get_model_vars_dataframe()

# Run 2 (same seed)
orch2 = SeasonOrchestrator(config, store_data=False)
orch2.run_day()
result2 = orch2.last_model_run.datacollector.get_model_vars_dataframe()

# Compare
import pandas as pd
pd.testing.assert_frame_equal(result1, result2)
print("✓ Single-run reproducibility: PASS")
```

**Expected:** Test passes (dataframes are identical)

---

### Test 2: Parallel vs Sequential Reproducibility
```python
from season.parallel_runner import run_parameter_sweep
from season.configs import make_season_config

# Create 5 identical configs with different seeds
configs = [
    make_season_config(
        season_id=f"test_{i}",
        seed=1000 + i,
        n_days=1,
        max_persons=50,
        road_path="data/roads/lcc_road.parquet",
        ecs_path="data/vehicle_counts/lcc_expected_counts.csv",
    )
    for i in range(5)
]

# Sequential execution
results_seq = []
for config in configs:
    orch = SeasonOrchestrator(config, store_data=False)
    orch.run_day()
    results_seq.append(orch._compute_season_summary())

# Parallel execution
from joblib import Parallel, delayed
def run_scenario(config):
    orch = SeasonOrchestrator(config, store_data=False)
    orch.run_day()
    return orch._compute_season_summary()

results_par = Parallel(n_jobs=5)(
    delayed(run_scenario)(config) for config in configs
)

# Compare results
import numpy as np
for seq, par in zip(results_seq, results_par):
    for key in seq:
        if isinstance(seq[key], (int, float)):
            np.testing.assert_almost_equal(seq[key], par[key], decimal=10)

print("✓ Parallel vs Sequential reproducibility: PASS")
```

**Expected:** Test passes (parallel results match sequential exactly)

---

### Test 3: Different Seeds Produce Different Results
```python
# Ensure RNG is actually being used (not constant)
config1 = make_season_config(season_id="seed_test_1", seed=111, n_days=1, max_persons=50, ...)
config2 = make_season_config(season_id="seed_test_2", seed=222, n_days=1, max_persons=50, ...)

orch1 = SeasonOrchestrator(config1, store_data=False)
orch1.run_day()
result1 = orch1.last_model_run.datacollector.get_model_vars_dataframe()

orch2 = SeasonOrchestrator(config2, store_data=False)
orch2.run_day()
result2 = orch2.last_model_run.datacollector.get_model_vars_dataframe()

# Results should be different (stochastic model)
assert not result1.equals(result2), "Different seeds produced identical results!"
print("✓ Different seeds produce different results: PASS")
```

**Expected:** Test passes (results differ due to different RNG states)

---

## Why This Fix is Safe

### No Behavior Change
- The distribution of noise remains exactly the same: `Normal(0, 0.1)`
- Only the SOURCE of randomness changes (global state → model RNG)
- Vehicles still brake with the same statistical properties

### Preserves Single-Run Behavior
- For a single model run, results are already deterministic (given a seed)
- This fix doesn't change single-run outputs
- It ENABLES deterministic behavior across multiple runs

### No Performance Impact
- `model.rng.normal()` has identical performance to `np.random.normal()`
- Both use the same underlying NumPy C code
- No measurable overhead

---

## Success Criteria
✅ Single-run reproducibility test passes
✅ Parallel vs sequential test passes
✅ Different seeds produce different results (RNG not broken)
✅ No other `np.random.*` calls exist outside of `model.rng`
✅ All existing model outputs remain unchanged (for same seed)

---

## Estimated Time
- **Fix:** 2 minutes (one line change)
- **Testing:** 15-30 minutes (run verification tests)
- **Total:** 20-35 minutes

---

## Implementation Checklist
- [ ] Change line 193 in `vehicle_agent.py`
- [ ] Run grep to verify no other issues
- [ ] Run Test 1: Single-run reproducibility
- [ ] Run Test 2: Parallel vs sequential (after parallel_runner.py is implemented)
- [ ] Run Test 3: Different seeds test
- [ ] Document fix in commit message
- [ ] Update parallel processing plan status (unblock implementation)

---

## Commit Message Template
```
Fix RNG state leakage in VehicleAgent braking noise

Problem: VehicleAgent.less_smooth_brake() used np.random.normal()
instead of model.rng.normal(), causing non-reproducible outputs
in parallel execution due to global RNG state interference.

Solution: Changed line 193 to use self.model.rng.normal(0, .1)

Impact: Enables deterministic parallel processing while preserving
all existing model behavior for single runs.

Testing: Verified single-run reproducibility and parallel vs sequential
matching with identical seeds.

Fixes: Blocking issue for parallel processing implementation
```

---

## Related Plans
This fix unblocks:
- **`plans/02_parallel_processing.md`** - Can now implement deterministic parallel execution

This fix does NOT affect:
- **`plans/01_bus_service_cost_reporting.md`** - Independent feature
- **`plans/03_enhanced_dynamic_tolling.md`** - Independent feature

---

## Notes
- This is the ONLY RNG issue in the codebase
- The rest of the model already uses proper RNG practices
- Consider adding a linting rule to catch future `np.random.*` usage
- Could add a CI test that runs parallel reproducibility checks
