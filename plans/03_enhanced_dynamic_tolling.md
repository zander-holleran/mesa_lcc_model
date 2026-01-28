# Feature Plan: Enhanced Dynamic Tolling System

## Overview
Enhance the existing tolling system with improved flow-based signals, toll rounding, and controlled update frequency to enable more realistic and policy-relevant congestion pricing experiments.

## Motivation
The current tolling system is functional but limited:
- Flow signal exists but is inflexible (fixed window)
- Volume signal updates every step (unrealistic)
- Toll values are unrounded floats ($3.47382 instead of $3.50)
- All mechanisms have different update behaviors

These limitations make it difficult to:
- Simulate realistic toll policies (operators don't update tolls every second)
- Test flow-based pricing (requires tunable averaging windows)
- Communicate results to stakeholders (need round dollar amounts)
- Compare mechanisms fairly (different update frequencies confound results)

## Design Decisions

### 1. Flow Signal Window: **Configurable Parameter (1-10 minutes)**
**Choice:** Make flow window a parameter with reasonable range 1-10 minutes (60-600 steps)

**Implications:**
- **Pros:**
  - Flexibility to test responsiveness vs. stability trade-offs
  - Can match real-world toll operator update rates
  - Allows sensitivity analysis on window size
  - One parameter handles both signal collection and averaging

- **Cons:**
  - More parameters to tune
  - Requires documentation on reasonable ranges
  - Longer windows increase memory/state requirements

**Implementation approach:**
- Store a **rolling window** of vehicle spawn counts
- Calculate flow as `sum(window) / len(window)` (vehicles per step)
- Window updates each step, toll updates every `update_frequency` steps

**Memory consideration:**
- Max window: 600 steps (10 minutes)
- Storage per step: 1 integer (spawn count)
- Total: ~2.4 KB per model (negligible)

---

### 2. Flow Metric: **All Vehicles (Cars + Buses)**
**Choice:** Count both cars and buses entering the system

**Implications:**
- **Pros:**
  - More accurate representation of road congestion
  - Buses contribute to traffic (slower, larger following distance)
  - Avoids perverse incentive (tolls ignore buses even if they cause delays)
  - Consistent with volume signal (which counts all vehicles)

- **Cons:**
  - Buses are policy levers, not exogenous demand (creates feedback loop)
  - Empty buses contribute to flow signal but carry no passengers

**Implementation:**
- Replace `car_counter` usage with new `vehicle_counter` (incremented for both cars and buses)
- Alternatively: track separately and weight by Passenger Car Equivalents (PCE)

**Note:** The flow signal should represent **new vehicle arrivals**, not departures or vehicles on road. This matches real-world ramp metering and entry pricing.

---

### 3. Toll Rounding: **Nearest $0.10 (10 cents)**
**Choice:** Round all toll values to nearest 10 cents after calculation

**Implications:**
- **Pros:**
  - Realistic (toll operators use discrete price points)
  - Easier to communicate to stakeholders
  - Reduces cognitive load for agents (if modeling toll perception later)
  - Matches most real-world toll systems

- **Cons:**
  - Slightly less smooth optimization (discrete instead of continuous)
  - Could create price plateaus (e.g., tolls stay at $5.00 for a range of volumes)
  - May interact with mechanism parameters (slope, thresholds)

**Rounding function:**
```python
def round_toll(toll: float, increment: float = 0.10) -> float:
    return round(toll / increment) * increment
```

**Examples:**
- $3.47 → $3.50
- $5.03 → $5.00
- $0.94 → $0.90

**Edge cases:**
- Negative tolls (subsidies): Round towards zero or away from zero? **Decision: Round to nearest (standard).**
- Zero toll: Remains $0.00

---

### 4. Update Frequency: **1 Minute for All Mechanisms**
**Choice:** Decouple signal calculation from toll updates. All mechanisms update every 60 steps (1 minute).

**Implications:**
- **Pros:**
  - Consistent comparison across mechanisms (same update rate)
  - Realistic (toll operators don't adjust every second)
  - Reduces computational overhead (9 out of 10 steps skip toll calculation)
  - Allows separate tuning of signal window and update rate

- **Cons:**
  - Volume signal becomes less responsive (was real-time, now 1-minute lag)
  - Requires refactoring update logic
  - Users must understand distinction between signal window and update frequency

**Key distinction:**
- **Signal window:** Time range for averaging measurements (flow only)
- **Update frequency:** How often the toll price changes (all mechanisms)

**Example:**
- Flow mechanism with 5-minute window, 1-minute update frequency:
  - Steps 0-59: Accumulate spawn data in 5-minute rolling window, don't update toll
  - Step 60: Calculate flow from last 300 steps (5 min), update toll
  - Steps 61-119: Accumulate data, don't update
  - Step 120: Recalculate, update toll
  - ...

---

## Implementation Subtasks

### Subtask 1: Add Rolling Window for Flow Signal
**File:** `traffic/model/tolling.py`
**Changes:**

1. Add helper class for rolling window:
```python
from collections import deque

class RollingWindow:
    """Fixed-size rolling window for efficient averaging."""
    def __init__(self, maxlen: int):
        self.window = deque(maxlen=maxlen)
        self.maxlen = maxlen

    def append(self, value):
        self.window.append(value)

    def mean(self):
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def __len__(self):
        return len(self.window)
```

2. Modify `get_flow_signal()`:
```python
def get_flow_signal(model, flow_window_steps):
    """
    Calculate average vehicle spawns per step over rolling window.

    Args:
        model: TrafficModel instance
        flow_window_steps: Size of rolling window (e.g., 300 for 5 minutes)

    Returns:
        float: Average vehicles per step, or None if not ready
    """
    step = model.schedule.steps

    # Initialize rolling window on first call
    if not hasattr(model, "flow_window"):
        model.flow_window = RollingWindow(maxlen=flow_window_steps)
        model.last_vehicle_counter = model.vehicle_counter

    # Calculate spawns this step
    spawned_this_step = model.vehicle_counter - model.last_vehicle_counter
    model.last_vehicle_counter = model.vehicle_counter

    # Add to rolling window
    model.flow_window.append(spawned_this_step)

    # Return average flow (vehicles per step)
    if len(model.flow_window) < flow_window_steps:
        return None  # Window not full yet

    return model.flow_window.mean()
```

**Estimated lines changed:** 40-50

---

### Subtask 2: Add Vehicle Counter (All Vehicles)
**File:** `traffic/model/traffic_model.py`
**Changes:**

1. Add counter initialization:
```python
def __init__(self, ...):
    # ... existing code ...
    self.vehicle_counter = 0  # Tracks all vehicles (cars + buses)
```

2. Update counter in generation functions:

**File:** `traffic/model/generate.py`

```python
def generate_person(model):
    # ... existing code ...
    if new_tp.mode == "car":
        # ... create car ...
        model.car_counter += 1
        model.vehicle_counter += 1  # NEW

def generate_new_bus(model):
    # ... existing code ...
    model.bus_counter += 1
    model.vehicle_counter += 1  # NEW
```

**Estimated lines changed:** 5-10

---

### Subtask 3: Implement Toll Rounding Function
**File:** `traffic/model/tolling.py`
**Changes:**

```python
def round_toll(toll: float, increment: float = 0.10) -> float:
    """
    Round toll to nearest increment.

    Args:
        toll: Raw toll value
        increment: Rounding increment (default: $0.10)

    Returns:
        Rounded toll

    Examples:
        >>> round_toll(3.47, 0.10)
        3.50
        >>> round_toll(5.03, 0.10)
        5.00
        >>> round_toll(12.46, 0.25)
        12.50
    """
    if increment <= 0:
        raise ValueError(f"Increment must be positive, got {increment}")

    return round(toll / increment) * increment
```

**Estimated lines:** 15-20

---

### Subtask 4: Implement Unified Update Frequency
**File:** `traffic/model/tolling.py`
**Changes:**

1. Modify `update_tolls()` to check update frequency:
```python
def update_tolls(model):
    """
    Update tolls based on mechanism and update frequency.

    Returns:
        float: Current toll (may be unchanged)
    """
    mech = model.toll_mechanism
    spec = MECHANISMS.get(mech)

    # Static mechanism: never update
    if spec is None:
        return model.current_toll_car

    # Check if it's time to update (frequency check)
    step = model.schedule.steps
    update_freq = model.toll_params.get("update_frequency_steps", 60)  # Default: 1 minute

    if not hasattr(model, "last_toll_update_step"):
        model.last_toll_update_step = step

    elapsed = step - model.last_toll_update_step

    # Not time to update yet
    if elapsed < update_freq:
        return model.current_toll_car

    # Time to update: collect signal
    params = model.toll_params
    signal = spec["signal"](model, **spec["signal_args"](params))

    # Signal not ready (e.g., flow window not full)
    if signal is None:
        return model.current_toll_car

    # Calculate raw toll
    raw_toll = spec["tx"](signal, **spec["tx_args"](params))

    # Apply rounding
    toll_increment = model.toll_params.get("toll_rounding_increment", 0.10)
    rounded_toll = round_toll(raw_toll, toll_increment)

    # Update model state
    model.current_toll_car = rounded_toll
    model.last_toll_update_step = step

    return rounded_toll
```

2. Update `MECHANISMS` dict to pass flow window parameter:
```python
"flow": dict(
    signal=get_flow_signal,
    signal_args=lambda p: dict(
        flow_window_steps=p.get("flow_window_steps", 300)  # Default: 5 minutes
    ),
    tx=piecewise_linear_tx,
    tx_args=lambda p: dict(
        min_threshold=p.get("min_threshold", 1.0),
        slope=p.get("slope", 10.0),
        base=p.get("base_price", 0.0),
    ),
),
```

**Estimated lines changed:** 30-40

---

### Subtask 5: Update Toll Parameters in Configs
**File:** `season/configs.py`
**Changes:**

Update `make_season_config()` to include new toll parameters:
```python
def make_season_config(
    # ... existing params ...
    toll_mechanism: Optional[str] = None,
    toll_params: Optional[Dict[str, Any]] = None,
    # ... existing params ...
):
    # Set default toll_params with new fields
    if toll_params is None:
        toll_params = {}

    # Add defaults for new parameters
    toll_params.setdefault("car", 0.0)
    toll_params.setdefault("bus", 0.0)
    toll_params.setdefault("update_frequency_steps", 60)  # 1 minute
    toll_params.setdefault("toll_rounding_increment", 0.10)  # 10 cents

    # For flow mechanism, set default window
    if toll_mechanism == "flow":
        toll_params.setdefault("flow_window_steps", 300)  # 5 minutes

    # ... rest of function ...
```

**Estimated lines changed:** 10-15

---

### Subtask 6: Add Toll Configuration Examples
**File:** `notebooks/single_day_season.ipynb`
**Changes:** Add cells demonstrating new toll parameters

```python
# Example 1: Flow-based tolling with 3-minute window
config = make_season_config(
    season_id="flow_toll_3min",
    toll_mechanism="flow",
    toll_params={
        "car": 0.0,  # Base price (overridden by mechanism)
        "bus": 0.0,
        "flow_window_steps": 180,  # 3 minutes
        "update_frequency_steps": 60,  # Update every minute
        "toll_rounding_increment": 0.10,  # Round to nearest dime
        "min_threshold": 1.0,  # Flow threshold (vehicles/step)
        "slope": 10.0,  # $10 per vehicle/step above threshold
        "base_price": 2.0,  # Base toll when above threshold
    },
    # ... other params ...
)

# Example 2: Volume-based tolling with 1-minute updates
config = make_season_config(
    season_id="volume_toll_1min",
    toll_mechanism="volume",
    toll_params={
        "car": 0.0,
        "bus": 0.0,
        "update_frequency_steps": 60,  # Update every minute (NEW)
        "toll_rounding_increment": 0.25,  # Round to nearest quarter
        "min_threshold": 50,  # Volume threshold (vehicles on road)
        "slope": 0.10,  # $0.10 per vehicle above threshold
        "base_price": 5.0,  # Base toll when above threshold
    },
    # ... other params ...
)
```

**Estimated lines:** 30-40 (notebook cells)

---

### Subtask 7: Update Documentation
**File:** `README.md`
**Changes:** Document new tolling parameters

```markdown
### Dynamic Tolling

Three mechanisms are supported:

1. **Static:** Fixed toll regardless of conditions
   ```python
   toll_mechanism="static",
   toll_params={"car": 10.0, "bus": 0.0}
   ```

2. **Volume-based:** Toll increases with vehicles currently on road
   ```python
   toll_mechanism="volume",
   toll_params={
       "min_threshold": 50,       # Vehicles on road before toll activates
       "slope": 0.10,              # Toll increase per vehicle above threshold
       "base_price": 5.0,          # Base toll when above threshold
       "update_frequency_steps": 60,      # Update every 60 steps (1 minute)
       "toll_rounding_increment": 0.10,   # Round to nearest $0.10
   }
   ```

3. **Flow-based:** Toll increases with vehicle arrival rate
   ```python
   toll_mechanism="flow",
   toll_params={
       "flow_window_steps": 300,   # Rolling window (5 minutes = 300 steps)
       "min_threshold": 1.0,        # Flow threshold (vehicles/step)
       "slope": 10.0,               # Toll increase per vehicle/step above threshold
       "base_price": 2.0,           # Base toll when above threshold
       "update_frequency_steps": 60,      # Update every 60 steps
       "toll_rounding_increment": 0.10,   # Round to nearest $0.10
   }
   ```

**Key parameters:**
- `update_frequency_steps`: How often toll price changes (default: 60 = 1 minute)
- `toll_rounding_increment`: Rounding precision (default: 0.10 = 10 cents)
- `flow_window_steps`: Averaging window for flow signal (default: 300 = 5 minutes)
```

**Estimated lines:** 40-50

---

## Critical Files to Modify
1. **traffic/model/tolling.py** - Core tolling logic (rolling window, rounding, update frequency)
2. **traffic/model/traffic_model.py** - Add vehicle_counter
3. **traffic/model/generate.py** - Increment vehicle_counter for cars and buses
4. **season/configs.py** - Default toll parameters
5. **notebooks/single_day_season.ipynb** - Example usage
6. **README.md** - Documentation

## Testing & Verification

### Unit Tests: Rounding Function
```python
def test_round_toll():
    assert round_toll(3.47, 0.10) == 3.50
    assert round_toll(5.03, 0.10) == 5.00
    assert round_toll(0.94, 0.10) == 0.90
    assert round_toll(12.46, 0.25) == 12.50
    assert round_toll(0.0, 0.10) == 0.0
    assert round_toll(-2.47, 0.10) == -2.50  # Negative tolls (subsidies)

    # Edge cases
    assert round_toll(0.05, 0.10) == 0.10  # Round half up
    assert round_toll(0.049, 0.10) == 0.0

    print("✓ Rounding tests passed")
```

### Unit Tests: Rolling Window
```python
def test_rolling_window():
    window = RollingWindow(maxlen=3)

    window.append(1)
    assert window.mean() == 1.0

    window.append(2)
    assert window.mean() == 1.5

    window.append(3)
    assert window.mean() == 2.0

    window.append(4)  # Pushes out 1
    assert window.mean() == 3.0
    assert len(window) == 3

    print("✓ Rolling window tests passed")
```

### Integration Tests: Flow Signal
```python
def test_flow_signal_integration():
    # Create model with flow tolling
    config = make_season_config(
        season_id="flow_test",
        toll_mechanism="flow",
        toll_params={
            "flow_window_steps": 60,  # 1 minute window
            "update_frequency_steps": 60,  # Update every minute
            "min_threshold": 1.0,
            "slope": 10.0,
            "base_price": 2.0,
        },
        max_persons=100,
        n_days=1,
    )

    orchestrator = SeasonOrchestrator(config, store_data=False)
    orchestrator.run_day()

    model = orchestrator.last_model_run

    # Check that flow window was created and used
    assert hasattr(model, "flow_window"), "Flow window not initialized"
    assert len(model.flow_window) > 0, "Flow window not populated"

    # Check that tolls were updated (not static)
    model_data = model.datacollector.get_model_vars_dataframe()
    toll_values = model_data["current_toll_car"].unique()
    assert len(toll_values) > 1, "Toll never changed (should be dynamic)"

    # Check that tolls are rounded
    for toll in toll_values:
        assert toll == round_toll(toll, 0.10), f"Toll {toll} not rounded to $0.10"

    print("✓ Flow signal integration test passed")
```

### Integration Tests: Update Frequency
```python
def test_update_frequency():
    config = make_season_config(
        season_id="freq_test",
        toll_mechanism="volume",
        toll_params={
            "update_frequency_steps": 120,  # Update every 2 minutes
            "min_threshold": 10,
            "slope": 0.5,
            "base_price": 5.0,
        },
        max_steps=500,
        max_persons=100,
    )

    orchestrator = SeasonOrchestrator(config, store_data=False)
    orchestrator.run_day()

    model = orchestrator.last_model_run
    model_data = model.datacollector.get_model_vars_dataframe()

    # Tolls should only change every 120 steps
    toll_changes = model_data["current_toll_car"].diff().abs() > 0.001
    change_steps = model_data[toll_changes].index.tolist()

    # Check that changes occur at multiples of 120
    for step in change_steps:
        assert step % 120 == 0, f"Toll changed at step {step}, not a multiple of 120"

    print("✓ Update frequency test passed")
```

### Comparison Tests: Before vs After
```python
def test_backward_compatibility():
    """Ensure changes don't break existing toll configurations."""

    # Old-style static toll config (should still work)
    config_old = make_season_config(
        season_id="static_old",
        toll_mechanism="static",
        toll_params={"car": 10.0, "bus": 0.0},
        max_persons=50,
    )

    orchestrator = SeasonOrchestrator(config_old, store_data=False)
    orchestrator.run_day()

    # Check toll stayed at $10.00
    model_data = orchestrator.last_model_run.datacollector.get_model_vars_dataframe()
    assert (model_data["current_toll_car"] == 10.0).all(), "Static toll changed unexpectedly"

    print("✓ Backward compatibility test passed")
```

## Potential Risks & Considerations

### Risk 1: Flow Window Memory
**Issue:** Large flow windows (10 minutes = 600 steps) require storing 600 integers

**Impact:** Minimal - 600 integers × 4 bytes = 2.4 KB per model

**Mitigation:** Not a concern for typical usage. Only becomes issue if running 1000+ models in parallel.

---

### Risk 2: Update Frequency Interacts with Window Size
**Issue:** If update frequency > flow window, signal is stale

**Example:**
- Flow window: 60 steps (1 minute)
- Update frequency: 300 steps (5 minutes)
- Toll is based on flow from 4-5 minutes ago (stale)

**Mitigation:**
- Document recommended relationship: `update_frequency <= flow_window`
- Add validation warning if violated:
  ```python
  if update_freq > flow_window:
      warnings.warn(f"Update frequency ({update_freq}) exceeds flow window ({flow_window}). "
                    "Toll signal will be stale.")
  ```

---

### Risk 3: Rounding Creates Plateaus
**Issue:** Tolls may stay constant over ranges of volumes/flows due to rounding

**Example:**
- Raw tolls: $4.91, $4.93, $4.97, $5.03, $5.07, $5.11
- Rounded: $4.90, $4.90, $5.00, $5.00, $5.10, $5.10
- Creates three discrete price levels instead of smooth increase

**Implications:**
- Reduces toll responsiveness slightly
- May affect optimal toll calculation (discrete vs continuous optimization)
- More realistic (matches real-world practice)

**Mitigation:** Generally beneficial for realism. If continuous optimization needed, set `toll_rounding_increment=0.01` (round to penny).

---

### Risk 4: Vehicle Counter vs Car Counter
**Issue:** Changing from car-only to all-vehicles changes flow signal values

**Impact:** Existing flow toll configurations will behave differently (higher flow due to buses)

**Mitigation:**
- Document the change clearly
- Adjust default parameters if needed (e.g., increase `min_threshold`)
- Provide migration guide for existing configs

---

### Risk 5: Update Frequency Applies to All Mechanisms
**Issue:** Volume signal was real-time (updated every step), now updates every minute

**Impact:** Volume-based tolls are less responsive than before

**Implications:**
- More realistic (toll operators don't update instantly)
- May reduce effectiveness of volume-based congestion pricing
- Allows fair comparison between volume and flow mechanisms

**Mitigation:**
- Document the change
- Allow per-mechanism update frequency if needed (future extension)
- Test that volume tolls still effectively reduce congestion

---

## Design Trade-offs Summary

| Decision | Realistic | Flexible | Complex | Performance |
|----------|-----------|----------|---------|-------------|
| Configurable flow window | ✓✓ | ✓✓✓ | ✓ | ○ (minimal memory) |
| All vehicles in flow | ✓✓✓ | ○ | ○ | ✓✓✓ |
| Toll rounding | ✓✓✓ | ✓ | ○ | ✓✓✓ |
| Unified update frequency | ✓✓✓ | ✓✓ | ✓ | ✓✓ (less compute) |

Legend: ✓✓✓ = Excellent, ✓✓ = Good, ✓ = Adequate, ○ = Neutral

---

## Parameter Recommendations

### Conservative (Stable Tolls)
```python
toll_params={
    "flow_window_steps": 600,          # 10 minutes (long average)
    "update_frequency_steps": 120,     # 2 minutes (infrequent updates)
    "toll_rounding_increment": 0.25,   # Round to quarter (coarse)
    "min_threshold": 2.0,              # High threshold (tolls rarely activate)
    "slope": 5.0,                      # Gentle slope
}
```
**Use case:** Risk-averse policies, initial testing

---

### Moderate (Balanced)
```python
toll_params={
    "flow_window_steps": 300,          # 5 minutes (default)
    "update_frequency_steps": 60,      # 1 minute (default)
    "toll_rounding_increment": 0.10,   # Round to dime (default)
    "min_threshold": 1.0,              # Moderate threshold
    "slope": 10.0,                     # Moderate slope
}
```
**Use case:** Most scenarios, realistic operator behavior

---

### Aggressive (Responsive)
```python
toll_params={
    "flow_window_steps": 60,           # 1 minute (short average)
    "update_frequency_steps": 30,      # 30 seconds (frequent updates)
    "toll_rounding_increment": 0.10,   # Round to dime
    "min_threshold": 0.5,              # Low threshold (tolls activate early)
    "slope": 20.0,                     # Steep slope
}
```
**Use case:** Testing toll effectiveness, automated systems

---

## Future Extensions

1. **Passenger Car Equivalents (PCE):** Weight buses differently in flow signal (e.g., 1 bus = 2 cars)
2. **Adaptive tolling:** Use reinforcement learning to optimize toll parameters
3. **Toll caps:** Maximum toll limits for equity
4. **Time-of-day pricing:** Different parameters by hour of day
5. **Multi-segment tolling:** Different tolls for different road sections
6. **Toll zones:** Cordon pricing around high-demand areas

## Success Criteria
✅ Flow signal uses configurable rolling window (1-10 minutes)
✅ Flow signal counts all vehicles (cars + buses)
✅ Tolls rounded to nearest $0.10 (configurable)
✅ All toll mechanisms update every 1 minute (configurable)
✅ Backward compatibility maintained for existing configs
✅ No performance degradation (<1% overhead)
✅ Documentation includes parameter recommendations
✅ Example notebooks demonstrate new features

## Estimated Implementation Time
- **Subtask 1:** 2-3 hours (rolling window implementation)
- **Subtask 2:** 1 hour (vehicle counter)
- **Subtask 3:** 30 minutes (rounding function)
- **Subtask 4:** 2-3 hours (unified update frequency)
- **Subtask 5:** 1 hour (config updates)
- **Subtask 6:** 1-2 hours (examples)
- **Subtask 7:** 1 hour (documentation)
- **Testing & Validation:** 2-3 hours
- **Total:** 11-15 hours

## Implementation Order (Recommended)
1. Start with rounding (simplest, independent)
2. Add vehicle counter (small change, enables flow improvements)
3. Implement rolling window for flow signal
4. Refactor update frequency (most complex, touches all mechanisms)
5. Update configs and documentation
6. Write tests and validate
