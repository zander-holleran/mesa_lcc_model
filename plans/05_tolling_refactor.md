# Feature Plan: Tolling System Refactor

## Overview
Refactor the tolling system into a composable **Signal -> Transform -> Toll** architecture using callable dataclass objects. Separate the concept of a car road toll (dynamic, congestion-management) from a bus user fee (fixed, per-person fare). Replace the flat `toll_params` dict and `MECHANISMS` registry with typed, composable config objects.

## Motivation
The current tolling implementation (`tolling.py`, ~95 lines) works but is clunky to extend:

1. **Signal functions mix "what" with "when"** — `get_flow_signal` manages its own update timing (checking elapsed steps, returning `None`), while `get_volume_signal` has no such gating. Adding a new signal requires deciding whether to bake timing into it.

2. **State management via `hasattr` on the model** — signals stash state (`model.last_toll_update_step`, `model.last_toll_update_car_counter`) using `hasattr` checks. This is fragile and would collide if two mechanisms needed similar state.

3. **Flat `toll_params` dict is a grab-bag** — static toll values (`"car"`, `"bus"`), signal params (`"update_toll_every_n"`), and transform params (`"min_threshold"`, `"slope"`, `"base_price"`) share a single namespace. Adding a new mechanism means stuffing more keys into the same dict.

4. **`signal_args`/`tx_args` lambdas are ceremony** — each mechanism entry has lambda extractors that exist solely because params are jammed into one dict.

5. **Only one transform exists** — `piecewise_linear_tx` is the only option. Adding new transform shapes requires writing a function, a MECHANISMS entry, and the lambda wiring.

6. **Bus toll conflation** — `current_toll_bus` is used both as a person-level mode choice cost (conceptually a user fee) and recorded on the `BusAgent` as `toll_paid` (conceptually a road toll the vehicle pays). Buses will never be tolled; passengers may pay a fixed user fee.

## Design

### Architecture

```
Signal (reads model state, returns a number)
    |
Transform (callable object, maps signal -> raw toll)
    |
Universal Wrappers (rounding, cap, floor — applied after any transform)
    |
Posted Toll (model.current_toll_car, read by person agents)
```

Any signal composes with any transform. Adding a new signal or transform doesn't require touching existing code.

### Signals

Signals are callable objects that read the model and return a numeric value (or `None` if not ready). They own their own state.

**Initial set:**
- `VolumeSignal` — current vehicle count on road (`len(model.vehicles_list)`)
- `FlowSignal(window_steps)` — rolling average vehicle arrival rate over a window

**Future candidates (not in this PR):**
- `SpeedSignal` — average speed of driving vehicles
- `DensitySignal` — vehicles per unit road length

**Interface:**
```python
@dataclass
class VolumeSignal:
    """Current number of vehicles on the road."""
    def __call__(self, model) -> float:
        return len(model.vehicles_list)

@dataclass
class FlowSignal:
    """Rolling average vehicle arrival rate (vehicles per step)."""
    window_steps: int = 300  # 5 minutes at 1 step/sec

    def __post_init__(self):
        self._window = deque(maxlen=self.window_steps)
        self._last_vehicle_count = None

    def __call__(self, model) -> float | None:
        current_count = model.car_counter + model.bus_counter
        if self._last_vehicle_count is None:
            self._last_vehicle_count = current_count
            return None

        spawned = current_count - self._last_vehicle_count
        self._last_vehicle_count = current_count
        self._window.append(spawned)

        if len(self._window) < self.window_steps:
            return None

        return sum(self._window) / len(self._window)
```

Key design choices:
- State lives on the signal object, not on the model (no more `hasattr` checks)
- Signal objects must be **reset between days** since they carry state (see Integration section)
- Return `None` when signal is not ready (e.g., flow window not full) — `update_tolls` preserves current toll in this case

### Transforms

Transforms are callable objects that map a signal value to a raw toll amount. They may be stateless or stateful.

**Initial set:**
- `PiecewiseLinearTransform(threshold, slope, base)` — current behavior
- `StepTransform(threshold, toll)` — binary: $0 below threshold, fixed amount above
- `PITransform(target, kp, ki, toll_min, toll_max)` — feedback controller

**Interface:**
```python
@dataclass
class PiecewiseLinearTransform:
    """Toll = base + slope * (signal - threshold) when signal > threshold, else 0."""
    threshold: float = 100.0
    slope: float = 0.05
    base: float = 5.0

    def __call__(self, signal: float) -> float:
        if signal <= self.threshold:
            return 0.0
        return self.base + self.slope * (signal - self.threshold)


@dataclass
class StepTransform:
    """Flat toll when signal exceeds threshold, else 0."""
    threshold: float = 100.0
    toll: float = 10.0

    def __call__(self, signal: float) -> float:
        return self.toll if signal > self.threshold else 0.0


@dataclass
class PITransform:
    """Proportional-integral feedback controller.

    Drives signal toward a target. Toll accumulates when signal stays
    above target and decreases when signal drops below.
    """
    target: float = 30.0
    kp: float = 0.5
    ki: float = 0.05
    toll_min: float = 0.0
    toll_max: float = 50.0

    def __post_init__(self):
        self._integral = 0.0
        self._prev_error = None

    def __call__(self, signal: float) -> float:
        error = signal - self.target
        self._integral += error

        # Anti-windup: clamp integral so it can't accumulate beyond
        # what would produce toll_min/toll_max by itself
        if self.ki != 0:
            max_integral = self.toll_max / self.ki
            self._integral = max(min(self._integral, max_integral), -max_integral)

        toll = self.kp * error + self.ki * self._integral

        return max(self.toll_min, min(self.toll_max, toll))
```

Key design choices:
- Stateful transforms (PITransform) carry their own state, not the model's
- Stateful transforms must be **reset between days** (see Integration section)
- All transforms return a raw toll; rounding/capping are applied universally afterward
- `toll_min`/`toll_max` on PITransform are transform-specific bounds that interact with anti-windup logic. These are distinct from the universal `cap`/`floor` on TollConfig (though in practice you'd use one or the other, not both)

### TollConfig

The top-level config object that ties signal, transform, and universal wrappers together.

```python
@dataclass
class TollConfig:
    """Complete toll specification: signal + transform + wrappers."""
    signal: Any          # callable(model) -> float | None
    transform: Any       # callable(signal) -> float
    update_every_n_steps: int = 1      # how often to recalculate
    rounding: float | None = None       # round to nearest increment (e.g., 0.10)
    cap: float | None = None            # maximum toll
    floor: float | None = None          # minimum nonzero toll (once threshold crossed)

    @staticmethod
    def static(car: float = 0.0) -> "TollConfig":
        """Convenience: fixed toll, no signal or transform needed."""
        return TollConfig(
            signal=None,
            transform=None,
            _static_toll=car,
        )

    def reset(self):
        """Reset signal/transform state between days."""
        if hasattr(self.signal, '__post_init__'):
            self.signal.__post_init__()
        if hasattr(self.transform, '__post_init__'):
            self.transform.__post_init__()
```

**Static toll alternative:**

The static case (no dynamic pricing) needs a clean path. Two options:

*Option A:* `TollConfig.static(car=10.0)` class method that creates a special-cased config with no signal/transform. `update_tolls` checks for this and returns the fixed value.

*Option B:* Static is just a `ConstantTransform(toll=10.0)` with a dummy signal. No special case needed — it flows through the same pipeline.

**Recommendation:** Option B is cleaner architecturally (no special cases), but Option A is more intuitive for the user. We'll implement Option A with `_static_toll` attribute, since static tolls are the common case and shouldn't require importing signal/transform classes.

### Bus User Fee (Separate Concept)

Buses are never tolled. Passengers may pay a fixed per-person user fee.

**Current state (broken):**
- `model.current_toll_bus` set from `toll_params["bus"]` at init
- Used in `TrafficPersonAgent.decide_mode()` as bus cost in generalized cost function
- Recorded on `BusAgent.toll_paid` (wrong — bus vehicle doesn't pay a road toll)

**New design:**
- **Rename** `model.current_toll_bus` to `model.bus_user_fee` — a fixed per-person fare
- **Set** from a new top-level config parameter `bus_user_fee: float = 0.0` on `SeasonConfig` (not inside TollConfig — it's a separate concept)
- **Remove** `toll_paid` from `BusAgent` (buses don't pay road tolls)
- **Person agent** reads `model.bus_user_fee` in `decide_mode()` as the bus cost
- **Person agent** records `toll_paid = model.bus_user_fee` when riding the bus (for trip log consistency — this is what the person paid, even though the bus didn't)

This is a clean conceptual split:
- `TollConfig` governs **car road tolls** (dynamic, congestion pricing)
- `bus_user_fee` governs **bus fares** (fixed, per person)

## Config Interface

### User-Facing API (in notebooks)

```python
from traffic.model.tolling import TollConfig, FlowSignal, PiecewiseLinearTransform, PITransform

# Static toll
make_season_config(
    toll=TollConfig.static(car=10.0),
    bus_user_fee=0.0,
    ...
)

# Flow-based piecewise linear
make_season_config(
    toll=TollConfig(
        signal=FlowSignal(window_steps=300),
        transform=PiecewiseLinearTransform(threshold=1.0, slope=10.0, base=5.0),
        update_every_n_steps=60,
        rounding=0.25,
        cap=25.0,
    ),
    bus_user_fee=0.0,
    ...
)

# Volume-based PI controller
make_season_config(
    toll=TollConfig(
        signal=VolumeSignal(),
        transform=PITransform(target=30, kp=0.5, ki=0.05, toll_max=25.0),
        update_every_n_steps=60,
        rounding=0.10,
    ),
    bus_user_fee=0.0,
    ...
)
```

### Parameter Sweeps

```python
# Sweep transform params with fixed signal
configs = []
for slope in [0.03, 0.05, 0.10]:
    for threshold in [50, 100, 150]:
        configs.append(make_season_config(
            toll=TollConfig(
                signal=FlowSignal(window_steps=300),
                transform=PiecewiseLinearTransform(
                    threshold=threshold, slope=slope, base=5.0
                ),
                rounding=0.25,
            ),
            ...
        ))

# Sweep across transform types
transforms = [
    PiecewiseLinearTransform(threshold=100, slope=0.05, base=5.0),
    StepTransform(threshold=100, toll=10.0),
    PITransform(target=30, kp=0.5, ki=0.05),
]
configs = [
    make_season_config(
        toll=TollConfig(signal=VolumeSignal(), transform=tx, rounding=0.25),
        ...
    )
    for tx in transforms
]
```

## Implementation Subtasks

### Subtask 1: Build Signal and Transform Classes
**File:** `traffic/model/tolling.py`

Replace the current contents with:
1. Signal classes: `VolumeSignal`, `FlowSignal`
2. Transform classes: `PiecewiseLinearTransform`, `StepTransform`, `PITransform`
3. `TollConfig` dataclass with `.static()` convenience method and `.reset()`
4. `round_toll()` helper function
5. New `update_tolls(model)` function that uses the new architecture

The new `update_tolls` logic:
```python
def update_tolls(model):
    toll_config = model.toll_config

    # Static toll: return fixed value
    if toll_config.signal is None:
        return model.current_toll_car

    # Cadence check
    if model.steps % toll_config.update_every_n_steps != 0:
        return model.current_toll_car

    # Signal
    signal = toll_config.signal(model)
    if signal is None:
        return model.current_toll_car

    # Transform
    raw_toll = toll_config.transform(signal)

    # Universal wrappers
    toll = raw_toll
    if toll_config.floor is not None and toll > 0:
        toll = max(toll, toll_config.floor)
    if toll_config.cap is not None:
        toll = min(toll, toll_config.cap)
    if toll_config.rounding is not None:
        toll = round_toll(toll, toll_config.rounding)

    model.current_toll_car = toll
    return toll
```

**Estimated lines:** ~150-180 (replacing current ~95 lines)

### Subtask 2: Update TrafficModel
**File:** `traffic/model/traffic_model.py`

Changes:
1. Replace `toll_mechanism` and `toll_params` constructor args with `toll_config: TollConfig`
2. Replace `self.toll_mechanism`, `self.toll_params` with `self.toll_config`
3. Initialize `self.current_toll_car` from `toll_config` (static value or 0.0 for dynamic)
4. Replace `self.current_toll_bus` with `self.bus_user_fee`
5. `update_tolls()` method stays the same (delegates to `tolling.update_tolls(self)`)

**Estimated lines changed:** ~15

### Subtask 3: Separate Bus User Fee
**Files:** `traffic/agents/traffic_person_agent.py`, `traffic/agents/bus_agent.py`

Changes in `traffic_person_agent.py`:
1. In `decide_mode()`, change `self.model.current_toll_bus` to `self.model.bus_user_fee`
2. When recording `toll_paid` for bus trips, use `self.model.bus_user_fee`

Changes in `bus_agent.py`:
1. Remove `self.toll_paid = model.current_toll_bus`
2. Bus vehicles have no toll. The person records what they paid.

### Subtask 4: Update SeasonConfig and make_season_config
**File:** `season/configs.py`

Changes:
1. Replace `toll_mechanism: Optional[str]` and `toll_params: Dict` on `SeasonConfig` with `toll_config: TollConfig`
2. Add `bus_user_fee: float = 0.0` to `SeasonConfig`
3. Update `make_season_config()`:
   - Replace `toll_mechanism` and `toll_params` params with `toll: TollConfig = TollConfig.static()`
   - Add `bus_user_fee: float = 0.0` param
4. Pass through to `SeasonConfig`

**Estimated lines changed:** ~15

### Subtask 5: Update SeasonOrchestrator
**File:** `season/season_orchestrator.py`

Changes:
1. In `_build_model()`, pass `toll_config=self.config.toll_config` and `bus_user_fee=self.config.bus_user_fee` instead of `toll_mechanism` and `toll_params`
2. **Reset toll state between days:** call `self.config.toll_config.reset()` at the start of each `run_day()`. This resets signal windows, PI integrals, etc. so each day starts fresh
3. Update `run_day_temp()` if kept — replace `self.config.toll_params['car']` reference

**Estimated lines changed:** ~10

### Subtask 6: Update Reporting
**File:** `traffic/model/reporting.py`

Changes:
1. `"current_toll_car"` reporter stays the same (reads `m.current_toll_car`)
2. Optionally add `"bus_user_fee"` reporter if useful for analysis

**Estimated lines changed:** ~2

### Subtask 7: Update Notebooks
**File:** `notebooks/single_day_season.ipynb`

Update the notebook to use the new config interface. Show examples of:
- Static toll
- Flow-based piecewise linear
- Volume-based PI controller

## Files Modified

| File | Nature of Change |
|------|-----------------|
| `traffic/model/tolling.py` | **Rewrite** — new signal/transform/config classes |
| `traffic/model/traffic_model.py` | Update constructor to accept `TollConfig` + `bus_user_fee` |
| `traffic/agents/traffic_person_agent.py` | `current_toll_bus` -> `bus_user_fee` |
| `traffic/agents/bus_agent.py` | Remove `toll_paid` |
| `season/configs.py` | Replace `toll_mechanism`/`toll_params` with `TollConfig` + `bus_user_fee` |
| `season/season_orchestrator.py` | Pass new config, add `.reset()` between days |
| `traffic/model/reporting.py` | Minor — rename if needed |
| `notebooks/single_day_season.ipynb` | Update config examples |

## Testing & Verification

### Unit Tests: Transforms
```python
def test_piecewise_linear():
    tx = PiecewiseLinearTransform(threshold=100, slope=0.05, base=5.0)
    assert tx(50) == 0.0           # below threshold
    assert tx(100) == 0.0          # at threshold
    assert tx(120) == 6.0          # base + slope * 20
    assert tx(200) == 10.0         # base + slope * 100

def test_step():
    tx = StepTransform(threshold=100, toll=10.0)
    assert tx(99) == 0.0
    assert tx(100) == 0.0
    assert tx(101) == 10.0

def test_pi_transform():
    tx = PITransform(target=30, kp=1.0, ki=0.0, toll_min=0, toll_max=50)
    assert tx(30) == 0.0           # at target
    assert tx(40) == 10.0          # above target
    assert tx(20) == 0.0           # below target, clamped to 0

def test_pi_integral_accumulation():
    tx = PITransform(target=30, kp=0.0, ki=1.0, toll_min=0, toll_max=50)
    tx(35)  # error = 5, integral = 5
    tx(35)  # error = 5, integral = 10
    assert tx(35) == 15.0  # ki * integral = 1.0 * 15

def test_pi_anti_windup():
    tx = PITransform(target=30, kp=0.0, ki=1.0, toll_min=0, toll_max=10)
    for _ in range(100):
        result = tx(1000)  # huge error
    assert result == 10.0  # capped at toll_max
```

### Unit Tests: Signals
```python
def test_volume_signal(mock_model):
    sig = VolumeSignal()
    mock_model.vehicles_list = [1, 2, 3]
    assert sig(mock_model) == 3

def test_flow_signal_returns_none_until_window_full():
    sig = FlowSignal(window_steps=3)
    # simulate 2 steps — not enough
    mock_model.car_counter = 1; mock_model.bus_counter = 0
    assert sig(mock_model) is None  # first call: init
    mock_model.car_counter = 2
    assert sig(mock_model) is None  # window has 1 entry, needs 3

def test_flow_signal_returns_average():
    sig = FlowSignal(window_steps=3)
    # fill window with 3 steps of 1 vehicle each
    # ... (simulate 4 calls: init + 3 data points)
    result = sig(mock_model)
    assert result == 1.0  # avg of [1, 1, 1]
```

### Unit Tests: TollConfig + round_toll
```python
def test_round_toll():
    assert round_toll(3.47, 0.10) == 3.50
    assert round_toll(5.03, 0.10) == 5.00
    assert round_toll(12.46, 0.25) == 12.50
    assert round_toll(0.0, 0.10) == 0.0

def test_static_config():
    tc = TollConfig.static(car=10.0)
    assert tc.signal is None
    assert tc._static_toll == 10.0

def test_cap_and_floor():
    tc = TollConfig(
        signal=VolumeSignal(),
        transform=PiecewiseLinearTransform(threshold=0, slope=1.0, base=0),
        cap=20.0,
        floor=2.0,
    )
    # When signal is 5 -> raw toll is 5.0 -> floor applies? No, 5 > 2, so toll = 5
    # When signal is 1 -> raw toll is 1.0 -> floor applies: toll = 2.0
    # When signal is 25 -> raw toll is 25.0 -> cap applies: toll = 20.0
```

### Integration: Full Day Run
```python
def test_dynamic_toll_changes_during_day():
    """Verify toll updates during simulation with flow-based pricing."""
    config = make_season_config(
        toll=TollConfig(
            signal=FlowSignal(window_steps=60),
            transform=PiecewiseLinearTransform(threshold=0.5, slope=10.0, base=2.0),
            update_every_n_steps=60,
            rounding=0.10,
        ),
        ...
    )
    orch = SeasonOrchestrator(config, store_data=False)
    orch.run_day()
    model_data = orch.last_model_run.datacollector.get_model_vars_dataframe()
    toll_values = model_data["current_toll_car"].unique()
    assert len(toll_values) > 1  # toll changed at least once

def test_static_toll_stays_constant():
    config = make_season_config(toll=TollConfig.static(car=10.0), ...)
    orch = SeasonOrchestrator(config, store_data=False)
    orch.run_day()
    model_data = orch.last_model_run.datacollector.get_model_vars_dataframe()
    assert (model_data["current_toll_car"] == 10.0).all()
```

### Determinism Check
Run `tests/optimization_check.py` before and after to confirm behavioral equivalence for the static toll case (most common existing usage).

## Potential Risks

### Risk 1: Signal/Transform State Across Days
**Issue:** Stateful signals (FlowSignal) and transforms (PITransform) carry state. If not reset between days, day 2 starts with day 1's accumulated integral / flow window.

**Mitigation:** `TollConfig.reset()` called at start of each day in `SeasonOrchestrator.run_day()`. This reinitializes all stateful internals via `__post_init__`.

### Risk 2: Pickle Serialization of TollConfig
**Issue:** `SeasonOrchestrator._save_config()` pickles the `SeasonConfig`. Callable objects with `deque` state need to be picklable.

**Mitigation:** All proposed classes (dataclasses with deque/float state) are picklable by default. No lambdas or closures. Verify with a test.

### Risk 3: PI Transform Tuning Difficulty
**Issue:** PI controllers require careful tuning of Kp and Ki. Bad values cause oscillation or sluggish response.

**Mitigation:** Document recommended starting values. This is inherent to the approach, not a code risk. Could add a "PI tuning guide" notebook later.

### Risk 4: update_every_n_steps Interacts with FlowSignal window
**Issue:** If `update_every_n_steps` > `FlowSignal.window_steps`, the signal may be stale by the time the toll updates.

**Mitigation:** Document the relationship. Optionally add a validation warning in `TollConfig.__post_init__`.

## Implementation Order

1. **Subtask 1:** Build signal/transform/TollConfig classes in `tolling.py` (foundation — everything depends on this)
2. **Subtask 2:** Update `TrafficModel` to accept `TollConfig`
3. **Subtask 3:** Separate bus user fee in person/bus agents
4. **Subtask 4:** Update `SeasonConfig` and `make_season_config`
5. **Subtask 5:** Update `SeasonOrchestrator` (wiring + reset between days)
6. **Subtask 6:** Update reporting
7. **Subtask 7:** Update notebooks
8. **Testing:** Unit tests, integration tests, determinism check

Subtasks 2-4 can be done together as they're tightly coupled. Subtask 1 must come first.

## Future Extensions (Not In This PR)

- **Additional signals:** SpeedSignal, DensitySignal, QueueLengthSignal
- **Additional transforms:** ExponentialTransform, TieredTransform (bracket-based), IncrementalLookupTransform (MnPASS-style)
- **Toll presets:** `TollConfig.preset("aggressive_flow")` for common configurations
- **Per-step toll logging:** Record signal value alongside toll for debugging/analysis
- **Multi-signal transforms:** transforms that accept multiple signals (e.g., density + speed)
- **Time-of-day modulation:** different transform params by hour
