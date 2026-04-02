# TrafficModel Lifecycle

The `TrafficModel` class in `traffic/model/traffic_model.py` is a Mesa `Model` subclass that manages a single simulated day. This page documents its initialization, step execution order, and termination.

---

## Initialization

`TrafficModel.__init__()` sets up the following in order:

1. **RNG**: `np.random.default_rng(seed)` for deterministic randomness
2. **Agent class registry**: Maps string keys to agent classes (`'car'` → `CarAgent`, etc.)
3. **Season person pool**: Copies the list of `SeasonPerson` objects for this day
4. **Model parameters**: `max_steps`, `max_persons`, `collect_every_n`, `start_hr`, etc.
5. **Toll configuration**: `TollConfig` instance and initial toll value
6. **Driver distributions**: Truncated normals for `acceptable_over`, `ideal_distance_multiplier`, `curve_response`
7. **Acceleration cache**: Pre-computes 101 acceleration functions (one per integer percentile)
8. **Canyon closures**: Converts closure dict to a sorted DataFrame
9. **Bus parameters**: `bus_interval`, `bus_capacity`, randomized first departure step
10. **Crash parameters**: `crashes_per_100k_vmt`, remainder tracker
11. **ContinuousSpace**: Mesa spatial container sized to road bounds + 10,000 unit buffer
12. **Road segments**: `init_road_segments()` creates `RoadSegmentAgent` instances and pre-computes numpy arrays (`rs_pos`, `rs_distance`, `rs_speed_limit`, `rs_curvature`, etc.)
13. **HybridDataCollector**: Initialized with the provided or default `HybridCollectorConfig`
14. **Agent lists**: Empty `vehicles_list`, `blockers_list`, `traffic_persons_list`

---

## Step Execution Order

Each call to `step()` executes these operations in sequence:

### 1. Toll Update
```python
self.update_tolls()
```
Calls `tolling.update_tolls(model)`. For static tolls, this is a no-op. For dynamic tolls, the signal reads model state, the transform maps it to a raw toll, and wrappers (rounding, cap, floor) are applied. Cadence is controlled by `update_every_n_steps`.

### 2. Compute p_generate
```python
if self.traffic_percentile:
    self.p_generate = self.expected_counts_seconds.iloc[self.start_step, self.traffic_percentile]
    self.start_step += 1
```
Indexes into the expected counts schedule for the current time-of-day and traffic percentile. `start_step` advances each step to track simulated time.

### 3. Generate Person
```python
gen.generate_person(self)
```
Bernoulli draw at probability `p_generate`. If successful, picks a `SeasonPerson` from the pool, creates a `TrafficPersonAgent` (which chooses mode), and either spawns a linked `CarAgent` or adds the person to `at_bus_stop`.

### 4. Generate Bus
```python
gen.generate_new_bus(self)
```
Checks if the bus interval timer has fired. If so, creates a `BusAgent`, boards waiting passengers from `at_bus_stop` in FCFS order up to `bus_capacity`.

### 5. Vehicle Physics (the kernel)
```python
self.update_next_agents()          # sort all vehicles+blockers by distance, set next_agent and gap
self.do_adjust_status(all_vehicles) # set status based on next_agent
# filter to driving/slowing vehicles only
self.do_adjust_speed(driving_vehicles)   # phase 1: all compute speed
self.do_move_along_path(driving_vehicles) # phase 2: all move
self.do_calculate_time_lost(all_vehicles) # track delay
```

This is the performance-critical section. The two-phase speed-then-move separation ensures no order-dependent race conditions.

### 6. Crash Computation
```python
self.crashes, self.remainder = self.should_crash_randomized_rounding(...)
```
Uses randomized rounding to convert the continuous crash rate (`crashes_per_100k_vmt`) into discrete crash events. The `remainder` carries fractional crash probability across steps to maintain the correct long-run rate.

### 7. Generate Crash / Canyon Closure
```python
gen.generate_crash(self)
gen.generate_canyon_closure(self)
```
If `crashes > 0`, creates a `BlockerAgent` at a random occupied road segment with a random duration (60--300 seconds). Canyon closures fire when the current step reaches a pre-scheduled `closure_step`.

### 8. Blocker Tick
```python
self.do_tick(self.blockers_list)
```
Decrements each blocker's `self_distruct_timer`. When it reaches 0, the blocker removes itself and resets all vehicles' `next_agent` references.

### 9. Data Collection
```python
self.datacollector.collect(self)
```
The `HybridDataCollector` checks each tier's cadence and collects accordingly: Tier 1 scalars/histograms, Tier 2 spatial snapshots, Tier 3 events (logged separately during generation), Tier 4 full snapshots.

### 10. Termination Checks
```python
self.max_steps_check()
self.max_persons_check()
```
Sets `self.running = False` if either condition is met: max steps reached, or pool exhausted and all persons arrived.

---

## run_model()

The outer loop is straightforward:

```python
def run_model(self):
    for _ in tqdm(range(self.max_steps), desc="Simulating", unit="step"):
        if not getattr(self, "running", True):
            break
        self.step()
```

A `tqdm` progress bar tracks step count. The loop exits early when `self.running` is set to `False` by either termination check.

---

## Termination Conditions

| Condition | Trigger | Meaning |
|-----------|---------|---------|
| Pool exhausted + all arrived | `not season_person_pool and not traffic_persons_list` | Every person has completed their trip |
| Max steps | `steps >= max_steps` | Safety cap (default 50,000 ≈ 14 hours simulated) |

Both checks run at the **end** of each step, after all generation, physics, and collection.
