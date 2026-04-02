# Generation & Lifecycle

This page covers cross-cutting model behavior that doesn't belong to vehicles, roads, or buses alone: how persons and vehicles are spawned, how demand is computed, and when the simulation ends.

---

## Person Spawning Logic

Each step, `generate_person()` in `traffic/model/generate.py` attempts to spawn a new person:

1. **Bernoulli draw**: `random() < p_generate` -- if the draw fails, no person is generated this step
2. **Pick from pool**: `pick_season_person_for_trip()` randomly selects a `SeasonPerson` from `season_person_pool` and removes it
3. **Create agent**: A `TrafficPersonAgent` is created, which immediately calls `decide_mode()` to choose car or bus
4. **Mode-specific action**:
    - **Car**: A `CarAgent` is created and double-linked with the person (`person.vehicle = car`, `car.passengers.append(person)`)
    - **Bus**: The person is added to `model.at_bus_stop` to wait for the next bus

---

## How p_generate Is Computed

The per-step arrival probability comes from one of two sources:

**From traffic_percentile (typical)**: When `traffic_percentile` is set, the model indexes into the expected counts schedule (ECS) -- a CSV with time-varying vehicle counts per second across percentiles:

```python
self.p_generate = self.expected_counts_seconds.iloc[self.start_step, self.traffic_percentile]
self.start_step += 1
```

This means demand varies across the day following real-world traffic patterns. The `start_step` counter advances each step, tracking simulated time-of-day.

**Direct (override)**: If `p_generate` is set directly and `traffic_percentile` is `None`, the probability stays fixed throughout the simulation.

---

## Empty Car Spawning After Pool Exhaustion

When `season_person_pool` is empty (all persons have been assigned), the model doesn't simply stop generating vehicles. If there are still `TrafficPersonAgent` instances in transit (bus passengers not yet arrived), the model continues spawning **empty cars** to maintain realistic traffic conditions:

```python
if season_person is None and any_unfinished:
    # keep generating empty cars to maintain traffic
    new_car = CarAgent.create_agents(model=model, n=1)[0]
    # empty car: no passengers to link
```

This ensures that the last bus passengers don't experience an artificially empty road. Empty cars have all the normal driver traits but carry no passengers and generate no trip records.

The model stops generating entirely only when the pool is empty **and** no persons remain in transit.

---

## Bus Dispatch Timing

Bus generation follows a separate clock from person generation:

1. **First bus**: Dispatches at a random step between 0 and `bus_interval * 60`
2. **Subsequent buses**: Dispatches exactly `bus_interval * 60` steps after the previous one
3. **Continues after pool exhaustion**: Buses keep dispatching as long as there are persons waiting at the bus stop, even if `max_persons` has been reached

```python
# Early exit: only stop if pool is exhausted AND no one is waiting
if model.person_counter >= model.max_persons and not model.at_bus_stop:
    return
```

---

## Model Termination Logic

The simulation ends when **either** of these conditions is met:

### Pool exhausted + all persons arrived
```python
def max_persons_check(self):
    if not self.season_person_pool and not self.traffic_persons_list:
        self.running = False
```

Both conditions must be true: no persons left to assign **and** no persons currently traveling. This ensures every person completes their trip before the model stops.

### Max steps reached
```python
def max_steps_check(self):
    if self.steps >= self.max_steps:
        self.running = False
```

This is a safety cap to prevent runaway simulations. Default: `50,000` steps (~14 hours of simulated time).

Both checks run at the **end** of each step, after all generation, movement, and data collection are complete.
