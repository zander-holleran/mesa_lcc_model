# Bus Service

The simulation models a fixed-route bus service running on the same road as private vehicles. Buses follow the same physics as cars but with conservative driving parameters.

---

## Dispatch Scheduling

Buses are dispatched at fixed intervals controlled by `bus_interval` (minutes). Key behavior:

- **First departure** is randomized: a random step between 0 and `bus_interval * 60` seconds
- **Subsequent departures** occur exactly `bus_interval * 60` steps apart
- **Empty buses are dispatched** even when no passengers are waiting -- this maintains realistic road conditions and ensures buses are positioned for future riders
- When `bus_interval = 0`, bus service is disabled entirely

```python
# From generate.py
self.bus_first_departure = self.random.randint(0, self.bus_interval * 60)
self.next_bus_step = self.bus_first_departure
# After each dispatch:
model.next_bus_step += model.bus_interval * 60
```

---

## Boarding

Persons who choose bus mode are added to `model.at_bus_stop` -- a global queue. When a bus spawns, it boards passengers in **FCFS (first-come, first-served)** order up to `bus_capacity`:

```python
n_board = min(capacity, len(model.at_bus_stop))
boarding_passengers = model.at_bus_stop[:n_board]
model.at_bus_stop = model.at_bus_stop[n_board:]
```

Each boarding passenger gets linked to the bus:

- `passenger.vehicle = new_bus` -- the person knows which bus they're on
- `new_bus.passengers.append(passenger)` -- the bus knows its riders
- `passenger.board_step = model.steps` -- records when boarding occurred

Passengers who don't board (queue overflow) wait for the next bus.

---

## Bus-Person Relationships

The bus-person relationship is a **double link**:

- Each `TrafficPersonAgent` with `mode="bus"` has a `.vehicle` reference pointing to their `BusAgent`
- Each `BusAgent` has a `.passengers` list containing all boarded `TrafficPersonAgent` instances

This mirroring is important for the arrival handoff. When the bus reaches the end of the road, `BusAgent.vehicle_to_tp_info_pass()` iterates over all passengers and transfers trip data:

```python
def vehicle_to_tp_info_pass(self):
    for tp in self.passengers:
        tp.toll_paid = self.model.bus_user_fee  # passengers pay user fee, not road toll
        tp.board_step = self.created_at_step
        tp.arrive_step = self.model.steps
        tp.cumtime_lost_sec = self.cumtime_lost_sec
        tp.tp_to_sp_info_pass()  # triggers belief update on SeasonPerson
```

Note that `BusAgent` **overrides** the base `VehicleAgent.vehicle_to_tp_info_pass()` to charge the `bus_user_fee` instead of the vehicle toll.

---

## Driving Characteristics

Buses use fixed conservative parameters (not drawn from distributions):

| Parameter | Value | Compared to cars |
|-----------|-------|-----------------|
| `acceptable_over` | 0 | Cars: ~3 mph mean |
| `performance` | 0.1 | Cars: uniform 0--1 |
| `ideal_distance_multiplier` | 2.0 | Cars: ~1.0 mean |
| `curve_responce` | 0.9 | Cars: ~0.95 mean |
| `toll_paid` | 0.0 | Cars: `current_toll_car` |

This makes buses slower to accelerate, more cautious in following distance, and strictly adherent to speed limits.

---

## Bus User Fee

Bus passengers pay a `bus_user_fee` (set in `SeasonConfig`) instead of the road toll. This fee enters the generalized cost calculation when a person is deciding between car and bus -- higher bus fees make bus less attractive relative to car.

Default: `0.0` (free bus service).
