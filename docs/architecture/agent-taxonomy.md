# Agent Taxonomy

The simulation uses six agent types across two categories: Mesa Agents (participate in the step loop) and data objects (persist across days).

---

## Summary Table

| Agent | Mesa Agent? | Defined in | Created by | Lifetime | Key state |
|-------|:-----------:|------------|------------|----------|-----------|
| **SeasonPerson** | No (dataclass) | `season/persons.py` | `PopulationParams.create_season_persons()` | Entire season | traits, history, beliefs |
| **TrafficPersonAgent** | Yes | `traffic/agents/traffic_person_agent.py` | `generate_person()` | One day (arrival → trip completion) | mode, status, travel times |
| **CarAgent** | Yes | `traffic/agents/car_agent.py` | `generate_person()` when mode=car | Until end of road | driver traits, speed, position |
| **BusAgent** | Yes | `traffic/agents/bus_agent.py` | `generate_new_bus()` | Until end of road | passengers list, fixed traits |
| **BlockerAgent** | Yes | `traffic/agents/blocker_agent.py` | `generate_crash()` / `generate_canyon_closure()` | Until timer expires | self_distruct_timer, blocker_type |
| **RoadSegmentAgent** | Yes | `traffic/agents/road_segment_agent.py` | `init_road_segments()` at model init | Entire day | speed_limit, curvature, position |

---

## SeasonPerson

A **dataclass** (not a Mesa Agent) representing a persistent individual across a season.

**Created once** by `PopulationParams.create_season_persons()` at season start. The same population is passed into each day's `TrafficModel`.

**Core traits** (drawn from distributions):

- `value_of_time`, `experience_weight_car`, `experience_weight_bus`
- `prior_car`, `prior_bus`, `time_decay_rate`, `prior_weight`
- `uncertainty_multiplier`, `travel_propensity`

**Evolving state:**

- `history` -- list of trip records (one per trip taken)
- `expected_tt_car`, `expected_tt_bus` -- recomputed between days
- `travel_time_uncertainty_car`, `travel_time_uncertainty_bus` -- recomputed between days
- `prior_car`, `prior_bus` -- slowly updated after each trip

---

## TrafficPersonAgent

A Mesa Agent representing a person making a single trip on a given day.

**Created** by `generate_person()` when the Bernoulli draw succeeds and a `SeasonPerson` is available.

**At creation:**

- Takes a snapshot of the parent `SeasonPerson`'s beliefs and traits
- Calls `decide_mode()` to choose car or bus via generalized cost comparison

**Linking:**

- If car: double-linked with a new `CarAgent` (`person.vehicle = car`, `car.passengers.append(person)`)
- If bus: added to `model.at_bus_stop` queue, linked to a `BusAgent` at boarding time

**At trip completion:** `tp_to_sp_info_pass()` computes derived metrics (wait time, onboard time, realized cost) and calls `SeasonPerson.record_experience()`.

---

## CarAgent

A Mesa Agent representing a private vehicle. Extends `VehicleAgent`.

**Created** alongside a `TrafficPersonAgent` when mode is car, or as an empty car when the person pool is exhausted.

**Randomized traits** (drawn from model-level truncated normal distributions):

- `acceptable_over` -- overspeed tolerance (~3 mph mean)
- `ideal_distance_multiplier` -- following distance preference (~1.0 mean)
- `curve_responce` -- curvature sensitivity (~0.95 mean)
- `performance` -- acceleration percentile (uniform 0--1)

**Pays** `model.current_toll_car` at creation.

---

## BusAgent

A Mesa Agent representing a scheduled transit bus. Extends `VehicleAgent`.

**Created** by `generate_new_bus()` on a fixed interval schedule.

**Fixed conservative parameters:** `acceptable_over=0`, `performance=0.1`, `ideal_distance_multiplier=2`, `curve_responce=0.9`.

**Passengers:** Boards from `model.at_bus_stop` in FCFS order up to `bus_capacity`. Overrides `vehicle_to_tp_info_pass()` to charge `bus_user_fee` instead of road toll.

---

## BlockerAgent

A Mesa Agent representing a temporary road obstruction.

**Created** by `generate_crash()` (random location, 60--300 second duration) or `generate_canyon_closure()` (scheduled location and duration).

**Key behavior:**

- `self_distruct_timer` counts down each step via `tick()`
- When timer reaches 0, `self_distruct()` removes the blocker and resets all vehicles' `next_agent` references
- On creation, also resets all vehicles' `next_agent` -- this forces gap re-evaluation
- Has `speed=0`, so vehicles behind it brake to a stop

**Types:** `"crash"` or `"canyon_closure"` (stored in `status`).

---

## RoadSegmentAgent

A Mesa Agent representing a static waypoint on the road.

**Created** by `init_road_segments()` at model initialization from the road GeoDataFrame.

**Properties:** `speed_limit`, `curvature`, `road_section`, `distance_traveled`, `linked_coord`.

**Role:** Vehicles use these as directional targets for navigation. The speed limit and curvature at each segment inform driver decisions via the look-ahead averaging in `get_speed_limit()`. Road segments do not actively participate in the step loop -- they are reference data.
