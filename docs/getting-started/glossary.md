# Glossary

A complete reference for domain-specific terminology used throughout the LCC Traffic Model codebase.

---

## Simulation Hierarchy

Step
:   One tick of the simulation clock, representing **1 second** of simulated time. Each step executes: toll update, person/bus generation, vehicle physics (speed adjustment + movement), crash generation, blocker tick, and data collection.

Day
:   A single simulation run within a season. Each day is one `TrafficModel` execution. Configured via a `DayParams` object that specifies traffic demand, bus interval, crash rate, and closures for that day.

Season
:   A sequence of simulated days sharing a **persistent population**. Persons accumulate travel experience and update their beliefs between days. Configured via `SeasonConfig`, executed by `SeasonOrchestrator`. Defined in `season/`.

---

## Agent Types

SeasonPerson
:   A **dataclass** (not a Mesa Agent) representing an individual person that persists across all days within a season. Holds core traits (`value_of_time`, experience weights, priors, decay rates) and accumulates a full history of trips taken. Beliefs about expected travel time for each mode are recomputed between days via `update_beliefs_from_history()`. Defined in `season/persons.py`.

TrafficPersonAgent
:   A **Mesa Agent** created when a person arrives on a given day. Takes a snapshot of its parent `SeasonPerson`'s beliefs and traits, then chooses a mode (car vs. bus) via `decide_mode()` using a generalized cost comparison. On trip completion, passes realized trip data back to the `SeasonPerson` via `tp_to_sp_info_pass()`. Defined in `traffic/agents/traffic_person_agent.py`.

CarAgent
:   A **Mesa Agent** representing a single private vehicle. Extends `VehicleAgent`. Has randomized driver traits drawn from model-level distributions: `acceptable_over`, `curve_responce`, `performance`, `ideal_distance_multiplier`. Pays the current road toll at time of creation. Defined in `traffic/agents/car_agent.py`.

BusAgent
:   A **Mesa Agent** representing a scheduled transit bus. Extends `VehicleAgent`. Uses fixed conservative driving parameters: `acceptable_over=0`, `performance=0.1`, `ideal_distance_multiplier=2`, `curve_responce=0.9`. Passengers board from the `model.at_bus_stop` queue. Buses do not pay road tolls; instead, each passenger pays the `bus_user_fee`. Defined in `traffic/agents/bus_agent.py`.

BlockerAgent
:   A **Mesa Agent** representing a temporary road obstruction -- either a crash or a canyon closure. Has a `self_distruct_timer` that counts down each step; the blocker is removed when it reaches zero. On creation and destruction, all vehicles reset their `next_agent` reference to force re-evaluation of gaps. Defined in `traffic/agents/blocker_agent.py`.

RoadSegmentAgent
:   A **Mesa Agent** representing a single waypoint on the road. Properties include `speed_limit`, `curvature`, `road_section`, and `distance_traveled`. Vehicles use these as **directional targets**, not exclusive occupancy zones -- multiple vehicles can be near the same segment simultaneously. Defined in `traffic/agents/road_segment_agent.py`.

---

## Person & Belief Parameters

value_of_time
:   Dollar value per minute of travel time. Used in the generalized cost calculation to convert travel time into a monetary cost. Default distribution: `lognorm(s=0.64, scale=40/60)` (~$0.67/min median). Defined on `PopulationParams` in `season/configs.py`.

experience_weight_car / experience_weight_bus
:   Multiplier on the travel-time component of generalized cost for each mode. Car default is `1.0` (scalar). Bus default is drawn from `skewnorm(6, loc=1.15, scale=0.3)`, reflecting higher perceived disutility of bus travel time. Used in `compute_expected_generalized_cost()`.

prior_car / prior_bus
:   Initial expected travel time (minutes) for each mode **before any experience**. Defaults: `prior_car=22.0`, `prior_bus=60.0`. Updated slowly after each trip via `slow_prior_update(prior, realized_tt, eta=0.05)`.

uncertainty_multiplier
:   Personal risk-aversion factor that scales the travel-time uncertainty penalty in mode choice. In the generalized cost formula, `effective_tt = expected_tt - uncertainty_multiplier * uncertainty`. Higher values make uncertain modes less attractive. Default: `1.0`.

time_decay_rate
:   Exponential decay rate for weighting past experiences. Each observation's weight is `exp(-time_decay_rate * age_in_days)`. Higher values discount older trips more heavily, creating a shorter memory. Default: `0.1`.

prior_weight
:   Strength of the prior relative to observed data when computing expected travel time. The expected travel time blends prior and data: `(prior_weight * prior + W * mu_data) / (prior_weight + W)` where `W` is the sum of time-decayed observation weights. Default: `1.0`.

travel_propensity
:   Frequency or willingness to make trips. Used in population draw ordering. Default: `1.0`.

expected_tt_car / expected_tt_bus
:   A person's current expected travel time (minutes) for each mode. Computed by `compute_experience_beliefs()` as a weighted average of the prior and exponentially time-decayed past trip observations. Recomputed at the start of each day.

travel_time_uncertainty_car / travel_time_uncertainty_bus
:   Perceived variability in travel time for each mode. Base uncertainty is `1 / total_weight` (inversely proportional to experience). Inflated by staleness: `base_unc * (1 + staleness_scale * staleness)` where `staleness` is days since the last trip on that mode.

---

## Vehicle & Driver Parameters

acceptable_over
:   How many mph above the posted speed limit a driver considers acceptable. The driver's **implicit speed limit** is `speed_limit + acceptable_over`. Drawn from a truncated normal distribution (mean ~3 mph) for cars. Buses use `0`. Defined in `traffic/agents/car_agent.py`.

curve_responce
:   How aggressively a driver slows for road curvature (note: original spelling retained). Drawn from a truncated normal distribution for cars. Buses use `0.9`. Higher values produce more speed reduction on curves. Defined in `traffic/agents/car_agent.py`.

performance
:   A percentile value (0--1) indexing into the acceleration curve cache. Each vehicle gets a unique acceleration function based on this value. Higher percentile = faster acceleration. Buses use `0.1`. Drawn uniformly for cars via `rng.random()`. Defined in `traffic/agents/car_agent.py`.

ideal_distance_multiplier
:   Multiplier on the baseline desired following distance. Higher values = more cautious following. Drawn from a truncated normal distribution (mean ~1.0) for cars. Buses use `2.0`. Defined in `traffic/agents/car_agent.py`.

accel_curve
:   A function mapping current speed (mph) to acceleration (m/s^2). Generated by `build_empirical_accel_function()` in `vehicle_agent.py` using piecewise segments derived from real-world stop-sign acceleration data with skew-normal distributions. The model pre-computes 101 curves (one per integer percentile 0--100) in `accel_curve_cache`.

---

## Generation Parameters

p_generate
:   Per-step probability of spawning a new person (Bernoulli parameter). Either set directly or derived from `traffic_percentile` by indexing into the expected counts schedule (ECS). Range: 0--1, where each step represents 1 second.

max_persons
:   Cap on total persons generated in a day. When the pool is exhausted, the model may continue spawning empty cars to maintain realistic congestion while active bus passengers are still in transit.

traffic_percentile
:   Selects a column from the expected vehicle counts CSV, controlling demand intensity. For example, 50th percentile = typical day, 90th percentile = heavy day. The selected column provides time-varying `p_generate` values across the simulation.

season_person_pool
:   List of pre-allocated `SeasonPerson` objects ready for assignment to trips. Size is `min(population_size, max_persons)`. Persons are removed one-by-one as they are assigned to `TrafficPersonAgent` instances.

at_bus_stop
:   Queue of `TrafficPersonAgent` instances waiting for the next bus. Persons who choose bus mode are added here at creation. When a bus spawns, it boards passengers from this queue in FCFS order up to `bus_capacity`.

---

## Tolling System

TollConfig
:   Complete toll specification composing a Signal, Transform, and optional wrappers. For static tolls, use `TollConfig.static(car=10.0)`. For dynamic tolls, provide a `signal` and `transform`. Wrappers include `rounding`, `cap`, `floor`, and `update_every_n_steps`. Resets signal/transform state between days via `reset()`. Defined in `traffic/model/tolling.py`.

Signal
:   A callable that reads model state and returns a numeric value (or `None` if not ready). Fed into a Transform to determine the toll. Two built-in signals exist: `VolumeSignal` and `FlowSignal`.

Transform
:   A callable that maps a signal value to a raw toll amount. Three built-in transforms exist: `PiecewiseLinearTransform`, `StepTransform`, and `PITransform`.

VolumeSignal
:   Returns the current number of active vehicles on the road: `len(model.vehicles_list)`. Always returns a value immediately. Defined in `traffic/model/tolling.py`.

FlowSignal
:   Returns a rolling average of vehicle arrivals per step over a configurable window. Default window: `300` steps (5 minutes at 1 step/sec). Returns `None` until the window is full. Defined in `traffic/model/tolling.py`.

PiecewiseLinearTransform
:   `toll = base + slope * (signal - threshold)` when `signal > threshold`, else `0`. Parameters: `threshold` (default 100), `slope` (default 0.05), `base` (default 5.0). Defined in `traffic/model/tolling.py`.

StepTransform
:   Binary toll: flat `toll` amount when `signal > threshold`, else `0`. Parameters: `threshold` (default 100), `toll` (default 10.0). Defined in `traffic/model/tolling.py`.

PITransform
:   Proportional-integral feedback controller that drives the signal toward a `target`. Toll accumulates when the signal exceeds the target and resets when it falls below (if `reset_integral_on_target=True`). Includes anti-windup clamping. Parameters: `target`, `kp`, `ki`, `toll_min`, `toll_max`. Defined in `traffic/model/tolling.py`.

current_toll_car
:   The active toll ($/vehicle) charged to cars at the current step. Updated by `update_tolls()` each step based on the `TollConfig`. Attribute on `TrafficModel`.

bus_user_fee
:   Fee ($/trip) charged to bus passengers instead of the vehicle toll. Set in `SeasonConfig`. Default: `0.0`.

---

## Data Collection (HybridDataCollector)

HybridCollectorConfig
:   Configuration dataclass controlling the 4-tier data collection system. Each tier can be independently enabled/disabled with its own cadence and limits. Defined in `traffic/model/hybrid_collector.py`.

Tier 1 -- Aggregate Metrics
:   Per-step scalar values and histograms collected at a configurable interval (`tier1_interval`). Scalars include: step, current toll, vehicle count, active cars/buses, bus riders waiting, mode share, finished count, recent travel time average. Histograms include speed distribution and speed-limit delta. Pre-allocated numpy arrays for performance.

Tier 2 -- Sampled Spatial Data
:   Agent position and state snapshots at regular intervals (`tier2_sample_interval`). Captures position, speed, status, gap, ideal gap, and driving action for up to `tier2_max_agents_per_sample` vehicles per sample. Used for generating animations.

Tier 3 -- Event Log
:   Discrete event records for crashes and canyon closures. Fields include event type, step, segment index, position, duration, and vehicles on road at the time. Always lightweight.

Tier 4 -- Full Snapshots
:   Complete model state dumps at configurable intervals or on crash events. Disabled by default (`tier4_enabled=False`) to prevent memory bloat. Limited by `tier4_max_snapshots`.

collect_every_n
:   How often to collect Tier 2 spatial data, in steps. Default: `10`. Higher values reduce memory usage and animation resolution.

---

## Vehicle Dynamics

gap
:   Current following distance in meters to the next entity ahead (vehicle or blocker). Determines whether gap-based braking (prevent-pass) is triggered.

ideal_gap
:   Desired following distance in meters, based on current speed and `ideal_distance_multiplier`. When `gap < ideal_gap`, the vehicle begins braking.

driving_action
:   Encoded integer describing the vehicle's current acceleration/deceleration state: accelerate (0), coast (1), slow_accel (2), speed_limit_brake (3), smooth_brake (4), prevent_pass (5).

status
:   Vehicle operational state: driving (0), slowing (1), crash (2), canyon_closure (3), arrived (4).

implicit_sl
:   The driver's personal speed limit: `posted_speed_limit + acceptable_over`. The vehicle brakes when exceeding this value and accelerates when below it.

cumtime_lost
:   Cumulative seconds a vehicle spends below a threshold speed due to congestion or braking. Measures delay attributable to traffic conditions rather than road geometry.

---

## Trip Output Terms

realized_tt
:   Total travel time for a completed trip in minutes. Equals `wait_time + onboard_time`. Recorded to the `SeasonPerson`'s history for belief updating.

realized_cost
:   Generalized cost of the completed trip in dollars. Calculated as `(total_travel_time * value_of_time * experience_weight) + toll_paid`.

wait_time
:   Minutes between person creation and vehicle boarding. For car drivers, this is typically near zero. For bus riders, this reflects time spent at the bus stop.

onboard_time
:   Minutes from boarding (vehicle departure) to arrival at destination. Calculated as `(arrive_step - board_step) / 60`.

toll_paid
:   Total toll charged for the trip in dollars. For cars: the `current_toll_car` at time of vehicle creation. For bus passengers: the `bus_user_fee`.

cumtime_lost_sec
:   Cumulative delay in seconds, transferred from the vehicle to the `TrafficPersonAgent` at trip completion. Converted to minutes as `cumtime_lost_min` when recorded to the `SeasonPerson`.

---

## Configuration Objects

SeasonConfig
:   Master configuration object for a multi-day season run. Contains season-level `TrafficModel` parameters (`max_steps`, `max_persons`, `collect_every_n`, `start_hr`, `bus_capacity`), road/data paths, toll configuration, per-day parameters (`day_params` list), population parameters, and collector configuration. Created by `make_season_config()`. Defined in `season/configs.py`.

DayParams
:   Per-day parameter specification containing `day_index`, `day_seed`, `traffic_percentile`, `bus_interval`, `crashes_per_100k_vmt_input`, and optional `canyon_closures`. Converted to `TrafficModel` keyword arguments via `to_model_kwargs()`. Defined in `season/configs.py`.

PopulationParams
:   Defines distributions for drawing `SeasonPerson` traits. Each field (e.g., `value_of_time`, `experience_weight_bus`) can be a scalar or a frozen scipy distribution with a `.rvs()` method. Generates the full population via `create_season_persons()`. Defined in `season/configs.py`.

ScheduleSpecs
:   Specifies how a day-varying parameter is realized across days within a season. Three modes: `static` (same value each day), `dist` (random draw from a scipy distribution each day), `list` (explicit per-day values). Realized via `realize(rng, n_days)`. Defined in `season/configs.py`.

make_season_config()
:   Factory function that assembles a `SeasonConfig` from high-level inputs. Takes schedule specs, realizes per-day values, builds the `DayParams` list, and returns a complete `SeasonConfig`. Defined in `season/configs.py`.
