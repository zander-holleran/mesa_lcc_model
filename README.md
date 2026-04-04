# Little Cottonwood Canyon Traffic Model – README

## Overview and Purpose

This project is an agent-based simulation of vehicle traffic in Little Cottonwood Canyon (LCC), Utah, built on the [Mesa](https://mesa.readthedocs.io/) framework. It combines person-level trip decisions with vehicle dynamics to explore how behavior, road geometry, and policy interventions interact over both single days and multi-day **seasons**.

- **Season concept:** A season strings together multiple simulated days that share a population with persistent preferences and learning parameters. Season-level configuration is defined in `season/configs.py`, and execution is orchestrated through `SeasonOrchestrator` in `season/season_orchestrator.py`.
- **Single-day focus:** The recommended entry point for experimenting with parameters or running analyses is `notebooks/single_day_season.ipynb`, which spins up a one-day season and provides analysis/animation helpers. A longer multi-day example lives in `notebooks/season_run.ipynb`.

The model captures short-to-medium-term traffic patterns (e.g., a single morning) and scales to multi-day experiments (e.g., toll or bus schedule changes across a season), making it suitable for scenario testing and policy exploration.

---

## Project Structure

Key folders and files:

- `notebooks/single_day_season.ipynb`: Primary notebook for running and analyzing a single-day season run (recommended starting point).
- `notebooks/season_run.ipynb`: Example of running multiple days in sequence with the season orchestrator.
- `traffic/`: Core traffic simulation (agents, model, utilities).
- `season/`: Season configuration, person generation, and orchestration logic.
- `collect_external_data/`: Scripts for loading or preprocessing road and vehicle count data.
- `pyproject.toml`: Project metadata and direct dependencies.
- `environment.yml`: Full conda environment snapshot for exact reproduction.

Historical notebooks such as `main.ipynb` or `batch_run.ipynb` are no longer the primary workflow; prefer the season notebooks above for up-to-date examples.

---

## Core Concepts and Mechanics

### Road Geometry

The road is structured from a `GeoDataFrame` (`road_gdf`) of evenly spaced points (~50m apart), each represented by a `RoadSegmentAgent`. These serve **only as reference targets** for vehicles — **vehicles do not “occupy” segments**. Instead, they move along continuous space using segment points as directional waypoints.

- Vehicles are affected by properties (e.g., curvature, speed limit) of the segment they are near.
- `vehicles_here` lists in `RoadSegmentAgent` can contain multiple vehicles because it reflects **estimation**, not exclusivity.

### Vehicle Movement and Speed Control

Each vehicle determines its speed using:

- **Gap-based braking**: If too close to a vehicle ahead, it slows down based on a `prevent_pass` mechanism — no two vehicles end up in the same position in a step.
- **Speed-limit braking**: If over their **implicit speed limit** (a personal interpretation of the posted limit using `acceptable_over`), they apply consistent braking logic.
- **Curve-based braking**: Adjusts speed based on curvature of upcoming road segment and individual `curve_responce`.
- **Acceleration**: Follows an individualized acceleration curve derived from percentile-based vehicle performance data.

> **Note:** Acceleration and braking are not fixed to vehicles. **Random braking variation** occurs each braking instance, simulating inconsistent human responses.

### Separation of Logic Steps

`adjust_speed()` and `move_along_path()` are intentionally separated to preserve logical consistency:

- All agents compute speed decisions first (before any movement).
- Then all agents move, ensuring no agent's motion influences another’s decision in the same step.

This design avoids race conditions where earlier agents in the loop would move and affect the perceived environment of others.

---

## Road Closures

Road segments can start as closed (`road_closed=True`), effectively halting progress beyond them. This is done by setting the segment’s speed limit to 0. Vehicles:

- Will **not be removed** upon encountering a closure.
- Will **pause at the closure point** until the road reopens.
- Resume movement automatically when the segment’s `road_closed` flag is cleared (e.g., due to avalanche clearance).

---

## Vehicle and Bus Generation

Vehicles are generated over time using:

- **Probability-based spawning** (`p_generate`), controlled by a traffic profile or set directly.
- **Control limit** via `max_persons` — not to simulate finite demand but to control the length of model runs (e.g., when testing new features).
- **Bus dispatch** occurs on a fixed schedule. Even if no passengers are waiting, **empty buses are still dispatched** for realism.

### Person Generation

Arrivals each step follow a Bernoulli draw (`p_generate`) until a per-day cap (`max_persons`) is reached. After the cap, the model may inject empty cars to preserve realistic congestion when active trips remain.

Season runs pre-generate a reusable population of `SeasonPerson` records via `PopulationParams` (see `season/configs.py`). Each arrival pulls the next SeasonPerson from a shuffled draw order and creates a `TrafficPersonAgent` that:

- Chooses a mode using generalized cost comparisons that combine expected travel time, a penalty for travel-time uncertainty, the person’s value of time, experience weights for car vs. bus, and any tolls in effect.
- If the mode is **car**, spawns a linked `CarAgent`; if **bus**, joins the global bus-stop queue until the next scheduled bus boards passengers.
- On trip completion, writes realized travel time, tolls, and cumulative delay back to the SeasonPerson, which updates its mode-specific priors and uncertainty for future days.

---

## Model Output

Two primary forms of output:

1. **DataCollector**:
   - Tracks metrics like average travel time, speed vs. limits, total cars/buses, etc.
   - Stored as a time-indexed DataFrame.

2. **`finished_agents`**:
   - List of dictionaries describing every vehicle that reached the end of the road or completed its journey.
   - Fields include:
     - `steps_taken`, `distance_traveled`, `car_interactions`
     - Driver traits: `acceptable_over`, `performance`, `curve_responce`

Season runs also capture per-day outputs to `data/season_outputs/<season_id>/`, enabling cross-day comparisons for a single population.

---

## How to Run

The fastest way to explore the model is through the season notebooks:

1. **Single day (recommended):** Open `notebooks/single_day_season.ipynb` and run cells in order. The notebook constructs a one-day season via `make_season_config(...)` and then runs `SeasonOrchestrator.run_day()`. You can manipulate:
   - **Demand and operations:** `traffic_percentile_schedule`, `bus_interval_schedule`, `n_days` (even if running one day), `max_persons`, `collect_every_n`.
   - **Policy levers:** `toll_mechanism` and `toll_params` for static or pigouvian tolls.
   - **Timing:** `start_hr` for morning start times and `canyon_closures_schedule` for forced closures.
   - **Data sources:** `road_path` and `ecs_path` for the road network and expected vehicle counts.
   - **Population characteristics:** `PopulationParams` (size, bus priors, seeds) to control the season’s reusable persons.

2. **Multi-day season:** `notebooks/season_run.ipynb` demonstrates running multiple days in sequence with the same population using `SeasonOrchestrator.run_season()`. Start here when testing interventions that require several simulated days.

If you need to drive the model directly (outside notebooks), the core API is exposed via `TrafficModel` in `traffic/model/traffic_model.py`.

---

## Summary

This Mesa model offers a flexible and empirically grounded framework for studying traffic behavior in constrained mountain road systems like Little Cottonwood Canyon. By combining per-person decision making with persistent season populations and realistic vehicle dynamics, it enables rich exploration of congestion, policy interventions, and system behavior across both single days and full seasons.
