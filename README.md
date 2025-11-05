
# Little Cottonwood Canyon Traffic Model – README 

## Overview and Purpose

This project is an agent-based simulation of vehicle traffic in Little Cottonwood Canyon (LCC), Salt Lake City, using the [Mesa](https://mesa.readthedocs.io/) framework. The model aims to simulate how individual driving behavior, road geometry, and traffic interventions interact to produce system-wide effects like congestion or smooth flow.

Unlike equation-based models, Mesa allows for **heterogeneous agents** (cars, buses, etc.) that each follow individual rules and parameters. Traffic has **emergent properties**, and agent-based simulation is especially suited to capturing these dynamic interactions.

The current model accurately captures short-to-medium-term traffic patterns (e.g., simulating a few hours of canyon traffic) and is ready for scenario testing (e.g., adding bus lines, changing driver behavior, modeling road closures).

---

## Project Structure

Key folders and files:

- `main.ipynb`: Interactive demo for launching and inspecting the model.
- `batch_run.ipynb`: Run multiple scenarios or parameter sweeps.
- `agents/`: Agent classes (`CarAgent`, `BusAgent`, `RoadSegmentAgent`, etc.).
- `model/`: Core model logic (`TrafficModel`, agent generation, data collection).
- `utils/`: Helper functions for animation, unit conversions, and distributions.
- `other_notebooks/`: Data preparation and analysis notebooks (e.g. acceleration profiling).
- `README_UPDATED.md`: This file.

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

Each person chooses to drive or wait for a bus:

- Decision is based on a `car_preference` probability or bus stop crowding.
- If driving, a new `CarAgent` is created.
- If waiting, they are counted in `at_bus_stop` and ride the next bus up the canyon.

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

---

## How to Run

Basic workflow:
```python
from mesa_lcc_model.model.traffic_model import TrafficModel
model = TrafficModel(
    road_gdf=road_gdf,
    ecs_df=ecs_df,
    max_steps=36000,
    traffic_percentile=50,
    car_preference=0.7,
    bus_interval=15,
    bus_capacity=30,
    canyon_open_step=18000,
    closed_sections={4}
)
model.run_model()

df = model.datacollector.get_model_vars_dataframe()
results = pd.DataFrame(model.finished_agents)
```

To visualize or animate the output, see `animation_utils.py`.

---

## Technical Notes for Developers

- **Position Tracking**: Vehicles move in continuous space and use road segment points only for navigation logic.
- **Randomness**: Only some randomness is persistent (e.g., `acceptable_over`). Braking variation is random **each instance**, not per vehicle.
- **Extending Behavior**: Easily subclass agents or add new ones (e.g., `TruckAgent`).
- **Closure logic**: Vehicles pause at closures; no need to simulate disappearance or rerouting.
---

## Summary

This Mesa model offers a highly flexible and empirically grounded framework for studying traffic behavior in constrained mountain road systems like Little Cottonwood Canyon. By carefully modeling driver heterogeneity and road features, it enables rich exploration of congestion, policy interventions, and vehicle interactions with a strong balance of realism and performance.
