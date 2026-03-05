# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Mesa-based agent-based simulation of vehicle traffic in Little Cottonwood Canyon (LCC), Utah. It models person-level trip decisions, vehicle dynamics, road geometry, tolling, and bus service across single days and multi-day "seasons."

## Running the Model

The primary entry points are Jupyter notebooks (not scripts):
- `notebooks/single_day_season.ipynb` — recommended starting point; runs a one-day season
- `notebooks/season_run.ipynb` — demonstrates multi-day runs with a persistent population

To run notebooks:
```bash
jupyter lab
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Testing

Tests live in `tests/`. Run them directly with Python:

```bash
# Determinism check (quick, ~1000 persons, 1 day)
python tests/optimization_check.py --verify

# Performance tests
python tests/performance_free_flow.py --verify --runs 1
python tests/performance_congestion.py --verify --runs 1

# Seasonal determinism
python tests/season_determinism.py
python tests/day_determinism.py
```

### Optimization Protocol

For perf-only changes that must not alter model behavior:

1. Save baselines before changes:
   ```bash
   python tests/optimization_check.py --save-baseline
   python tests/performance_free_flow.py --save-baseline --runs 1
   python tests/performance_congestion.py --save-baseline --runs 1
   ```
2. Make changes
3. Verify determinism: `python tests/optimization_check.py --verify` — **all outputs must match**
4. Verify performance: run the performance tests with `--verify`
5. Clean up: run each test with `--clean`

Baseline files are stored in `tests/baselines/`.

## Architecture

### Layer Structure

```
season/              # Multi-day orchestration
  configs.py         # SeasonConfig, PopulationParams, make_season_config()
  season_orchestrator.py  # SeasonOrchestrator: runs days, persists outputs
  persons.py         # SeasonPerson dataclass (persistent across days)

traffic/
  model/
    traffic_model.py # TrafficModel (Mesa Model subclass) — core simulation
    generate.py      # Vehicle/bus spawning logic
    init_helpers.py  # Road and agent initialization
    tolling.py       # Signal -> Transform -> TollConfig composable toll system
    hybrid_collector.py  # 4-tier data collection (aggregate/spatial/event/snapshot)
  agents/
    vehicle_agent.py      # Base vehicle physics (speed, braking, movement)
    car_agent.py          # Car-specific behavior
    bus_agent.py          # Bus dispatch and boarding
    traffic_person_agent.py  # Mode choice logic (car vs bus)
    road_segment_agent.py # Road geometry waypoints
    blocker_agent.py      # Road closure blocker
  utils/
    unit_conversion_utils.py
    distribution_utils.py
    analysis_utils.py
    animation_utils.py

collect_external_data/  # Scripts for loading road geometry and vehicle count data
data/                   # Season outputs (season_outputs/<season_id>/)
tests/                  # Determinism and performance tests
plans/                  # Feature planning docs (read before implementing)
notes/                  # Schema diagrams and crash notes
```

### Key Design Decisions

**Road geometry:** `road_gdf` is a GeoDataFrame of evenly spaced points (~50m apart), each a `RoadSegmentAgent`. Vehicles use these as directional waypoints, not exclusive occupancy zones — multiple vehicles can be near the same segment simultaneously.

**Step separation:** `adjust_speed()` and `move_along_path()` are intentionally separate Mesa steps. All agents compute speed first, then all move. This prevents race conditions where earlier movers would affect later agents' decisions in the same step.

**Season persons:** `SeasonPerson` objects persist across days within a season, accumulating travel time experience and updating mode-choice priors. They are created once by `PopulationParams.create_season_persons()` and passed into `TrafficModel` via `season_persons`.

**Tolling:** `TollConfig` composes a `Signal` (reads model state) with a `Transform` (maps signal to toll amount). Use `TollConfig.static(car=X)` for flat tolls or configure `PigouvianTransform` for dynamic tolling.

**Data collection:** `HybridDataCollector` (4 tiers) is the preferred collection system over Mesa's built-in `DataCollector`. Tier 1 = per-step scalars/histograms; Tier 2 = sampled spatial data for animations; Tier 3 = crash/closure events; Tier 4 = full snapshots (off by default).

**Output:** Season outputs go to `data/season_outputs/<season_id>/`. Per-day trip logs and belief state logs are stored as parquet files.

## Development Workflow

**Before any feature:** Create a plan in `plans/feature-name.md` with current state analysis, proposed solution, trade-offs. Wait for approval before coding.

**Branching:**
- `feature/description` — new features
- `fix/description` — bug fixes
- `refactor/description` — refactoring
- `perf/description` — performance optimizations

Always branch from `main` and open a PR; do not commit directly to `main`.
