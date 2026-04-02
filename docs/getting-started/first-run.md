# First Run

This guide walks through `notebooks/season_run.ipynb` -- the recommended entry point for running a multi-day season simulation with a persistent population.

---

## Step 1: Setup & Data Prep

The first two cells handle working directory detection and data file preparation:

```python
# Cell 1: Set project root via git
PROJECT_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)
os.chdir(PROJECT_ROOT)
```

```python
# Cell 2: Download/verify external data
from collect_external_data.expected_counts import get_expected_counts
from collect_external_data.road_geom import get_road_geometry

get_road_geometry()
get_expected_counts()
```

These functions check if the road geometry and vehicle count data exist locally, and download them if not.

---

## Step 2: Imports

The notebook imports from three layers of the project:

```python
# Season orchestration
from season.persons import SeasonPerson
from season.configs import (ScheduleSpecs, SeasonConfig, DayParams,
                            make_season_config, PopulationParams)
from season.season_orchestrator import SeasonOrchestrator

# Data collection
from traffic.model.hybrid_collector import HybridCollectorConfig

# Tolling system
from traffic.model.tolling import (
    TollConfig, VolumeSignal, FlowSignal,
    PiecewiseLinearTransform, StepTransform, PITransform
)
```

See the [Glossary](glossary.md) for definitions of each of these.

---

## Step 3: Schedule Specifications

`ScheduleSpecs` controls how day-varying parameters are realized across the season. Three modes are available:

| Mode | Description | Example |
|------|-------------|---------|
| `static` | Same value every day | `ScheduleSpecs("static", value=30)` |
| `dist` | Random draw from a scipy distribution each day | `ScheduleSpecs("dist", dist=norm(50, 5))` |
| `list` | Explicit per-day values (length must match `n_days`) | `ScheduleSpecs("list", value=[80, 82, 85])` |

The notebook defines three schedules:

```python
traffic_percentile_schedule = ScheduleSpecs(
    mode='list',
    value=[80, 82, 85],  # increasing demand across 3 days
)

bus_interval_schedule = ScheduleSpecs(
    mode='static',
    value=30,  # bus every 30 minutes
)

crashes_schedule = ScheduleSpecs(
    mode='static',
    value=0,  # no crashes
)
```

---

## Step 4: Building the Season Config

`make_season_config()` is the factory function that assembles a complete `SeasonConfig`. Here is the notebook's configuration with annotations:

```python
config = make_season_config(
    # --- Identity ---
    season_id='speed_test2',
    run_description='',
    seed=33,

    # --- Scale ---
    n_days=3,
    max_steps=99999,
    max_persons=99999,       # no artificial person cap
    collect_every_n=60,      # Tier 2 spatial data every 60 steps

    # --- Time ---
    start_hr=8,              # simulation begins at 8:00 AM

    # --- Bus ---
    bus_capacity=60,

    # --- Data paths ---
    road_path='data/roads/hw210_sl_and_curvs.parquet',
    ecs_path='data/vehicle_counts/expected_counts_seconds.csv',

    # --- Population ---
    population_params=PopulationParams(
        population_size=3000,
        prior_car=22.0,      # initial expected car travel time (min)
        prior_bus=40.0,      # initial expected bus travel time (min)
        time_decay_rate=0.1,
        prior_weight=1.0,
        uncertainty_multiplier=1.0,
    ),

    # --- Tolling ---
    toll=TollConfig(
        signal=VolumeSignal(),
        transform=PITransform(
            target=300, kp=0.5, ki=0.05,
            toll_min=0, toll_max=50
        ),
        update_every_n_steps=60,
        rounding=0.10,
    ),
    bus_user_fee=0.0,

    # --- Data collection ---
    hybrid_collector_config=HybridCollectorConfig(
        max_steps=100000,
        tier1_enabled=False,
        tier2_enabled=False,
        tier3_enabled=False,
        tier4_enabled=False,
        # ... (tier-specific settings)
    ),

    # --- Schedules ---
    traffic_percentile_schedule=traffic_percentile_schedule,
    bus_interval_schedule=bus_interval_schedule,
    crashes_schedule=crashes_schedule,
    canyon_closures_schedule=None,
)
```

!!! tip "Quick experiments"
    For faster runs, reduce `population_size` and `max_persons` to ~500, and set `n_days=1`.

---

## Step 5: Toll Strategy Examples

The notebook includes 5 example toll configurations. Swap any of these into the `toll=` parameter:

**1. Static toll** -- fixed price, no signal needed:
```python
toll=TollConfig.static(car=10.0)
```

**2. Volume-based piecewise linear** -- toll scales with vehicle count:
```python
toll=TollConfig(
    signal=VolumeSignal(),
    transform=PiecewiseLinearTransform(threshold=100, slope=0.05, base=5.0),
    update_every_n_steps=60,
    rounding=0.25,
)
```

**3. Flow-based piecewise linear** -- toll scales with arrival rate:
```python
toll=TollConfig(
    signal=FlowSignal(window_steps=300),  # 5-min rolling window
    transform=PiecewiseLinearTransform(threshold=1.0, slope=10.0, base=2.0),
    update_every_n_steps=60,
    rounding=0.25,
    cap=25.0,
)
```

**4. Volume-based step toll** -- binary on/off:
```python
toll=TollConfig(
    signal=VolumeSignal(),
    transform=StepTransform(threshold=100, toll=10.0),
    update_every_n_steps=60,
)
```

**5. Volume-based PI controller** -- feedback-driven:
```python
toll=TollConfig(
    signal=VolumeSignal(),
    transform=PITransform(target=300, kp=0.5, ki=0.05, toll_min=0, toll_max=50),
    update_every_n_steps=60,
    rounding=0.10,
)
```

See [Tolling System](../architecture/tolling-system.md) for detailed documentation of each component.

---

## Step 6: Run the Season

```python
orchestrator = SeasonOrchestrator(season_config=config, store_data=True)
orchestrator.run_season()
```

This:

1. Creates the population of `SeasonPerson` objects from `PopulationParams`
2. Iterates over each day in `day_params`
3. Updates person beliefs between days
4. Builds and runs a `TrafficModel` for each day
5. Persists outputs to `data/season_outputs/<season_id>/`

---

## Step 7: Reading Outputs

After the run, outputs are saved as parquet files:

```python
parquet_path = PROJECT_ROOT / "data" / "season_outputs" / "speed_test2" / "day_0_model_ts.parquet"
df_day0 = pd.read_parquet(parquet_path)
```

The output directory contains:

| File | Contents |
|------|----------|
| `day_N_model_ts.parquet` | Tier 1 time series for day N |
| `trip_log.parquet` | Per-trip records across all days |
| `day_summary.parquet` | Aggregate metrics per day |
| `season_person_log.parquet` | Person belief snapshots after each day |
| `sp_day_summary.parquet` | Person-day aggregate summaries |
| `config.json` | The season configuration used |

You can also access the last model run directly:

```python
orchestrator.last_model_run  # the TrafficModel instance from the final day
```

---

## What to Try Next

- **Increase demand**: Change `traffic_percentile_schedule` to higher values (e.g., 90+)
- **Enable crashes**: Set `crashes_schedule = ScheduleSpecs("static", value=5)` for ~5 crashes per 100k VMT
- **Add canyon closures**: Use `canyon_closures_schedule` to simulate avalanche control
- **Try different tolling**: Swap in any of the 5 toll configurations above
- **Enable data collection**: Set `tier1_enabled=True` (and other tiers) in the `HybridCollectorConfig` to capture detailed metrics
- **Run more days**: Increase `n_days` and watch beliefs evolve across the season
