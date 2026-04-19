# Parallel Parameter Sweeps

## Overview

The parallel sweep system runs many independent season configurations across CPU cores to explore parameter spaces efficiently. It builds on the existing `SeasonOrchestrator` without modifying its core logic.

## Object Interaction

```
notebook (sweep_run.ipynb)
  |
  +-- build_sweep_configs(sweep_space, fixed, base_seed)
  |     +-- itertools.product -> list[SeasonConfig]
  |           +-- make_season_config() per combo
  |
  +-- ParallelSweepRunner(configs, max_workers)
        |
        +-- ProcessPoolExecutor
        |     +-- Worker 1: run_one_season(config) -> dict
        |     |     +-- SeasonOrchestrator(config, store_data=False, silent=True)
        |     |           +-- loads road_gdf, ecs_df from paths in config
        |     |           +-- creates SeasonPersons from PopulationParams
        |     |           +-- run_season() -> self.season_summary
        |     |                 +-- per day: _build_model() -> TrafficModel -> run_model()
        |     +-- Worker 2: ...
        |     +-- Worker N: ...
        |
        +-- as_completed() -> collect dicts -> pd.DataFrame
        +-- _save_outputs() -> data/sweep_outputs/sweep_YYYYMMDD_HHMMSS/
```

**Key design points:**

- Each worker is fully independent -- no shared state between processes.
- `SeasonConfig` is the only object that crosses the process boundary (via pickle). All components (`PopulationParams`, `TollConfig`, `ScheduleSpecs`, scipy frozen dists) are confirmed picklable.
- Workers load data files independently from paths stored in `SeasonConfig` (road_gdf ~41 KB, ecs_df ~47 MB per worker).
- No file I/O from workers -- all results flow back as dicts to the main process. The runner writes output once at the end.
- `silent=True` suppresses all tqdm and print output in workers. A single outer progress bar in the runner tracks completions.

## Sweep Collector Tiers

Sweeps should disable unnecessary data collection to minimize per-worker overhead. Three pre-built configs are available in `season/parallel.py`:

| Constant | What's Collected | Size/Season | Use Case |
|----------|-----------------|-------------|----------|
| `SWEEP_SUMMARIES_ONLY` | Day summaries + season summary only (tier1 disabled) | ~50 KB | Fast sweeps, coarse comparison across configs |
| `SWEEP_FULL_TIER1` | Tier 1 scalars, window scalars, histograms every 60 steps | ~500 KB+ | Debugging, deep-dive on specific configs |
| *future: SWEEP_SUMMARIES_PLUS_DISTRIBUTIONS* | Summaries + percentile-based distributional metrics | ~100-200 KB | Richer analysis without full time series |

Tier 2 (spatial), Tier 3 (events), and Tier 4 (snapshots) are **always disabled** in sweeps.

## Data Scale Examples

| Sweep | Configs | Per-Config (summaries) | Total on Disk | Est. Wall Time (10 workers) |
|-------|---------|----------------------|---------------|---------------------------|
| Quick: 3x3 bus_interval x pop, 3 days, 1500p | 9 | ~50 KB | ~450 KB | ~1.5 min |
| Medium: 3x3x2 (+ toll), 10 days, 1500p | 18 | ~50 KB | ~900 KB | ~4 min |
| Large: 5x4x3 (bus x pop x toll), 30 days, 3000p | 60 | ~50 KB | ~3 MB | ~30 min |

For large-population runs (7000 persons), reduce `max_workers` to 6 due to ~1-1.5 GB RAM per worker.

## Where Data Lives

Sweep outputs are automatically saved to `data/sweep_outputs/sweep_YYYYMMDD_HHMMSS/`. Timestamp-based naming ensures older sweeps are never overwritten.

Each sweep directory contains:

| File | Contents |
|------|----------|
| `sweep_results.parquet` | Combined DataFrame -- one row per season config, all season_summary metrics + swept param values |
| `sweep_config.json` | Git commit, branch, dirty flag, max_workers, all serialized `SeasonConfig` dicts |
| `failures.json` | List of `{season_id, error}` (only present if workers failed) |

The `sweep_config.json` stores enough information to reproduce the sweep: the exact git commit the model was on and every `SeasonConfig` used (serialized via `dataclasses.asdict()`).

To reload a past sweep:
```python
df = pd.read_parquet("data/sweep_outputs/sweep_20260409_143000/sweep_results.parquet")
```

## Analysis Workflow

All sweep analysis happens in `notebooks/sweep_run.ipynb`.

The returned DataFrame has columns for:
- Every swept parameter (e.g., `population_size`, `bus_interval_schedule`)
- All `season_summary` metrics (e.g., `avg_tt`, `share_bus`, `total_rev`, `wall_time_seconds`)
- Runtime metadata (`total_steps`, `avg_steps_per_second`, etc.)

Typical analysis patterns:

```python
# Pivot table: avg travel time by bus interval and population
df.pivot_table(values="avg_tt", index="population_size", columns="bus_interval_schedule")

# Filter to specific conditions
df[df["population_size"] == 1500].sort_values("avg_tt")
```

For single-run deep dives (animations, per-step time series), re-run a specific config with `store_data=True` via `notebooks/season_run.ipynb`.

## Memory and Worker Scaling

| Workers | Est. RAM (1500p) | Est. RAM (7000p) | Headroom (32 GB) |
|---------|-----------------|-----------------|------------------|
| 8 | 2.0-3.6 GB | 8-12 GB | Comfortable / Tight |
| 10 | 2.5-4.5 GB | 10-15 GB | Comfortable / Too much |
| 6 | 1.5-2.7 GB | 6-9 GB | Plenty / OK |

Default: `max_workers = min(os.cpu_count() - 2, len(configs))`. Override in `ParallelSweepRunner(configs, max_workers=N)`.
