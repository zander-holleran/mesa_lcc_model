# Parallel Parameter-Sweep Runner — Research Findings

**Date:** 2026-04-08  
**Phase:** Research  
**Status:** Complete — ready for planning phase

---

## 1. Parallelization Strategy

### Recommendation: `concurrent.futures.ProcessPoolExecutor`

| Option | Verdict | Rationale |
|--------|---------|-----------|
| **ProcessPoolExecutor** | **Use this** | Stdlib, clean `submit()`+`as_completed()` API, negligible overhead for 30-90s tasks |
| multiprocessing.Pool | Equivalent | Less ergonomic API; `imap_unordered` works but futures are cleaner |
| joblib | Skip | Not installed, loky overhead benefits sub-second tasks (sklearn), not 30-90s runs |
| Ray | Skip | Cluster-scale tool, ~200MB deps, massive overkill for single-machine |

### Process Isolation: Confirmed Safe

All components are **fully picklable** (verified):
- `SeasonConfig`, `TollConfig`, `PopulationParams`, `HybridCollectorConfig`, `BusCostConfig`
- scipy frozen distributions (`lognorm`, `skewnorm`, `norm`) — roundtrip cleanly
- `SeasonConfig` serializes to ~18KB

**Architecture pattern:** Do NOT pickle `SeasonOrchestrator`. Pickle `SeasonConfig` (stores paths, not data). Each worker constructs its own `SeasonOrchestrator`, which loads `road_gdf` and `ecs_df` independently.

### GIL / CPU-bound Confirmation

Mesa stepping is pure Python CPU-bound work. `multiprocessing` (process-level parallelism) bypasses the GIL entirely. Threading would give zero benefit here. Each worker process gets its own Python interpreter and GIL.

### macOS `spawn` Method

Python 3.8+ on macOS defaults to `spawn` (not `fork`). **Do NOT switch to `fork`** — Apple's Objective-C runtime is not fork-safe, causes intermittent crashes with numpy/scipy/geopandas.

**Key implication:** The worker function must be defined in a `.py` module (e.g., `season/parallel.py`), not in a notebook cell. Functions defined in notebook cells cannot be pickled by `spawn`. The notebook imports and calls this module.

---

## 2. Parameter Space API Design

### Recommendation: `itertools.product` with fixed/sweep dict pattern

No new dependencies needed. The pattern:

```python
sweep_space = {
    "bus_interval_schedule": [ScheduleSpecs("static", v) for v in [15, 30, 60]],
    "population_size": [500, 1000, 1500],
    "toll": [TollConfig.static(car=0), TollConfig.static(car=10)],
}
fixed = dict(n_days=10, seed=42, start_hr=7, ...)
```

A helper generates the cross-product, calling `make_season_config()` for each combo with auto-generated `season_id` strings encoding the varied parameters.

### Sweep Types Supported
- **Grid search**: full cross-product of all sweep dims
- **Single-axis sweep**: one key in `sweep_space`, everything else in `fixed`
- **Specific combos**: pass an explicit list of override dicts

### Seed Handling
Use `base_seed + sweep_index` to ensure each config gets a unique, reproducible seed while the sweep itself is deterministic.

### Integration with `make_season_config()`
The helper unpacks `{**fixed, **combo}` into `make_season_config(**kwargs)`. `PopulationParams` fields that vary (e.g., `population_size`) need the helper to construct `PopulationParams(population_size=combo["population_size"])` before passing to the factory.

---

## 3. Data Storage at Scale

### Current Per-Season Output (3 days, 1500 persons, tier1 only)

| File | Size | Purpose |
|------|------|---------|
| season_person_log.parquet | 264 KB | Per-person state snapshots per day |
| trip_log.parquet | 132 KB | Per-trip records |
| day_summary.parquet | 24 KB | Per-day aggregate metrics |
| day_N_model_ts.parquet (×3) | 20-24 KB each | Tier 1 time series per day |
| season_summary.parquet | 20 KB | Season-level summary |
| sp_day_summary.parquet | 12 KB | SeasonPerson day summaries |
| season_config.json | 8 KB | Config snapshot |
| **Total** | **~528 KB** | |

### Projections

| Sweep Size | Per-Season | Total Disk | Notes |
|------------|-----------|------------|-------|
| 30 configs × 3 days | 528 KB | **15 MB** | Comfortable |
| 100 configs × 10 days | ~1.5 MB | **150 MB** | Fine |
| 200 configs × 30 days, 7k persons | ~12 MB | **2.4 GB** | Manageable but watch it |

### Largest Conceivable Run (30 days, 7000 persons)

Scaling from baseline (3 days, 1500 persons):
- `season_person_log`: scales with persons × days → 264KB × (7000/1500) × (30/3) ≈ **12.3 MB**
- `trip_log`: scales with persons × days → 132KB × (7000/1500) × (30/3) ≈ **6.2 MB**
- `day_N_model_ts`: scales with days → 22KB × 30 ≈ **660 KB**
- Other files: ~100 KB
- **Estimated total per season: ~19 MB**

### Sweep Collection Tiers

Sweeps should always use `store_data=False` on individual workers. All results flow back to the main process as dicts/DataFrames. The sweep runner writes the combined output once at the end.

The `SWEEP_COLLECTOR_CONFIG` defines three levels of detail:

| Level | Name | What's Collected | Estimated Size/Season | Use Case |
|-------|------|-----------------|----------------------|----------|
| 1 | `summaries_only` | Day summaries + season summary | ~50 KB | Fast sweeps, coarse comparison across configs |
| 2 | `summaries_plus_distributions` | Level 1 + per-day and per-season distributional metrics | ~100-200 KB | Richer analysis without full time series |
| 3 | `full_tier1` | Level 2 + all tier1 scalars/window_scalars (model time series) | ~500 KB+ | Debugging, deep-dive on specific configs |

**Tier2/tier3/tier4 are always disabled in sweeps.** No spatial data, no event logs, no snapshots.

#### Level 2: Distributional Metrics (future work, design notes)

The distributional metrics would capture the shape of key metric distributions at the day and season level without storing per-trip or per-step data. Structure:

```
Index: (day_index, metric_name)  — or (metric_name,) for season-level
Columns: min, p10, p25, p50, p75, p90, max, mean, var
```

Candidate metrics for distributional summaries:
- `realized_tt`, `realized_cost`, `toll_paid`, `cumtime_lost_min` (per-trip distributions)
- `value_of_time`, `prior_car`, `prior_bus` (per-person distributions)
- `vehicle_count`, `current_toll` (per-step distributions from tier1)

This is relatively minimal data (one row per metric per day, ~9 columns) but captures far more distributional detail than the current mean-only summaries. The existing tier1 histograms (`implicit_sl_delta`, `speed_mps`) are an analogue but use a fixed-bin structure that doesn't generalize well across metrics. The percentile-based approach is metric-agnostic and produces a clean, flat DataFrame.

**Status:** Not yet implemented. Would be added to `_compute_day_summary()` and `_compute_season_summary()` as an optional output. The sweep runner would collect these alongside the existing summary dicts.

### Other Data Recommendations

1. **Primary analysis artifact: combined season_summary DataFrame** — the sweep runner collects all `season_summary` dicts into a single DataFrame. This is the thing you analyze. Per-run file I/O is eliminated entirely for sweeps.

---

## 4. Runtime Metadata

### What Already Exists

| Metric | Where | How |
|--------|-------|-----|
| `self.steps` | TrafficModel (Mesa base) | Auto-incremented per step |
| `created_counts` | TrafficModel:43 | `defaultdict(int)` — counts by agent type |
| `vehicle_count` | tier1 scalar | `len(m.vehicles_list)` per collection step |
| `active_cars` | tier1 scalar | Already collected |
| `active_buses` | tier1 scalar | Already collected |
| `total_finished` | tier1 scalar | Already collected |
| tqdm steps/s | run_model():275 | Displayed but not captured |

### New Metrics to Add (to season_summary dict)

| Metric | Source | Implementation |
|--------|--------|----------------|
| `wall_time_seconds` | `time.perf_counter()` around `run_season()` | Wrap in sweep worker |
| `total_steps` | Sum of `tm.steps` across days | Add accumulator in `run_season()` |
| `avg_steps_per_second` | `total_steps / wall_time_seconds` | Computed post-run |
| `avg_steps_per_second_per_vehicle` | `total_vehicle_steps / wall_time_seconds` | Computed post-run |

### Vehicle-Step Counting

Add `self.total_vehicle_steps = 0` to `TrafficModel.__init__` and increment by `len(self.vehicles_list)` each step. Zero overhead (one `len()` call per step), exact, no dependency on tier1 collection interval.

### Removing the Legacy `batchrun` Flag

`batchrun` is a legacy construct from pre-season batch runs. It should be **removed entirely** before implementing parallelization, to avoid confusion with the new parallel sweep system.

**Complete audit of `batchrun` effects (3 total):**

| Location | Effect | Status |
|----------|--------|--------|
| `traffic_model.py:142` | `tier2_enabled=not self.batchrun` in fallback collector config | **Dead code** — season runs always pass an explicit `hybrid_collector_config`, bypassing this fallback |
| `analysis_utils.py:154` | `vehicle_agent_data_time_series()` returns `None` if `batchrun=True` | **Redundant** — tier2 being disabled already means the function returns `None` (empty DataFrame check on line 159) |
| `traffic_model.py:275` | tqdm `run_model()` loop | **NOT gated by batchrun** — tqdm always runs regardless |

**Removal plan (pre-requisite for parallelization):**
1. Remove `batchrun` param from `TrafficModel.__init__` and `self.batchrun`
2. Remove `batch_run` from `SeasonConfig` and `make_season_config()`
3. Remove `batchrun=self.config.batch_run` from `SeasonOrchestrator._build_model()`
4. Remove `if model.batchrun == True` guard in `analysis_utils.py:154` (the empty-DataFrame check already handles this)
5. Update tests that pass `batch_run=True` (`optimization_check.py:56`, `season_determinism.py:45`, `day_determinism.py:45`, `perf_utils.py:100,120`)

### Suppressing tqdm in Parallel Workers

tqdm is **not** gated by any flag — it always runs in `run_model()`. For parallel sweeps, N workers printing concurrent progress bars will flood output.

**Solution:** Add a `silent: bool = False` param to `TrafficModel.__init__`, pass `disable=self.silent` to the tqdm call. The sweep worker sets `silent=True`. This is a clean, purpose-built flag that replaces the defunct batchrun's intended (but never implemented) tqdm suppression.

---

## 5. Progress Reporting

### Recommendation: `submit()` + `as_completed()` + `tqdm.auto`

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

with ProcessPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(run_one_season, cfg): cfg.season_id for cfg in configs}
    with tqdm(total=len(futures), desc="Sweep") as pbar:
        for future in as_completed(futures):
            result = future.result()  # raises if worker crashed
            pbar.update(1)
            pbar.set_postfix(last=futures[future])
```

- `tqdm.auto` (v4.67.1, already installed) auto-detects Jupyter vs terminal
- Shows "12/50 seasons complete" style progress with ETA
- Each tick fires when a worker finishes — real-time updates
- `set_postfix(last=season_id)` shows which config just completed

**Do NOT use per-worker sub-bars** — known issues with tqdm + multiprocessing in Jupyter (#485, #1133). One outer bar tracking completions is reliable.

---

## 6. Memory Analysis

### Per-Worker Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| Python interpreter | ~80 MB | Base overhead per spawn |
| ecs_df (CSV load) | 47 MB | Loaded independently per worker |
| road_gdf | 41 KB | Negligible |
| Model runtime (agents, arrays) | ~100-300 MB | Depends on population_size |
| **Total per worker** | **~250-450 MB** | Conservative estimate |

### Worker Scaling

| Workers | Estimated RAM | Headroom (of 32GB) |
|---------|--------------|-------------------|
| 8 | 2.0-3.6 GB | Plenty |
| 10 | 2.5-4.5 GB | Comfortable |
| 12 | 3.0-5.4 GB | Fine, but machine less responsive |

**For 7000-person runs**: agent memory scales linearly. Estimate ~1-1.5 GB per worker → 6 workers max for large configs.

### Recommendation
- Default: `max_workers = min(os.cpu_count() - 2, len(configs))` → 10 on this machine
- For large configs (7k persons): reduce to 6 workers
- Make it a parameter the user sets in the notebook

---

## 7. Wall-Time Projections

| Sweep | Tasks | Per-Task | Sequential | 10 Workers | Speedup |
|-------|-------|----------|-----------|-----------|---------|
| 30 configs × 3d × 1500p | 30 | ~40s | 20 min | ~2 min | ~10x |
| 100 configs × 10d × 1500p | 100 | ~2 min | 3.3 hr | ~20 min | ~10x |
| 50 configs × 30d × 7000p | 50 | ~15 min | 12.5 hr | ~1.3 hr | ~10x |

**Expected speedup: ~N_workers × 0.9** (0.9 accounts for spawn overhead and load imbalance). With 10 workers, expect ~9x speedup.

---

## 8. Key Risks & Gotchas

| Risk | Mitigation |
|------|------------|
| Worker function in notebook cell won't pickle (macOS spawn) | Define in `season/parallel.py`, import in notebook |
| tqdm per-step bars flood output from N workers | New `silent=True` flag on TrafficModel, use single outer completion bar |
| `season_summary_log.csv` concurrent writes from workers | Don't write CSV from workers; collect results in main process, write once |
| Memory pressure with large configs + many workers | Make `max_workers` configurable, document guidance for 7k-person runs |
| One worker crash kills the whole sweep | Wrap `future.result()` in try/except, log failures, continue sweep |
| Non-unique `season_id` across sweep | Auto-generate IDs encoding sweep parameters |

---

## 9. Recommended File Structure

```
season/
  parallel.py          # NEW — worker function, sweep grid builder, ParallelSweepRunner
  configs.py           # Existing — no changes needed to SeasonConfig/make_season_config
  season_orchestrator.py  # Minimal changes: wall_time, total_steps, optional person log

notebooks/
  season_run.ipynb     # Existing — single-run workflow, unchanged
  sweep_run.ipynb      # NEW — dedicated sweep notebook (approved; already created)
```

`sweep_run.ipynb` is modeled after `season_run.ipynb`: same pwd/autoreload setup, display options, and data-existence checks. It adds sweep-specific cells: `SWEEP_COLLECTOR_CONFIG` constants, a sweep definition block (`sweep_space` + `fixed` dicts → `itertools.product` → list of `SeasonConfig`), a run cell (sequential placeholder → `ParallelSweepRunner` once `season/parallel.py` exists), and a results cell. The notebook file may be created and edited during the initial dev step.

---

## 10. Implementation Priority

1. **Remove `batchrun`** — pre-requisite cleanup before any parallel work
2. **Runtime metadata** (wall_time, total_steps, steps/s) — small change to `SeasonOrchestrator`, useful even without parallelism
3. **Worker function** in `season/parallel.py` — accepts `SeasonConfig`, returns summary dict
4. **Sweep grid builder** — `itertools.product` helper generating list of `SeasonConfig`
5. **`ParallelSweepRunner`** — wraps `ProcessPoolExecutor` + tqdm + result collection
6. **`sweep_run.ipynb`** — wire in `ParallelSweepRunner`, finalize results cell
