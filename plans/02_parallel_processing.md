# Feature Plan: Parallel Processing for Parameter Sweeps and Monte Carlo Simulations

## Overview
Enable parallel execution of multiple independent model runs to accelerate parameter sweeps and Monte Carlo simulations while maintaining exact reproducibility.

## Motivation
Currently, running multiple scenarios (e.g., testing 10 toll levels × 5 bus intervals × 10 random seeds = 500 runs) requires sequential execution. This is slow and limits the scope of sensitivity analysis and uncertainty quantification.

**Target use cases:**
1. **Parameter sweeps:** Test different policy configurations (toll mechanisms, bus intervals, traffic levels)
2. **Monte Carlo simulations:** Run the same configuration with multiple random seeds to quantify stochastic uncertainty

**Out of scope:**
- Parallelizing days within a single season run (complex due to learning dynamics, minimal benefit)
- Real-time parallel agent updates within a single step (unnecessary, breaks determinism)

## Design Decisions

### 1. Parallelization Level: **Independent Season Runs**
**Choice:** Parallelize at the coarsest grain - entire `SeasonOrchestrator.run_season()` executions

**Implications:**
- **Pros:**
  - Trivially parallelizable (no shared state between runs)
  - Exact reproducibility guaranteed (each run uses its own seed)
  - No code changes to core model logic
  - Works with existing SeasonConfig system
  - Scales linearly with number of cores

- **Cons:**
  - Cannot speed up a single long season run
  - Requires enough independent scenarios to fill cores
  - Peak memory usage increases (multiple models in memory)

**Architecture:**
```
joblib.Parallel(n_jobs=-1)(
    delayed(run_single_scenario)(config)
    for config in scenario_configs
)
```

### 2. Reproducibility: **Deterministic Seeding**
**Choice:** Each parallel run uses a unique, deterministic seed derived from base seed + scenario index

**Implications:**
- **Pros:**
  - Runs are reproducible regardless of execution order
  - Parallel and sequential results are identical
  - Easy to re-run specific scenarios
  - Debugging is straightforward

- **Cons:**
  - Requires careful seed management
  - Must avoid RNG state leakage between runs

**Seed Strategy:**
```python
base_seed = 12345
for i, config in enumerate(scenario_configs):
    config.seed = base_seed + i  # Deterministic, unique seed per scenario
```

### 3. Computing Resource: **Local Machine (Multiple Cores)**
**Choice:** Use `joblib` library with `loky` backend for local multiprocessing

**Implications:**
- **Pros:**
  - Works on laptops/desktops without infrastructure setup
  - Automatic load balancing across cores
  - Progress tracking with `tqdm` integration
  - Memory-efficient with lazy evaluation
  - Cross-platform (Windows, Mac, Linux)

- **Cons:**
  - Limited to local core count (~4-16 cores typical)
  - Not scalable to hundreds of runs (HPC would be better)
  - Shared memory constraints

**Alternative considered:** `multiprocessing.Pool` (rejected: less flexible, no tqdm integration)

**Future extension:** Add cluster support via `dask` or `ray` for large-scale sweeps

## Implementation Subtasks

### Subtask 1: Create Parallel Execution Utility
**File:** `season/parallel_runner.py` (new file)
**Purpose:** Wrapper functions for parallel scenario execution

**Code:**
```python
from joblib import Parallel, delayed
from tqdm import tqdm
from season.season_orchestrator import SeasonOrchestrator
from season.configs import SeasonConfig
import warnings

def run_single_scenario(config: SeasonConfig, store_data: bool = True,
                        output_root_dir: str = "data/season_outputs"):
    """
    Run a single scenario (season) and return its summary.

    This function is designed to be called in parallel via joblib.
    It creates a SeasonOrchestrator, runs the season, and returns results.

    Args:
        config: SeasonConfig object with all parameters
        store_data: Whether to save outputs to disk
        output_root_dir: Root directory for outputs

    Returns:
        dict: Season summary with season_id and all metrics
    """
    # Suppress warnings in parallel workers (optional)
    warnings.filterwarnings('ignore')

    orchestrator = SeasonOrchestrator(
        season_config=config,
        store_data=store_data,
        output_root_dir=output_root_dir
    )

    orchestrator.run_season()

    # Return summary + dataframes for aggregation
    return {
        "season_id": config.season_id,
        "seed": config.seed,
        "summary": orchestrator._compute_season_summary(),
        "day_summaries": orchestrator.get_day_summary_df(),
        "trip_log": orchestrator.get_trip_log_df() if store_data else None,
    }


def run_parameter_sweep(
    config_generator,
    n_jobs: int = -1,
    store_data: bool = True,
    output_root_dir: str = "data/season_outputs",
    verbose: int = 10,
):
    """
    Run multiple scenarios in parallel.

    Args:
        config_generator: Iterable of SeasonConfig objects
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        store_data: Whether to save individual run outputs
        output_root_dir: Root directory for outputs
        verbose: Verbosity level for joblib (higher = more output)

    Returns:
        list: List of result dictionaries from each scenario
    """
    configs = list(config_generator)

    print(f"Running {len(configs)} scenarios with n_jobs={n_jobs}")
    print(f"Estimated total runs: {sum(c.n_days for c in configs)} model-days")

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(run_single_scenario)(config, store_data, output_root_dir)
        for config in tqdm(configs, desc="Scenarios")
    )

    return results


def run_monte_carlo(
    base_config: SeasonConfig,
    n_replications: int = 10,
    base_seed: int = None,
    n_jobs: int = -1,
    store_data: bool = True,
    output_root_dir: str = "data/season_outputs",
):
    """
    Run Monte Carlo simulations with different random seeds.

    Args:
        base_config: Template SeasonConfig to replicate
        n_replications: Number of Monte Carlo runs
        base_seed: Base seed for replication (uses base_config.seed if None)
        n_jobs: Number of parallel jobs
        store_data: Whether to save outputs
        output_root_dir: Root directory for outputs

    Returns:
        list: Results from all replications
    """
    if base_seed is None:
        base_seed = base_config.seed

    # Create configs with different seeds
    configs = []
    for i in range(n_replications):
        config = copy.deepcopy(base_config)
        config.seed = base_seed + i
        config.season_id = f"{base_config.season_id}_mc{i:03d}"
        configs.append(config)

    print(f"Running {n_replications} Monte Carlo replications")
    return run_parameter_sweep(configs, n_jobs, store_data, output_root_dir)
```

**Estimated lines:** 100-120

---

### Subtask 2: Create Configuration Generator Utilities
**File:** `season/config_generators.py` (new file)
**Purpose:** Helper functions to generate SeasonConfig lists for common sweep patterns

**Code:**
```python
from season.configs import make_season_config, SeasonConfig, PopulationParams
from itertools import product
import copy

def generate_toll_sweep(
    base_config_params: dict,
    toll_levels: list,
    toll_mechanism: str = "static",
    base_seed: int = 123,
):
    """
    Generate configs for toll level sweep.

    Example:
        configs = generate_toll_sweep(
            base_config_params={...},
            toll_levels=[0, 5, 10, 15, 20],
            toll_mechanism="static",
            base_seed=1000,
        )
    """
    configs = []
    for i, toll in enumerate(toll_levels):
        params = copy.deepcopy(base_config_params)
        params["toll_mechanism"] = toll_mechanism
        params["toll_params"] = {"car": toll, "bus": 0.0}
        params["seed"] = base_seed + i
        params["season_id"] = f"toll_{toll_mechanism}_{toll:.1f}_seed{base_seed + i}"

        config = make_season_config(**params)
        configs.append(config)

    return configs


def generate_grid_sweep(
    base_config_params: dict,
    param_grid: dict,
    base_seed: int = 123,
):
    """
    Generate configs for full factorial parameter grid.

    Example:
        configs = generate_grid_sweep(
            base_config_params={...},
            param_grid={
                "toll_params": [{"car": 0}, {"car": 10}, {"car": 20}],
                "bus_interval_schedule": [
                    ScheduleSpecs("static", 15),
                    ScheduleSpecs("static", 30),
                    ScheduleSpecs("static", 60),
                ],
            },
            base_seed=2000,
        )
    """
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    configs = []
    for i, combination in enumerate(product(*param_values)):
        params = copy.deepcopy(base_config_params)

        # Override with grid values
        for name, value in zip(param_names, combination):
            params[name] = value

        params["seed"] = base_seed + i
        params["season_id"] = f"grid_sweep_{i:04d}"

        config = make_season_config(**params)
        configs.append(config)

    return configs
```

**Estimated lines:** 80-100

---

### Subtask 3: Add Result Aggregation Utilities
**File:** `season/parallel_runner.py` (extend)
**Purpose:** Combine results from parallel runs into summary dataframes

**Code:**
```python
import pandas as pd

def aggregate_sweep_results(results: list) -> pd.DataFrame:
    """
    Aggregate results from parameter sweep into single DataFrame.

    Args:
        results: List of result dicts from run_parameter_sweep()

    Returns:
        DataFrame with one row per scenario, all summary metrics as columns
    """
    rows = []
    for result in results:
        row = {
            "season_id": result["season_id"],
            "seed": result["seed"],
        }
        row.update(result["summary"])
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_day_summaries(results: list) -> pd.DataFrame:
    """
    Combine day summaries from all scenarios.

    Returns:
        DataFrame with day_index, season_id, and all day metrics
    """
    dfs = []
    for result in results:
        df = result["day_summaries"]
        if df is not None:
            df["season_id"] = result["season_id"]
            df["seed"] = result["seed"]
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else None
```

**Estimated lines:** 40-50

---

### Subtask 4: Create Example Notebooks
**File:** `notebooks/parallel_examples.ipynb` (new file)
**Purpose:** Demonstrate parallel execution patterns

**Sections:**
1. **Single-core baseline** (for comparison)
2. **Parallel toll sweep** (5 toll levels, 4 seeds each = 20 runs)
3. **Monte Carlo simulation** (1 config, 50 seeds)
4. **Full factorial grid** (3 tolls × 3 bus intervals × 5 seeds = 45 runs)
5. **Result visualization** (comparing across scenarios)

**Example cells:**
```python
# Example: Parallel toll sweep
from season.parallel_runner import run_parameter_sweep
from season.config_generators import generate_toll_sweep

base_params = {
    "season_id": "toll_sweep",
    "run_description": "Testing toll levels 0-20",
    "n_days": 1,
    "max_persons": 200,
    "road_path": "data/roads/lcc_road.parquet",
    "ecs_path": "data/vehicle_counts/lcc_expected_counts.csv",
    # ... other params
}

configs = generate_toll_sweep(
    base_config_params=base_params,
    toll_levels=[0, 5, 10, 15, 20],
    base_seed=5000,
)

results = run_parameter_sweep(configs, n_jobs=-1)

# Aggregate and visualize
import pandas as pd
df = aggregate_sweep_results(results)
df.plot(x="toll_levels", y=["avg_tt", "percent_bus_share"], subplots=True)
```

**Estimated lines:** 200-300 (notebook cells + markdown)

---

### Subtask 5: Update Documentation
**File:** `README.md`
**Changes:** Add section on parallel execution

**Content:**
```markdown
## Parallel Execution

For parameter sweeps and Monte Carlo simulations, use the parallel execution utilities:

\`\`\`python
from season.parallel_runner import run_parameter_sweep, run_monte_carlo
from season.config_generators import generate_toll_sweep

# Generate configs for toll sweep
configs = generate_toll_sweep(
    base_config_params={...},
    toll_levels=[0, 5, 10, 15, 20],
    base_seed=1000,
)

# Run in parallel (uses all CPU cores)
results = run_parameter_sweep(configs, n_jobs=-1)

# Aggregate results
df = aggregate_sweep_results(results)
\`\`\`

See `notebooks/parallel_examples.ipynb` for detailed examples.

**Performance:** Parallel execution scales linearly with CPU cores. A 20-scenario sweep that takes 60 minutes sequentially completes in ~8 minutes on an 8-core machine.
```

**Estimated lines:** 20-30

---

### Subtask 6: Add Requirements
**File:** `requirements.txt`
**Changes:** Add parallel processing dependencies

```
joblib>=1.3.0
tqdm>=4.66.0
```

**Note:** Both are lightweight libraries with no heavy dependencies

---

## Critical Files to Modify/Create
1. **season/parallel_runner.py** (NEW) - Core parallel execution logic
2. **season/config_generators.py** (NEW) - Config generation utilities
3. **notebooks/parallel_examples.ipynb** (NEW) - Usage examples
4. **README.md** - Documentation update
5. **requirements.txt** - Add dependencies

## Testing & Verification

### Reproducibility Tests
**Critical:** Parallel and sequential results must be identical

```python
import numpy as np

# Test 1: Single scenario, run twice
config = make_season_config(season_id="test", seed=999, ...)
result1 = run_single_scenario(config)
result2 = run_single_scenario(config)

assert result1["summary"] == result2["summary"], "Single run not reproducible!"

# Test 2: Sequential vs parallel
configs = [make_season_config(seed=1000+i, ...) for i in range(5)]

# Sequential
results_seq = [run_single_scenario(c) for c in configs]

# Parallel
results_par = run_parameter_sweep(configs, n_jobs=5)

# Compare
for seq, par in zip(results_seq, results_par):
    np.testing.assert_equal(seq["summary"], par["summary"])

print("✓ Parallel results match sequential results exactly")
```

### Performance Tests
```python
import time

# Benchmark: 10 scenarios, sequential vs parallel
configs = [make_season_config(seed=2000+i, n_days=1, max_persons=100, ...)
           for i in range(10)]

# Sequential
start = time.time()
results = run_parameter_sweep(configs, n_jobs=1)
seq_time = time.time() - start

# Parallel (4 cores)
start = time.time()
results = run_parameter_sweep(configs, n_jobs=4)
par_time = time.time() - start

speedup = seq_time / par_time
print(f"Sequential: {seq_time:.1f}s")
print(f"Parallel (4 cores): {par_time:.1f}s")
print(f"Speedup: {speedup:.2f}x")

# Expect speedup close to 4x (minus overhead)
assert speedup > 3.0, "Parallel speedup insufficient"
```

### Memory Tests
```python
import psutil
import os

# Monitor memory during parallel execution
process = psutil.Process(os.getpid())

configs = [make_season_config(seed=i, ...) for i in range(20)]

mem_before = process.memory_info().rss / 1024**2  # MB

results = run_parameter_sweep(configs, n_jobs=-1)

mem_after = process.memory_info().rss / 1024**2  # MB
mem_increase = mem_after - mem_before

print(f"Memory increase: {mem_increase:.1f} MB")
print(f"Per scenario: {mem_increase / len(configs):.1f} MB")

# Warn if excessive
if mem_increase > 5000:  # 5 GB
    print("⚠ High memory usage - consider reducing n_jobs or batch size")
```

## Potential Risks & Considerations

### Risk 1: RNG State Leakage
**Issue:** If any code uses global RNG state (e.g., `np.random.rand()` without `rng`), parallel runs could interfere.

**Current status:** Model uses `model.rng = np.random.default_rng(seed)` properly seeded. VehicleAgent has one instance of `np.random.normal()` without RNG argument (line 193).

**Mitigation:**
- Audit all `np.random` calls to ensure they use `model.rng`
- Replace `np.random.normal()` with `model.rng.normal()`
- Add RNG state check in reproducibility tests

**Priority:** HIGH - must fix before deploying parallelization

---

### Risk 2: Memory Constraints
**Issue:** Running N jobs in parallel creates N model instances in memory simultaneously.

**Typical memory usage:**
- Single model: ~50-200 MB (depends on max_persons, collect_every_n)
- 8 parallel jobs: ~400-1600 MB total
- Acceptable on most modern machines

**Mitigation:**
- Set `n_jobs` conservatively (leave 1-2 cores free)
- Use `store_data=False` for large sweeps to reduce memory
- Process results in batches for very large sweeps

---

### Risk 3: Disk I/O Contention
**Issue:** If all parallel jobs write to disk simultaneously, I/O could become bottleneck.

**Current behavior:** Each job writes to separate `data/season_outputs/{season_id}/` directory

**Mitigation:**
- Generally not a problem for local SSDs
- For HPC with shared filesystems, consider writing to local temp directory then copying

---

### Risk 4: Progress Tracking Overhead
**Issue:** `tqdm` progress bars in parallel can interfere with each other

**Mitigation:**
- `joblib` handles this automatically with `verbose` parameter
- Disable model-level progress bars (`batchrun=True`) in parallel runs

---

### Risk 5: Exception Handling
**Issue:** If one scenario fails, should entire sweep abort or continue?

**Current behavior:** `joblib` by default raises exception and stops all jobs

**Mitigation:**
```python
# Option 1: Continue on error (collect exceptions)
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(run_single_scenario_safe)(config)  # Wraps with try/except
    for config in configs
)

# Option 2: Use joblib's built-in error handling
results = Parallel(n_jobs=-1, return_exceptions=True)(...)
```

**Recommendation:** Fail fast for debugging, continue on error for production sweeps

---

## Performance Expectations

### Speedup Formula
Theoretical speedup with P cores:
```
Speedup = P / (1 + overhead)
```

**Overhead sources:**
- Process spawning (~0.1-0.5s per job)
- Data serialization (config objects)
- RNG initialization
- Disk I/O (if storing data)

**Typical speedup:**
- 4 cores: ~3.5x faster
- 8 cores: ~6-7x faster
- 16 cores: ~12-14x faster

**Diminishing returns:** Beyond 8-12 cores, overhead increases. Best for 4-16 core machines.

### Benchmark Scenarios
**Scenario 1: Small sweep (20 runs, 1 day each)**
- Sequential: ~20 minutes (1 min/run)
- Parallel (8 cores): ~3 minutes
- **Speedup: 6.7x**

**Scenario 2: Large sweep (100 runs, 5 days each)**
- Sequential: ~500 minutes (~8 hours)
- Parallel (8 cores): ~75 minutes (~1.25 hours)
- **Speedup: 6.7x**

**Scenario 3: Monte Carlo (50 seeds, 1 day each)**
- Sequential: ~50 minutes
- Parallel (16 cores): ~4 minutes
- **Speedup: 12.5x**

## Future Extensions

1. **Distributed computing:** Add `dask` or `ray` backend for cluster execution
2. **Checkpointing:** Save intermediate results to resume interrupted sweeps
3. **Dynamic scheduling:** Prioritize fast-running scenarios first
4. **Result caching:** Skip scenarios that have already been run
5. **Adaptive sampling:** Use Bayesian optimization to explore parameter space efficiently

## Success Criteria
✅ Parallel execution produces identical results to sequential execution
✅ Speedup scales linearly with core count (within 80% efficiency)
✅ Memory usage remains reasonable (<2 GB per core)
✅ Example notebooks demonstrate all common use cases
✅ Documentation clearly explains when to use parallel vs sequential
✅ No RNG state leakage between parallel jobs
✅ Exception handling allows graceful failure recovery

## Estimated Implementation Time
- **Subtask 1:** 2-3 hours (parallel_runner.py)
- **Subtask 2:** 1-2 hours (config_generators.py)
- **Subtask 3:** 1 hour (result aggregation)
- **Subtask 4:** 2-3 hours (example notebook)
- **Subtask 5-6:** 1 hour (docs + requirements)
- **Testing & Validation:** 2-3 hours (reproducibility critical)
- **Total:** 9-13 hours

## Implementation Notes
- Start with reproducibility tests BEFORE writing parallel code
- Fix RNG issues in VehicleAgent first (blocking issue)
- Test on small sweeps (5-10 runs) before scaling up
- Benchmark memory usage with your typical config size
- Consider adding a `--dry-run` flag to preview sweep size before executing
