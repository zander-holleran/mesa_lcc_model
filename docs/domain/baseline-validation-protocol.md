# Baseline Validation Protocol

Reproducible procedure for validating the ABM's emergent speed-density relationship against classical traffic flow theory parameterized from empirical literature.

## Endpoint

Overlay simulation-fitted and literature-fitted Greenshields, Greenberg, and Underwood speed-density curves. If ABM 95% CI bands overlap substantially with literature 95% CI bands, the model produces macroscopically realistic traffic dynamics.

## Step 1: Run the Simulation Sweep

Open `notebooks/sweep_run.ipynb`. Configure as follows.

### Collector Config (Tier 2 only, fast)

```python
from traffic.model.hybrid_collector import HybridCollectorConfig

VALIDATION_TIER2 = HybridCollectorConfig(
    max_steps=100000,
    tier1_enabled=False,
    tier2_enabled=True,
    tier2_sample_interval=10,
    tier2_max_samples=5000,
    tier2_max_agents_per_sample=9999,
    tier3_enabled=False,
    tier4_enabled=False,
)
```

### Sweep Configuration

Run one sweep per traffic percentile. Percentiles: **20, 40, 60, 70, 80, 85, 90, 95, 97**.

```python
sweep_space = dict(
    seed=[42, 43, 44, 45, 46, 47],
    bus_interval_schedule=[ScheduleSpecs("static", 30)],
)

fixed = dict(
    run_description="baseline_validation",
    n_days=1,
    start_hr=7,
    population_size=3000,
    traffic_percentile_schedule=ScheduleSpecs("static", <PERCENTILE>),
    crashes_schedule=ScheduleSpecs("static", 0),
    max_steps=99999,
    max_persons=99999,
    bus_capacity=60,
    bus_user_fee=0.0,
    hybrid_collector_config=VALIDATION_TIER2,
)
```

Set `<PERCENTILE>` to each value in turn (or automate via an outer loop). Total: 9 percentiles x 6 seeds = 54 runs.

### What Each Run Produces

- Season output directory: `data/season_outputs/<season_id>/`
- Tier 2 parquet files within day-level subdirectories
- Sweep summary: `data/sweep_outputs/sweep_YYYYMMDD_HHMMSS/sweep_results.parquet`

## Step 2: Locate Output Files

Tier 2 data lives in each season's day output directory. Note the paths from the sweep results or `runner.output_dir`. Copy, symlink, or reference these paths directly in the analysis notebook.

## Step 3: Fit Models in the Benchmark Notebook

Open `empirical_benchmark/classical_traffic_models_benchmark.ipynb`.

### Per-Run Processing

For each run's Tier 2 parquet:

```python
import empirical_benchmark.empirical_benchmark_helpers as ebh

# 1. Load and preprocess
df = pd.read_parquet("<path_to_tier2.parquet>")
df["dist_step_m"] = (
    df.groupby("AgentID")["distance_traveled"]
    .diff().fillna(0).clip(lower=0)
)

# 2. Compute Edie's k-v-q
edie = ebh.compute_edie_kvq(df, road_length_m=19950,
                             n_spatial=20, n_temporal=30,
                             tier2_sample_interval=10)
k = edie["k_vehmi"].values
v = edie["v_mph"].values

# 3. Fit three models
gs = ebh.fit_model(ebh.greenshields, k, v, p0=[35, 130], model_name="Greenshields")
gr = ebh.fit_model(ebh.greenberg,    k, v, p0=[25, 130], model_name="Greenberg")
uw = ebh.fit_model(ebh.underwood,    k, v, p0=[35, 25],  model_name="Underwood")
```

Collect results across all runs:

```python
all_results = {"greenshields": [], "greenberg": [], "underwood": []}
for run_path in run_paths:
    # ... load, preprocess, fit (as above) ...
    all_results["greenshields"].append(gs)
    all_results["greenberg"].append(gr)
    all_results["underwood"].append(uw)
```

### Aggregate and Compare

```python
# Aggregate fitted params into GaussianDist per parameter
sim_dists = ebh.aggregate_fit_results(all_results)

# Build sim CI bands
K_PLOT = np.linspace(0.5, 250, 500)
sim_ci = {}
for model_key, (model_fn, param_keys) in {
    "greenshields": (ebh.greenshields, ["v_f", "k_j"]),
    "greenberg":    (ebh.greenberg,    ["v_0", "k_j"]),
    "underwood":    (ebh.underwood,    ["v_f", "k_0"]),
}.items():
    dists = [sim_dists[model_key][p] for p in param_keys]
    sim_ci[model_key] = ebh.mc_ci_band(model_fn, dists, K_PLOT)

# Literature CI bands (from earlier in the notebook)
lit_ci = {
    "greenshields": (gs_mean, gs_lo, gs_hi),
    "greenberg":    (gr_mean, gr_lo, gr_hi),
    "underwood":    (uw_mean, uw_lo, uw_hi),
}

# Comparison figure
fig, axes = ebh.plot_kv_ci_comparison(K_PLOT, lit_ci, sim_ci,
    save_path="notebooks/figures/kv_validation_comparison.png")
```

## Step 4: Interpret Results

- **Visual check:** Do the ABM (orange) CI bands overlap substantially with the literature (blue/pink/green) CI bands?
- **Parameter check:** For each parameter (v_f, k_j, v_0, k_0), does the sim mean fall within the literature 95% CI?
- **Expected outcomes:**
  - v_f and k_j: should align well with literature ranges
  - k_0 and v_0: may show wider sim variance due to fewer constraining observations at low density
- **Validation passes** if band overlap is substantial across all three models

## Key Parameters Reference

| Parameter | Value | Notes |
|---|---|---|
| road_length_m | 19,950 | ~12.4 miles, SR-210 canyon road |
| tier2_sample_interval | 10 | 10 steps = 10 seconds between samples |
| n_spatial | 20 | Edie spatial bins |
| n_temporal | 30 | Edie temporal bins |
| bus_interval | 30 | 30-minute headway (baseline) |
| crashes | 0 | Disabled for clean baseline |
| seeds | 42-47 | 6 replicates per percentile |
| percentiles | 20,40,60,70,80,85,90,95,97 | 9 demand levels |
