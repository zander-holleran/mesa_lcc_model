"""
Shared performance benchmarking infrastructure.

Provides scenario configs, collector presets, and benchmark utilities
used by performance_congestion.py, performance_free_flow.py, and compare_collectors.py.
"""

import sys
import time
from pathlib import Path


import pandas as pd
from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config, PopulationParams, ScheduleSpecs

BASELINE_DIR = Path(__file__).parent / "baselines"
DEFAULT_SEED = 99999

# All collector preset names. "pre" means no DataCollectionConfig (all tiers off by default).
COLLECTOR_PRESETS = ["pre", "small", "medium", "large", "matched", "lean"]


# ── Collector config registry ─────────────────────────────────────────────────

def get_collector_config(size: str):
    """
    Return a DataCollectionConfig for the given preset name, or None for 'pre'/'medium'.

    Presets:
        pre / medium : None (use TrafficModel default — all tiers off)
        small        : tier1 only, every 10 steps, minimal scalars
        large        : all tiers enabled, every step
        matched      : replicates old reporting.py (tier1 every 100 steps)
        lean         : leaner than old system (every 300 steps, 3 scalars)
    """
    from traffic.model.hybrid_collector import DataCollectionConfig, Tier1Config, Tier2Config, Tier4Config

    if size == "small":
        return DataCollectionConfig(
            tier1=Tier1Config(
                interval=10,
                scalars=["step", "vehicle_count", "persons_finished"],
                histograms=[],
            ),
        )
    elif size == "large":
        return DataCollectionConfig(
            tier1=Tier1Config(interval=1),
            tier2=Tier2Config(sample_interval=5, max_samples=10000),
            tier3=True,
            tier4=Tier4Config(snapshot_interval=500, snapshot_on_crash=True),
        )
    elif size == "matched":
        return DataCollectionConfig(
            tier1=Tier1Config(
                interval=100,
                scalars=[
                    "step", "current_toll", "vehicle_count",
                    "persons_at_bus_stop", "bus_mode_share_recent",
                ],
                histograms=[],
            ),
        )
    elif size == "lean":
        return DataCollectionConfig(
            tier1=Tier1Config(
                interval=300,
                scalars=["step", "vehicle_count", "persons_finished"],
                histograms=[],
            ),
        )
    else:  # "pre" or "medium" — use TrafficModel's built-in default
        return None


# ── Scenario config factories ─────────────────────────────────────────────────

def get_congestion_config(seed: int = DEFAULT_SEED):
    """Standard config for the congestion performance scenario (90th percentile, 2 days)."""
    return make_season_config(
        season_id="perf_congestion",
        run_description="Performance test - congestion",
        seed=seed,
        n_days=2,
        max_persons=3000,
        max_steps=50000,
        road_path="data/roads/hw210_sl_and_curvs.parquet",
        ecs_path="data/vehicle_counts/expected_counts_seconds.csv",
        traffic_percentile_schedule=ScheduleSpecs("static", 90),
        bus_interval_schedule=ScheduleSpecs("static", 30),
        crashes_schedule=ScheduleSpecs("static", 5),
        population_params=PopulationParams(population_size=3000),
    )


def get_free_flow_config(seed: int = DEFAULT_SEED):
    """Standard config for the free-flow performance scenario (50th percentile, 2 days)."""
    return make_season_config(
        season_id="perf_free_flow",
        run_description="Performance test - free flow",
        seed=seed,
        n_days=2,
        max_persons=3000,
        max_steps=50000,
        road_path="data/roads/hw210_sl_and_curvs.parquet",
        ecs_path="data/vehicle_counts/expected_counts_seconds.csv",
        traffic_percentile_schedule=ScheduleSpecs("static", 50),
        bus_interval_schedule=ScheduleSpecs("static", 30),
        crashes_schedule=ScheduleSpecs("static", 5),
        population_params=PopulationParams(population_size=3000),
    )


SCENARIOS = {
    "congestion": get_congestion_config,
    "free_flow": get_free_flow_config,
}


# ── Core benchmark utilities ──────────────────────────────────────────────────

def get_baseline_paths(scenario: str, collector_label: str):
    """Return (metrics_path, trip_log_path) for a scenario + collector label."""
    return (
        BASELINE_DIR / f"performance_{scenario}_{collector_label}_metrics.csv",
        BASELINE_DIR / f"performance_{scenario}_{collector_label}_trip_log.parquet",
    )


def run_single_benchmark(make_config, collector_config=None):
    """Run one benchmark iteration. Returns (metrics_dict, trip_log)."""
    import tempfile
    config = make_config()
    if collector_config is not None:
        config.data_collection = collector_config

    start = time.perf_counter()
    orch = SeasonOrchestrator(config, output_root_dir=tempfile.mkdtemp(), silent=True)
    orch.run_season()
    elapsed = time.perf_counter() - start

    trip_log = orch.get_trip_log_df()
    total_trips = len(trip_log) if trip_log is not None else 0
    total_steps = orch.last_model_run.steps if orch.last_model_run else 0

    return {
        "elapsed_sec": elapsed,
        "total_trips": total_trips,
        "total_steps": total_steps,
        "steps_per_sec": total_steps / elapsed if elapsed > 0 else 0,
        "trips_per_sec": total_trips / elapsed if elapsed > 0 else 0,
    }, trip_log


def run_performance_test(make_config, scenario_label: str, num_runs: int = 3,
                         verbose: bool = True, collector_config=None,
                         collector_label: str = "pre") -> tuple:
    """
    Run N benchmark iterations and return (results_dict, trip_log).

    results_dict includes avg/min/max timing and step/trip counts.
    """
    if verbose:
        print("=" * 60)
        print(f"PERFORMANCE TEST: {scenario_label.upper()} [{collector_label}]")
        print("=" * 60)
        print(f"Number of runs: {num_runs}")
        print()

    times, all_steps, all_trips = [], [], []
    trip_log = None

    for i in range(num_runs):
        if verbose:
            print(f"Run {i + 1}/{num_runs}...")

        metrics, trip_log = run_single_benchmark(make_config, collector_config)
        times.append(metrics["elapsed_sec"])
        all_steps.append(metrics["total_steps"])
        all_trips.append(metrics["total_trips"])

        if verbose:
            print(f"  Completed in {metrics['elapsed_sec']:.2f}s, "
                  f"{metrics['total_steps']} steps, "
                  f"{metrics['steps_per_sec']:.0f} steps/s")

    avg_time = sum(times) / len(times)
    total_steps = all_steps[-1]
    total_trips = all_trips[-1]

    results = {
        "scenario": scenario_label,
        "collector": collector_label,
        "num_runs": num_runs,
        "avg_time_sec": avg_time,
        "min_time_sec": min(times),
        "max_time_sec": max(times),
        "total_steps": total_steps,
        "total_trips": total_trips,
        "avg_steps_per_sec": total_steps / avg_time if avg_time > 0 else 0,
        "trips_per_sec": total_trips / avg_time if avg_time > 0 else 0,
    }

    if verbose:
        print()
        print("=" * 60)
        print(f"RESULTS [{collector_label}]")
        print("=" * 60)
        print(f"Total steps:    {total_steps}")
        print(f"Total trips:    {total_trips}")
        print()
        print(f"Average time:   {avg_time:.2f}s")
        print(f"Min time:       {min(times):.2f}s")
        print(f"Max time:       {max(times):.2f}s")
        print()
        print(f"Avg steps/sec:  {results['avg_steps_per_sec']:.0f}")
        print(f"Trips/second:   {results['trips_per_sec']:.0f}")
        print("=" * 60)

    return results, trip_log


def save_baseline(make_config, scenario: str, num_runs: int = 3, verbose: bool = True,
                  collector_label: str = "pre", collector_config=None) -> bool:
    """Save baseline performance metrics + trip log for later comparison."""
    if verbose:
        print("=" * 60)
        print(f"SAVING BASELINE: {scenario.upper()} [{collector_label}]")
        print("=" * 60)
        print()

    BASELINE_DIR.mkdir(exist_ok=True)
    metrics_path, trip_log_path = get_baseline_paths(scenario, collector_label)

    results, trip_log = run_performance_test(
        make_config, scenario, num_runs=num_runs, verbose=verbose,
        collector_config=collector_config, collector_label=collector_label,
    )

    pd.DataFrame([results]).to_csv(metrics_path, index=False)
    trip_log.to_parquet(trip_log_path)

    if verbose:
        print()
        print(f"Baseline saved to {BASELINE_DIR}/")
        print(f"  - {metrics_path.name}")
        print(f"  - {trip_log_path.name}")
        print("=" * 60)
        print("BASELINE SAVED SUCCESSFULLY")
        print("=" * 60)

    return True


def verify_against_baseline(make_config, scenario: str, num_runs: int = 3,
                             verbose: bool = True, collector_label: str = "pre",
                             collector_config=None) -> bool:
    """
    Run current config and verify outputs match the saved 'pre' baseline.

    Always compares against the 'pre' baseline regardless of collector_label.
    """
    if verbose:
        print("=" * 60)
        print(f"VERIFYING AGAINST BASELINE: {scenario.upper()} [{collector_label}]")
        print("=" * 60)
        print()

    metrics_path, trip_log_path = get_baseline_paths(scenario, "pre")
    if not metrics_path.exists():
        print(f"ERROR: No 'pre' baseline found for scenario '{scenario}'.")
        print("Run with --save-baseline first.")
        return False

    baseline = pd.read_csv(metrics_path).iloc[0].to_dict()
    baseline_trip_log = pd.read_parquet(trip_log_path)

    if verbose:
        print(f"Pre baseline: {baseline['avg_time_sec']:.2f}s avg, "
              f"{baseline['avg_steps_per_sec']:.0f} steps/s")
        print()

    results, trip_log = run_performance_test(
        make_config, scenario, num_runs=num_runs, verbose=verbose,
        collector_config=collector_config, collector_label=collector_label,
    )

    all_passed = True

    if verbose:
        print()
        print("-" * 40)
        print("OUTPUT VERIFICATION")
        print("-" * 40)

    if results["total_trips"] != baseline["total_trips"]:
        if verbose:
            print(f"FAIL: Trip count differs "
                  f"(baseline: {baseline['total_trips']}, current: {results['total_trips']})")
        all_passed = False

    if results["total_steps"] != baseline["total_steps"]:
        if verbose:
            print(f"FAIL: Step count differs "
                  f"(baseline: {baseline['total_steps']}, current: {results['total_steps']})")
        all_passed = False

    try:
        pd.testing.assert_frame_equal(
            trip_log.reset_index(drop=True),
            baseline_trip_log.reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
        )
        if verbose and all_passed:
            print("PASS: Outputs identical to baseline")
    except AssertionError as e:
        if verbose:
            print(f"FAIL: Trip log differs")
            print(f"  {str(e)[:200]}")
        all_passed = False

    if verbose:
        print()
        print("-" * 40)
        print("PERFORMANCE COMPARISON")
        print("-" * 40)
        _print_perf_comparison(baseline, results)
        print()
        print("=" * 60)
        if all_passed:
            print("PERFORMANCE CHECK: PASSED")
            print("Outputs are identical to baseline.")
        else:
            print("PERFORMANCE CHECK: FAILED")
            print("Outputs differ from baseline!")
        print("=" * 60)

    return all_passed


def clean_baseline(scenario: str, collector_label: str = "pre", verbose: bool = True) -> bool:
    """Remove baseline files for a scenario + collector label."""
    if verbose:
        print(f"Removing baseline for '{scenario}' [{collector_label}]...")

    metrics_path, trip_log_path = get_baseline_paths(scenario, collector_label)
    removed = False
    for f in [metrics_path, trip_log_path]:
        if f.exists():
            f.unlink()
            if verbose:
                print(f"  Removed {f.name}")
            removed = True

    if not removed and verbose:
        print("  No baseline files found")

    return True


def _print_perf_comparison(baseline: dict, results: dict):
    baseline_time = baseline["avg_time_sec"]
    current_time = results["avg_time_sec"]
    speedup = (baseline_time - current_time) / baseline_time * 100

    baseline_sps = baseline["avg_steps_per_sec"]
    current_sps = results["avg_steps_per_sec"]
    sps_change = (current_sps - baseline_sps) / baseline_sps * 100

    print(f"Baseline time:      {baseline_time:.2f}s")
    print(f"Current time:       {current_time:.2f}s")
    if speedup > 0:
        print(f"Time change:        {speedup:.1f}% faster")
    elif speedup < 0:
        print(f"Time change:        {-speedup:.1f}% slower")
    else:
        print(f"Time change:        No change")

    print()
    print(f"Baseline steps/s:   {baseline_sps:.0f}")
    print(f"Current steps/s:    {current_sps:.0f}")
    if sps_change > 0:
        print(f"Throughput change:  {sps_change:.1f}% faster")
    elif sps_change < 0:
        print(f"Throughput change:  {-sps_change:.1f}% slower")
    else:
        print(f"Throughput change:  No change")


# ── Multi-config comparison ───────────────────────────────────────────────────

def compare_collector_configs(scenario: str, configs_to_run: list = None,
                               num_runs: int = 1, verbose: bool = True,
                               seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """
    Run multiple collector configs, verify determinism against 'pre', print comparison table.

    'pre' is always run first as the reference, even if not explicitly listed.
    Each other config's trip log is compared against 'pre' for output correctness.

    Args:
        scenario:       "congestion" or "free_flow"
        configs_to_run: collector preset names to run; defaults to ["pre", "matched", "lean"]
        num_runs:       timing iterations per config (default 1)
        verbose:        print progress and table
        seed:           random seed

    Returns:
        DataFrame with one row per config: config, avg_time_sec, steps_per_sec,
        total_steps, total_trips, vs_pre_pct, output_match.
    """
    if configs_to_run is None:
        configs_to_run = ["pre", "matched", "lean"]

    # Ensure "pre" runs first as reference; deduplicate while preserving order.
    ordered = ["pre"] + [c for c in configs_to_run if c != "pre"]

    make_config = lambda: SCENARIOS[scenario](seed=seed)

    if verbose:
        print("=" * 60)
        print(f"COLLECTOR COMPARISON: {scenario.upper()}")
        print(f"Configs:        {', '.join(ordered)}")
        print(f"Runs per config: {num_runs}")
        print("=" * 60)
        print()

    all_rows = []
    reference_trip_log = None
    reference_steps = None
    reference_trips = None
    reference_sps = None

    for label in ordered:
        collector_config = get_collector_config(label)

        if verbose:
            print(f"Running [{label}]...")

        # First run gets the trip log; subsequent runs only used for timing.
        metrics, trip_log = run_single_benchmark(make_config, collector_config)
        times = [metrics["elapsed_sec"]]
        for _ in range(num_runs - 1):
            m, _ = run_single_benchmark(make_config, collector_config)
            times.append(m["elapsed_sec"])

        avg_time = sum(times) / len(times)
        sps = metrics["total_steps"] / avg_time if avg_time > 0 else 0

        if verbose:
            print(f"  {avg_time:.1f}s avg  |  {sps:.0f} steps/s")

        # Determinism check against "pre"
        if label == "pre":
            reference_trip_log = trip_log
            reference_steps = metrics["total_steps"]
            reference_trips = metrics["total_trips"]
            reference_sps = sps
            output_match = "(ref)"
        else:
            match = (
                metrics["total_trips"] == reference_trips
                and metrics["total_steps"] == reference_steps
            )
            if match and reference_trip_log is not None:
                try:
                    pd.testing.assert_frame_equal(
                        trip_log.reset_index(drop=True),
                        reference_trip_log.reset_index(drop=True),
                        check_exact=False,
                        rtol=1e-10,
                    )
                except AssertionError:
                    match = False
            output_match = "PASS" if match else "FAIL"

        vs_pre_pct = (
            (sps - reference_sps) / reference_sps * 100
            if reference_sps and label != "pre"
            else None
        )

        all_rows.append({
            "config": label,
            "avg_time_sec": avg_time,
            "steps_per_sec": sps,
            "total_steps": metrics["total_steps"],
            "total_trips": metrics["total_trips"],
            "vs_pre_pct": vs_pre_pct,
            "output_match": output_match,
        })

    df = pd.DataFrame(all_rows)

    if verbose:
        print()
        print("=" * 60)
        print(f"RESULTS: {scenario.upper()}")
        print("=" * 60)
        col_w = max(len(c) for c in ordered) + 2
        header = f"{'Config':<{col_w}} | {'Steps/s':>8} | {'vs Pre':>10} | Output"
        print(header)
        print("-" * len(header))
        for _, row in df.iterrows():
            vs = f"{row['vs_pre_pct']:+.1f}%" if row["vs_pre_pct"] is not None else "(ref)"
            print(f"{row['config']:<{col_w}} | {row['steps_per_sec']:>8.0f} | {vs:>10} | {row['output_match']}")
        print("=" * 60)
        fails = df[df["output_match"] == "FAIL"]
        if len(fails) > 0:
            print(f"OUTPUT MISMATCH: {', '.join(fails['config'].tolist())} differ from pre!")
        else:
            print("All outputs match 'pre' reference.")
        print("=" * 60)

    return df
