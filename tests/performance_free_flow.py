"""
Performance test under free-flow traffic conditions.

Benchmarks model execution speed with moderate traffic (50th percentile).
Supports baseline save/verify workflow for tracking performance changes.

Configuration:
- 3000 persons
- 2 days
- traffic_percentile=50 (free-flow)
- bus_interval=30 mins
- crashes_per_100k_vmt=5

Usage:
    python tests/performance_free_flow.py                              # Run benchmark (medium collector)
    python tests/performance_free_flow.py --save-baseline              # Save as baseline (label: pre)
    python tests/performance_free_flow.py --verify --collector small   # Compare to pre baseline using small config
    python tests/performance_free_flow.py --clean                      # Remove baseline for current label
    python tests/performance_free_flow.py --runs 5                     # More runs for stable average
    python tests/performance_free_flow.py --collector {small,medium,large}  # Select collector config
"""

import time
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config, PopulationParams, ScheduleSpecs

BASELINE_DIR = Path(__file__).parent / "baselines"
SEED = 99999


def get_baseline_paths(collector_label: str):
    """Return (metrics_path, trip_log_path) for the given collector label."""
    return (
        BASELINE_DIR / f"performance_free_flow_{collector_label}_metrics.csv",
        BASELINE_DIR / f"performance_free_flow_{collector_label}_trip_log.parquet",
    )


def get_collector_config(size: str):
    """Return a HybridCollectorConfig for the given size label, or None for 'medium'."""
    from traffic.model.hybrid_collector import HybridCollectorConfig
    if size == "small":
        return HybridCollectorConfig(
            tier1_interval=10,
            tier1_scalars=["step", "vehicle_count", "total_finished"],
            tier1_histograms=[],
            tier2_enabled=False,
            tier3_enabled=False,
            tier4_enabled=False,
        )
    elif size == "large":
        return HybridCollectorConfig(
            tier1_interval=1,
            tier2_enabled=True,
            tier2_sample_interval=5,
            tier2_max_samples=10000,
            tier3_enabled=True,
            tier4_enabled=True,
            tier4_snapshot_interval=500,
            tier4_snapshot_on_crash=True,
        )
    else:  # "medium" — matches current default behavior
        return None


def get_config():
    """Return the standard config for free-flow performance test."""
    return make_season_config(
        season_id="perf_free_flow",
        run_description="Performance test - free flow",
        seed=SEED,
        n_days=2,
        max_persons=3000,
        max_steps=50000,
        collect_every_n=100,
        batch_run=True,
        road_path="data/roads/hw210_sl_and_curvs.parquet",
        ecs_path="data/vehicle_counts/expected_counts_seconds.csv",
        traffic_percentile_schedule=ScheduleSpecs("static", 50),
        bus_interval_schedule=ScheduleSpecs("static", 30),
        crashes_schedule=ScheduleSpecs("static", 5),
        population_params=PopulationParams(population_size=3000),
    )


def run_single_benchmark(collector_config=None):
    """Run a single benchmark and return metrics."""
    config = get_config()
    if collector_config is not None:
        config.hybrid_collector_config = collector_config

    start = time.perf_counter()
    orch = SeasonOrchestrator(config, store_data=False)
    orch.run_season()
    elapsed = time.perf_counter() - start

    trip_log = orch.get_trip_log_df()
    total_trips = len(trip_log) if trip_log is not None else 0

    # Get total steps from last model (for single season, this is cumulative)
    # For multi-day runs, we track total trips as the throughput metric
    total_steps = orch.last_model_run.steps if orch.last_model_run else 0

    return {
        "elapsed_sec": elapsed,
        "total_trips": total_trips,
        "total_steps": total_steps,
        "steps_per_sec": total_steps / elapsed if elapsed > 0 else 0,
        "trips_per_sec": total_trips / elapsed if elapsed > 0 else 0,
    }, trip_log


def run_performance_test(num_runs: int = 3, verbose: bool = True,
                         collector_config=None, collector_label: str = "pre") -> dict:
    """
    Run performance benchmark under free-flow conditions.

    Args:
        num_runs: Number of runs to average
        verbose: Print progress and results
        collector_config: Optional HybridCollectorConfig to inject
        collector_label: Label for display (pre/small/medium/large)

    Returns:
        Dictionary with performance metrics
    """
    if verbose:
        print("=" * 60)
        print(f"PERFORMANCE TEST: FREE FLOW (50th percentile) [{collector_label}]")
        print("=" * 60)
        print(f"Config: 3000 persons, 2 days, traffic_percentile=50, seed={SEED}")
        print(f"Number of runs: {num_runs}")
        print()

    times = []
    all_steps = []
    all_trips = []
    trip_log = None

    for i in range(num_runs):
        if verbose:
            print(f"Run {i + 1}/{num_runs}...")

        metrics, trip_log = run_single_benchmark(collector_config=collector_config)
        times.append(metrics["elapsed_sec"])
        all_steps.append(metrics["total_steps"])
        all_trips.append(metrics["total_trips"])

        if verbose:
            print(f"  Completed in {metrics['elapsed_sec']:.2f}s, "
                  f"{metrics['total_steps']} steps, "
                  f"{metrics['steps_per_sec']:.0f} steps/s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    total_steps = all_steps[-1]  # Same for all runs (deterministic)
    total_trips = all_trips[-1]

    results = {
        "scenario": "free_flow",
        "traffic_percentile": 50,
        "persons": 3000,
        "days": 2,
        "num_runs": num_runs,
        "avg_time_sec": avg_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
        "total_steps": total_steps,
        "total_trips": total_trips,
        "avg_steps_per_sec": total_steps / avg_time if avg_time > 0 else 0,
        "trips_per_sec": total_trips / avg_time if avg_time > 0 else 0,
    }

    if verbose:
        print()
        print("=" * 60)
        print("RESULTS: FREE FLOW")
        print("=" * 60)
        print(f"Total steps:    {total_steps}")
        print(f"Total trips:    {total_trips}")
        print()
        print(f"Average time:   {avg_time:.2f}s")
        print(f"Min time:       {min_time:.2f}s")
        print(f"Max time:       {max_time:.2f}s")
        print()
        print(f"Avg steps/sec:  {results['avg_steps_per_sec']:.0f}")
        print(f"Trips/second:   {results['trips_per_sec']:.0f}")
        print("=" * 60)

    return results, trip_log


def save_baseline(num_runs: int = 3, verbose: bool = True,
                  collector_label: str = "pre", collector_config=None) -> bool:
    """Save baseline performance metrics for later comparison."""
    if verbose:
        print("=" * 60)
        print(f"SAVING PERFORMANCE BASELINE: FREE FLOW [{collector_label}]")
        print("=" * 60)
        print()

    BASELINE_DIR.mkdir(exist_ok=True)
    metrics_path, trip_log_path = get_baseline_paths(collector_label)

    results, trip_log = run_performance_test(
        num_runs=num_runs, verbose=verbose,
        collector_config=collector_config, collector_label=collector_label,
    )

    pd.DataFrame([results]).to_csv(metrics_path, index=False)
    trip_log.to_parquet(trip_log_path)

    if verbose:
        print()
        print(f"Baseline saved to {BASELINE_DIR}/")
        print(f"  - {metrics_path.name}")
        print(f"  - {trip_log_path.name}")
        print()
        print("=" * 60)
        print("BASELINE SAVED SUCCESSFULLY")
        print("=" * 60)

    return True


def verify_against_baseline(num_runs: int = 3, verbose: bool = True,
                            collector_label: str = "pre", collector_config=None) -> bool:
    """Verify current performance against saved baseline."""
    if verbose:
        print("=" * 60)
        print(f"VERIFYING AGAINST PERFORMANCE BASELINE: FREE FLOW [{collector_label}]")
        print("=" * 60)
        print()

    metrics_path, trip_log_path = get_baseline_paths("pre")
    if not metrics_path.exists():
        print("ERROR: No 'pre' baseline found. Run with --save-baseline on main branch first.")
        return False

    # Load pre baseline
    baseline = pd.read_csv(metrics_path).iloc[0].to_dict()
    baseline_trip_log = pd.read_parquet(trip_log_path)

    if verbose:
        print(f"Pre baseline: {baseline['avg_time_sec']:.2f}s avg, "
              f"{baseline['avg_steps_per_sec']:.0f} steps/s")
        print()

    results, trip_log = run_performance_test(
        num_runs=num_runs, verbose=verbose,
        collector_config=collector_config, collector_label=collector_label,
    )

    # Compare outputs (determinism check)
    all_passed = True

    if verbose:
        print()
        print("-" * 40)
        print("OUTPUT VERIFICATION")
        print("-" * 40)

    # Check trip counts match
    if results["total_trips"] != baseline["total_trips"]:
        if verbose:
            print(f"FAIL: Trip count differs (baseline: {baseline['total_trips']}, "
                  f"current: {results['total_trips']})")
        all_passed = False

    if results["total_steps"] != baseline["total_steps"]:
        if verbose:
            print(f"FAIL: Step count differs (baseline: {baseline['total_steps']}, "
                  f"current: {results['total_steps']})")
        all_passed = False

    # Check trip log matches
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

    # Performance comparison
    if verbose:
        print()
        print("-" * 40)
        print("PERFORMANCE COMPARISON")
        print("-" * 40)
        baseline_time = baseline['avg_time_sec']
        current_time = results['avg_time_sec']
        speedup = (baseline_time - current_time) / baseline_time * 100

        baseline_sps = baseline['avg_steps_per_sec']
        current_sps = results['avg_steps_per_sec']
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

    if verbose:
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


def clean_baseline(verbose: bool = True, collector_label: str = "pre") -> bool:
    """Remove baseline files for the given collector label."""
    if verbose:
        print(f"Removing baseline files for label '{collector_label}'...")

    metrics_path, trip_log_path = get_baseline_paths(collector_label)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Free-flow performance test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow for pre vs post HybridCollector comparison:
  1. On main branch:  python tests/performance_free_flow.py --save-baseline
  2. On hybrid branch: python tests/performance_free_flow.py --verify --collector small
                       python tests/performance_free_flow.py --verify --collector medium
                       python tests/performance_free_flow.py --verify --collector large
        """
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current run as baseline (label: pre when no --collector)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify against pre baseline")
    parser.add_argument("--clean", action="store_true",
                        help="Remove baseline files for current collector label")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--collector", choices=["small", "medium", "large"], default=None,
                        help="HybridCollectorConfig size preset (omit to use pre/default)")
    args = parser.parse_args()

    if args.seed != SEED:
        SEED = args.seed

    collector_label = args.collector if args.collector else "pre"
    collector_config = get_collector_config(args.collector) if args.collector else None

    if args.clean:
        clean_baseline(verbose=not args.quiet, collector_label=collector_label)
        sys.exit(0)

    if args.save_baseline:
        save_baseline(num_runs=args.runs, verbose=not args.quiet,
                      collector_label=collector_label, collector_config=collector_config)
        sys.exit(0)

    if args.verify:
        passed = verify_against_baseline(num_runs=args.runs, verbose=not args.quiet,
                                         collector_label=collector_label,
                                         collector_config=collector_config)
        sys.exit(0 if passed else 1)

    # Default: just run the benchmark
    run_performance_test(num_runs=args.runs, verbose=not args.quiet,
                         collector_config=collector_config, collector_label=collector_label)
