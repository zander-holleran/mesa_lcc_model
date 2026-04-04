"""
Optimization verification test.

Verifies that code changes (optimizations) produce identical outputs to a saved baseline.

Workflow:
    1. Before making optimization changes, save a baseline:
       python tests/optimization_check.py --save-baseline

    2. Make your optimization changes to the code

    3. Verify outputs match the baseline:
       python tests/optimization_check.py --verify

Configuration:
- 1000 persons
- 1 day
- bus_interval=4 mins
- crashes enabled
- seed=12345

Usage:
    python tests/optimization_check.py --save-baseline  # Save baseline outputs
    python tests/optimization_check.py --verify         # Compare against baseline
    python tests/optimization_check.py --clean          # Remove baseline files
"""

import time
import pandas as pd
import sys
from pathlib import Path


from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config, PopulationParams, ScheduleSpecs

# Baseline file paths
BASELINE_DIR = Path(__file__).parent / "baselines"
BASELINE_TRIP_LOG = BASELINE_DIR / "optimization_baseline_trip_log.parquet"
BASELINE_MODEL_TS = BASELINE_DIR / "optimization_baseline_model_ts.parquet"
BASELINE_META = BASELINE_DIR / "optimization_baseline_meta.csv"

SEED = 12345


def get_config():
    """Return the standard config for optimization checks."""
    return make_season_config(
        season_id="optimization_check",
        run_description="Optimization verification test",
        seed=SEED,
        n_days=1,
        max_persons=1000,
        max_steps=50000,
        collect_every_n=10,
        batch_run=True,
        road_path="data/roads/hw210_sl_and_curvs.parquet",
        ecs_path="data/vehicle_counts/expected_counts_seconds.csv",
        traffic_percentile_schedule=ScheduleSpecs("static", 50),
        bus_interval_schedule=ScheduleSpecs("static", 4),
        crashes_schedule=ScheduleSpecs("static", 5),
        population_params=PopulationParams(population_size=1000),
    )


def run_model():
    """Run the model and return outputs with timing."""
    config = get_config()

    start = time.perf_counter()
    orch = SeasonOrchestrator(config, store_data=False)
    orch.run_day()
    elapsed = time.perf_counter() - start

    model = orch.last_model_run
    model_ts = model.datacollector.get_model_vars_dataframe()
    trip_log = orch.get_trip_log_df()

    meta = {
        "steps": model.steps,
        "crashes": model.total_crashes,
        "finished": len(model.finished_agents),
        "elapsed_sec": elapsed,
    }

    return model_ts, trip_log, meta


def save_baseline(verbose: bool = True) -> bool:
    """Save baseline outputs for later comparison."""
    if verbose:
        print("=" * 60)
        print("SAVING OPTIMIZATION BASELINE")
        print("=" * 60)
        print(f"Config: 1000 persons, 1 day, bus_interval=4min, seed={SEED}")
        print()

    # Create baseline directory if needed
    BASELINE_DIR.mkdir(exist_ok=True)

    if verbose:
        print("Running model...")

    model_ts, trip_log, meta = run_model()

    if verbose:
        print(f"  Steps: {meta['steps']}")
        print(f"  Crashes: {meta['crashes']}")
        print(f"  Finished: {meta['finished']}")
        print(f"  Time: {meta['elapsed_sec']:.2f}s")
        print()

    # Save outputs
    model_ts.to_parquet(BASELINE_MODEL_TS)
    trip_log.to_parquet(BASELINE_TRIP_LOG)
    pd.DataFrame([meta]).to_csv(BASELINE_META, index=False)

    if verbose:
        print(f"Baseline saved to {BASELINE_DIR}/")
        print(f"  - {BASELINE_MODEL_TS.name}")
        print(f"  - {BASELINE_TRIP_LOG.name}")
        print(f"  - {BASELINE_META.name}")
        print()
        print("=" * 60)
        print("BASELINE SAVED SUCCESSFULLY")
        print("=" * 60)

    return True


def verify_against_baseline(verbose: bool = True) -> bool:
    """Verify current outputs match the saved baseline."""
    if verbose:
        print("=" * 60)
        print("VERIFYING AGAINST OPTIMIZATION BASELINE")
        print("=" * 60)
        print(f"Config: 1000 persons, 1 day, bus_interval=4min, seed={SEED}")
        print()

    # Check baseline exists
    if not BASELINE_TRIP_LOG.exists():
        print("ERROR: No baseline found. Run with --save-baseline first.")
        return False

    # Load baseline
    baseline_model_ts = pd.read_parquet(BASELINE_MODEL_TS)
    baseline_trip_log = pd.read_parquet(BASELINE_TRIP_LOG)
    baseline_meta = pd.read_csv(BASELINE_META).iloc[0].to_dict()

    if verbose:
        print(f"Baseline: {baseline_meta['steps']} steps, {baseline_meta['elapsed_sec']:.2f}s")
        print()
        print("Running model...")

    model_ts, trip_log, meta = run_model()

    if verbose:
        print(f"  Steps: {meta['steps']}")
        print(f"  Crashes: {meta['crashes']}")
        print(f"  Finished: {meta['finished']}")
        print(f"  Time: {meta['elapsed_sec']:.2f}s")
        print()

    # Compare results
    all_passed = True

    # Check basic stats match
    for key in ["steps", "crashes", "finished"]:
        if meta[key] != baseline_meta[key]:
            if verbose:
                print(f"FAIL: {key} differs (baseline: {baseline_meta[key]}, current: {meta[key]})")
            all_passed = False

    if all_passed and verbose:
        print("PASS: Basic stats match")

    # Check model time series
    try:
        pd.testing.assert_frame_equal(
            model_ts.reset_index(drop=True),
            baseline_model_ts.reset_index(drop=True)
        )
        if verbose:
            print("PASS: Model time series identical")
    except AssertionError as e:
        if verbose:
            print(f"FAIL: Model time series differ")
            print(f"  {str(e)[:200]}")
        all_passed = False

    # Check trip log
    try:
        pd.testing.assert_frame_equal(
            trip_log.reset_index(drop=True),
            baseline_trip_log.reset_index(drop=True),
            check_exact=False,
            rtol=1e-10,
        )
        if verbose:
            print("PASS: Trip log identical")
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
        baseline_time = baseline_meta['elapsed_sec']
        current_time = meta['elapsed_sec']
        speedup = (baseline_time - current_time) / baseline_time * 100

        print(f"Baseline time:  {baseline_time:.2f}s")
        print(f"Current time:   {current_time:.2f}s")
        if speedup > 0:
            print(f"Speedup:        {speedup:.1f}% faster")
        elif speedup < 0:
            print(f"Slowdown:       {-speedup:.1f}% slower")
        else:
            print(f"Change:         No change")

    if verbose:
        print()
        print("=" * 60)
        if all_passed:
            print("OPTIMIZATION CHECK: PASSED")
            print("Outputs are identical to baseline.")
        else:
            print("OPTIMIZATION CHECK: FAILED")
            print("Outputs differ from baseline - optimization changed behavior!")
        print("=" * 60)

    return all_passed


def clean_baseline(verbose: bool = True) -> bool:
    """Remove baseline files."""
    if verbose:
        print("Removing baseline files...")

    removed = False
    for f in [BASELINE_MODEL_TS, BASELINE_TRIP_LOG, BASELINE_META]:
        if f.exists():
            f.unlink()
            if verbose:
                print(f"  Removed {f.name}")
            removed = True

    if BASELINE_DIR.exists() and not any(BASELINE_DIR.iterdir()):
        BASELINE_DIR.rmdir()
        if verbose:
            print(f"  Removed {BASELINE_DIR.name}/")

    if not removed and verbose:
        print("  No baseline files found")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimization verification test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. python tests/optimization_check.py --save-baseline
  2. Make optimization changes to the code
  3. python tests/optimization_check.py --verify
        """
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save current outputs as baseline"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify outputs match baseline"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove baseline files"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    args = parser.parse_args()

    if not any([args.save_baseline, args.verify, args.clean]):
        parser.print_help()
        sys.exit(1)

    if args.clean:
        clean_baseline(verbose=not args.quiet)
        sys.exit(0)

    if args.save_baseline:
        save_baseline(verbose=not args.quiet)
        sys.exit(0)

    if args.verify:
        passed = verify_against_baseline(verbose=not args.quiet)
        sys.exit(0 if passed else 1)
