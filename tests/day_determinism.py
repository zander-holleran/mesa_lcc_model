"""
Day-level determinism test.

Verifies that running the model twice with the same seed produces identical outputs
for a single day simulation.

Configuration:
- 1000 persons
- 1 day
- bus_interval=4 mins (ensures at least one bus departs)
- crashes enabled

Usage:
    python tests/day_determinism.py
"""

import pandas as pd
import sys
from pathlib import Path


from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config, PopulationParams, ScheduleSpecs


def run_day_determinism_test(seed: int = 12345, verbose: bool = True) -> bool:
    """
    Run determinism test for a single day.

    Args:
        seed: Random seed for reproducibility
        verbose: Print progress and results

    Returns:
        True if test passes, False otherwise
    """
    config = make_season_config(
        season_id="day_determinism_test",
        run_description="Day-level determinism test",
        seed=seed,
        n_days=1,
        max_persons=1000,
        max_steps=50000,
        collect_every_n=10,
        batch_run=True,
        road_path="data/roads/hw210_sl_and_curvs.parquet",
        ecs_path="data/vehicle_counts/expected_counts_seconds.csv",
        traffic_percentile_schedule=ScheduleSpecs("static", 50),
        bus_interval_schedule=ScheduleSpecs("static", 4),  # 4 min interval
        crashes_schedule=ScheduleSpecs("static", 5),
        population_params=PopulationParams(population_size=1000),
    )

    if verbose:
        print("=" * 60)
        print("DAY DETERMINISM TEST")
        print("=" * 60)
        print(f"Config: 1000 persons, 1 day, bus_interval=4min, seed={seed}")
        print()

    # Run 1
    if verbose:
        print("Running model (run 1)...")
    orch1 = SeasonOrchestrator(config, store_data=False)
    orch1.run_day()
    model1 = orch1.last_model_run

    model_ts1 = model1.datacollector.get_model_vars_dataframe()
    trip_log1 = orch1.get_trip_log_df()

    run1_stats = {
        "steps": model1.steps,
        "crashes": model1.total_crashes,
        "finished": len(model1.finished_agents),
    }

    if verbose:
        print(f"  Run 1: steps={run1_stats['steps']}, crashes={run1_stats['crashes']}, finished={run1_stats['finished']}")

    # Run 2
    if verbose:
        print("Running model (run 2)...")
    orch2 = SeasonOrchestrator(config, store_data=False)
    orch2.run_day()
    model2 = orch2.last_model_run

    model_ts2 = model2.datacollector.get_model_vars_dataframe()
    trip_log2 = orch2.get_trip_log_df()

    run2_stats = {
        "steps": model2.steps,
        "crashes": model2.total_crashes,
        "finished": len(model2.finished_agents),
    }

    if verbose:
        print(f"  Run 2: steps={run2_stats['steps']}, crashes={run2_stats['crashes']}, finished={run2_stats['finished']}")
        print()

    # Compare results
    all_passed = True

    # Check basic stats match
    if run1_stats != run2_stats:
        if verbose:
            print("FAIL: Basic stats differ between runs")
            print(f"  Run 1: {run1_stats}")
            print(f"  Run 2: {run2_stats}")
        all_passed = False

    # Check model time series
    try:
        pd.testing.assert_frame_equal(
            model_ts1.reset_index(drop=True),
            model_ts2.reset_index(drop=True)
        )
        if verbose:
            print("PASS: Model time series identical")
    except AssertionError as e:
        if verbose:
            print(f"FAIL: Model time series differ")
            print(f"  {str(e)[:200]}")
        all_passed = False

    # Check trip log (use tolerance for floating point columns)
    try:
        if trip_log1 is not None and trip_log2 is not None:
            pd.testing.assert_frame_equal(
                trip_log1.reset_index(drop=True),
                trip_log2.reset_index(drop=True),
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

    if verbose:
        print()
        print("=" * 60)
        if all_passed:
            print("DAY DETERMINISM TEST: PASSED")
        else:
            print("DAY DETERMINISM TEST: FAILED")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run day-level determinism test")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    passed = run_day_determinism_test(seed=args.seed, verbose=not args.quiet)
    sys.exit(0 if passed else 1)
