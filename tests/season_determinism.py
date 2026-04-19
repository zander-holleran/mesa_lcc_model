"""
Season-level determinism test.

Verifies that running the model twice with the same seed produces identical outputs
across a multi-day season simulation.

Configuration:
- 1000 persons
- 4 days
- bus_interval=4 mins (ensures at least one bus departs)
- crashes enabled

Usage:
    python tests/season_determinism.py
"""

import pandas as pd
import sys
import tempfile
from pathlib import Path


from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config, PopulationParams, ScheduleSpecs


def run_season_determinism_test(seed: int = 12345, verbose: bool = True) -> bool:
    """
    Run determinism test for a full season (4 days).

    Args:
        seed: Random seed for reproducibility
        verbose: Print progress and results

    Returns:
        True if test passes, False otherwise
    """
    config = make_season_config(
        season_id="season_determinism_test",
        run_description="Season-level determinism test",
        seed=seed,
        n_days=4,
        max_persons=1000,
        max_steps=50000,
        road_path="data/roads/hw210_sl_and_curvs.parquet",
        ecs_path="data/vehicle_counts/expected_counts_seconds.csv",
        traffic_percentile_schedule=ScheduleSpecs("static", 50),
        bus_interval_schedule=ScheduleSpecs("static", 4),  # 4 min interval
        crashes_schedule=ScheduleSpecs("static", 5),
        population_params=PopulationParams(population_size=1000),
    )

    if verbose:
        print("=" * 60)
        print("SEASON DETERMINISM TEST")
        print("=" * 60)
        print(f"Config: 1000 persons, 4 days, bus_interval=4min, seed={seed}")
        print()

    # Run 1
    if verbose:
        print("Running season (run 1)...")
    orch1 = SeasonOrchestrator(config, output_root_dir=tempfile.mkdtemp(), silent=True)
    orch1.run_season()

    trip_log1 = orch1.get_trip_log_df()
    day_summary1 = orch1.get_day_summary_df()
    sp_log1 = orch1.get_season_person_log_df()

    if verbose:
        print(f"  Run 1: {len(trip_log1)} trips, {len(day_summary1)} day summaries")

    # Run 2
    if verbose:
        print("Running season (run 2)...")
    orch2 = SeasonOrchestrator(config, output_root_dir=tempfile.mkdtemp(), silent=True)
    orch2.run_season()

    trip_log2 = orch2.get_trip_log_df()
    day_summary2 = orch2.get_day_summary_df()
    sp_log2 = orch2.get_season_person_log_df()

    if verbose:
        print(f"  Run 2: {len(trip_log2)} trips, {len(day_summary2)} day summaries")
        print()

    # Compare results
    all_passed = True

    # Check trip log
    try:
        if trip_log1 is not None and trip_log2 is not None:
            pd.testing.assert_frame_equal(
                trip_log1.reset_index(drop=True),
                trip_log2.reset_index(drop=True)
            )
        if verbose:
            print("PASS: Trip log identical")
    except AssertionError as e:
        if verbose:
            print(f"FAIL: Trip log differs")
            print(f"  {str(e)[:200]}")
        all_passed = False

    # Check day summaries
    try:
        if day_summary1 is not None and day_summary2 is not None:
            pd.testing.assert_frame_equal(
                day_summary1.reset_index(drop=True),
                day_summary2.reset_index(drop=True)
            )
        if verbose:
            print("PASS: Day summaries identical")
    except AssertionError as e:
        if verbose:
            print(f"FAIL: Day summaries differ")
            print(f"  {str(e)[:200]}")
        all_passed = False

    # Check season person log (excluding 'history' column which contains lists)
    try:
        if sp_log1 is not None and sp_log2 is not None:
            # Compare only scalar columns
            cols_to_compare = [c for c in sp_log1.columns if c != "history"]
            pd.testing.assert_frame_equal(
                sp_log1[cols_to_compare].reset_index(drop=True),
                sp_log2[cols_to_compare].reset_index(drop=True)
            )
        if verbose:
            print("PASS: Season person log identical")
    except AssertionError as e:
        if verbose:
            print(f"FAIL: Season person log differs")
            print(f"  {str(e)[:200]}")
        all_passed = False

    if verbose:
        print()
        print("=" * 60)
        if all_passed:
            print("SEASON DETERMINISM TEST: PASSED")
        else:
            print("SEASON DETERMINISM TEST: FAILED")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run season-level determinism test")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    passed = run_season_determinism_test(seed=args.seed, verbose=not args.quiet)
    sys.exit(0 if passed else 1)
