"""
Performance test under free-flow traffic conditions.

Benchmarks model execution speed with moderate traffic (50th percentile).

Configuration:
- 3000 persons
- 3 days
- traffic_percentile=50 (free-flow)
- bus_interval=30 mins
- crashes_per_100k_vmt=5

Usage:
    python tests/performance_free_flow.py
    python tests/performance_free_flow.py --runs 5  # More runs for stable average
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from season.season_orchestrator import SeasonOrchestrator
from season.configs import make_season_config, PopulationParams, ScheduleSpecs


def run_performance_test(
    seed: int = 99999,
    num_runs: int = 3,
    verbose: bool = True
) -> dict:
    """
    Run performance benchmark under free-flow conditions.

    Args:
        seed: Random seed for reproducibility
        num_runs: Number of runs to average
        verbose: Print progress and results

    Returns:
        Dictionary with performance metrics
    """
    config = make_season_config(
        season_id="perf_free_flow",
        run_description="Performance test - free flow",
        seed=seed,
        n_days=3,
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

    if verbose:
        print("=" * 60)
        print("PERFORMANCE TEST: FREE FLOW (50th percentile)")
        print("=" * 60)
        print(f"Config: 3000 persons, 3 days, traffic_percentile=50")
        print(f"Number of runs: {num_runs}")
        print()

    times = []
    total_steps = 0
    total_trips = 0

    for i in range(num_runs):
        if verbose:
            print(f"Run {i + 1}/{num_runs}...")

        start = time.perf_counter()
        orch = SeasonOrchestrator(config, store_data=False)
        orch.run_season()
        elapsed = time.perf_counter() - start

        times.append(elapsed)

        # Capture stats from last run
        trip_log = orch.get_trip_log_df()
        total_trips = len(trip_log) if trip_log is not None else 0

        # Sum steps across all days
        # (We only have access to last model, so estimate from trips)
        if verbose:
            print(f"  Completed in {elapsed:.2f}s, {total_trips} trips")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    results = {
        "scenario": "free_flow",
        "traffic_percentile": 50,
        "persons": 3000,
        "days": 3,
        "num_runs": num_runs,
        "avg_time_sec": avg_time,
        "min_time_sec": min_time,
        "max_time_sec": max_time,
        "total_trips": total_trips,
        "trips_per_sec": total_trips / avg_time if avg_time > 0 else 0,
    }

    if verbose:
        print()
        print("=" * 60)
        print("RESULTS: FREE FLOW")
        print("=" * 60)
        print(f"Average time:   {avg_time:.2f}s")
        print(f"Min time:       {min_time:.2f}s")
        print(f"Max time:       {max_time:.2f}s")
        print(f"Total trips:    {total_trips}")
        print(f"Trips/second:   {results['trips_per_sec']:.0f}")
        print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run free-flow performance test")
    parser.add_argument("--seed", type=int, default=99999, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    results = run_performance_test(
        seed=args.seed,
        num_runs=args.runs,
        verbose=not args.quiet
    )
