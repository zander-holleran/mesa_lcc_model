"""
Performance test under congested traffic conditions.

Benchmarks model execution speed with heavy traffic (90th percentile).
Supports baseline save/verify workflow for tracking performance changes.

Configuration:
- 3000 persons
- 2 days
- traffic_percentile=90 (congested)
- bus_interval=30 mins
- crashes_per_100k_vmt=5

Usage:
    python tests/performance_congestion.py                              # Run benchmark
    python tests/performance_congestion.py --save-baseline              # Save as 'pre' baseline
    python tests/performance_congestion.py --verify --collector matched # Compare to pre baseline
    python tests/performance_congestion.py --clean                      # Remove baseline for label
    python tests/performance_congestion.py --runs 3                     # More runs for stable avg
    python tests/performance_congestion.py --collector {small,medium,large,matched,lean}
"""

import sys
from pathlib import Path


from perf_utils import (
    get_collector_config, get_congestion_config,
    run_performance_test, save_baseline, verify_against_baseline, clean_baseline,
    DEFAULT_SEED, COLLECTOR_PRESETS,
)

SCENARIO = "congestion"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Congestion performance test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Save pre baseline:  python tests/performance_congestion.py --save-baseline
  2. Verify any config:  python tests/performance_congestion.py --verify --collector matched
  3. Clean up:           python tests/performance_congestion.py --clean

For multi-config comparison in one shot, use compare_collectors.py instead.
        """
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current run as 'pre' baseline")
    parser.add_argument("--verify", action="store_true",
                        help="Run and verify outputs match 'pre' baseline")
    parser.add_argument("--clean", action="store_true",
                        help="Remove baseline files for current collector label")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--collector", choices=COLLECTOR_PRESETS[1:], default=None,
                        help="Collector preset (omit for 'pre'/default)")
    args = parser.parse_args()

    seed = args.seed
    make_config = lambda: get_congestion_config(seed=seed)
    collector_label = args.collector or "pre"
    collector_config = get_collector_config(args.collector) if args.collector else None

    if args.clean:
        clean_baseline(SCENARIO, collector_label, verbose=not args.quiet)
        sys.exit(0)

    if args.save_baseline:
        save_baseline(make_config, SCENARIO, args.runs, not args.quiet,
                      collector_label, collector_config)
        sys.exit(0)

    if args.verify:
        passed = verify_against_baseline(make_config, SCENARIO, args.runs, not args.quiet,
                                         collector_label, collector_config)
        sys.exit(0 if passed else 1)

    # Default: just run the benchmark
    run_performance_test(make_config, SCENARIO, args.runs, not args.quiet,
                         collector_config, collector_label)
