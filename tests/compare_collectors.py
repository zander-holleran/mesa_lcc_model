"""
Evaluate multiple HybridCollectorConfig presets against a scenario in a single run.

Always runs 'pre' (no HybridCollectorConfig) as the reference. Each other config
is checked for output determinism against 'pre', and timing is compared.

Usage:
    # Compare default set (pre, matched, lean) on congestion scenario:
    python tests/compare_collectors.py

    # Select scenario and configs:
    python tests/compare_collectors.py --scenario free_flow
    python tests/compare_collectors.py --scenario congestion --collectors pre small medium large matched lean

    # Control runs per config and seed:
    python tests/compare_collectors.py --runs 2 --seed 42

    # Save results to CSV:
    python tests/compare_collectors.py --out results.csv

Available scenarios:   congestion, free_flow
Available collectors:  pre, small, medium, large, matched, lean
"""

import sys
from pathlib import Path


from perf_utils import (
    compare_collector_configs, SCENARIOS, COLLECTOR_PRESETS, DEFAULT_SEED,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare HybridCollectorConfig presets for a scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        default="congestion",
        help="Traffic scenario to benchmark (default: congestion)",
    )
    parser.add_argument(
        "--collectors",
        nargs="+",
        choices=COLLECTOR_PRESETS,
        default=["pre", "matched", "lean"],
        metavar="COLLECTOR",
        help=(
            "Collector presets to compare. 'pre' is always run as reference even if omitted. "
            f"Choices: {COLLECTOR_PRESETS} (default: pre matched lean)"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of timing iterations per config (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="PATH",
        help="Save comparison table to CSV at this path",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args()

    df = compare_collector_configs(
        scenario=args.scenario,
        configs_to_run=args.collectors,
        num_runs=args.runs,
        verbose=not args.quiet,
        seed=args.seed,
    )

    if args.out:
        out_path = Path(args.out)
        df.to_csv(out_path, index=False)
        if not args.quiet:
            print(f"Results saved to {out_path}")

    # Exit non-zero if any config failed output check
    fails = df[df["output_match"] == "FAIL"]
    sys.exit(1 if len(fails) > 0 else 0)
