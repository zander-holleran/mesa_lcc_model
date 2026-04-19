"""
Helpers for constructing traffic_percentile schedules in three categories:
  - static: same TP every day
  - normal: TP drawn from a normal distribution (clipped to 1-100)
  - realistic: TP sequence extracted from real 2023 SR-210 count data

Each helper returns a ScheduleSpecs object ready for make_season_config().
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import truncnorm

from season.configs import ScheduleSpecs


# ---------------------------------------------------------------------------
# Load + cache real traffic data
# ---------------------------------------------------------------------------

_REAL_TP_CACHE: pd.DataFrame | None = None


def _load_real_tp(
    csv_path: str = "data/vehicle_counts/210_real_count_data.csv",
) -> pd.DataFrame:
    """Load real count data and compute per-day traffic percentile.

    Returns DataFrame indexed by date with columns: TOTAL, tp, month.
    """
    global _REAL_TP_CACHE
    if _REAL_TP_CACHE is not None:
        return _REAL_TP_CACHE

    df = pd.read_csv(csv_path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="mixed")

    # upbound lanes only (PLane*)
    up_df = df[df["LANE"].str.startswith("P")]
    hour_cols = [c for c in df.columns if c.startswith("H")]
    daily = up_df.groupby("DATE")[hour_cols].sum()
    daily["TOTAL"] = daily.sum(axis=1)
    daily = daily.sort_index()
    daily["tp"] = np.ceil(daily["TOTAL"].rank(pct=True) * 100).astype(int)
    daily["month"] = daily.index.month
    daily.index.name = "date"

    _REAL_TP_CACHE = daily
    return _REAL_TP_CACHE


# ---------------------------------------------------------------------------
# Category 1: Static
# ---------------------------------------------------------------------------

def static_schedule(tp: int) -> ScheduleSpecs:
    """Same traffic percentile every day.

    Category label: 'static'
    """
    return ScheduleSpecs(mode="static", value=tp)


# ---------------------------------------------------------------------------
# Category 2: Normal distribution
# ---------------------------------------------------------------------------

def normal_schedule(mean: float, std: float = 10.0) -> ScheduleSpecs:
    """Traffic percentile drawn from truncated N(mean, std) bounded to [1, 100].

    Category label: 'normal'
    """
    a, b = (1 - mean) / std, (100 - mean) / std
    frozen = truncnorm(a, b, loc=mean, scale=std)
    return ScheduleSpecs(mode="dist", dist=frozen, round_to_int=True)


# ---------------------------------------------------------------------------
# Category 3: Realistic (from real SR-210 data)
# ---------------------------------------------------------------------------

def realistic_schedule(
    n_days: int,
    target_mean_tp: float,
    tolerance: float = 5.0,
    months: list[int] | None = None,
    csv_path: str = "data/vehicle_counts/210_real_count_data.csv",
    verbose: bool = True,
) -> ScheduleSpecs:
    """Extract an n_day TP sequence from real 2023 SR-210 count data near a target mean.

    Selects a window of consecutive observed days (in calendar order, continuing
    through gaps in the data) whose mean TP is closest to target_mean_tp.

    For example, if the data has observations on Jan 3, 4, 5, 7, 8 (gap on Jan 6),
    a 5-day window starting at Jan 3 yields the TPs for those 5 observed days
    in the order they occurred.

    Parameters
    ----------
    n_days : Number of days in the schedule.
    target_mean_tp : Desired mean traffic percentile for the window.
    tolerance : Warning threshold — prints a warning if best window deviates more.
    months : Optional filter — only consider days in these months (1-12).
        Use [1,2,3,4,12] for ski season, [5,6,7,8,9,10,11] for off-season.
    csv_path : Path to the real count data CSV.
    verbose : Print the selected date range and TP values.

    Returns
    -------
    ScheduleSpecs with mode="list" containing the TP sequence.
    """
    daily = _load_real_tp(csv_path)

    if months is not None:
        daily = daily[daily["month"].isin(months)]

    if len(daily) < n_days:
        raise ValueError(
            f"Only {len(daily)} days available after filtering "
            f"(need {n_days}). Try relaxing the months filter."
        )

    tp_values = daily["tp"].values
    dates = daily.index

    best_start = 0
    best_diff = float("inf")

    for i in range(len(tp_values) - n_days + 1):
        window_mean = tp_values[i : i + n_days].mean()
        diff = abs(window_mean - target_mean_tp)
        if diff < best_diff:
            best_diff = diff
            best_start = i

    if best_diff > tolerance:
        import warnings
        warnings.warn(
            f"Best window mean TP ({tp_values[best_start:best_start+n_days].mean():.1f}) "
            f"deviates {best_diff:.1f} from target {target_mean_tp}. "
            f"Consider adjusting target or months filter.",
            UserWarning,
        )

    selected = daily.iloc[best_start : best_start + n_days]
    tp_list = selected["tp"].tolist()
    date_start = selected.index[0].strftime("%Y-%m-%d")
    date_end = selected.index[-1].strftime("%Y-%m-%d")
    actual_mean = np.mean(tp_list)

    # identify gaps in the selected window
    sel_dates = pd.Series(selected.index)
    gaps = sel_dates.diff().dt.days
    gap_days = gaps[gaps > 1]

    if verbose:
        print(f"Realistic schedule: {date_start} to {date_end} ({n_days} observed days)")
        print(f"  Mean TP: {actual_mean:.1f} (target: {target_mean_tp})")
        print(f"  Range: [{min(tp_list)}, {max(tp_list)}], std: {np.std(tp_list):.1f}")
        if not gap_days.empty:
            print(f"  Gaps: {len(gap_days)} gap(s) in date coverage "
                  f"(largest: {int(gap_days.max())-1} missing day(s))")
        print(f"  TPs: {tp_list}")

    return ScheduleSpecs(mode="list", value=tp_list)


# ---------------------------------------------------------------------------
# Category label helper
# ---------------------------------------------------------------------------

def schedule_category(spec: ScheduleSpecs) -> str:
    """Return the category label for a ScheduleSpecs: 'static', 'normal', or 'realistic'."""
    if spec.mode == "static":
        return "static"
    elif spec.mode == "list":
        return "realistic"
    elif spec.mode == "dist":
        return "normal"
    return "unknown"


def schedule_summary(spec: ScheduleSpecs, n_days: int | None = None) -> dict:
    """Return a summary dict for a schedule (useful for sweep analysis metadata).

    Keys: schedule_category, schedule_mean_tp, schedule_min_tp, schedule_max_tp,
    schedule_tp_std, schedule_varies.
    """
    cat = schedule_category(spec)

    if spec.mode == "static":
        v = spec.value
        return {
            "schedule_category": cat,
            "schedule_mean_tp": float(v),
            "schedule_min_tp": float(v),
            "schedule_max_tp": float(v),
            "schedule_tp_std": 0.0,
            "schedule_varies": False,
        }

    if spec.mode == "list":
        vals = spec.value
        return {
            "schedule_category": cat,
            "schedule_mean_tp": float(np.mean(vals)),
            "schedule_min_tp": float(min(vals)),
            "schedule_max_tp": float(max(vals)),
            "schedule_tp_std": float(np.std(vals)),
            "schedule_varies": True,
        }

    if spec.mode == "dist":
        mean = spec.dist.mean()
        std = spec.dist.std()
        return {
            "schedule_category": cat,
            "schedule_mean_tp": float(np.clip(mean, 1, 100)),
            "schedule_min_tp": float(np.clip(mean - 2 * std, 1, 100)),
            "schedule_max_tp": float(np.clip(mean + 2 * std, 1, 100)),
            "schedule_tp_std": float(std),
            "schedule_varies": True,
        }

    return {"schedule_category": "unknown"}


def plot_schedule(spec: ScheduleSpecs, n_days: int = 14, seed: int = 42, title: str | None = None):
    """Line plot of TP by day (left) and histogram of TP values (right)."""
    rng = np.random.default_rng(seed)
    vals = spec.realize(rng, n_days)
    cat = schedule_category(spec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [2, 1]})

    ax1.plot(range(len(vals)), vals, marker="o", markersize=4, linewidth=1.2)
    ax1.axhline(np.mean(vals), color="red", linestyle="--", linewidth=0.8, label=f"mean={np.mean(vals):.0f}")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Traffic Percentile")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.hist(vals, bins=range(0, 105, 5), orientation="horizontal", color="steelblue", edgecolor="white")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Count")
    ax2.set_yticklabels([])
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title or f"{cat} schedule (n={n_days}, mean={np.mean(vals):.0f})", fontsize=11)
    plt.tight_layout()
    plt.show()
