import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from IPython.display import display

import re
import numpy as np
from pprint import pprint

#============================================ LOAD DATA ============================================
BASE_DIR = Path('data/outputs/seasons')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DATA_ITEM_TITLES = {
    'season_summary': 'Season Summary - one row per SEASON, with aggregate metrics',
    'day_summary': 'Day Summary - one row per DAY, with aggregate metrics',
    'trip_log': 'Trip Log - one row per TRIP, with trip-level outcomes',
    'season_person_log': 'Season Person Log - one row per PERSON, with season-level outcomes',
    'sp_day_summary': 'SP Day Summary - one row per DAY, with social planner metrics',
    'model_ts': 'Model Time Series - one dataframe per DAY, with collected Tier 1 metrics',
    'spatial': 'Spatial Data - one dataframe per DAY, with collected spatial metrics',
    'road_gdf': 'Road GeoDataFrame - road geometry and segment attributes for the run',
}
 # e.g., 'my_season_id'

def list_runs(base_dir: Path = BASE_DIR):
    if not base_dir.exists():
        return []
    return sorted([p.name for p in base_dir.iterdir() if p.is_dir()])

def _read_parquet_optional(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path)

def load_model_ts(run_dir: Path):
    model_ts = {}
    if not run_dir.exists():
        return model_ts
    pattern = re.compile(r'^day_(\d+)_model_ts\.parquet$')
    for p in sorted(run_dir.glob('day_*_model_ts.parquet')):
        m = pattern.match(p.name)
        if not m:
            continue
        day_idx = int(m.group(1))
        model_ts[day_idx] = pd.read_parquet(p)
    return model_ts

def load_spatial(run_dir: Path):
    spatial = {}
    if not run_dir.exists():
        return spatial
    pattern = re.compile(r'^day_(\d+)_spatial\.parquet$')
    for p in sorted(run_dir.glob('day_*_spatial.parquet')):
        m = pattern.match(p.name)
        if not m:
            continue
        day_idx = int(m.group(1))
        spatial[day_idx] = pd.read_parquet(p)
    return spatial

def load_road_gdf(run_dir: Path):
    config_path = run_dir / 'season_config.json'
    if not config_path.exists():
        return None

    cfg = json.loads(config_path.read_text())
    road_path = cfg.get('road_path')
    if not road_path:
        return None

    return gpd.read_parquet(PROJECT_ROOT / road_path)

def load_run(run_id: str, base_dir: Path = BASE_DIR, verbose: bool = True):
    run_dir = base_dir / run_id

    data = {
        'run_dir': run_dir,
        'trip_log': _read_parquet_optional(run_dir / 'trip_log.parquet'),
        'day_summary': _read_parquet_optional(run_dir / 'day_summary.parquet'),
        'season_person_log': _read_parquet_optional(run_dir / 'season_person_log.parquet'),
        'sp_day_summary': _read_parquet_optional(run_dir / 'sp_day_summary.parquet'),
        'model_ts': load_model_ts(run_dir),
        'spatial': load_spatial(run_dir),
        'road_gdf': load_road_gdf(run_dir),
        'season_summary': _read_parquet_optional(run_dir / 'season_summary.parquet'),
    }

    if data['spatial'] and data['trip_log'] is not None:
        data['spatial'] = merge_season_person_ids(data['spatial'], data['trip_log'])

    if verbose:
        run_data_keys = {
            k: (
                list(v.keys()) if k in {'model_ts', 'spatial'}
                else (None if v is None else getattr(v, 'shape', None))
            )
            for k, v in data.items()
        }
        pprint(run_data_keys)

    return data


def merge_season_person_ids(spatial_df_dict: dict, trip_log: pd.DataFrame) -> dict:
    """Merge season_person_id from trip_log into each spatial DataFrame.

    Note: trip_log only contains entries for vehicles that reached end_of_road().
    Vehicles that got stuck and never completed their trip will have no trip_log
    entry, so they will appear as has_person=False even if they carried a person.

    Args:
        spatial_df_dict: dict {day_index: DataFrame} (e.g. run_data['spatial']).
        trip_log: DataFrame with columns vehicle_id, season_person_id, day_index.

    Returns:
        New dict {day_index: DataFrame} with season_person_id column added (left join;
        NaN for bus agents or rows with no matching trip record).
    """
    id_map = (
        trip_log[['day_index', 'vehicle_id', 'season_person_id']]
        .dropna(subset=['vehicle_id'])
        .drop_duplicates(subset=['day_index', 'vehicle_id'])
        .astype({'vehicle_id': int})
    )

    result = {}
    for day_index, spatial_df in spatial_df_dict.items():
        day_map = (
            id_map[id_map['day_index'] == day_index][['vehicle_id', 'season_person_id']]
            .rename(columns={'vehicle_id': 'AgentID'})
        )
        merged = spatial_df.merge(day_map, on='AgentID', how='left')
        merged['has_person'] = merged['season_person_id'].notna()
        result[day_index] = merged

    return result


def load_run_data_item(run_data: dict, key: str, day_index: int | None = None):
    print('=' * 100)

    if key not in run_data:
        print(f'No data found for key: {key}')
        print('=' * 100)
        return None

    value = run_data[key]
    print(RUN_DATA_ITEM_TITLES.get(key, f'{key} data'))

    if isinstance(value, dict):
        if day_index is None:
            print('Returning all days')
            print('=' * 100)
            if not value:
                print(f'No {key} data')
                return value

            first_day = sorted(value)[0]
            print(f'Displaying head for day {first_day}')
            display(value[first_day].head(3))
            return value

        print(f'Returning day {day_index}')
        print('=' * 100)
        day_value = value.get(day_index)
        if day_value is None:
            print(f'No {key} data for day {day_index}')
            return None
        display(day_value.head(3))
        return day_value

    print('=' * 100)
    if value is None:
        print(f'No {key} data')
        return None

    if hasattr(value, 'head'):
        display(value.head(3))
    else:
        display(value)
    return value


#============================================ SWEEP ANALYSIS ============================================

def _parse_scipy_dist(val):
    """Extract a human-readable repr and median from a serialized scipy dist or scalar.

    Returns (repr_or_value, median_or_None).
    """
    if isinstance(val, (int, float)):
        return val, float(val)
    if isinstance(val, dict) and val.get("_type") == "rv_continuous_frozen":
        dist_info = val.get("dist", {})
        name = dist_info.get("name", "unknown")
        args = val.get("args", [])
        kwds = val.get("kwds", {})
        parts = [str(a) for a in args] + [f"{k}={v}" for k, v in kwds.items()]
        repr_str = f"{name}({', '.join(parts)})"
        # best-effort median
        loc = kwds.get("loc", 0)
        scale = kwds.get("scale", 1)
        median = None
        if name == "lognorm" and loc == 0:
            median = scale
        elif name == "norm":
            median = loc
        return repr_str, median
    return str(val), None


def _flatten_config(cfg: dict) -> dict:
    """Flatten a season_config.json dict into a single-level dict for DataFrame columns."""
    row = {}

    # --- top-level SeasonConfig scalars ---
    for key in (
        "season_id", "run_description", "seed", "n_days",
        "max_steps", "max_persons", "max_concurrent_vehicles",
        "start_hr", "bus_capacity", "road_path", "ecs_path",
        "bus_user_fee", "sweep_id", "car_preference",
    ):
        row[key] = cfg.get(key)

    # --- toll config ---
    tc = cfg.get("toll_config") or {}
    static_toll = tc.get("_static_toll")
    transform = tc.get("transform")
    signal = tc.get("signal")

    if static_toll is not None:
        toll_type = "static"
    elif transform is None:
        toll_type = "none"
    else:
        _type = transform.get("_type", "") if isinstance(transform, dict) else ""
        if "PI" in _type:
            toll_type = "PI"
        elif "Piecewise" in _type:
            toll_type = "piecewise"
        elif "Step" in _type:
            toll_type = "step"
        else:
            toll_type = _type or "unknown"

    row["toll_type"] = toll_type
    row["toll_static_amount"] = static_toll
    row["toll_update_every_n_steps"] = tc.get("update_every_n_steps")
    row["toll_rounding"] = tc.get("rounding")
    row["toll_cap"] = tc.get("cap")
    row["toll_floor"] = tc.get("floor")

    # signal fields
    if isinstance(signal, dict):
        row["toll_signal_type"] = signal.get("_type")
        row["toll_signal_window_steps"] = signal.get("window_steps")
    else:
        row["toll_signal_type"] = None
        row["toll_signal_window_steps"] = None

    # transform fields — initialize all to None, then populate from transform dict
    transform_fields = {
        "toll_target": None, "toll_kp": None, "toll_ki": None,
        "toll_toll_min": None, "toll_toll_max": None,
        "toll_reset_integral_on_target": None,
        "toll_threshold": None, "toll_slope": None, "toll_base": None,
        "toll_step_amount": None,
    }
    if isinstance(transform, dict):
        field_map = {
            "target": "toll_target", "kp": "toll_kp", "ki": "toll_ki",
            "toll_min": "toll_toll_min", "toll_max": "toll_toll_max",
            "reset_integral_on_target": "toll_reset_integral_on_target",
            "threshold": "toll_threshold", "slope": "toll_slope",
            "base": "toll_base", "toll": "toll_step_amount",
        }
        for src_key, dst_key in field_map.items():
            if src_key in transform:
                transform_fields[dst_key] = transform[src_key]
        # auto-extract any unknown transform fields
        for k, v in transform.items():
            if k != "_type" and k not in field_map:
                col = f"toll_{k}"
                if col not in transform_fields:
                    transform_fields[col] = v
    row.update(transform_fields)

    # --- population params ---
    pp = cfg.get("population_params") or {}
    row["pop_population_size"] = pp.get("population_size")
    distributable = [
        "value_of_time", "experience_weight_car", "experience_weight_bus",
        "prior_car", "prior_bus", "time_decay_rate", "prior_weight",
        "uncertainty_multiplier", "travel_propensity",
    ]
    for field in distributable:
        val = pp.get(field)
        repr_val, median = _parse_scipy_dist(val) if val is not None else (None, None)
        row[f"pop_{field}"] = repr_val
        row[f"pop_{field}_median"] = median

    # --- day params (aggregated) ---
    day_params = cfg.get("day_params") or []
    for field, col_base in [
        ("traffic_percentile", "day_traffic_percentile"),
        ("bus_interval", "day_bus_interval"),
        ("crashes_per_100k_vmt_input", "day_crashes"),
    ]:
        values = [dp.get(field) for dp in day_params if dp.get(field) is not None]
        if not values:
            row[col_base] = None
            row[f"{col_base}_min"] = None
            row[f"{col_base}_max"] = None
            row[f"{col_base}_std"] = None
            row[f"{col_base}_varies"] = None
        elif len(set(values)) == 1:
            row[col_base] = values[0]
            row[f"{col_base}_min"] = values[0]
            row[f"{col_base}_max"] = values[0]
            row[f"{col_base}_std"] = 0.0
            row[f"{col_base}_varies"] = False
        else:
            row[col_base] = np.mean(values)
            row[f"{col_base}_min"] = min(values)
            row[f"{col_base}_max"] = max(values)
            row[f"{col_base}_std"] = float(np.std(values))
            row[f"{col_base}_varies"] = True

    # --- bus cost config ---
    bc = cfg.get("bus_cost_config")
    bc_fields = [
        "layover_recovery_pct", "return_trip_factor", "spare_ratio",
        "bus_purchase_price", "useful_life_years",
        "driver_hourly_rate", "relief_factor",
        "ops_cost_per_hour", "overhead_multiplier",
        "service_hours_per_day", "service_days_per_year",
    ]
    for field in bc_fields:
        row[f"bus_cost_cfg_{field}"] = bc.get(field) if isinstance(bc, dict) else None

    # --- data collection ---
    dc = cfg.get("data_collection") or {}
    row["dc_tier1"] = dc.get("tier1") is not None and dc.get("tier1") is not False
    row["dc_tier2"] = dc.get("tier2") is not None and dc.get("tier2") is not False
    row["dc_tier3"] = bool(dc.get("tier3"))
    row["dc_tier4"] = dc.get("tier4") is not None and dc.get("tier4") is not False

    return row


def _load_config_json(config_path: Path) -> dict | None:
    """Load and sanitize a season_config.json file."""
    try:
        text = config_path.read_text()
        # scipy dists serialize NaN/Infinity as JS literals
        text = text.replace(": NaN", ": null")
        text = text.replace(": Infinity", ": 1e308")
        text = text.replace(": -Infinity", ": -1e308")
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None


def compute_season_metrics(
    trip_log: pd.DataFrame,
    warmup_days: int = 0,
    median_vot: float | None = None,
) -> dict | None:
    """Recompute season-level metrics from trip_log, optionally excluding warmup days.

    Parameters
    ----------
    trip_log : DataFrame with columns from SeasonOrchestrator trip log.
    warmup_days : Number of initial days to exclude (day_index >= warmup_days).
    median_vot : Median value-of-time for VOT-standardized cost.
        If None, defaults to 0.0 (VOT-standardized metrics = toll only).

    Returns dict matching _aggregate_trips schema plus days_included/warmup_days_excluded,
    or None if no trips remain.
    """
    from season.season_orchestrator import SeasonOrchestrator

    df = trip_log[trip_log["day_index"] >= warmup_days]
    if df.empty:
        return None

    if median_vot is None:
        median_vot = 0.0

    metrics = SeasonOrchestrator._aggregate_trips(df, median_vot)
    if metrics is None:
        return None

    metrics["days_included"] = int(df["day_index"].nunique())
    metrics["warmup_days_excluded"] = warmup_days
    return metrics


def build_sweep_analysis_df(
    season_ids: list[str] | None = None,
    base_dir: Path = BASE_DIR,
    warmup_days: int = 3,
) -> pd.DataFrame:
    """Build a wide DataFrame joining config params with outcome metrics across seasons.

    Works incrementally — safely skips seasons still running (missing trip_log.parquet).
    Each row = one completed season with all flattened config columns + aggregate metrics.

    Parameters
    ----------
    season_ids : Specific season IDs to load, or None to scan base_dir.
    base_dir : Root directory containing season subdirectories.
    warmup_days : Number of initial days to exclude when computing metrics.
    """
    if season_ids is None:
        season_ids = list_runs(base_dir)

    rows = []
    skipped = []

    for sid in season_ids:
        season_dir = base_dir / sid
        config_path = season_dir / "season_config.json"
        trip_log_path = season_dir / "trip_log.parquet"

        if not config_path.exists() or not trip_log_path.exists():
            skipped.append(sid)
            continue

        cfg = _load_config_json(config_path)
        if cfg is None:
            skipped.append(sid)
            continue

        row = _flatten_config(cfg)

        # extract median_vot from config
        pp = cfg.get("population_params") or {}
        vot_val = pp.get("value_of_time")
        if vot_val is not None:
            _, median_vot = _parse_scipy_dist(vot_val)
            if median_vot is None:
                median_vot = 0.0
        else:
            median_vot = 0.0

        # compute warmup-excluded metrics
        trip_log = pd.read_parquet(trip_log_path)
        metrics = compute_season_metrics(trip_log, warmup_days=warmup_days, median_vot=median_vot)
        if metrics is not None:
            row.update(metrics)

        # merge any extra columns from season_summary (includes _sweep_params)
        season_summary_path = season_dir / "season_summary.parquet"
        if season_summary_path.exists():
            ss = pd.read_parquet(season_summary_path).iloc[0].to_dict()
            for k, v in ss.items():
                if k not in row:
                    row[k] = v

        # extract bus operational inputs from day_summary for post-hoc recalculation
        day_summary_path = season_dir / "day_summary.parquet"
        if day_summary_path.exists():
            day_df = pd.read_parquet(day_summary_path)
            post_warmup = day_df[day_df["day_index"] >= warmup_days]
            if not post_warmup.empty:
                row["bus_avg_onboard_time"] = post_warmup["avg_onboard_time_bus"].mean()
                row["bus_avg_steps_per_day"] = post_warmup["steps"].mean() if "steps" in post_warmup.columns else None
            else:
                row["bus_avg_onboard_time"] = None
                row["bus_avg_steps_per_day"] = None
        else:
            row["bus_avg_onboard_time"] = None
            row["bus_avg_steps_per_day"] = None

        rows.append(row)

    if skipped:
        print(f"Skipped {len(skipped)} incomplete/in-progress seasons"
              + (f": {skipped[:5]}..." if len(skipped) > 5 else f": {skipped}"))

    return pd.DataFrame(rows)


def recompute_bus_costs(
    analysis_df: pd.DataFrame,
    bus_cost_config=None,
) -> pd.DataFrame:
    """Recalculate bus cost columns using alternative BusCostConfig coefficients.

    Uses simulation-derived columns already in analysis_df:
      - bus_avg_onboard_time (avg bus onboard time in minutes)
      - day_bus_interval (headway in minutes)
      - bus_avg_steps_per_day (avg model steps per day)

    Returns a copy of analysis_df with bus_cost_* columns replaced.
    """
    from traffic.model.bus_system_cost import BusCostConfig, compute_bus_costs

    if bus_cost_config is None:
        bus_cost_config = BusCostConfig.default()

    df = analysis_df.copy()

    # columns that compute_bus_costs produces
    cost_cols = [
        "bus_cost_cycle_time_min", "bus_cost_active_buses", "bus_cost_total_fleet",
        "bus_cost_per_hour", "bus_cost_labor_per_hour", "bus_cost_ops_per_hour",
        "bus_cost_capital_per_hour", "bus_cost_model_run", "bus_cost_daily", "bus_cost_annual",
    ]
    for col in cost_cols:
        if col not in df.columns:
            df[col] = None

    for idx, row in df.iterrows():
        headway = row.get("day_bus_interval")
        onboard = row.get("bus_avg_onboard_time")
        steps = row.get("bus_avg_steps_per_day")

        if (
            headway is not None and headway > 0
            and onboard is not None and not np.isnan(onboard)
            and steps is not None and not np.isnan(steps)
        ):
            costs = compute_bus_costs(
                config=bus_cost_config,
                avg_one_way_tt_min=float(onboard),
                headway_min=int(headway),
                model_steps=int(steps),
            )
            if costs is not None:
                for k, v in costs.items():
                    df.at[idx, k] = v
            else:
                for col in cost_cols:
                    df.at[idx, col] = None
        else:
            for col in cost_cols:
                df.at[idx, col] = None

    return df


# #============================================ PLOTTING FUNCTIONS #============================================
def plot_realized_cost_means_with_total(trip_log):
    # mean by day and mode
    summary_mode = (
        trip_log
        .groupby(["day_index", "mode"], as_index=False)["realized_cost"]
        .mean()
    )

    # overall mean by day (all modes combined)
    summary_total = (
        trip_log
        .groupby("day_index", as_index=False)["realized_cost"]
        .mean()
        .rename(columns={"realized_cost": "mean_cost_total"})
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    color_map = {"car": "red", "bus": "blue"}

    # mode-specific lines
    for mode in summary_mode["mode"].unique():
        sub = summary_mode[summary_mode["mode"] == mode].sort_values("day_index")
        ax.plot(
            sub["day_index"],
            sub["realized_cost"],
            label=f"{mode} mean",
            color=color_map.get(mode, None),
        )

    # total line (all persons, all modes)
    sub_tot = summary_total.sort_values("day_index")
    ax.plot(
        sub_tot["day_index"],
        sub_tot["mean_cost_total"],
        label="total mean",
        color="black",
        linestyle="--",
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Realized cost")
    ax.set_title("Average realized cost by mode and overall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_realized_cost_boxplots(trip_log):
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.boxplot(
        data=trip_log,
        x="day_index",
        y="realized_cost",
        hue="mode",
        palette={"car": "red", "bus": "blue"},
        ax=ax,
    )

    xticks = ax.get_xticks()
    for i in range(len(xticks) - 1):
        mid = (xticks[i] + xticks[i + 1]) / 2
        ax.axvline(mid, color="black", linewidth=1, linestyle="-", alpha=0.6)

    ax.set_xlabel("Day")
    ax.set_ylabel("Realized cost")
    ax.set_title("Realized cost by mode and day")
    ax.legend(title="Mode")

    plt.tight_layout()
    plt.show()


def _load_start_hr(run_id: str, output_root: str = "data/outputs/seasons") -> int:
    config_path = Path(output_root) / run_id / "season_config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f).get("start_hr", 7)
    return 7


def _fmt_hour(h: float) -> str:
    hour = int(h) % 24
    suffix = "AM" if hour < 12 else "PM"
    display = hour if hour <= 12 else hour - 12
    display = 12 if display == 0 else display
    return f"{display} {suffix}"


def plot_model_ts_interactive(model_ts, run_id: str, output_root: str = "data/outputs/seasons"):
    start_hr = _load_start_hr(run_id, output_root)

    dfs = []
    for day, df in model_ts.items():
        tmp = df.copy()
        tmp["day_index"] = day
        dfs.append(tmp)
    data = pd.concat(dfs, ignore_index=True)

    data["hour_of_day"] = start_hr + data["step"] / 3600.0

    exclude = {"step", "hour_of_day", "day_index"}
    metrics = [c for c in data.columns if c not in exclude and pd.api.types.is_numeric_dtype(data[c])]
    start_metric = "vehicle_count"
    days = sorted(data["day_index"].unique())
    n_days = len(days)

    colors = []
    for i in range(n_days):
        t = i / (n_days - 1) if n_days > 1 else 0
        colors.append(f"rgb({int(255*(1-t))},0,{int(255*t)})")

    # build whole-hour tick marks spanning the data
    h_min = data["hour_of_day"].min()
    h_max = data["hour_of_day"].max()
    tick_vals = list(range(int(h_min), int(h_max) + 2))
    tick_text = [_fmt_hour(h) for h in tick_vals]

    fig = go.Figure()

    for i, day in enumerate(days):
        sub = data[data["day_index"] == day]
        fig.add_trace(
            go.Scatter(
                x=sub["hour_of_day"],
                y=sub[start_metric],
                mode="lines",
                name=f"Day {day}",
                line=dict(color=colors[i]),
                customdata=sub[["step"]].values,
                hovertemplate="<b>%{fullData.name}</b><br>Time: %{x:.2f}h<br>Value: %{y}<br>Step: %{customdata[0]}<extra></extra>",
            )
        )

    buttons = []
    for metric in metrics:
        new_ys = [data[data["day_index"] == day][metric] for day in days]
        buttons.append(
            dict(
                label=metric,
                method="update",
                args=[{"y": new_ys}, {"yaxis": {"title": metric}}],
            )
        )

    fig.update_layout(
        title=dict(text="Model time series by day", y=0.97, yanchor="top"),
        width=1000,
        margin=dict(t=100),
        xaxis=dict(
            title="Time of day",
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        yaxis_title=start_metric,
        legend_title_text="Day (click to show/hide)",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                type="dropdown",
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top",
                showactive=True,
            )
        ],
    )

    fig.show()


def plot_single_day_metrics(
    model_ts,
    run_id: str,
    output_root: str = "data/outputs/seasons",
    default_metrics: list[str] | None = None,
):
    """Plot multiple metrics for a single day, with a slider to select the day."""
    start_hr = _load_start_hr(run_id, output_root)

    if default_metrics is None:
        default_metrics = ["vehicle_count", "recent_travel_time_avg"]

    dfs = []
    for day, df in model_ts.items():
        tmp = df.copy()
        tmp["day_index"] = day
        dfs.append(tmp)
    data = pd.concat(dfs, ignore_index=True)
    data["hour_of_day"] = start_hr + data["step"] / 3600.0

    all_metrics = [
        "vehicle_count",
        "current_toll",
        "active_cars",
        "active_buses",
        "persons_at_bus_stop",
        "persons_finished",
        "persons_pool_remaining",
        "persons_in_transit",
        "recent_travel_time_avg",
        "p_generate",
        "rolling_count_vehicles_generated",
        "rolling_count_persons_generated",
    ]
    all_metrics = [m for m in all_metrics if m in data.columns]
    days = sorted(data["day_index"].unique())

    h_min = data["hour_of_day"].min()
    h_max = data["hour_of_day"].max()
    tick_vals = list(range(int(h_min), int(h_max) + 2))
    tick_text = [_fmt_hour(h) for h in tick_vals]

    # colour palette for metrics
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    fig = go.Figure()

    # Add one trace per (metric, day). Only the first day's traces are visible.
    first_day = days[0]
    for m_idx, metric in enumerate(all_metrics):
        color = palette[m_idx % len(palette)]
        visible_by_default = metric in default_metrics
        for day in days:
            sub = data[data["day_index"] == day]
            fig.add_trace(
                go.Scatter(
                    x=sub["hour_of_day"],
                    y=sub[metric],
                    mode="lines",
                    name=metric,
                    line=dict(color=color),
                    visible=visible_by_default if day == first_day else False,
                    legendgroup=metric,
                    showlegend=(day == first_day),
                )
            )

    n_metrics = len(all_metrics)
    n_days = len(days)
    # total traces = n_metrics * n_days
    # trace index for metric m_idx, day d_idx = m_idx * n_days + d_idx

    # Build slider steps — each step shows traces for one day
    steps = []
    for d_idx, day in enumerate(days):
        visibility = []
        for m_idx, metric in enumerate(all_metrics):
            for dd_idx in range(n_days):
                if dd_idx == d_idx:
                    # Use "legendonly" for non-default metrics so they appear
                    # togglable in the legend but aren't drawn initially
                    if metric in default_metrics:
                        visibility.append(True)
                    else:
                        visibility.append("legendonly")
                else:
                    visibility.append(False)
        steps.append(
            dict(
                method="restyle",
                args=[{"visible": visibility}],
                label=str(day),
            )
        )

    fig.update_layout(
        title=f"Day {first_day} — model time series",
        xaxis=dict(title="Time of day", tickvals=tick_vals, ticktext=tick_text),
        yaxis_title="Value",
        legend_title_text="Metric (click to toggle)",
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Day: "),
                pad=dict(t=60),
                steps=steps,
            )
        ],
    )

    fig.show()
