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
