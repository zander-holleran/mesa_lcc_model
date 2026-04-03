import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

import re
import numpy as np

#============================================ LOAD DATA ============================================
BASE_DIR = Path('data/season_outputs')
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

def load_run(run_id: str, base_dir: Path = BASE_DIR):
    run_dir = base_dir / run_id
    data = {
        'run_dir': run_dir,
        'trip_log': _read_parquet_optional(run_dir / 'trip_log.parquet'),
        'day_summary': _read_parquet_optional(run_dir / 'day_summary.parquet'),
        'season_person_log': _read_parquet_optional(run_dir / 'season_person_log.parquet'),
        'sp_day_summary': _read_parquet_optional(run_dir / 'sp_day_summary.parquet'),
        'model_ts': load_model_ts(run_dir),
    }
    return data


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


def _load_start_hr(run_id: str, output_root: str = "data/season_outputs") -> int:
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


def plot_model_ts_interactive(model_ts, run_id: str, output_root: str = "data/season_outputs"):
    start_hr = _load_start_hr(run_id, output_root)

    dfs = []
    for day, df in model_ts.items():
        tmp = df.copy()
        tmp["day_index"] = day
        dfs.append(tmp)
    data = pd.concat(dfs, ignore_index=True)

    data["hour_of_day"] = start_hr + data["step"] / 3600.0

    metrics = [
        "vehicle_count",
        "current_toll",
        "active_cars",
        "active_buses",
        "bus_riders_waiting",
        "total_finished",
        "recent_travel_time_avg",
        "p_generate",
        "rolling_count_vehicles_generated",
        "rolling_count_persons_generated",
    ]
    metrics = [m for m in metrics if m in data.columns]
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
        title="Model time series by day",
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
                x=1.05,
                xanchor="left",
                y=1,
                yanchor="top",
                showactive=True,
            )
        ],
    )

    fig.show()
