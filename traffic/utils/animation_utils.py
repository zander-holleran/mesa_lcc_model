import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from shapely.geometry import LineString
from scipy.stats import gaussian_kde
from traffic.utils import unit_conversion_utils as uc
# -~-~-~-~-~-~-~-~-~-~-~-~ animations -~-~-~-~-~-~-~-~-~-~-~-~
def make_scale_legend(ax, wx, wy):
    #half baked make scale function 
    # currently it calls ax.plot, this stays around on the plot. To fully fix ill need to update it instead. Currently its just commented out 
    """
    Draws a scale legend on the given axes at position (wx, wy).
    """
    base_x = wx
    base_y = wy + 70  # position slightly above current view

    bar_length = 200
    tick_every = 50
    tick_height = 20
    label_offset = 30

    # Draw the main bar
    ax.add_patch(Rectangle(
        (base_x, base_y), bar_length, 5, color="black"
    ))

    # Draw tick marks & labels
    for i in range(0, bar_length + 1, tick_every):
        tx = base_x + i
        ax.plot([tx, tx], [base_y, base_y + tick_height], color="black", linewidth=1)
        if i % (tick_every * 2) == 0:  # label every 100 m
            ax.text(tx, base_y + tick_height + label_offset,
                    f"{i} m", ha="center", va="bottom", fontsize=8)


def _build_color_scheme(series):
    """
    Analyze a Series and return a color scheme dict.

    Returns a dict with keys:
      'mode'       : 'discrete' | 'continuous'
      'color_map'  : {value: color} for discrete, None for continuous
      'cmap'       : matplotlib Colormap for continuous, None for discrete
      'norm'       : mcolors.Normalize for continuous, None for discrete
    """
    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if n_unique < 7:
        unique_vals = sorted(series.dropna().unique(), key=str)
        palette = sns.color_palette("tab10", n_colors=max(len(unique_vals), 1))
        color_map = {v: palette[i] for i, v in enumerate(unique_vals)}
        return {'mode': 'discrete', 'color_map': color_map, 'cmap': None, 'norm': None}

    if is_numeric:
        vmin, vmax = series.min(), series.max()
        if vmin < 0 and vmax > 0:
            cmap = plt.cm.RdBu_r
        else:
            cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return {'mode': 'continuous', 'color_map': None, 'cmap': cmap, 'norm': norm}

    # Many unique string values — fall back to husl discrete
    unique_vals = sorted(series.dropna().unique(), key=str)
    palette = sns.color_palette("husl", n_colors=len(unique_vals))
    color_map = {v: palette[i] for i, v in enumerate(unique_vals)}
    return {'mode': 'discrete', 'color_map': color_map, 'cmap': None, 'norm': None}


def animate_traffic(spatial_df, road_gdf, interval=60, step_skip=1, watch=None, zoom=1, color_by=None):
    """
    Create and return an HTML animation of vehicle movement along a road.

    Parameters:
    - spatial_df: pd.DataFrame with columns ['Step', 'AgentID', 'pos', 'status', ...]
    - road_gdf: GeoDataFrame of road points
    - interval: milliseconds between frames
    - step_skip: sample every nth step
    - watch: AgentID to focus on (optional)
    - zoom: zoom factor when watching an agent
    - color_by: column name in spatial_df to use for vehicle color. If None, defaults
                to status-based coloring. Automatically picks a continuous colormap
                (viridis / RdBu_r) or discrete palette (tab10) based on the data.

    Returns:
    - HTML animation object
    """
    if watch and watch not in spatial_df.AgentID.unique():
        raise ValueError('select a valid agent_id')

    if color_by is not None and color_by not in spatial_df.columns:
        raise ValueError(f"color_by column '{color_by}' not found in spatial_df")

    spatial_df = spatial_df.copy()
    spatial_df['x'] = spatial_df['pos'].apply(lambda p: p[0])
    spatial_df['y'] = spatial_df['pos'].apply(lambda p: p[1])

    road_line = LineString(road_gdf.geometry.tolist())
    x_road, y_road = road_line.xy
    steps = sorted(set(spatial_df["Step"]))[::step_skip]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_road, y_road, color='black', linewidth=2, label="Road")
    scat_all = ax.scatter([], [], s=20)
    scat_watch = ax.scatter([], [], s=40, color='purple', label='Watched')

    if watch is None:
        ax.set_xlim(min(x_road) - 50, max(x_road) + 50)
        ax.set_ylim(min(y_road) - 50, max(y_road) + 50)

    # --- Color scheme ---
    _status_colors = {
        "driving": "grey", "crash": "red",
        "slowing": "yellow", "canyon_closure": "blue",
    }

    if color_by is not None:
        scheme = _build_color_scheme(spatial_df[color_by])
        if scheme['mode'] == 'continuous':
            sm = plt.cm.ScalarMappable(cmap=scheme['cmap'], norm=scheme['norm'])
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(color_by, fontsize=9)
        else:
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=str(v),
                           markerfacecolor=c, markersize=8)
                for v, c in scheme['color_map'].items()
            ]
            ax.legend(handles=handles, title=color_by, loc='upper right', fontsize=12,
                      bbox_to_anchor=(0.97, 0.97), borderaxespad=1.5)

    def _get_colors(step_df_other):
        if color_by is None:
            return step_df_other["status"].map(_status_colors).fillna("gray").values
        if scheme['mode'] == 'discrete':
            return step_df_other[color_by].map(scheme['color_map']).fillna("gray").values
        vals = pd.to_numeric(step_df_other[color_by], errors='coerce')
        return scheme['cmap'](scheme['norm'](vals.fillna(scheme['norm'].vmin).values))

    def init():
        scat_all.set_offsets(np.empty((0, 2)))
        scat_watch.set_offsets(np.empty((0, 2)))
        return scat_all, scat_watch

    def update(frame):
        step_df = spatial_df[spatial_df["Step"] == frame]

        if watch is not None:
            step_df_watch = step_df[step_df["AgentID"] == watch]
            step_df_other = step_df[step_df["AgentID"] != watch]
        else:
            step_df_watch = step_df[step_df["AgentID"] == -1]
            step_df_other = step_df

        scat_all.set_offsets(step_df_other[['x', 'y']].values)
        scat_all.set_color(_get_colors(step_df_other))

        if step_df_watch is not None and not step_df_watch.empty:
            scat_watch.set_offsets(step_df_watch[['x', 'y']].values)
            wx, wy = step_df_watch.iloc[0][['x', 'y']]
            tot_x_dist = max(x_road) - min(x_road)
            window = tot_x_dist / 2 / zoom
            ax.set_xlim(wx - window, wx + window)
            ax.set_ylim(wy - window, wy + window)
        else:
            scat_watch.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Step {frame}", fontsize=14)
        return scat_all, scat_watch

    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval)
    plt.close()
    return HTML(anim.to_jshtml())


def animate_relative_distance(vehicle_df, agent_id, distance_behind, color_by='driving_action'):
    """
    Animate the relative positions of vehicles behind a reference agent.

    Parameters:
    - vehicle_df (pd.DataFrame): Contains ['Step', 'AgentID', 'distance_traveled']
    - agent_id (int): The reference agent ID
    - distance_behind (float): The maximum distance behind the reference agent to visualize

    Returns:
    - HTML animation object
    """

     # --- df and agent_id checks ---
    base_required = {"AgentID", "Step", "distance_traveled", "gap_m", "ideal_gap_m"}

    if color_by not in ("status", "driving_action"):
        raise ValueError("color_by must be 'status' or 'driving_action'")

    required_cols = base_required | {color_by}
    missing = [c for c in required_cols if c not in vehicle_df.columns]

    if missing:
        raise ValueError(f"vehicle_df is missing required columns: {missing}")

    if not agent_id in vehicle_df.AgentID.unique():
        raise ValueError('select a valid agent_id')
    
    # Filter to only include agents >= agent_id
    vehicle_df = vehicle_df[vehicle_df["AgentID"] >= agent_id].copy()

    # Determine reference distances per step
    ref_distances = (
        vehicle_df[vehicle_df["AgentID"] == agent_id]
        .set_index("Step")["distance_traveled"]
        .rename("ref_distance")
    )

    # Merge reference distances
    vehicle_df = vehicle_df.merge(ref_distances, on="Step")
    vehicle_df["distance_behind_ref"] = vehicle_df["ref_distance"] - vehicle_df["distance_traveled"]

    # Filter to within distance_behind
    vehicle_df = vehicle_df[vehicle_df["distance_behind_ref"] <= distance_behind]

    driving_action_colors = {
        "coast": "gray",
        'jitter':'purple',
        "slow_accelerate":'lightgreen',
        "accelerate": "green",
        "smooth_break": "orange",
        "speed_limit_break":"yellow",
        "prevent_pass": "red",
    }
    status_colors = {
        "driving": "gray",
        "crash": "red",
        "slowing": "yellow",
        "canyon_closure": "blue"
    }

    if color_by == 'status':
        label_colors = status_colors
    elif color_by == 'driving_action':
        label_colors = driving_action_colors
    else:
        raise ValueError("color_by must be 'status' or 'driving_action'")
    
    # Text label storage
    text_labels = []

    
    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 2))
    scat = ax.scatter([], [], s=60)
    ax.set_xlim(-5, distance_behind+30)
    ax.set_ylim(-1, 3)
    ax.invert_xaxis()  # <-- this line flips the x-axis
    ax.set_yticks([])
    ax.axvline(x= 0, color='red', linestyle='--', label='Reference Agent')
    ax.legend(loc='upper left')

    # Add a legend for driving actions
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=8)
               for label, color in label_colors.items()]
    ax.legend(handles=handles, loc='upper left', title=color_by)

    # All unique steps
    steps = sorted(vehicle_df["Step"].unique())

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame):
        nonlocal text_labels
        # Clear existing text
        for txt in text_labels:
            txt.remove()
        text_labels = []

        
        step_df = vehicle_df[vehicle_df["Step"] == frame]
        xs = step_df["distance_behind_ref"]
        coords = np.column_stack([xs, np.zeros(len(xs))])
        scat.set_offsets(coords)
        colors = step_df[color_by].map(label_colors).fillna("black")
        scat.set_color(colors)

        #print(vehicle_df.head())

        # Add text labels
        for _, row in step_df.iterrows():
            if row["AgentID"] != agent_id:
                gap = int(round(row["gap_m"])) if np.isfinite(row["gap_m"]) else "NA"
                ideal_gap = int(round(row["ideal_gap_m"])) if np.isfinite(row["ideal_gap_m"]) else "NA"
                text = f"{gap}/{ideal_gap}m"
                label = ax.text(row["distance_behind_ref"], 0.6, text,
                                ha='center', va='bottom', fontsize=7)
                text_labels.append(label)
                
        ax.set_title(f"Step {frame}")
        return scat,

    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=100)
    plt.close()
    return HTML(anim.to_jshtml())


SPATIAL_METRIC_UNITS = {
    'density':           'veh/mi',
    'speed':             'mph',
    'speed_mps':         'm/s',
    'gap_m':             'm',
    'ideal_gap_m':       'm',
    'distance_traveled': 'mi',
}


def animate_smoothed_spatial_metric(
    spatial_df,
    road_gdf,
    interval=100,
    step_skip=1,
    bandwidth=0.3,
    curve_resp=0.5,
    metric='density',
    max_categories=None,
):
    """
    Animate a smoothed quantitative or categorical metric along the 1D road axis.

    Parameters:
    - spatial_df: Tier 2 HybridCollector output — requires ['Step', 'distance_traveled', 'status']
    - road_gdf: GeoDataFrame with columns ['distance_traveled', 'speed_limit', 'curvature']
    - interval: ms between animation frames
    - step_skip: sample every Nth step
    - bandwidth: smoothing bandwidth in miles (Gaussian kernel for all modes)
    - curve_resp: driver curve response factor (0–1) for curve-adjusted speed limit strip
    - metric: 'density' (default) or any column name in spatial_df.
              Numeric columns → Nadaraya-Watson locally-weighted mean.
              Non-numeric (string/categorical) columns → stacked smoothed proportions.
    - max_categories: (categorical mode only) keep the top-N categories by frequency;
              everything else is lumped into 'other'. None = no limit.

    Returns:
    - HTML animation object
    """
    if metric != 'density' and metric not in spatial_df.columns:
        raise ValueError(f"metric '{metric}' not found in spatial_df columns and is not 'density'")

    # --- 1. Data prep ---
    df = spatial_df[
        (spatial_df['distance_traveled'] >= 0) &
        (spatial_df['status'] != 'arrived')
    ].copy()
    df['dist_mi'] = uc.meters_to_miles(df['distance_traveled'])

    steps = sorted(df['Step'].unique())[::step_skip]

    road_dist_mi = uc.meters_to_miles(road_gdf['distance_traveled'].values)
    x_min, x_max = road_dist_mi.min(), road_dist_mi.max()
    x_grid = np.linspace(x_min, x_max, 500)

    # --- 2. Detect mode ---
    is_categorical = (
        metric != 'density' and
        (
            pd.api.types.is_bool_dtype(df[metric]) or
            pd.api.types.is_object_dtype(df[metric]) or
            isinstance(df[metric].dtype, pd.CategoricalDtype)
        )
    )

    # --- 3. Categorical setup: resolve categories, apply max_categories cap ---
    categories = []
    cat_colors = []
    if is_categorical:
        val_counts = df[metric].value_counts()
        if max_categories is not None and len(val_counts) > max_categories:
            top_cats = val_counts.index[:max_categories].tolist()
            other_cats = val_counts.index[max_categories:].tolist()
            print(
                f"[animate_smoothed_spatial_quant_metric] '{metric}': "
                f"{len(other_cats)} categories lumped into 'other': {other_cats}"
            )
            df[metric] = df[metric].where(df[metric].isin(top_cats), other='other')
            categories = top_cats + ['other']
        else:
            categories = val_counts.index.tolist()

        palette = sns.color_palette('tab10', n_colors=len(categories))
        cat_colors = {
            cat: ('#aaaaaa' if cat == 'other' else palette[i])
            for i, cat in enumerate(categories)
        }

    # --- 4. y-axis label and y_max ---
    unit = SPATIAL_METRIC_UNITS.get(metric, '')
    if metric == 'density':
        y_label = f"Density ({unit})"
    elif is_categorical:
        y_label = f"Density by {metric} (veh/mi)"
    else:
        y_label = f"{metric} ({unit})" if unit else metric

    if is_categorical:
        sample_steps = steps[::max(1, len(steps) // 50)]
        y_max = 0.0
        for s in sample_steps:
            pts_s = df.loc[df['Step'] == s, 'dist_mi'].values
            if len(pts_s) < 2:
                continue
            kde_s = gaussian_kde(pts_s, bw_method=bandwidth)
            y_vals_s = kde_s(x_grid) * len(pts_s)
            y_max = max(y_max, y_vals_s.max())
        y_max = y_max * 1.1 if y_max > 0 else 10.0
    else:
        sample_steps = steps[::max(1, len(steps) // 50)]
        y_max = 0.0
        for s in sample_steps:
            pts = df.loc[df['Step'] == s, 'dist_mi'].values
            if len(pts) < 2:
                continue
            if metric == 'density':
                kde = gaussian_kde(pts, bw_method=bandwidth)
                y_vals = kde(x_grid) * len(pts)
            else:
                vals = pd.to_numeric(df.loc[df['Step'] == s, metric], errors='coerce').values
                mask = np.isfinite(vals)
                if mask.sum() < 2:
                    continue
                K = np.exp(-0.5 * ((x_grid[:, None] - pts[mask][None, :]) / bandwidth) ** 2)
                denom = K.sum(axis=1)
                y_vals = np.where(denom > 0, (K @ vals[mask]) / denom, np.nan)
            finite = y_vals[np.isfinite(y_vals)]
            if len(finite):
                y_max = max(y_max, finite.max())
        y_max = y_max * 1.1 if y_max > 0 else 10.0

    # --- 5. Curve-adjusted speed limit for road strip (mirrors vehicle_kernel.py:117-122) ---
    sl = road_gdf['speed_limit'].values.astype(float)
    curv = road_gdf['curvature'].values.astype(float)
    curve_effect = np.clip(curv / 90.0, 0.0, 1.0)
    speed_effect = np.clip((sl - 10.0) / (60.0 - 10.0), 0.0, 1.0)
    speed_effect[sl <= 15.0] = 0.0
    curve_sl = sl * (1.0 - curve_resp * curve_effect * speed_effect)

    # --- 6. Figure layout ---
    fig, (ax_metric, ax_speed) = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={'height_ratios': [5, 1]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.05)

    line, = ax_metric.plot([], [], lw=2, color='steelblue')
    ax_metric.set_xlim(x_min, x_max)
    ax_metric.set_ylim(0, y_max)
    ax_metric.set_ylabel(y_label)
    ax_metric.set_xticks(np.arange(np.ceil(x_min), np.floor(x_max) + 1, 1.0))
    ax_metric.grid(axis='x', linestyle='--', alpha=0.3)

    if is_categorical:
        line.set_visible(False)
        handles = [
            plt.Rectangle((0, 0), 1, 1, fc=cat_colors[c], label=c)
            for c in categories
        ]
        ax_metric.legend(handles=handles, title=metric, loc='upper right',
                         fontsize=8, framealpha=0.8)

    fill_container = []

    # --- 7. Speed limit strip (static, drawn once) ---
    x_edges = np.concatenate([road_dist_mi, [road_dist_mi[-1] + (road_dist_mi[-1] - road_dist_mi[-2])]])
    y_edges = np.array([0.0, 1.0])
    mesh = ax_speed.pcolormesh(
        x_edges, y_edges, curve_sl[np.newaxis, :],
        cmap='gray_r',
        vmin=sl.min(),
        vmax=sl.max(),
        shading='flat',
    )
    ax_speed.set_yticks([])
    ax_speed.set_xlabel("Distance (miles)")
    ax_speed.set_xlim(x_min, x_max)
    cbar = fig.colorbar(mesh, ax=ax_speed, orientation='vertical', pad=0.01, fraction=0.02)
    cbar.set_label("Curve-adj SL (mph)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    # --- 8. Animation ---
    def _nw_kernel(pts, vals, x_grid, bw):
        """Nadaraya-Watson kernel regression on a grid."""
        K = np.exp(-0.5 * ((x_grid[:, None] - pts[None, :]) / bw) ** 2)
        denom = K.sum(axis=1)
        safe_denom = np.where(denom > 0, denom, 1.0)
        return np.where(denom > 0, (K @ vals) / safe_denom, np.nan)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        step_mask = df['Step'] == frame
        pts = df.loc[step_mask, 'dist_mi'].values
        n = len(pts)
        title_suffix = '' if metric == 'density' else f' | {metric}'
        ax_metric.set_title(f"Step {frame} — {n} vehicles{title_suffix}", fontsize=12)

        for artist in fill_container:
            artist.remove()
        fill_container.clear()

        if n < 2:
            line.set_data([], [])
            return line,

        if is_categorical:
            kde = gaussian_kde(pts, bw_method=bandwidth)
            density = kde(x_grid) * n

            cumulative = np.zeros(len(x_grid))
            cat_vals = df.loc[step_mask, metric].values
            for cat in categories:
                indicators = (cat_vals == cat).astype(float)
                prop = _nw_kernel(pts, indicators, x_grid, bandwidth)
                prop = np.nan_to_num(prop, nan=0.0)
                band = prop * density
                new_cumulative = cumulative + band
                fill_container.append(
                    ax_metric.fill_between(
                        x_grid, cumulative, new_cumulative,
                        color=cat_colors[cat], alpha=0.8,
                    )
                )
                cumulative = new_cumulative
            fill_container.append(
                ax_metric.plot(x_grid, cumulative, color='black', lw=0.8, alpha=0.5)[0]
            )

        elif metric == 'density':
            kde = gaussian_kde(pts, bw_method=bandwidth)
            y_smooth = kde(x_grid) * n
            line.set_data(x_grid, y_smooth)
            fill_container.append(
                ax_metric.fill_between(x_grid, y_smooth, alpha=0.2, color='steelblue')
            )

        else:
            vals = pd.to_numeric(df.loc[step_mask, metric], errors='coerce').values
            mask = np.isfinite(vals)
            if mask.sum() < 2:
                line.set_data([], [])
                return line,
            y_smooth = _nw_kernel(pts[mask], vals[mask], x_grid, bandwidth)
            line.set_data(x_grid, y_smooth)
            fill_container.append(
                ax_metric.fill_between(x_grid, y_smooth, alpha=0.2, color='steelblue')
            )

        return line,

    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval)
    plt.close()
    return HTML(anim.to_jshtml())


def animate_traffic_with_speed_delta_highlight(spatial_df, road_gdf, model_ts, interval=60, step_skip=1, watch=None, zoom=1, highlight_width=100):
    """
    Animation of vehicle movement along a road, with a static speed delta plot and a moving highlight rectangle.
    The speed delta plot is drawn once (with seaborn), and a rectangle highlights a window of steps as the animation progresses.
    """
    import matplotlib.gridspec as gridspec

    if watch and not watch in spatial_df.AgentID.unique():
        raise ValueError('select a valid agent_id')

    # Prepare vehicle positions
    spatial_df['x'] = spatial_df['pos'].apply(lambda p: p[0])
    spatial_df['y'] = spatial_df['pos'].apply(lambda p: p[1])
    road_line = LineString(road_gdf.geometry.tolist())
    x_road, y_road = road_line.xy
    steps = sorted(set(spatial_df["Step"]))[::step_skip]

    # Prepare cumulative data for speed delta plot
    sl_delta_data = spatial_df.loc[spatial_df.status == 'driving'].sort_values('Step')
    model_ts = model_ts.sort_values('Step')

    # Set up side-by-side plots
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1.5])
    ax_anim = fig.add_subplot(gs[0])
    ax_delta = fig.add_subplot(gs[1])

    # Road plot setup
    ax_anim.plot(x_road, y_road, color='black', linewidth=2, label="Road")
    scat_all = ax_anim.scatter([], [], s=20, label='Vehicles')
    scat_watch = ax_anim.scatter([], [], s=40, color='purple', label='Watched')
    if watch is None:
        ax_anim.set_xlim(min(x_road) - 50, max(x_road) + 50)
        ax_anim.set_ylim(min(y_road) - 50, max(y_road) + 50)

    # --- Static speed delta plot ---
    ax_delta.set_ylabel("Speed Δ (mph)", fontsize=12)
    ax_delta.set_ylim(-40, 5)
    ax_delta2 = ax_delta.twinx()
    ax_delta2.set_ylabel("Vehicle Volume", fontsize=10, rotation=270, labelpad=15)
    ax_delta2.set_ylim(0, model_ts['volume'].max()*3)
    ax_delta2.yaxis.set_label_position("right")
    ax_delta2.yaxis.tick_right()
    ax_delta.set_xlabel("Step")

    N = 10  # downsample factor
    window = 20  # rolling window for smoothing

    sl_delta_data_plot = sl_delta_data.iloc[::N].copy()
    if len(sl_delta_data_plot) > window:
        sl_delta_data_plot['smooth_delta'] = sl_delta_data_plot['implicit_sl_delta'].rolling(window, min_periods=1).mean()
        sns.lineplot(data=sl_delta_data_plot, x='Step', y='smooth_delta', ax=ax_delta, color='blue', label="Speed Δ (smoothed)")
    else:
        sns.lineplot(data=sl_delta_data_plot, x='Step', y='implicit_sl_delta', ax=ax_delta, color='blue', label="Speed Δ (driving)")

    model_ts_plot = model_ts
    sns.lineplot(x=model_ts_plot['Step'], y=model_ts_plot['volume'], ax=ax_delta2, color='red', label="Vehicle Volume")
    ax_delta2.fill_between(model_ts_plot['Step'], model_ts_plot['volume'], color='red', alpha=0.3)

    ax_delta.legend(loc='upper left')
    ax_delta2.legend(loc='upper right')
    ax_delta.set_title("Implicit Speed Δ vs. Vehicle Volume")

    # --- Highlight rectangle ---
    y_min, y_max = ax_delta.get_ylim()
    highlight_rect = Rectangle((0, y_min), highlight_width, y_max - y_min, color='orange', alpha=0.2)
    ax_delta.add_patch(highlight_rect)

    def init():
        scat_all.set_offsets(np.empty((0, 2)))
        scat_watch.set_offsets(np.empty((0, 2)))
        highlight_rect.set_xy((0, y_min))
        return scat_all, scat_watch, highlight_rect

    def update(frame):
        # Animation plot
        step_df = spatial_df[spatial_df["Step"] == frame]
        if watch is not None:
            step_df_watch = step_df[step_df["AgentID"] == watch]
            step_df_other = step_df[step_df["AgentID"] != watch]
        else:
            step_df_watch = step_df[step_df["AgentID"] == -1]
            step_df_other = step_df

        colors = step_df_other["status"].map({
            "driving": "grey",
            "crash": "red",
            "slowing": "yellow",
            "canyon_closure": "blue"
        }).fillna("gray")
        scat_all.set_offsets(step_df_other[['x', 'y']].values)
        scat_all.set_color(colors.values)
        if step_df_watch is not None and not step_df_watch.empty:
            scat_watch.set_offsets(step_df_watch[['x', 'y']].values)
            wx, wy = step_df_watch.iloc[0][['x', 'y']]
            tot_x_dist = max(x_road) - min(x_road)
            window_anim = tot_x_dist/2/zoom 
            ax_anim.set_xlim(wx - window_anim, wx + window_anim)
            ax_anim.set_ylim(wy - window_anim, wy + window_anim)
        else:
            scat_watch.set_offsets(np.empty((0, 2)))
        ax_anim.set_title(f"Step {frame}", fontsize=14)

        # Move highlight rectangle
        rect_x = frame - highlight_width // 2
        # Clamp to plot limits
        rect_x = max(rect_x, sl_delta_data_plot['Step'].min())
        rect_x = min(rect_x, sl_delta_data_plot['Step'].max() - highlight_width)
        highlight_rect.set_xy((rect_x, y_min))

        return scat_all, scat_watch, highlight_rect

    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval)
    plt.close()
    return HTML(anim.to_jshtml())