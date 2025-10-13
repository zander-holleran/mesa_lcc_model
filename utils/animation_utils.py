import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from shapely.geometry import LineString


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
    ax.add_patch(patches.Rectangle(
        (base_x, base_y), bar_length, 5, color="black"
    ))

    # Draw tick marks & labels
    for i in range(0, bar_length + 1, tick_every):
        tx = base_x + i
        ax.plot([tx, tx], [base_y, base_y + tick_height], color="black", linewidth=1)
        if i % (tick_every * 2) == 0:  # label every 100 m
            ax.text(tx, base_y + tick_height + label_offset,
                    f"{i} m", ha="center", va="bottom", fontsize=8)


def animate_traffic(cars_full, road_gdf, interval=60, step_skip=1, watch=None, zoom=1):
    """
    Create and return an HTML animation of vehicle movement along a road.

    Parameters:
    - cars_full: pd.DataFrame with columns ['Step', 'AgentID', 'x', 'y', 'type']
    - road_gdf: GeoDataFrame of road points
    - interval: milliseconds between frames
    - step_skip: sample every nth step
    - watch: AgentID to focus on (optional)

    Returns:
    - HTML animation object
    """
    cars_full['x'] = cars_full['pos'].apply(lambda p: p[0])
    cars_full['y'] = cars_full['pos'].apply(lambda p: p[1])
    
    # Prepare the road line
    road_line = LineString(road_gdf.geometry.tolist())
    x_road, y_road = road_line.xy

    # Get all unique steps to animate
    steps = sorted(set(cars_full["Step"]))[::step_skip]

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_road, y_road, color='black', linewidth=2, label="Road")
    scat_all = ax.scatter([], [], s=20, label='Vehicles')
    scat_watch = ax.scatter([], [], s=40, color='purple', label='Watched')

    # Set axis limits if not watching
    if watch is None:
        ax.set_xlim(min(x_road) - 50, max(x_road) + 50)
        ax.set_ylim(min(y_road) - 50, max(y_road) + 50)

    def init():
        scat_all.set_offsets(np.empty((0, 2)))
        scat_watch.set_offsets(np.empty((0, 2)))
        return scat_all, scat_watch

    def update(frame):
        step_df = cars_full[cars_full["Step"] == frame]

        # Plot all vehicles except the watched one
        if watch is not None:
            step_df_watch = step_df[step_df["AgentID"] == watch]
            step_df_other = step_df[step_df["AgentID"] != watch]
        else:
            step_df_watch = step_df[step_df["AgentID"] == -1]  # empty
            step_df_other = step_df

        # Assign colors by type
        colors = step_df_other["status"].map({
            "driving": "grey",
            "crash": "red",
            "slowing": "yellow",
            "canyon_closure": "blue"
        }).fillna("gray")

        # Set offsets
        scat_all.set_offsets(step_df_other[['x', 'y']].values)
        scat_all.set_color(colors.values)

        
        if step_df_watch is not None and not step_df_watch.empty:
            scat_watch.set_offsets(step_df_watch[['x', 'y']].values)
            wx, wy = step_df_watch.iloc[0][['x', 'y']]
            tot_x_dist = max(x_road) - min(x_road) # the window is the distance of the road in the x axis/2 (because you have to add and subtract to WX and WY)/ devided by the zoom
            window = tot_x_dist/2/zoom 
            ax.set_xlim(wx - window, wx + window)
            ax.set_ylim(wy - window, wy + window)

            #make_scale_legend(ax, wx, wy)
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



