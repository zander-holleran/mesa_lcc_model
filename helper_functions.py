# -~-~-~-~-~-~-~-~-~-~-~-~ imports -~-~-~-~-~-~-~-~-~-~-~-~
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import seaborn as sns

import pandas as pd
import numpy as np
import math
from scipy.stats import truncnorm

from shapely.geometry import LineString
from IPython.display import HTML
import matplotlib.patches as patches


def get_mps(mph):
    """
    Convert miles per hour to meters per second.
    """
    return mph * 1609.34 / 3600

def get_mph(mps):
    """
    Convert meters per second to miles per hour.
    """
    return mps * 3600 / 1609.34

def meters_to_feet(meters):
    """
    Convert meters to feet.
    """
    return meters * 3.28084

def feet_to_meters(feet):
    """
    Convert feet to meters.
    """
    return feet / 3.28084

def meters_to_miles(meters):
    """
    Convert meters to miles.
    """
    return meters / 1609.34

def make_truncnorm(upper, lower, var, mean=None):
    '''
    If a mean is not passed the average of upper and lower will be used
    '''
    if not mean:
        mean = (upper+lower)/2

    return truncnorm((lower - mean)/var, (upper - mean)/var, loc=mean, scale=var)


# -~-~-~-~-~-~-~-~-~-~-~-~ general analysis-~-~-~-~-~-~-~-~-~-~-~-~
def make_colored_road_plot(road_gdf, color_var):
    fig, ax = plt.subplots(figsize=(15, 6))
    road_gdf.plot(column=color_var, cmap='viridis', legend=True, ax=ax, markersize=30)
    
    # Optional: Add axis title and labels
    ax.set_title(f"SR210 colored by {color_var}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True)
    plt.show()

def plot_param_grid_heatmap(var1_vals, var2_vals, func, param1_name, param2_name, fixed_params=None, round_to=2):
    '''
    Analyze effects of two parameters on the output of any function.

    Parameters:
        var1_vals: sequence of values for first varying parameter
        var2_vals: sequence of values for second varying parameter
        func: function to evaluate
        param1_name: string name of the first varying parameter
        param2_name: string name of the second varying parameter
        fixed_params: dict of other parameter values to pass to func
        round_to: rounding for displayed values
        label1: label for x-axis
        label2: label for y-axis
        title: title for plot
    '''
    fixed_params = fixed_params or {}
    
    data = []
    for y in var2_vals:
        row = []
        for x in var1_vals:
            params = fixed_params.copy()
            params[param1_name] = x
            params[param2_name] = y
            row.append(func(**params))
        data.append(row)

    df = pd.DataFrame(data, index=[round(y, 2) for y in var2_vals], columns=[round(x, 2) for x in var1_vals])
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=f".{round_to}f", cmap="Reds", cbar_kws={"label": "Output"})
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title('Peram Grid Heatmap')
    plt.tight_layout()
    plt.show()



# -~-~-~-~-~-~-~-~-~-~-~-~ analysis of mesa model outputs -~-~-~-~-~-~-~-~-~-~-~-~
def make_driving_actions_plots(df, speed_change_col="speed_change", action_col="driving_action"):
    """
    Creates a side-by-side bar plot and box plot based on driving actions.

    Parameters:
    - df: pandas DataFrame containing the driving data
    - speed_change_col: column name representing speed changes (numeric)
    - action_col: column name representing driving action categories (str)
    """

    # Prepare the figure with two subplots side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(20, 5), sharex=True)

    # Bar plot: distribution of driving actions
    action_counts = df[action_col].value_counts(normalize=True).reset_index()
    action_counts.columns = [action_col, "percentage"]
    sns.barplot(data=action_counts, x=action_col, y="percentage", ax=axs[0])
    axs[0].set_title("Distribution of Driving Actions")
    axs[0].set_ylabel("Percentage")
    axs[0].set_xlabel("Driving Action")
    axs[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Box plot: speed change by driving action
    sns.boxenplot(data=df, x=action_col, y=speed_change_col, ax=axs[1], outlier_prop=0.00000001)
    axs[1].set_title("Speed Change by Driving Action")
    axs[1].set_ylabel("Speed Change")
    axs[1].set_xlabel("Driving Action")
    axs[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def make_travel_time_hist(finished_agents):
    '''
    Displays a nice histogram of the travel times

    Args:
        finished_agents (DataFrame): the output from make_finished_agents_df
    '''
    # Convert to minutes
    times = finished_agents.steps_taken / 60
    
    # Compute summary stats
    mean_time = times.mean()
    min_time = times.min()
    max_time = times.max()
    
    # --- Histogram ---
    plt.figure(figsize=(6, 3))
    sns.histplot(times)
    
    # Title and axis labels
    plt.title("Distribution of Travel Times", fontsize=16)
    plt.xlabel("Time to top (minutes)", fontsize=14)
    plt.ylabel("Number of Cars", fontsize=14)
    
    # Add stats box
    textstr = f"Mean: {mean_time:.1f} min\nMin: {min_time:.1f} min\nMax: {max_time:.1f} min"
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

    
def make_finished_agents_df(model):
    '''
    Takes - model.finished_agents - a list of dicts that represent the agent data at the final step
    Returns - DataFrame - properly formatted finished_agents
    '''
    finished_agents = model.finished_agents
    finished_agents = pd.DataFrame(finished_agents)
    finished_agents =finished_agents.sort_values(by='AgentID')
    #make a travel time
    finished_agents["follow_ft_at50mph"] = meters_to_feet(get_mps(50)) * finished_agents["ideal_distance_multiplier"]
    finished_agents["travel_time"] = pd.to_timedelta(finished_agents["steps_taken"], unit="s").astype(str).str.extract(r'(\d+:\d{2})')
    # rounding
    round_cols = ['distance_traveled', 'approx_average_mph','acceptable_over', 'ideal_distance_multiplier','follow_ft_at50mph']
    finished_agents[round_cols] = round(finished_agents[round_cols], 2)
    # make more cols
    finished_agents["steps_behind_previous"] = finished_agents["created_at_step"].diff().fillna(0).astype(int)
    finished_agents["pct_car_interactions"] = finished_agents.car_interactions/finished_agents.steps_taken

    print(f'N cars: {len(finished_agents)}, fake hours: {round(model.steps/3600,1)}')
    # Make the nice histogram
    make_travel_time_hist(finished_agents)
    
    return finished_agents

def make_vehicles_full_df(model):
    if model.log_agents ==False:
        return None
        
    agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
    vehicles_full = agent_data.loc[agent_data.AgentType.isin(['CarAgent', 'BusAgent'])].copy()
    
    # produce some lists of ids i might want to look at
    bus_ids = list(vehicles_full.loc[vehicles_full.AgentType == 'BusAgent'].AgentID.unique())
    slowest_ids = list(vehicles_full.groupby(by='AgentID', as_index=False).max('steps_taken').sort_values('steps_taken', ascending=False).AgentID[:5])
    
    print(f'''Bus IDs: {bus_ids}
    Slow IDs: {slowest_ids}''')

    # display the driving actions graphs
    make_driving_actions_plots(vehicles_full)

    return vehicles_full



# -~-~-~-~-~-~-~-~-~-~-~-~ animation -~-~-~-~-~-~-~-~-~-~-~-~

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
        colors = step_df_other["AgentType"].map({
            "CarAgent": "red",
            "BusAgent": "blue"
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


def animate_relative_distance(vehicle_df, agent_id, distance_behind):
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

    print()
    driving_action_colors = {
        "coast": "gray",
        'jitter':'purple',
        "slow_accelerate":'lightgreen',
        "accelerate": "green",
        "smooth_break": "orange",
        "speed_limit_break":"yellow",
        "prevent_pass": "red",
    }

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
               for label, color in driving_action_colors.items()]
    ax.legend(handles=handles, loc='upper left', title="Driving Action")

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
        colors = step_df["driving_action"].map(driving_action_colors).fillna("black")
        scat.set_color(colors)

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



