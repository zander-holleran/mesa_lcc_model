# -~-~-~-~-~-~-~-~-~-~-~-~ imports -~-~-~-~-~-~-~-~-~-~-~-~
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import seaborn as sns

import pandas as pd
import numpy as np
import math

from shapely.geometry import LineString
from IPython.display import HTML



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


# -~-~-~-~-~-~-~-~-~-~-~-~ analysis -~-~-~-~-~-~-~-~-~-~-~-~
def make_driving_actions_plots(df, speed_change_col="speed_change", action_col="driving_action"):
    """
    Creates a side-by-side bar plot and box plot based on driving actions.

    Parameters:
    - df: pandas DataFrame containing the driving data
    - speed_change_col: column name representing speed changes (numeric)
    - action_col: column name representing driving action categories (str)
    """

    # Prepare the figure with two subplots side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

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
    
    # Choose a point in meter‐coordinates to anchor your bar:
    base_x = min(x_road) + 50     # 50 m in from the left
    base_y = min(y_road) + 50     # 50 m up from the bottom
    
    bar_length = 1000             # 1 000 m
    #tick_every = 100              # mark every 100 m
    #tick_height = 20              # 20 m tall ticks
    label_offset = 30             # 30 m above the bar for text
    
    # Draw the main bar
    ax.add_patch(patches.Rectangle(
        (base_x, base_y),
        bar_length,        # width = 1 000 m
        5,                 # thickness = 5 m
        color="black"
    ))

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
        else:
            scat_watch.set_offsets(np.empty((0, 2)))

        ax.set_title(f"Step {frame}", fontsize=14)
        return scat_all, scat_watch

    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=interval)
    plt.close()
    return HTML(anim.to_jshtml())
