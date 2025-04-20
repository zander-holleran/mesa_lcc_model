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

# -~-~-~-~-~-~-~-~-~-~-~-~ gneneral -~-~-~-~-~-~-~-~-~-~-~-~
def get_dps(mph):
    '''
    Args: 
        mph (numeric): a speed value in miles per hour
    Returns:
        float: speed in degrees per secound
    '''
    mps =mph/3600.0 # 3600 secounds in an hour
    dps = mps/69 # 69 miles in a degree 
    return dps

def get_mph(dps): 
    '''
    Args: 
        dps (numeric): speed in degrees per secound
    Returns:
        float: speed in miles per hour
    '''
    dph =dps*3600.0 # 3600 mins in an hour
    mph = dph*69 # 69 miles in a degree 
    return mph

def degrees_to_feet(degrees):
    """
    Approximate conversion from degrees to feet in EPSG:4326.
    Args:
        degrees (float): degrees lat/lon
    Returns:
        float: distance in ft or None if the input is inf, -inf, or NaN 
    """
    if not math.isfinite(degrees):
        return None
    feet_per_degree = 364000  # 69 miles × 5280 ft/mile
    return round(degrees * feet_per_degree)

def get_mps(mph):
    """
    Convert miles per hour to meters per second.
    """
    return mph * 1609.34 / 3600

def mps_to_mph(mps):
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



# -~-~-~-~-~-~-~-~-~-~-~-~ analysis -~-~-~-~-~-~-~-~-~-~-~-~
def make_finished_agents_df(model_output):
    '''
    Takes - model.finished_agents - a list of dicts that represent the agent data at the final step
    Returns - DataFrame - properly formatted finished_agents
    '''
    finished_agents = pd.DataFrame(model_output)
    finished_agents =finished_agents.sort_values(by='AgentID')
    #make a travel time
    finished_agents["follow_ft_at50mph"] = degrees_to_feet(get_dps(50)) * finished_agents["ideal_distance_multiplier"]
    finished_agents["travel_time"] = pd.to_timedelta(finished_agents["steps_taken"], unit="s").astype(str).str.extract(r'(\d+:\d{2})')
    # rounding
    round_cols = ['distance_traveled', 'approx_average_mph','acceptable_over', 'ideal_distance_multiplier','follow_ft_at50mph']
    finished_agents[round_cols] = round(finished_agents[round_cols], 2)
    # make more cols
    finished_agents["steps_behind_previous"] = finished_agents["created_at_step"].diff().fillna(0).astype(int)
    finished_agents["pct_car_interactions"] = finished_agents.car_interactions/finished_agents.steps_taken
    return finished_agents


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


# -~-~-~-~-~-~-~-~-~-~-~-~ animation -~-~-~-~-~-~-~-~-~-~-~-~
def animate_traffic(cars_full, road_gdf, interval=60, step_skip=1, watch=None, zoom=1):
    """
    Create and return an HTML animation of vehicle movement along a road.

    Parameters:
    - cars_full: pd.DataFrame with columns ['Step', 'AgentID', 'x', 'y']
    - road_gdf: GeoDataFrame of points representing the road
    - interval: time between frames in milliseconds
    - step_skip: int, sample every nth step
    - watch: int or None, if an AgentID is passed, zoom in on that car

    Returns:
    - HTML animation object
    """
    
    road_line = LineString(road_gdf.geometry.tolist())
    x_road, y_road = road_line.xy

    steps = sorted(set(cars_full.Step))[::step_skip]

    fig, ax = plt.subplots(figsize=(8, 6))
    scat_all = ax.scatter([], [], s=20, color='red', label='Cars')
    scat_watch = ax.scatter([], [], s=30, color='blue', label='Watched Car', edgecolor='k', linewidth=0.05)
    ax.plot(x_road, y_road, color='black', linewidth=1)

    if watch is None:
        x_min, x_max = min(x_road) - 0.001, max(x_road) + 0.001
        y_min, y_max = min(y_road), max(y_road)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    print()

    def init():
        scat_all.set_offsets(np.empty((0, 2)))
        scat_watch.set_offsets(np.empty((0, 2)))
        return scat_all, scat_watch

    def update(frame):
        step_df = cars_full.loc[cars_full.Step == frame]

        if watch is not None:
            watched = step_df[step_df.AgentID == watch]
            others = step_df[step_df.AgentID != watch]
        else:
            watched = pd.DataFrame(columns=step_df.columns)
            others = step_df

        scat_all.set_offsets(others[['x', 'y']].values)
        scat_watch.set_offsets(watched[['x', 'y']].values)

        ax.set_title(f"Step {frame}", fontsize=14)

        if watch is not None and not watched.empty:
            wx, wy = watched.iloc[0][['x', 'y']]
            tot_x_dist = max(x_road) - min(x_road)
            # the window is the distance of the road in the x axis/2 (because you have to add and subtract to WX and WY)/ devided by the zoom
            window = tot_x_dist/2/zoom 
            ax.set_xlim(wx - window, wx + window)
            ax.set_ylim(wy - window, wy + window)

        return scat_all, scat_watch

    anim = FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, repeat=True, interval=interval)
    plt.close()
    return HTML(anim.to_jshtml())
