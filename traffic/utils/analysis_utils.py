import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from traffic.utils.unit_conversion_utils import get_mps, meters_to_feet  

# -~-~-~-~-~-~-~-~-~-~-~-~ general analysis - should be able to be used independent of main -~-~-~-~-~-~-~-~-~-~-~-~
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

# -~-~-~-~-~-~-~-~-~-~-~-~ road analysis -~-~-~-~-~-~-~-~-~-~-~-~
def plot_colored_road(road_gdf, color_var):
    fig, ax = plt.subplots(figsize=(15, 6))
    road_gdf.plot(column=color_var, cmap='viridis', legend=True, ax=ax, markersize=30)
    
    # Optional: Add axis title and labels
    ax.set_title(f"SR210 colored by {color_var}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True)
    plt.show()


# -~-~-~-~-~-~-~-~-~-~-~-~ used in main -~-~-~-~-~-~-~-~-~-~-~-~

##  -~-~-~-~-~-~-~-~-~-~-~-~ Stage 1: takes the model as input or used in function that do so -~-~-~-~-~-~-~-~-~-~-~-~
def plot_driving_actions(df, speed_change_col="speed_change", action_col="driving_action"): # used in clean_vehicle_agent_data
    """
    Creates a side-by-side bar plot and box plot based on driving actions.

    Parameters:
    - df: pandas DataFrame containing the driving data
    - speed_change_col: column name representing speed changes (numeric)
    - action_col: column name representing driving action categories (str)
    """

    # Prepare the figure with two subplots side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(20, 4), sharex=True)

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

def plot_travel_time_hist(finished_agents): # used in make_finished_agents_df
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
    
def finished_agents_summary_df(model, plots=True):
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
    if plots:
        plot_travel_time_hist(finished_agents)
    
    return finished_agents

def vehicle_agent_data_time_series(model, plots=True):
    if model.batchrun == True:
        return None
        
    agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
    vehicles_full = agent_data.loc[agent_data.AgentType.isin(['CarAgent', 'BusAgent'])].copy()
    vehicles_full['implicit_sl_delta'] = vehicles_full.speed - vehicles_full.implicit_speed_limit
    vehicles_full['posted_sl_delta'] = vehicles_full.speed - vehicles_full.posted_speed_limit

    # produce some lists of ids i might want to look at
    bus_ids = list(vehicles_full.loc[vehicles_full.AgentType == 'BusAgent'].AgentID.unique())
    slowest_ids = list(vehicles_full.groupby(by='AgentID', as_index=False).max('steps_taken').sort_values('steps_taken', ascending=False).AgentID[:5])
    
    print(f'''Bus IDs: {bus_ids}
    Slow IDs: {slowest_ids}''')

    # display the driving actions graphs
    if plots:
        plot_driving_actions(vehicles_full)

    return vehicles_full


def model_data_time_series(model):   
    model_df = model.datacollector.get_model_vars_dataframe()
    model_df.drop(columns='FinishedAgentsSummary', inplace=True) # get rid of FinishedAgentsSummary
    # create the density col
    
    ## this info here is static from the road processing section 
    section_miles_dict = {1: 3.664933, 2: 3.066007, 3: 3.065791, 4: 1.550799, 5: 0.962587}
    def calc_density(vol_dict):
        return {
            section: vol / section_miles_dict[section]
            for section, vol in vol_dict.items()
            if section in section_miles_dict and section_miles_dict[section] > 0
        }
    
    model_df["density_by_section"] = model_df["volume_by_section"].apply(calc_density)
    return model_df
 




    
##  -~-~-~-~-~-~-~-~-~-~-~-~ Stage 2: takes a stage one df as input -~-~-~-~-~-~-~-~-~-~-~-~
def plot_single_car_driving_actions(vehicles_full, issue_car_id):
    issue_car_df = vehicles_full[vehicles_full.AgentID == issue_car_id]
    fig = px.scatter(
        issue_car_df, 
        x="Step", 
        y="speed", 
        color="driving_action",
        opacity=0.5,
        title=f"Driving Actions for Car {issue_car_id}"
    )
    fig.update_layout(height=500, width=1000)  # Adjust size here
    fig.show()

def plot_agent_trajectories(vehicles_full, ids_list, y_var,step_range=(None, None)):
    """
    Plot agent trajectories over time using seaborn lineplot.

    Parameters:
    - df: pd.DataFrame with columns ['Step', 'AgentID', y_var]
    - y_var: str, the column to use on the y-axis

    Returns:
    - Displays a line plot where each line is one AgentID

    """
    
    df = vehicles_full.loc[vehicles_full.AgentID.isin(ids_list)]
    start, end = step_range

    if start is not None:
        df = df[df['Step'] >= start]
    if end is not None:
        df = df[df['Step'] <= end]
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Step", y=y_var, hue="AgentID", legend=False)

     # Add labels to the start of each line
    for agent_id in df['AgentID'].unique():
        agent_df = df[df['AgentID'] == agent_id]
        first_point = agent_df.iloc[0]
        label = f"Agent:{agent_id} - ({first_point['AgentType']})"
        plt.text(first_point["Step"], first_point[y_var]+.5, label, fontsize=8, ha='left', va='top')

    plt.title(f"Agent Trajectories of {y_var} over Time", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel(y_var)
    plt.tight_layout()
    plt.show()

def plot_mean_feature(vehicles_full, feature): 
    # Filter out rows where feature is not finite
    vehicles_clean = vehicles_full[np.isfinite(vehicles_full[feature])]
    # Then aggregate
    mean_over_time = vehicles_clean.groupby('Step', as_index=False)[feature].mean()
    sns.scatterplot(data=mean_over_time, x='Step', y=feature)
    plt.xlabel("Step")
    plt.ylabel(f'Mean {feature} by Step ')


def plot_section_trends(model_ts, col, window):
    """
    Creates an interactive Plotly line plot of a time-varying dictionary column from model output, 
    showing trends for each road section over time.

    Parameters:
    ----------
    model_ts : pandas.DataFrame
        The model output DataFrame containing a column of dictionaries (e.g., volume_by_section).
    
    col : str
        The name of the column containing dictionary values, where keys represent road sections
        and values are the variable of interest (e.g., volume(_by_section), average speed(_by_section)).
    
    window : int
        The window size for rolling mean smoothing applied to each section's time series.

    Returns:
    -------
    None. Displays an interactive Plotly line chart with zoom, pan, and legend filtering.
    """
    prefix = col.removesuffix("_by_section")
    # dict col to wide
    col_data = model_ts[col].apply(pd.Series) 
    col_data["Step"] = col_data.index 
    # wide to long
    col_data = col_data.melt(id_vars="Step", var_name="Section", value_name=prefix)
    # smooth things out, window is a peram 
    col_data[prefix] = (
        col_data
        .groupby("Section")[prefix]
        .transform(lambda x: x.rolling(window=window, center=True, min_periods=1).mean())
    )
    
    # Plotly interactive line plot
    fig = px.line(
        col_data,
        x="Step",
        y=prefix,
        color="Section",
        title=f"{prefix.replace('_', ' ').title()} Over Time by Section",
        labels={"Step": "Simulation Step", prefix: prefix.title()},
        template="plotly_white"
    )

    fig.update_layout(hovermode="x unified")
    fig.show()

from matplotlib.ticker import MaxNLocator

def plot_speed_delta(vehicles_full, model_ts):
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    # Plot primary data
    sl_delta_data = vehicles_full.loc[vehicles_full.status == 'driving']
    sns.lineplot(data=sl_delta_data, x='Step', y='implicit_sl_delta', ax=ax1)
    ax1.set_ylabel("Speed Δ (mph)", fontsize=12)
    ax1.set_ylim(-40, 5)
    num_ticks = len(ax1.get_yticks())

    # Create secondary axis
    ax2 = ax1.twinx()
    
    # Scale agent count to be visually smaller (still real values)
    sns.lineplot(x=model_ts['Step'], y=model_ts['volume'], ax=ax2, color='red',)
    # Fill the area under the line
    ax2.fill_between(model_ts['Step'], model_ts['volume'], color='red', alpha=0.3)
    # Set y-limits to make it visually compressed to, e.g., the bottom 20% of ax1
    new_ymax = model_ts['volume'].max()*3
    ax2.set_ylim(0, new_ymax)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=num_ticks -1 ))

    # Cleanup
    ax2.set_ylabel("Vehicle Volume", fontsize=10, rotation=270, labelpad=15)
    ax2.yaxis.tick_right()

    ax2.yaxis.set_label_position("right")
    
    plt.title("Implicit Speed Δ vs. Vehicle Volume")
    ax1.grid()
    plt.tight_layout()
    plt.show()


# -~-~-~-~-~-~-~-~-~-~-~-~ misc utils -~-~-~-~-~-~-~-~-~-~-~-~``
def agents_to_df(model, RoadSegmentAgent):
    # 1) collect the agents
    try:
        segs = list(model.agents.select(agent_type=RoadSegmentAgent))
    except AttributeError:
        # fallback if you're not using Mesa's AgentSet.select
        segs = [a for a in model.agents if isinstance(a, RoadSegmentAgent)]

    if not segs:
        return pd.DataFrame()  # nothing to report

    # 2) union of attribute names across agents (public only)
    keys = sorted({
        k for a in segs for k in vars(a).keys()
        if not k.startswith("_")
    } | {"unique_id"})  # ensure id is included if you use it

    # 3) light coercion so the DF is printable/savable
    def coerce(v):
        # shapely geometry → WKT (if present)
        if hasattr(v, "wkt"):
            return v.wkt
        # sets/tuples → lists
        if isinstance(v, (set, tuple)):
            return list(v)
        return v

    rows = []
    for a in segs:
        row = {}
        for k in keys:
            row[k] = coerce(getattr(a, k, None))
        rows.append(row)

    return pd.DataFrame(rows)