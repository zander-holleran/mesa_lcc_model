import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


from utils.unit_conversion_utils import get_mps, meters_to_feet  


# -~-~-~-~-~-~-~-~-~-~-~-~ general analysis - should be able to be used independent of project spicific data -~-~-~-~-~-~-~-~-~-~-~-~
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
def make_colored_road_plot(road_gdf, color_var):
    fig, ax = plt.subplots(figsize=(15, 6))
    road_gdf.plot(column=color_var, cmap='viridis', legend=True, ax=ax, markersize=30)
    
    # Optional: Add axis title and labels
    ax.set_title(f"SR210 colored by {color_var}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True)
    plt.show()

# -~-~-~-~-~-~-~-~-~-~-~-~ mesa model output helpers -~-~-~-~-~-~-~-~-~-~-~-~
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
    if model.batchrun ==True:
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