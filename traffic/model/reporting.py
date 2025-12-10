from mesa.datacollection import DataCollector
import numpy as np
from traffic.agents import VehicleAgent
from collections import defaultdict
import traffic.utils.unit_conversion_utils as uc


def get_average_time_to_top(model):
    steps = [a["steps_taken"] for a in model.finished_agents if "steps_taken" in a]
    if not steps:
        return np.nan
    return np.mean([s / 60 for s in steps])

def get_average_car_interactions(model):
    car_interactions = [a["car_interactions"] for a in model.finished_agents if "car_interactions" in a]
    if not car_interactions:
        return np.nan
    return np.mean(car_interactions)


# these two could be combind into one function with a type perameter added to the function 
def get_average_speed_relative_to_posted_sl(model):
    """
    Computes the average difference between posted speed limit and actual speed (in mph)
    for all active VehicleAgents in the model.
    """
    agents = model.agents.select(agent_type=VehicleAgent)
    
    if not agents:
        return float('nan')
    
    sl_deltas = [
        uc.get_mph(a.speed) - a.posted_speed_limit
        for a in agents
        if hasattr(a, 'posted_speed_limit') and hasattr(a, 'speed')
    ]

    return np.mean(sl_deltas) if sl_deltas else float('nan')

def get_average_speed_relative_to_implicit_sl(model):
    """
    Computes the average difference between posted speed limit and actual speed (in mph)
    for all active VehicleAgents in the model.
    """
    agents = model.agents.select(agent_type=VehicleAgent)
    
    if not agents:
        return float('nan')
    sl_deltas = [
        uc.get_mph(a.speed) - a.implicit_speed_limit 
        for a in agents
        if hasattr(a, 'implicit_speed_limit') and hasattr(a, 'speed')
    ]

    return np.mean(sl_deltas) if sl_deltas else float('nan')


def report_volume_by_section(model):
    section_counts = defaultdict(int)
    for segment in model.road_segments:
        section = segment.road_section
        section_counts[section] += len(segment.vehicles_here)

    #section_counts["total"] = sum(section_counts.values())
    return dict(section_counts)

def report_avg_speed_by_section(model):
    from collections import defaultdict

    speed_sums = defaultdict(float)
    count_by_section = defaultdict(int)

    for segment in model.road_segments:
        section = segment.road_section
        for vehicle in segment.vehicles_here:
            if hasattr(vehicle, "speed") and vehicle.speed is not None:
                speed_sums[section] += vehicle.speed
                count_by_section[section] += 1

    avg_speed_by_section = {
        section: uc.get_mph(speed_sums[section] / count_by_section[section])
        if count_by_section[section] > 0 else 0.0
        for section in range(1, 6)
    }

    return avg_speed_by_section

def get_recent_bus_mode_share(model):
    """
    Percent of traffic-persons created in the most-recent collection window who chose 'bus'.
    Uses model.collect_every_n to determine the window: [model.steps - collect_every_n + 1, model.steps].
    Returns percentage (0-100) or NaN if no persons were created in that window.
    """
    tp_cls = getattr(model, "agent_cls", {}).get("traffic_person", None)

    # get all traffic-person agents
    if tp_cls is not None:
        persons = list(model.agents.select(agent_type=tp_cls))
    else:
        # fallback: iterate agent set and filter by class name
        persons = [a for a in list(model.agents) if a.__class__.__name__ == "TrafficPersonAgent"]

    # determine window bounds
    window = max(1, getattr(model, "collect_every_n", 1))
    upper = getattr(model, "steps", 0)
    lower = max(0, upper - window + 1)

    # filter persons created in this window
    recent = [
        a for a in persons
        if getattr(a, "created_step", None) is not None and lower <= a.created_step <= upper
    ]

    if not recent:
        return float("nan")

    bus_count = sum(1 for a in recent if getattr(a, "mode", None) == "bus")
    return 100.0 * bus_count / len(recent)

agent_reporters={
    "AgentType": lambda a: a.__class__.__name__ ,
    'status': lambda a: a.status if isinstance(a, VehicleAgent) else None,
    'distance_traveled': lambda a: a.distance_traveled if hasattr(a, 'distance_traveled') else None,
    'driving_action': lambda a: a.driving_action if isinstance(a, VehicleAgent) else None,
    'speed_change': lambda a: uc.get_mph(a.speed_change) if isinstance(a, VehicleAgent) else None,
    'speed_change_mps2': lambda a: a.speed_change if isinstance(a, VehicleAgent) else None,
    'speed': lambda a: uc.get_mph(a.speed) if isinstance(a, VehicleAgent) else None,
    'speed_mps': lambda a: a.speed if isinstance(a, VehicleAgent) else None,
    'steps_taken': lambda a: a.steps_taken if isinstance(a, VehicleAgent) else None,
    'steps_taken': lambda a: a.steps_taken if isinstance(a, VehicleAgent) else None,
    'posted_speed_limit':lambda a: a.posted_speed_limit if isinstance(a, VehicleAgent) else None,
    'implicit_speed_limit':lambda a: a.implicit_speed_limit if isinstance(a, VehicleAgent) else None,
    'gap_m':lambda a: a.gap if isinstance(a, VehicleAgent) else None,
    'ideal_gap_m':lambda a: a.speed * a.ideal_distance_multiplier if isinstance(a, VehicleAgent) else None,
    "next_vehicle": lambda a: a.next_agent.unique_id if isinstance(a, VehicleAgent) and a.next_agent is not None else None,
    'pos':lambda a: a.pos if isinstance(a, VehicleAgent) else None # dont remove, need for visuals
}

model_reporters = {
   # "FinishedAgentsSummary": lambda m: None,  # required for finished agents to work
    'Step':lambda m: m.steps,
    "current_toll_car": lambda m: m.current_toll_car,
    "volume": lambda m: len(m.agents.select(agent_type=VehicleAgent)),
    "bus_mode_share_recent": get_recent_bus_mode_share,
    "at_bus_stop": lambda m: len(m.at_bus_stop),
    # "avg_time_to_top": get_average_time_to_top,
    # "avg_car_interactions": get_average_car_interactions,
    "avg_posted_sl_delta":get_average_speed_relative_to_posted_sl,
   # "avg_implicit_sl_delta":get_average_speed_relative_to_implicit_sl,
    # "person_counter": lambda m: m.person_counter,
    # "bus_counter": lambda m: m.bus_counter,
    # "car_counter": lambda m: m.car_counter,
    # "bus_riders": lambda m: m.bus_riders,
    # "volume_by_section": report_volume_by_section,
    # 'speed_by_section':report_avg_speed_by_section


}

