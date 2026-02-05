from mesa.datacollection import DataCollector
import numpy as np
from traffic.agents import VehicleAgent
from collections import defaultdict
import traffic.utils.unit_conversion_utils as uc


def get_average_time_to_top(model):
    total = 0.0
    n = 0

    for a in model.finished_agents:
        s = a.get("steps_taken")
        if s is not None:
            total += s
            n += 1

    if n == 0:
        return np.nan

    return (total / n) / 60.0


def get_average_car_interactions(model):
    total = 0.0
    n = 0

    for a in model.finished_agents:
        x = a.get("car_interactions")
        if x is not None:
            total += x
            n += 1

    return (total / n) if n else np.nan


def get_average_speed_relative_to_sl(model, sl_attr):
    agents = model.vehicles_list
    if not agents:
        return float("nan")

    total = 0.0
    n = 0

    for a in agents:
        # assume your VehicleAgents have these attributes; skip only if missing/None
        speed = getattr(a, "speed", None)
        sl = getattr(a, sl_attr, None)
        if speed is None or sl is None:
            continue

        total += uc.get_mph(speed) - sl
        n += 1

    return (total / n) if n else float("nan")

def get_average_speed_relative_to_posted_sl(model):
    return get_average_speed_relative_to_sl(model, "posted_speed_limit")

def get_average_speed_relative_to_implicit_sl(model):
    return get_average_speed_relative_to_sl(model, "implicit_speed_limit")



def get_recent_bus_mode_share(model):
    """
    Percent of traffic-persons created in the most-recent collection window who chose 'bus'.
    Window is [model.steps - model.collect_every_n + 1, model.steps].
    Returns percentage (0-100) or NaN if none were created in that window.

    Assumes:
      - model.traffic_persons_list exists (active persons: waiting or on road)
      - model.collect_every_n exists
      - model.steps exists
      - each person has .created_step and .mode
    """
    persons = model.traffic_persons_list
    if not persons:
        return float("nan")

    window = model.collect_every_n
    if window < 1:
        window = 1

    upper = model.steps
    lower = upper - window + 1
    if lower < 0:
        lower = 0

    recent_n = 0
    bus_n = 0

    for p in persons:
        cs = p.created_step
        if cs < lower or cs > upper:
            continue
        recent_n += 1
        if p.mode == "bus":
            bus_n += 1

    return (100.0 * bus_n / recent_n) if recent_n else float("nan")


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
    "bus_user_fee": lambda m: m.bus_user_fee,
    "volume": lambda m: len(m.vehicles_list),
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


}

