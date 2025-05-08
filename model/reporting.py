from mesa.datacollection import DataCollector
import numpy as np
from utils import unit_conversion_utils as uc  # for get_mph, etc.
from agents import VehicleAgent


def get_average_time_to_top(self):
        steps = [a["steps_taken"] for a in self.finished_agents if "steps_taken" in a]
        if not steps:
            return np.nan
        return np.mean([s / 60 for s in steps])

def get_average_car_interactions(self):
    car_interactions = [a["car_interactions"] for a in self.finished_agents if "car_interactions" in a]
    if not car_interactions:
        return np.nan
    return np.mean([c for c in car_interactions])
    

agent_reporters={
    "AgentType": lambda a: a.__class__.__name__ ,
    #'status': lambda a: a.status if isinstance(a, VehicleAgent) else None,
    
    'distance_traveled': lambda a: a.distance_traveled if hasattr(a, 'distance_traveled') else None,
    'driving_action': lambda a: a.driving_action if isinstance(a, VehicleAgent) else None,
    'speed_change': lambda a: uc.get_mph(a.speed_change) if isinstance(a, VehicleAgent) else None,
    'speed_change_mps2': lambda a: a.speed_change if isinstance(a, VehicleAgent) else None,
    'speed': lambda a: uc.get_mph(a.speed) if isinstance(a, VehicleAgent) else None,
    'speed_mps': lambda a: a.speed if isinstance(a, VehicleAgent) else None,
    'steps_taken': lambda a: a.steps_taken if isinstance(a, VehicleAgent) else None,
    'gap_m':lambda a: a.gap if isinstance(a, VehicleAgent) else None,
    'ideal_gap_m':lambda a: a.speed * a.ideal_distance_multiplier if isinstance(a, VehicleAgent) else None,
    "next_vehicle": lambda a: a.next_agent.unique_id if isinstance(a, VehicleAgent) and a.next_agent is not None else None,
    'pos':lambda a: a.pos if isinstance(a, VehicleAgent) else None # dont remove, need for visuals
}

model_reporters = {
    "avg_time_to_top": lambda m: get_average_time_to_top(m),
    "avg_car_interactions": lambda m: get_average_car_interactions(m),
    "bus_interval": lambda m: m.bus_interval,
    "bus_capacity": lambda m: m.bus_capacity,
    "bus_riders": lambda m: m.bus_riders,
    "person_counter": lambda m: m.person_counter,
    "bus_counter": lambda m: m.bus_counter,
    "car_counter": lambda m: m.car_counter,
    "p_generate": lambda m: m.p_generate,
    "car_preference": lambda m: m.car_preference,
    "FinishedAgentsSummary": lambda m: None  # Placeholder

}

