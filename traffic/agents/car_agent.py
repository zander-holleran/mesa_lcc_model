from traffic.agents.vehicle_agent import VehicleAgent
import numpy as np 

class CarAgent(VehicleAgent):
    """Represents a car moving in the canyon."""
    def __init__(self, model):
        super().__init__(model)
        self.status = "driving" 
        rng = model.rng  
        self.acceptable_over = model.dist_acceptable_over.rvs(rng)
        self.ideal_distance_multiplier = model.dist_ideal_distance_multiplier.rvs(rng)
        self.curve_responce = model.dist_curve_response.rvs(rng)
        self.performance = rng.random()
        self.accel_curve = model.accel_curve_cache[int(self.performance * 100)]

        self.toll_paid = model.current_toll_car  


