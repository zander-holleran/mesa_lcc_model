from agents.vehicle_agent import VehicleAgent, build_empirical_accel_function
import numpy as np 

        
class BusAgent(VehicleAgent):
    """Represents a bus moving in the canyon."""
    def __init__(self, model):
        super().__init__(model)
        self.status = "driving"  # the initial status of the car
        
        # speed perams
        self.acceptable_over = 0
        self.ideal_distance_multiplier = 2
        self.performance = .1
        self.accel_curve = build_empirical_accel_function(self.performance)
        self.curve_responce = .9 

