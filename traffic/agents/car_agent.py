from traffic.agents.vehicle_agent import VehicleAgent, build_empirical_accel_function
import traffic.utils.distribution_utils as du
import numpy as np 

class CarAgent(VehicleAgent):
    """Represents a car moving in the canyon."""
    def __init__(self, model):
        super().__init__(model)
        self.status = "driving"  # the initial status of the car
        
        # speed perams
        self.acceptable_over =  du.make_truncnorm(15, -2, 4, mean=3).rvs()  # this is a right skewed normal dist bounded by (-2,20)
        self.ideal_distance_multiplier = du.make_truncnorm(2, .6, .3, mean=1).rvs()
        self.performance = np.random.uniform()
        self.accel_curve = build_empirical_accel_function(self.performance)
        self.curve_responce = du.make_truncnorm(.95, .6, .1, mean=None).rvs() 