from traffic.agents.vehicle_agent import VehicleAgent, build_empirical_accel_function
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

        self.toll_paid = 0.0  # buses don't pay road tolls; passengers pay user fee

    def vehicle_to_tp_info_pass(self):
        """Override to charge passengers the bus_user_fee instead of vehicle toll."""
        for tp in getattr(self, "passengers", []):
            tp.toll_paid = self.model.bus_user_fee  # passengers pay user fee
            tp.board_step = self.created_at_step
            tp.arrive_step = self.model.steps
            tp.cumtime_lost_sec = self.cumtime_lost_sec

            tp.tp_to_sp_info_pass()