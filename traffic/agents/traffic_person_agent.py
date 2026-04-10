from mesa import Agent
from scipy.stats import lognorm, skewnorm, norm
from math import exp, sqrt
from typing import Sequence, Tuple
from season.persons import SeasonPerson
# from traffic.utils import compute_experience_beliefs, compute_generalized_cost

class TrafficPersonAgent(Agent):
    def __init__(self, model, season_person:SeasonPerson):
        super().__init__(model)

        self.model.created_counts[type(self).__name__] += 1 # temp 


        self.person_id = season_person.person_id
        self._season_person_ref = season_person  # robust link back for logging

        if season_person is None:
            raise ValueError("TrafficPersonAgent requires a SeasonPerson instance (got None).")
        if not isinstance(season_person, SeasonPerson):
            raise TypeError(f"season_person must be SeasonPerson, got {type(season_person)!r}")

        self.status = "traveling"  

        # traits / beliefs snapshot
        self.value_of_time = season_person.value_of_time
        self.uncertainty_multiplier = season_person.uncertainty_multiplier
        self.experience_weight_car = season_person.experience_weight_car
        self.experience_weight_bus = season_person.experience_weight_bus

        self.expected_tt_car = season_person.expected_tt_car
        self.expected_tt_bus = season_person.expected_tt_bus
        self.tt_unc_car = season_person.travel_time_uncertainty_car
        self.tt_unc_bus = season_person.travel_time_uncertainty_bus

        self.mode = self.decide_mode()
        self.vehicle = None
        
        # data collection – filled over the life of the trip
        self.created_step = self.model.steps         
        
        # Collected from the vehicle
        self.toll_paid = None                         # total toll paid
        self.board_step = None                        
        self.arrive_step = None                       
        self.cumtime_lost_sec = None

        self.wait_time = None                          # minutes
        self.onboard_time = None                       # minutes
        self.total_travel_time = None                  # minutes
       
    def compute_expected_generalized_cost(
            self,
            expected_travel_time,
            travel_time_uncertainty,
            value_of_time,
            experience_weight,
            toll,
        ):
        effective_tt = expected_travel_time - self.uncertainty_multiplier * travel_time_uncertainty
        return value_of_time * experience_weight * effective_tt + toll

    def decide_mode(self) -> str:
        car_cost = self.compute_expected_generalized_cost(
            expected_travel_time=self.expected_tt_car,
            travel_time_uncertainty=self.tt_unc_car,
            value_of_time=self.value_of_time,
            experience_weight=self.experience_weight_car,
            toll=self.model.current_toll_car,
        )

        bus_cost = self.compute_expected_generalized_cost(
            expected_travel_time=self.expected_tt_bus,
            travel_time_uncertainty=self.tt_unc_bus,
            value_of_time=self.value_of_time,
            experience_weight=self.experience_weight_bus,
            toll=self.model.bus_user_fee,
        )
        
        return "car" if car_cost <= bus_cost else "bus"
    
    # ---------- trip completion hook ----------

    def tp_to_sp_info_pass(self):
        """
        Called by the VehicleAgent when the vehicle reaches end_of_road().
        trip_summary is a dict with at least:
            - wait_time (minutes)
            - onboard_time (minutes)
            - total_travel_time (minutes)
        and can contain additional metrics later (congestion, stops, etc.).
        """
        wait_steps = max(0, self.board_step - self.created_step)
        onboard_steps =  max(0, self.arrive_step - self.board_step)

        wait_time = wait_steps / 60
        onboard_time = onboard_steps / 60
        total_tt = wait_time + onboard_time
        cumtime_lost_min =self.cumtime_lost_sec / 60

        weight = (lambda: self.experience_weight_bus if self.mode == "bus" else self.experience_weight_car)()
        realized_cost = (total_tt*self.value_of_time*weight)+self.toll_paid       


        # forward the full summary to SeasonPerson for belief updates / history
        sp = self._season_person_ref
        if sp:
            sp.record_experience(
                # all kwargs passed are saved to sp.history as a dict
                day_index=self.model.current_day,
                mode=self.mode,
                toll_paid=self.toll_paid,
                realized_tt=total_tt,
                wait_time=wait_time,
                onboard_time=onboard_time,
                cumtime_lost_min=cumtime_lost_min,
                realized_cost=realized_cost,
                vehicle_id=self.vehicle.unique_id if self.vehicle else None,
            )
            self.model.sp_finished_counter += 1
                
        self.status = "arrived"

        # Use getattr in case 'traffic_persons_list' is not present on the model (optional attribute)
        if self.model.traffic_persons_list is not None:
            try:
                self.model.traffic_persons_list.remove(self)
            except ValueError:
                pass
