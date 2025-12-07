from mesa import Agent
from scipy.stats import lognorm, skewnorm, norm
from math import exp, sqrt
from typing import Sequence, Tuple
# from traffic.utils import compute_experience_beliefs, compute_generalized_cost

class TrafficPersonAgent(Agent):
    def __init__(self, model, season_person):
        super().__init__(model)

        self.person_id = season_person.person_id
        self._season_person_ref = season_person  # robust link back for logging

        self.status = "traveling"  # or "arrived"

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
        
        # timing fields – filled over the life of the trip
        self.created_step = self.model.steps          # set when generated
        self.board_step = None                        # for bus users only
        self.wait_time = 0.0                          # minutes
        self.onboard_time = 0.0                       # minutes
        self.total_travel_time = 0.0                  # minutes

        self.cumtime_lost_sec = 0.0                   # seconds
        self.toll_paid = 0.0                          # total toll paid
       
    def compute_generalized_cost(
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
        car_cost = self.compute_generalized_cost(
            expected_travel_time=self.expected_tt_car,
            travel_time_uncertainty=self.tt_unc_car,
            value_of_time=self.value_of_time,
            experience_weight=self.experience_weight_car,
            toll=self.model.current_toll_car,
        )
        bus_cost = self.compute_generalized_cost(
            expected_travel_time=self.expected_tt_bus,
            travel_time_uncertainty=self.tt_unc_bus,
            value_of_time=self.value_of_time,
            experience_weight=self.experience_weight_bus,
            toll=self.model.current_toll_bus,
        )
        return "car" if car_cost <= bus_cost else "bus"
    
    # ---------- trip completion hook ----------

    def on_trip_completed(self, trip_summary: dict):
        """
        Called by the VehicleAgent when the vehicle reaches end_of_road().
        trip_summary is a dict with at least:
            - wait_time (minutes)
            - onboard_time (minutes)
            - total_travel_time (minutes)
        and can contain additional metrics later (congestion, stops, etc.).
        """
        self.status = "arrived"
        # store key metrics on the person
        self.wait_time = trip_summary.get("wait_time", 0.0)
        self.onboard_time = trip_summary.get("onboard_time", 0.0)
        self.total_travel_time = trip_summary.get(
            "total_travel_time",
            self.wait_time + self.onboard_time,
        )
        self.toll_paid = trip_summary.get("toll_paid", 0.0)
        self.cumtime_lost_sec = trip_summary.get("cumtime_lost_sec", 0.0)

        # forward the full summary to SeasonPerson for belief updates / history
        sp = self._season_person_ref
        if sp is not None:
            sp.record_experience(
                day_index=self.model.current_day,
                mode=self.mode,
                realized_tt=self.total_travel_time,
                toll_paid=self.toll_paid,
                cumtime_lost_sec=self.cumtime_lost_sec
                #**trip_summary,   # keeps it extensible
            )