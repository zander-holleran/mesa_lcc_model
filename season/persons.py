
from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np

@dataclass
class SeasonPerson:
    person_id: int
    season_id: str

    # core traits (drawn from PopulationParams)
    value_of_time: float
    experience_weight_car: float
    experience_weight_bus: float
    prior_car: float
    prior_bus: float
    time_decay_rate: float
    prior_weight: float
    uncertainty_multiplier: float
    travel_propensity: float

    forced_mode: bool = False
    assigned_mode: str | None = None

    # full history of realized experiences
    history: List[Dict[str, Any]] = field(default_factory=list)
    realized_costs: List[Dict[str, Any]] = field(default_factory=list)

    # derived belief summaries for each mode
    expected_tt_car: float = field(init=False)
    expected_tt_bus: float = field(init=False)
    travel_time_uncertainty_car: float = field(init=False)
    travel_time_uncertainty_bus: float = field(init=False)


    # initialize derived fields after creation
    def __post_init__(self):
        """
        Initialize expected_tt_* and uncertainty_* using current history (which will be empty at first).
        """
        self.update_beliefs_from_history(current_day=0)

    # -~-~-~-~-~-~-~-~-~-~-~-~   Helpers Start here   -~-~-~-~-~-~-~-~-~-~-~-~ #

    @staticmethod
    def slow_prior_update(prior, realized_tt, eta=0.05):
        """
        Update prior using a single realized travel time:
            prior_new = prior + eta * (realized_tt - prior)
        """
        return prior + eta * (realized_tt - prior)
    
    @staticmethod
    def compute_experience_beliefs(**kwargs):   
        travel_data = kwargs.get("travel_data", [])
        current_day = kwargs.get("current_day", 0)
        time_decay_rate = kwargs.get("time_decay_rate", 0.1)
        travel_time_prior = kwargs.get("travel_time_prior", 0.0)
        prior_weight = kwargs.get("prior_weight", 1.0)
        staleness_scale = kwargs.get("staleness_scale", 0.05)
        
        if not travel_data:
            expected_tt, travel_time_uncertainty = (travel_time_prior, 1.0/prior_weight)
            return expected_tt, travel_time_uncertainty
        
        # happy path: we have some travel data to work with
        days, tts = zip(*travel_data)
        days      = np.array(days, dtype=float)
        tts       = np.array(tts,  dtype=float)

        ages      = current_day - days
        weights   = np.exp(-time_decay_rate * ages)
        W         = float(weights.sum())

        if W > 0:
            mu_data = float((weights * tts).sum() / W)
        else:
            mu_data = travel_time_prior

        total_weight = prior_weight + W
        expected_tt  = (prior_weight * travel_time_prior + W * mu_data) / total_weight

        # base uncertainty ~ 1 / effective sample size
        base_unc = 1.0 / max(total_weight, 1e-8)

        # staleness: time since last observation
        last_day  = float(days.max())
        staleness = max(current_day - last_day, 0.0)

        # inflate uncertainty with staleness
        travel_time_uncertainty = base_unc * (1.0 + staleness_scale * staleness)

        return expected_tt, travel_time_uncertainty

    def update_beliefs_from_history(self, current_day:int):
        """
        Recompute expectations, uncertainties, and priors for both car and bus,
        using the COMPLETE history as it currently exists.
       
        """
        # build (day, tt) pairs by mode
        car_travel_data = [
            (rec["day_index"], rec["realized_tt"])
            for rec in self.history
            if rec["mode"] == "car"
        ]

        bus_travel_data = [
            (rec["day_index"], rec["realized_tt"])
            for rec in self.history
            if rec["mode"] == "bus"
        ]

        # --- CAR ---
        exp_car, unc_car = self.compute_experience_beliefs(
            travel_data=car_travel_data,
            current_day=current_day,
            time_decay_rate=self.time_decay_rate,
            travel_time_prior=self.prior_car,
            prior_weight=self.prior_weight,
        )

        self.expected_tt_car = exp_car
        self.travel_time_uncertainty_car = unc_car

        # --- BUS ---
        exp_bus, unc_bus = self.compute_experience_beliefs(
            travel_data=bus_travel_data,
            current_day=current_day,
            time_decay_rate=self.time_decay_rate,
            travel_time_prior=self.prior_bus,
            prior_weight=self.prior_weight,
        )

        self.expected_tt_bus = exp_bus
        self.travel_time_uncertainty_bus = unc_bus


    def record_experience(self,  **kwargs):
        """
        Append a single trip record to history, update the mode-specific prior
        with this realized travel time.
        """

        # copy of kwargs as the history record 
        record = {**kwargs}
        self.history.append(record)

        # extract and coerce values used to update priors
        mode = record["mode"]
        realized_tt = float(record["realized_tt"])

        # update mode-specific prior based on this single realization
        if mode == "car":
            self.prior_car = self.slow_prior_update(self.prior_car, realized_tt)
        elif mode == "bus":
            self.prior_bus = self.slow_prior_update(self.prior_bus, realized_tt)
        else:
            # if you ever introduce more modes, handle them here
            pass

    