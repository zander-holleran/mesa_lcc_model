from mesa import Agent
from scipy.stats import lognorm, skewnorm, norm
from math import exp, sqrt
from typing import Sequence, Tuple
# from traffic.utils import compute_experience_beliefs, compute_generalized_cost

class TrafficPersonAgent(Agent):
    def __init__(self, unique_id, model, person_data=None):
        super().__init__(unique_id, model)

        # If we got person_data from Season, use it
        if person_data is not None:
            self.person_id = person_data["person_id"]

            # VOT passed in as $/hour; convert once to $/minute
            self.value_of_time = person_data["value_of_time"] / 60.0
            self.experience_weight_car = person_data["experience_weight_car"]
            self.experience_weight_bus = person_data["experience_weight_bus"]
            self.prior_car = person_data["prior_car"]
            self.prior_bus = person_data["prior_bus"]
            self.time_decay_rate = person_data["time_decay_rate"]
            self.prior_weight = person_data["prior_weight"]
            self.uncertainty_multiplier = person_data["uncertainty_multiplier"]
        else:
            # Fallback for runs WITHOUT Season: simple defaults or draws
            self.person_id = unique_id
            self.value_of_time = lognorm(s=0.64 , scale=40 ).rvs() / 60.0
            self.experience_weight_car = 1.0
            self.experience_weight_bus = skewnorm(6, loc=1.15, scale=0.3).rvs()
            self.prior_car = 22.0
            self.prior_bus = 60.0
            self.time_decay_rate = 0.1
            self.prior_weight = 1.0
            self.uncertainty_multiplier = 1

        # For now, no history stored here – you can layer that in later

    def compute_experience_beliefs(
        travel_data: Sequence[Tuple[float, float]],
        current_day: float,
        time_decay_rate: float,
        travel_time_prior: float,
        prior_weight: float = 1.0,
        ) -> Tuple[float, float]:

        """
        Compute decay-weighted beliefs about travel time from past trip history.


        Parameters
        ----------
        travel_data : sequence of (day, travel_time)
            History of realized travel times for this agent and mode.
            Each element is a tuple (day_i, travel_time_i), with day_i <= current_day.
        current_day : float
            The day index at which beliefs are being evaluated (e.g., t in U_m(t)).
        time_decay_rate : float
            Exponential decay rate λ >= 0. Larger values mean older trips lose weight faster.
        travel_time_prior : float
            Prior mean travel time to which expected_travel_time will revert as the
            influence of past data decays away.
        prior_weight : float, default 1.0
            Effective weight of the prior in the weighted average. Acts like a pseudo-count:
            larger values keep beliefs closer to travel_time_prior.

        Returns
        -------
        expected_travel_time : float
            Decay-weighted mean travel time at current_day, shrunk toward travel_time_prior.
        travel_time_uncertainty : float
            A measure of uncertainty about the mean travel time, increasing as the total
            effective weight (prior_weight + W(t)) becomes small (e.g., when the mode
            hasn’t been used recently or often).
        """
        # No data at all: fall back to prior and a simple uncertainty proxy
        if not travel_data:
            expected_travel_time = travel_time_prior
            # Maximal uncertainty case; you can refine this later if you like.
            travel_time_uncertainty = 1.0 / max(prior_weight, 1e-8)
            return expected_travel_time, travel_time_uncertainty

        # Exponential decay weights based on how long ago each trip was
        weights = []
        times = []
        for day_i, travel_time_i in travel_data:
            age = current_day - day_i
            w_i = exp(-time_decay_rate * age) if time_decay_rate > 0 else 1.0
            weights.append(w_i)
            times.append(travel_time_i)

        W = sum(weights)

        # Decay-weighted mean with prior pull
        total_weight = prior_weight + W
        weighted_sum = prior_weight * travel_time_prior + sum(w * y for w, y in zip(weights, times))
        expected_travel_time = weighted_sum / total_weight

        # Decay-weighted squared deviations (including the prior as a pseudo-observation)
        S = prior_weight * (travel_time_prior - expected_travel_time) ** 2
        S += sum(w * (y - expected_travel_time) ** 2 for w, y in zip(weights, times))

        # Treat total_weight as an "effective sample size" to scale uncertainty:
        # more total weight -> smaller uncertainty; as weights decay, uncertainty grows.
        travel_time_uncertainty = sqrt(S) / total_weight if total_weight > 0 else 0.0

        return expected_travel_time, travel_time_uncertainty
    
    def compute_generalized_cost(
        expected_travel_time: float,
        travel_time_uncertainty: float,
        uncertainty_multiplier: float,
        value_of_time: float,
        experience_weight: float,
        toll: float = 0.0,
        ) -> float:
        """
        Compute generalized monetary cost for a mode from experience-based beliefs and a toll.

        Parameters
        ----------
        expected_travel_time : float
            Belief about mean travel time for this mode on this day (e.g., minutes).
        travel_time_uncertainty : float
            Belief about uncertainty in travel time (e.g., standard deviation, minutes).
        uncertainty_multiplier : float
            Scalar controlling how uncertainty contributes to the effective minutes:
            > 0 : uncertainty makes the mode worse (adds effective minutes).
            < 0 : uncertainty makes the mode better (subtracts effective minutes).
            = 0 : uncertainty ignored.
        value_of_time : float
            Monetary value per minute of effective travel time (e.g., dollars per minute).
        experience_weight : float
            Mode-/agent-specific weight λ_m that rescales perceived minutes in this mode:
            > 1 : minutes in this mode feel worse than baseline.
            < 1 : minutes in this mode feel better than baseline.
            = 1 : neutral.
        toll : float, default 0.0
            Monetary toll or fare for this mode on this day (e.g., dollars).

        Returns
        -------
        float
            Generalized cost C_m(t) in money units (e.g., dollars).
        """
        effictive_mins = expected_travel_time + (uncertainty_multiplier * travel_time_uncertainty) # Effective minutes from beliefs
        travel_time_cost = value_of_time * experience_weight * effictive_mins # Monetary equivalent of effective minutes
        generalized_cost =  travel_time_cost + toll
        return generalized_cost
    
    def get_beliefs(self, mode, day):
        if mode == "car":
            history = self.history_car
            prior = self.prior_car
        else:  # "bus"
            history = self.history_bus
            prior = self.prior_bus

        expected_tt, uncertainty_tt = self.compute_experience_beliefs(
            travel_data=history,
            current_day=day,
            time_decay_rate=self.time_decay_rate,
            travel_time_prior=prior,
            prior_weight=self.prior_weight,
        )
        return expected_tt, uncertainty_tt

    def compute_cost(self, mode, day, toll):
        expected_tt, uncertainty_tt = self.get_beliefs(mode, day)

        if mode == "car":
            experience_weight = self.experience_weight_car
        else:  # "bus"
            experience_weight = self.experience_weight_bus

        return self.compute_generalized_cost(
            expected_travel_time=expected_tt,
            travel_time_uncertainty=uncertainty_tt,
            uncertainty_multiplier=self.uncertainty_multiplier,
            value_of_time=self.value_of_time,
            experience_weight=experience_weight,
            toll=toll,
        )

    def decide_mode(self, day, toll_car=0.0, toll_bus=0.0):
        cost_car = self.compute_cost("car", day, toll_car)
        cost_bus = self.compute_cost("bus", day, toll_bus)
        # True = take_car, False = take_bus
        return cost_car < cost_bus

    