from mesa import Agent
import numpy as np
from scipy.stats import skewnorm


import traffic.utils.unit_conversion_utils as uc

# this has to be externial to VehicleAgent because each VehicleAgent gets one accel() function build_empirical_accel_function produces an accel(). 
# it would not work if VehicleAgent got build_empirical_accel_function
def build_empirical_accel_function(pctile, mean_shift=-.2 , var_streach=1.3):
    """
    Builds a function that estimates acceleration (in m/s²)
    given speed (in mph), using empirical acceleration data
    from real-world stop sign behavior.

    Returns:
        accel(speed_mph): callable function
    """
    
    if pctile < .07: trimmed_pctile = .07
    elif pctile > .95: trimmed_pctile = .95
    else: trimmed_pctile = pctile
    og_means = [1, 2.5, 2, 1.5]
    og_vars = [.35, .4, .4, .3]

    means = [i+mean_shift for i in og_means]
    var = [i*var_streach for i in og_vars]
    
    # differnet dists in m/s^s 
    dist0 = skewnorm(loc=means[0], scale=var[0], a=1)
    dist1 = skewnorm(loc=means[1], scale=var[1], a=1)
    dist2 = skewnorm(loc=means[2], scale=var[2], a=1)
    dist3 = skewnorm(loc=means[3], scale=var[3], a=1)

    # Acceleration values in G, time intervals in seconds
    segments = [
        {"start_t": 0, "end_t": 2, "accel_mpss": dist0.ppf(trimmed_pctile)},
        {"start_t": 2, "end_t": 4, "accel_mpss": dist1.ppf(trimmed_pctile)},
        {"start_t": 4, "end_t": 6, "accel_mpss": dist2.ppf(trimmed_pctile)},
        {"start_t": 6, "end_t": 8, "accel_mpss": dist3.ppf(trimmed_pctile)},
    ]

    # Convert to speed ranges in mph
    speed_bounds = [0]
    for seg in segments:
        delta_v_mps = seg["accel_mpss"] * (seg["end_t"] - seg["start_t"])
        delta_v_mph = delta_v_mps * 2.2  # convert m/s to mph
        speed_bounds.append(speed_bounds[-1] + delta_v_mph)

    # Pre-compute acceleration in m/s² for each segment
    accels_mps2 = [seg["accel_mpss"] for seg in segments]
    def accel(speed_mph):
        for i in range(len(speed_bounds) - 1):
            if speed_bounds[i] <= speed_mph < speed_bounds[i + 1]:
                return accels_mps2[i]
        return np.clip(accels_mps2[-1]*  (1-((speed_mph-speed_bounds[-1])/70)), 0,10)

    return accel


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- The actual VehicleAgent Class ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    
class VehicleAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.passengers: list = []

    def reset_next_agent(self):
        """No-op kept for BlockerAgent compatibility."""
        pass

    def vehicle_to_tp_info_pass(self):
        for tp in getattr(self, "passengers", []):
            tp.toll_paid = self.toll_paid
            tp.board_step = self.created_at_step
            tp.arrive_step = self.model.steps
            tp.cumtime_lost_sec = self.cumtime_lost_sec

            tp.tp_to_sp_info_pass()

    def end_of_road(self):
        m = self.model
        vs = m.vs
        slot = vs.vid_to_slot.get(self.unique_id)
        if slot is None:
            return   # already removed (guard against double-call)

        dist         = float(vs.dist[slot])
        steps        = int(vs.steps_taken[slot])
        interactions = int(vs.car_interactions[slot])
        cumtime      = float(vs.cumtime_lost[slot])
        toll         = float(vs.toll_paid[slot])
        performance  = float(vs.performance[slot])
        curve_resp   = float(vs.curve_resp[slot])
        acceptable_ov= float(vs.acceptable_ov[slot])
        ideal_dm     = float(vs.ideal_dm[slot])
        created      = int(vs.created_step[slot])
        veh_type     = int(vs.veh_type[slot])

        m.finished_agents.append({
            "AgentID":                 self.unique_id,
            "AgentType":               "CarAgent" if veh_type == 0 else "BusAgent",
            "created_at_step":         created,
            "steps_taken":             steps,
            "car_interactions":        interactions,
            "distance_traveled":       dist,
            "approx_average_mph":      uc.meters_to_miles(dist) / (steps / 3600) if steps > 0 else 0.0,
            "performance":             performance,
            "curve_responce":          curve_resp,
            "acceptable_over":         uc.get_mph(acceptable_ov),
            "ideal_distance_multiplier": ideal_dm,
            "cumtime_lost":            cumtime,
            "toll_paid":               toll,
        })

        # Set attributes needed by vehicle_to_tp_info_pass (and BusAgent override)
        self.toll_paid = toll
        self.created_at_step = created
        self.cumtime_lost_sec = cumtime

        # Notify passengers
        self.vehicle_to_tp_info_pass()

        # Slot cleanup
        vs.remove(self.unique_id)
        del m.vid_to_vehicle[self.unique_id]
        m.vehicles_list.remove(self)
        self.remove()   # Mesa Agent removal