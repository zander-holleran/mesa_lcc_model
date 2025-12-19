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
    trimmed_pctile = np.clip(pctile, .07, .95)
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
         # 1) freeze this vehicle's spawn and place it
        p_start = np.array(model.start_point, dtype=float)
        self.model.space.place_agent(self, tuple(p_start))
        
        # 2) freeze road segments and build path points (replace first point with p_start)
        self.road_segments = self.model.agents.select(agent_type=self.model.agent_cls['road'])
        self._rs = list(self.road_segments)                              # fixed order
        real_xy = np.array([rs.pos for rs in self._rs], float)    # points for road
        real_xy[0] = p_start                                      # replace first point
        self.path_xy = real_xy                                    # single source of truth (points)

         # 3) stuff for speed limit lookup
        self.N_ahead = 5
        self._seg_speed = np.fromiter((r.speed_limit for r in self._rs), dtype=float)  # m/s
        self._seg_curve = np.fromiter((r.curvature   for r in self._rs), dtype=float)
        w = np.array([1.0/(1+i) for i in range(self.N_ahead)], dtype=float)
        self._weights = w / w.sum()

        # 4) for move along path
        seg_vec = self.path_xy[1:] - self.path_xy[:-1]            # shape [n-1, 2]
        seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
        seg_len[seg_len == 0.0] = 1e-12                           # guard zero length
        self._seg_vec = seg_vec
        self._seg_len = seg_len
        self._seg_dir = seg_vec / seg_len[:, None]
        self.path_index = 0               # 0..(n-2): which segment we're on
        self._s = 0.0                     # distance traveled within current segment

        # ---- 5) For data collection ----
        self.status = "driving"
        self.speed = self.road_segments[0].speed_limit # the attribute is stored in m/s even tho the analysis is in mph
        self.distance_traveled = -model.space.get_distance(model.initial_start_point, tuple(p_start))  # (meters)
        self.break_cooldown = 0
        self.speed_change = 0
        self.created_at_step = self.model.steps
        self.steps_taken = 0 
        self.car_interactions = 0
        self.gap = 9_999_999 # distance to next vehicle (m) starts as effectively infinite
        self.ideal_distance_multiplier = 1.5
        self.ideal_gap=max(self.speed * self.ideal_distance_multiplier,5) # desired following distance (m)
        self.next_agent = None 
        self.driving_action = None # what the car did this step, accelerate, break, coast, etc
        self.posted_speed_limit = None # the posted speed limit of the road segment the car is currently on
        self.implicit_speed_limit = None   # the speed limit adjusted for curvature, acceptable_over
        self.speed_delta = 0  # difference between current speed and desired speed, usually neg (mpg)
        self.cumtime_lost_sec = 0.0  # cumulative time lost due to traffic (seconds)
        self.time_lost_sec = 0.0  # time lost that step (seconds)
        self.toll_paid = 0  # total toll paid by this vehicle, assigned by buses and cars

        # Speed control tuning parameters (can be overridden)
        self.acceptable_over = None
        self.curve_responce = None
        self.performance = .5
        self.accel_curve = build_empirical_accel_function(self.performance)

        # passengers list
        self.passengers = []

    def calculate_time_lost(self):
        self.speed_delta = uc.get_mph(self.speed) - self.implicit_speed_limit 
        frac_seconds_lost = max(0.0, -self.speed_delta / self.implicit_speed_limit)
        self.time_lost_sec = frac_seconds_lost
        self.cumtime_lost_sec += frac_seconds_lost


    def get_next_agent(self): 
        '''
        Used in the get_gap function
        Takes self, checks if a next_agent exists and status == driving, if so uses that, if not trys to find a new next agent. 
        '''
        # check if 1) next car is already saved & 2)it is driving. The secound check is to deal with when cars get removed at the end. This works because self.next_agent existing is tested first
        if self.next_agent and self.next_agent.status != 'arrived':
            return
        # Find the closest vehicle ahead (or None if there isn't one)
        vehicles_ahead = self.model.agents.select(
            lambda a: a.distance_traveled > self.distance_traveled,
            agent_type=(VehicleAgent, self.model.agent_cls['blocker']),
        )

        self.next_agent = min(vehicles_ahead, key=lambda a: a.distance_traveled, default=None)

    def reset_next_agent(self):
        '''used at verious points to clear the next_agent var'''
        self.next_agent = None

    def get_gap(self):
            """
            Returns:
                ideal_gap: float — the desired following distance (m)
                gap: float — distance to the closest vehicle ahead (m)

            Used in the adjust_speed function
            """
            ideal_gap = max(self.speed * self.ideal_distance_multiplier,5)
            # if a agent exists then measure the gap
            if self.next_agent: 
                gap = self.model.space.get_distance(self.pos, self.next_agent.pos)
            else:
                gap = 9999999 # effectively infinite gap if no next agent
            self.ideal_gap = ideal_gap
            self.gap = gap

    # Fast clip for scalars
    def _clip01(self, x: float) -> float:
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    def get_speed_limit(self):
        # ----- slice cached arrays (no list() rebuild) -----
        i = self.path_index
        j = i + self.N_ahead
        sl = self._seg_speed[i:j]   # posted speed limits ahead
        cv = self._seg_curve[i:j]   # curvatures ahead

        # weights truncated to actual slice length, renormalized (end-of-path safety)
        w = self._weights[: len(sl)]
        if w.size != self._weights.size:
            w = w / w.sum()

        # tiny dot products (faster than np.average for small N)
        avg_sl = float((sl * w).sum())
        avg_cv = float((cv * w).sum())

        # ----- curve_adjust (branchy, no numpy) -----
        curve_effect  = self._clip01(avg_cv/90.0)
        if avg_sl <= 15.0:
            speed_effect = 0.0
        else:
            speed_effect = self._clip01((avg_sl - 10.0) / (60.0 - 10.0))

        curve_speed_limit_mph = avg_sl * (1.0 - (self.curve_responce * curve_effect * speed_effect))
        implicit_speed_limit_mph = curve_speed_limit_mph + self.acceptable_over

        # save
        # posted speed limit of CURRENT segment (0-length slice safe)
        self.posted_speed_limit = float(self._seg_speed[i]) if i < self._seg_speed.size else float(self._seg_speed[-1])
        self.implicit_speed_limit = implicit_speed_limit_mph
        # (function returns None, same as before)
    
    def less_smooth_brake(self, gap, ideal_gap):
        """
        Simulate more realistic, human-like braking behavior.
        Returns a value between 0 and 1 indicating brake intensity.
        """
        if ideal_gap <= 0 or np.isnan(ideal_gap):
            return 0
        force = max((ideal_gap - gap) / ideal_gap, 0)
        # Squared to overreact when too close
        base = force ** 2
        
        # Add some human-like noise
        noise = np.random.normal(0, .1)
        break_pct =  np.clip(base + noise, 0, 1)

        deceleration = break_pct * 8 # <- this is acting as max decel in mps 
        return deceleration

    def speed_limit_brake(self, speed_limit, speed):
        '''
        used when the car is going over the speed limit, essentially the more your going over the sl the more you apply breaks
        '''
        if speed < speed_limit:
            # this should never be triggered but i added anyway to make sure it didnt trip an error
            return 0
        mph_over = uc.get_mph(speed)-uc.get_mph(speed_limit) 
        if mph_over > 7: 
            return 1.1 
        elif mph_over > 2:
            return .5
        elif mph_over > 0:
            return .2
   
    def adjust_status(self):
        '''
       used to set the status of the car, driving, canyon_closure, crash ect
        '''
        if not self.next_agent or self.next_agent.status == 'driving': # if no next agent or next agent is driving then you are driving
            self.status = 'driving'
            return 
        elif self.gap <= self.ideal_gap and self.next_agent.status != 'driving': # if you are too close to a non driving next agent then status is their status
            self.status = 'slowing'
            # additionally if you become blocked you take on a different status
            if self.speed < 1: 
                self.status = self.next_agent.status
            return
        else: # if you are far enough away from a non driving next agent then you are driving
            self.status = 'driving'
            return
            
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- The initial adjust speed ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
    def adjust_speed(self):
        ''' takes self from self uses'''
        implicit_speed_limit_mps =  uc.get_mps(self.implicit_speed_limit)        
        # save the current speed 
        old_speed = self.speed 
        
        # 1) measues the gap to the next vehicle, if less than the ideal gap, applies the smooth breaking
        if self.gap < self.ideal_gap:
            self.driving_action = 'smooth_break'
            self.car_interactions += 1
            self.break_cooldown = 5
            self.speed -= self.less_smooth_brake(gap=self.gap, ideal_gap=self.ideal_gap)

        # 3) if outside the jitter threashhold see if the car is above speed limit, if so break
        elif self.speed > implicit_speed_limit_mps:
            self.driving_action = 'speed_limit_break'
            self.break_cooldown = 3
            self.speed -= self.speed_limit_brake(speed_limit=implicit_speed_limit_mps, speed=self.speed)
            
        # 4) if outside the jitter threashhold & below speed limit & max speed then speed up 
        elif self.break_cooldown in [4,5]:
            self.driving_action = 'coast'
            self.break_cooldown -= 1 
        
        elif self.break_cooldown in [1,2,3]: # self.break_cooldown will be 3,2,1
            self.driving_action = 'slow_accelerate'
            self.speed+= self.accel_curve(uc.get_mph(self.speed)) * ((4-self.break_cooldown)/4)
            self.break_cooldown -= 1 

        else:
            self.driving_action = 'accelerate'
            self.speed+= self.accel_curve(uc.get_mph(self.speed))

        # overwrites
        if (self.next_agent is not None) and ((self.speed - self.next_agent.speed) > self.gap):
            self.driving_action = 'prevent_pass'
            self.break_cooldown = 5
            self.speed = self.next_agent.speed-1

        # dont go backwards
        self.speed = max(self.speed, 0)
        self.ideal_gap = max(self.speed * self.ideal_distance_multiplier,5)

        # new speed - old speed
        self.speed_change = self.speed - old_speed
              
    def move_along_path(self):
        """Advance along path by self.speed meters using a single path + index."""
        if self.speed <= 0.0:  # early out for not moving
            # pin at the last point
            self.steps_taken += 1
            return
                    
        d = float(self.speed)
        self.distance_traveled += d # bookkeeping
        i = self.path_index
        s = self._s

        # consume distance across segments
        while d > 0.0 and i < self._seg_len.size:
            rem = self._seg_len[i] - s
            if d < rem:
                s += d
                d = 0.0
            else:
                d -= rem
                i += 1
                s = 0.0

        
        if i >= self._seg_len.size: # if at end of road
            i = self._seg_len.size - 1
            s = self._seg_len[i]
            new_pos = self.path_xy[-1]
            self.end_of_road()
        else:
            start_of_seg = self.path_xy[i]
            new_pos = start_of_seg + self._seg_dir[i] * s # this identifies the new pos as the start of the seg + the direction of the seg * how far into the seg you are (s)

        # book keeping
        self.path_index = i
        self._s = s
        self.steps_taken += 1

        # finally move the agent
        self.model.space.move_agent(self, (float(new_pos[0]), float(new_pos[1])))


    def vehicle_to_tp_info_pass(self):
        for tp in getattr(self, "passengers", []):
            tp.toll_paid = self.toll_paid
            tp.board_step = self.created_at_step
            tp.arrive_step = self.model.steps
            tp.cumtime_lost_sec = self.cumtime_lost_sec

            # call the trip completed hook on the person
            tp.tp_to_sp_info_pass()

    def end_of_road(self):
        self.model.finished_agents.append({
            "AgentID": self.unique_id,
            'AgentType': self.__class__.__name__,
            "created_at_step": self.created_at_step,
            "steps_taken": self.steps_taken,
            "car_interactions": self.car_interactions, 
            "distance_traveled": self.distance_traveled, 
            "approx_average_mph": uc.meters_to_miles(self.distance_traveled)/(self.steps_taken/3600), 
            'performance': self.performance,
            'curve_responce': self.curve_responce,
            "acceptable_over": uc.get_mph(self.acceptable_over),
            "ideal_distance_multiplier":self.ideal_distance_multiplier, 
            "cumtime_lost": self.cumtime_lost_sec,
            "toll_paid": self.toll_paid,
            # Add more if needed
        })

        self.vehicle_to_tp_info_pass()

        self.status = "arrived"

        # remove vehicle from various model lists
        try: self._rs[self.path_index].vehicles_here.remove(self)       
        except Exception: pass
        try: self.model.space.remove_agent(self)                        
        except Exception:pass
        try: self.model.vehicles_list.remove(self)                     
        except ValueError: pass 
        self.remove() # remove from model