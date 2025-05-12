from mesa import Agent
import numpy as np
from scipy.stats import skewnorm

from agents.road_segment_agent import RoadSegmentAgent
import utils.unit_conversion_utils as uc


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


def vert_decel(slope_deg):
    # not used yet
    slope_rad = np.radians(slope_deg)
    vert_decel = 9.81* np.sin(slope_rad)
    return vert_decel



# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- The actual VehicleAgent Class ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    
class VehicleAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

        self.status = "driving"
        self.speed = uc.get_mps(1) # starting speed
        self.break_cooldown = 0
        
        # For data collection 
        self.speed_change = 0
        self.created_at_step = self.model.steps 
        self.steps_taken = 0 
        self.distance_traveled = 0
        self.car_interactions = 0
        self.gap = 0 
        self.next_agent = None
        self.driving_action = None
        self.posted_speed_limit = None
        self.implicit_speed_limit = None

        # Speed control tuning parameters (can be overridden)
        self.ideal_distance_multiplier = None
        self.acceptable_over = None
        self.curve_responce = None
        self.performance = .5
        self.accel_curve = build_empirical_accel_function(self.performance)

        # init all the road segement data 
        self.road_segments = self.model.agents.select(agent_type=RoadSegmentAgent)

        # Establish the Vehicles position 
        self.path = self.road_segments.get('position')
        self.path_index = 0

        self.model.space.place_agent(self, self.path[0])  # <-- here is the intial place agent

        
    def end_of_road(self):
        '''this is where one part of finished agents reporting occures, is paired with '''
        if self.path_index >= len(self.path) - 1:
            self.status = "arrived"
            
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
                "ideal_distance_multiplier":self.ideal_distance_multiplier
            # Add more if needed
            })
            self.remove() 
            return True


    def get_next_agent(self): 
        '''
        Used in the get_gap function
        Takes self, checks if a next_agent exists and status == driving, if so uses that, if not trys to find a new next agent. 
        '''
        # check if 1) next car is already saved & 2)it is driving. This works because self.next_agent existing is tested first
        if self.next_agent and self.next_agent.status == "driving":
             return
        
        # if the next agent does not exist look for a new next_agent
        other_vehicles = self.model.agents.select(agent_type=VehicleAgent)
        cars_ahead = [
            agent for agent in other_vehicles
            if agent.distance_traveled > self.distance_traveled
        ]
        
        # set the next agent to the next vehicle, if no next vehicle then set to None
        if cars_ahead:
            self.next_agent = min(cars_ahead, key=lambda agent: agent.distance_traveled)
        else:
            self.next_agent = None
            
        
    def get_gap(self):
            """
            Returns:
                ideal_gap: float — the desired following distance (deg)
                gap: float — distance to the closest vehicle ahead (deg)

            Used in the adjust_speed function
            """
            ideal_gap = max(self.speed * self.ideal_distance_multiplier,2)
            
            # run the get_new_next_agent function 
            self.get_next_agent()
            
            # if a agent exists then measure the gap
            if self.next_agent: 
                gap = self.model.space.get_distance(self.pos, self.next_agent.pos)
            else:
                gap = np.nan
            self.gap = gap
            return ideal_gap, gap
        
    def get_speed_limit(self):
        def curve_adjust(max_affect_pct=0.5, curvature=45, speed=60):
            '''
            max_affect_pct - float: max possible speed reduction proportion at extreme curve
            curvature - float: standardized curve (0-90 degrees ideally)
            speed - float: current vehicle speed in mph
            '''
            curve_effect = curvature/90  # normalized curvature
            curve_effect = np.clip(curve_effect, 0, 1)  # protect against overcurve
        
            if speed <= 15:
                speed_effect = 0  # no curve penalty below 15 mph
            else:
                speed_effect = (speed - 10) / (60 - 10)  # normalized to [0, 1] between 15 and 60 mph
                speed_effect = np.clip(speed_effect, 0, 1)  # protect against overspeed
            return speed * (1 - (max_affect_pct * curve_effect * speed_effect))

        # gather the data from the road segments
        # 1. Get the 5 agents
        next_road_agents = list(self.road_segments)[self.path_index:self.path_index+4]
        posted_limit = [agent.speed_limit for agent in next_road_agents]
        self.posted_speed_limit=posted_limit[0]
        curvatures = [agent.curvature for agent in next_road_agents]
        
        # weighted averages
        weights = np.array([1 / (1 + i) for i in range(len(posted_limit))])
        average_posted_limit = np.average(posted_limit, weights=weights)
        average_curvature = np.average(curvatures, weights=weights)

        # enter the info in to the curve adjust function 
        curve_speed_limit_mph = curve_adjust(self.curve_responce, average_curvature, average_posted_limit)
        implicit_speed_limit_mph = curve_speed_limit_mph + self.acceptable_over
        self.implicit_speed_limit = implicit_speed_limit_mph
        return  uc.get_mps(implicit_speed_limit_mph)

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
        #print(mph_over)
        if mph_over > 7: 
            return 1.1 
        elif mph_over > 2:
            return .5
        elif mph_over > 0:
            return .2
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- The initial adjust speed ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-
            
    def adjust_speed(self):
        ''' takes self from self uses'''
        ideal_gap, gap  = self.get_gap() 
        implicit_speed_limit = self.get_speed_limit()

        # save the current speed 
        old_speed = self.speed 
        
        # 1) measues the gap to the next vehicle, if less than the ideal gap, applies the smooth breaking
        if gap < ideal_gap:
            self.driving_action = 'smooth_break'
            self.car_interactions += 1
            self.break_cooldown = 5
            self.speed -= self.less_smooth_brake(gap=gap, ideal_gap=ideal_gap)

        # 3) if outside the jitter threashhold see if the car is above speed limit, if so break
        elif self.speed > implicit_speed_limit:
            self.driving_action = 'speed_limit_break'
            self.break_cooldown = 3
            self.speed -= self.speed_limit_brake(speed_limit=implicit_speed_limit, speed=self.speed)
            
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
        if (self.next_agent is not None) and ((self.speed - self.next_agent.speed) > gap):
            self.driving_action = 'prevent_pass'
            self.break_cooldown = 5
            self.speed = self.next_agent.speed-1

        # dont go backwards
        self.speed = max(self.speed, 0)
        
        # new speed - old speed
        self.speed_change = self.speed - old_speed
    
            
    def move_along_path(self):
        """Move the agent along its predefined path based on current speed."""
        distance_to_travel = self.speed # sets a local variable in the function 
        self.distance_traveled += distance_to_travel # adds distance_to_travel(from this step) to the overall distance traveled
        pos = np.array(self.pos) # current position 
        new_position = pos
        
        while distance_to_travel > 0 and not self.end_of_road():
            next_target = np.array(self.path[self.path_index + 1])
            direction = self.model.space.get_heading(pos, next_target)
            distance = self.model.space.get_distance(pos, next_target)
    
            if distance < distance_to_travel:
                self.path_index += 1    
                distance_to_travel -= distance
                pos = next_target
                new_position = pos
            else:
                step_vector = distance_to_travel * direction / distance
                new_position = pos + step_vector
                distance_to_travel = 0

        # after you have figured out where ur going to move to 
        # 1) add yourself to that road_segment's vehicles_here
        new_segment = self.road_segments[self.path_index]
        new_segment.vehicles_here.append(self)
        # 2) incroment the steps taken 
        self.steps_taken += 1  
        # 3) actually move the agent
        self.model.space.move_agent(self, tuple(new_position))


