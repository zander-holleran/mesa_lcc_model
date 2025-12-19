# genreal imports
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import my agents
from traffic.agents import VehicleAgent, BlockerAgent, BusAgent, CarAgent, RoadSegmentAgent, TrafficPersonAgent

# import my utils
from traffic.utils import unit_conversion_utils as uc  # for get_mph, etc.

# import other parts of model
import traffic.model.reporting as rep
import traffic.model.generate as gen 
import traffic.model.init_helpers as ih 
from collections import defaultdict


class TrafficModel(Model):
    """Mesa model simulating traffic on the canyon road with a car cap."""

    def __init__(self, road_gdf, ecs_df, max_steps=50000, seed=123, batchrun=False, collect_every_n=1, 
                 start_hr=5, traffic_percentile=None, p_generate=None, max_persons=50,
                 canyon_closures={},
                 bus_interval=30, car_preference=1, bus_capacity=30, 
                 crashes_per_100k_vmt_input=4,
                 toll_mechanism="static", toll_params=None,
                 season_persons=None,
                 current_day = 0
                 ):
        super().__init__(seed=seed)

        self.created_counts = defaultdict(int) # temp

        self._np_rng = np.random.default_rng(seed)

        self.agent_cls = {
            "vehicle": VehicleAgent,
            "blocker": BlockerAgent,
            'car': CarAgent,
            "bus": BusAgent,
            "road": RoadSegmentAgent,
            "traffic_person": TrafficPersonAgent
        }
        
        # --- SeasonPersons wiring ---
        self.season_person_pool = list(season_persons) or []

        #===== model perams =====
        self.current_day = current_day
        self.expected_counts_seconds = ecs_df
        self.max_steps = max_steps
        self.batchrun = batchrun
        self.collect_every_n = collect_every_n
        self.initial_start_point = tuple(road_gdf.iloc[0].geometry.coords[0])  # never changes
        self.start_point = tuple(road_gdf.iloc[0].geometry.coords[0])          # may be moved by "too_close" logic

        # ===== Tolling perams =====
        self.toll_mechanism = toll_mechanism
        self.toll_params = toll_params or {}

        self.current_toll_car = self.toll_params.get("car", 0.0)
        self.current_toll_bus = self.toll_params.get("bus", 0.0)

        # ===== car centric perams =====
        self.start_step = uc.sec_after_five(start_hr)
        self.traffic_percentile = traffic_percentile
        self.p_generate = p_generate  # Probability of new car each step
        self.max_persons = max_persons  # Maximum number of persons allowed

        # ===== canyon closure perams =====
        if not canyon_closures or len(canyon_closures) == 0:
            self.canyon_closures = pd.DataFrame([]) 
        else:
            self.canyon_closures = pd.DataFrame(canyon_closures).sort_values('closure_step').reset_index(drop=True)
            
        # ===== bus centric perams =====
        self.bus_interval = bus_interval
        if self.bus_interval == 0: 
            self.car_preference = 1 
        else: 
            self.car_preference = car_preference
        self.bus_capacity = bus_capacity
        self.bus_first_departure = self.random.randint(0, self.bus_interval * 60)  # Random step between 0 and 5 mins
        self.next_bus_step = self.bus_first_departure
        
        # ===== crash perams =====
        self.crashes_per_100k_vmt = crashes_per_100k_vmt_input
        self.remainder = 0 
        self.crashes = 0
        self.total_crashes = 0

        # ===== verious trackers =====
        self.too_close_counter = 0 
        self.person_counter = 0 
        self.bus_counter = 0 
        self.car_counter = 0 
        self.bus_riders = 0 
        self.at_bus_stop = []
        self.finished_agents = []

        # ===== Set up ContinuousSpace =====
        buffer = 10000
        minx, miny, maxx, maxy = road_gdf.total_bounds
        self.space = ContinuousSpace(x_min=minx - buffer, x_max=maxx + buffer, y_min=miny - buffer, y_max=maxy + buffer, torus=False)

        # set up road segments with a helper
        self.road_segments = ih.init_road_segments(
            model=self,
            road_gdf=road_gdf
        )

        # place the roadsegments on the space
        for agent, (x, y) in zip(self.road_segments, self.rs_pos):
            self.space.place_agent(agent, (x, y))

        # set up the reporters
        if self.batchrun: 
            self.datacollector = DataCollector(model_reporters = rep.model_reporters)
        else:
            self.datacollector = DataCollector(model_reporters = rep.model_reporters, agent_reporters = rep.agent_reporters)

        # agent lists
        self.vehicles_list = []
        self.blockers_list = []
        self.traffic_persons_list = []

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ END OF INIT -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    def max_persons_check(self):
    # Stop model when entire population has completed a trip 
        if not self.season_person_pool and not self.traffic_persons_list:
            print(f"{self.person_counter} people arrived stopping model.")
            self.running = False
    
    def max_steps_check(self):
        # Stop model at hard cap of steps
        if self.steps >= self.max_steps:
            print(f"Reached max step count ({self.max_steps}). Stopping model.")
            self.running = False

    def should_crash_randomized_rounding(
        self, 
        crashes_per_100k_vmt,
        num_cars, # len(self.active_vehicles)
        avg_speed_mps, # avg(self.active_vehicles.speed)
        remainder: float = 0.0, # self.remainder
        rng: np.random.Generator | None = None
        ):

        """
        Randomized-rounding crash generator with no external state container.

        Args:
            crashes_per_100k_vmt: target crashes per 100,000 vehicle-miles.
            num_cars: cars on the road this step.
            avg_speed_mps: average speed (m/s) this step.
            remainder: fractional remainder carried from the previous step.
            rng: numpy random Generator (optional).

        Returns:
            crashes: int
            new_remainder: float (carry this to the next call)
        """
        METERS_PER_100K_MILES = 160_934_400.0
        if rng is None:
            rng = np.random.default_rng()

        meters_this_step = float(num_cars) * float(avg_speed_mps)
        lambda_step = (crashes_per_100k_vmt / METERS_PER_100K_MILES) * meters_this_step

        total = remainder + lambda_step # this continuiously increases until a crash happens then resets 
        base = int(total)
        frac = total - base
        crashes = base + (1 if rng.random() < frac else 0)
        new_remainder = total - crashes  # keep only the fractional part
        return crashes, new_remainder
    
    def update_tolls(self):
        """Compute current tolls based on mechanism + model state."""
        mech = self.toll_mechanism

        if mech == "static":
           return # no change

        elif mech == "volume":
            self._update_tolls_volume()
    
    def _update_tolls_volume(self):
        """
        Simple volume-based toll calulator: linear increase above a threshold.
        """

        # these params come from SeasonConfig / DayParams via __init__
        threshold = self.toll_params.get("volume_threshold", 100)
        slope = self.toll_params.get("slope", 0.05)          
        base_price = self.toll_params.get("base_price", 5.0)
       
        volume = len(self.vehicles_list)  # current number of vehicles on the road

        if volume <= threshold:
            car_toll = 0.0
        else:
            car_toll = base_price + slope * (volume - threshold)
        
        self.current_toll_car = car_toll


    def update_next_agents(self):
        # 1) Collect candidates once (avoid AgentSet scans inside per-agent code)
        vehicles = self.vehicles_list
        blockers = self.blockers_list
        next_agents = vehicles + blockers

        if not next_agents:
            return

        # 2) Sort by along-road progress; tie-break by object id (stable)
        next_agents.sort(key=lambda a: (a.distance_traveled, id(a)))

        # Assign next_agent and gap for each vehicle (and blocker)
        last = len(next_agents) - 1
        for i, a in enumerate(next_agents):
            na = next_agents[i + 1] if i < last else None
            a.next_agent = na
            a.gap = (na.distance_traveled - a.distance_traveled) if na is not None else float("inf")
    
     # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ Agent method loopers  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    # vehicles
    def do_adjust_status(self, agents):
        for a in agents:
            a.adjust_status()


    def do_adjust_speed(self, agents):
        for a in agents:
            a.adjust_speed()

    def do_move_along_path(self, agents):
        for a in agents:
            a.move_along_path()

    def do_calculate_time_lost(self, agents):
        for a in agents:
            a.calculate_time_lost()
    # blockers
    def do_tick(self, agents):
            for a in agents:
                a.tick()

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ THE STEP  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    def step(self):
        # Toll update
        self.update_tolls()

        # establish what p_generate is going to be for that step
        if self.traffic_percentile:
            self.p_generate = self.expected_counts_seconds.iloc[self.start_step, self.traffic_percentile]
            self.start_step += 1
        
        # generate functions
        gen.generate_person(self)
        gen.generate_new_bus(self)

        # vehicle functions
        all_vehicles = self.vehicles_list
        self.update_next_agents()
        self.do_adjust_status(all_vehicles)  

        driving_vehicles = []
        for v in all_vehicles:
            if v.status == "driving" or v.status == "slowing":
                driving_vehicles.append(v)

        self.do_adjust_speed(driving_vehicles)
        self.do_move_along_path(driving_vehicles)
        self.do_calculate_time_lost(all_vehicles)

        # Blocker actions
        n = len(driving_vehicles)
        if n:
            avg_speed_mps = 0.0
            for veh in driving_vehicles:
                avg_speed_mps += veh.speed
            avg_speed_mps /= n
        else:
            avg_speed_mps = 1.0

        self.crashes, self.remainder = self.should_crash_randomized_rounding(
            crashes_per_100k_vmt=self.crashes_per_100k_vmt,
            num_cars=n,
            avg_speed_mps=avg_speed_mps,
            remainder=self.remainder,
            rng=self.random
        )

        self.total_crashes += self.crashes

        # will generate a blocker if crashes > 0
        gen.generate_crash(self)
        gen.generate_canyon_closure(self)
        
        self.do_tick(self.blockers_list)

        # Collect data
        if (self.steps % self.collect_every_n) == 0:
            self.datacollector.collect(self)

        # end of step house keeping            
        self.max_steps_check()
        self.max_persons_check()

    def run_model(self):
        for _ in tqdm(range(self.max_steps), desc="Simulating", unit="step"):
            if not getattr(self, "running", True):
                break
            self.step()
   



