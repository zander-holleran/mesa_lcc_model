# genreal imports
from mesa import Model
from mesa.space import ContinuousSpace
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import my agents
from traffic.agents import VehicleAgent, BlockerAgent, BusAgent, CarAgent, RoadSegmentAgent, TrafficPersonAgent
from traffic.agents.vehicle_agent import build_empirical_accel_function
from traffic.model import tolling
from traffic.model.tolling import TollConfig
# import my utils
from traffic.utils import unit_conversion_utils as uc  # for get_mph, etc.
import traffic.utils.distribution_utils as du


# import other parts of model
import traffic.model.generate as gen
import traffic.model.init_helpers as ih
from traffic.model.hybrid_collector import HybridDataCollector, HybridCollectorConfig
from collections import defaultdict


class TrafficModel(Model):
    """Mesa model simulating traffic on the canyon road with a car cap."""

    def __init__(self, road_gdf, ecs_df, max_steps=50000, seed=123,
                 start_hr=5, traffic_percentile=None, p_generate=None, max_persons=50,
                 canyon_closures={},
                 bus_interval=30, car_preference=1, bus_capacity=30,
                 crashes_per_100k_vmt_input=4,
                 toll_config: TollConfig = None,
                 bus_user_fee: float = 0.0,
                 season_persons=None,
                 current_day = 0,
                 hybrid_collector_config: HybridCollectorConfig = None,
                 max_concurrent_vehicles: int = 5000,
                 silent: bool = False,
                 ):
        super().__init__(seed=seed)

        self.silent = silent
        self.created_counts = defaultdict(int) # temp

        self.rng = np.random.default_rng(seed)

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
        self.initial_start_point = tuple(road_gdf.iloc[0].geometry.coords[0])  # never changes
        self.start_point = tuple(road_gdf.iloc[0].geometry.coords[0])          # may be moved by "too_close" logic

        # ===== Tolling perams =====
        self.toll_config = toll_config if toll_config is not None else TollConfig.static(car=0.0)
        self.current_toll_car = self.toll_config.get_initial_toll()
        self.bus_user_fee = bus_user_fee

        # ===== car centric perams =====
        self.start_step = uc.sec_after_five(start_hr)
        self.traffic_percentile = traffic_percentile
        self.p_generate = p_generate  # Probability of new car each step
        self.max_persons = max_persons  # Maximum number of persons allowed
        self.dist_acceptable_over = du.make_truncnorm(15, -2, 4, mean=3)
        self.dist_ideal_distance_multiplier = du.make_truncnorm(2, 0.6, 0.3, mean=1)
        self.dist_curve_response = du.make_truncnorm(0.95, 0.6, 0.1, mean=None)

        # precompute accel curves once (101 buckets)
        self.accel_curve_cache = [build_empirical_accel_function(p / 100) for p in range(101)]

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
        self._p_generate_frozen = False
        self._exception_car_fraction = None
        self.too_close_counter = 0
        self.person_counter = 0 
        self.bus_counter = 0 
        self.car_counter = 0 
        self.bus_riders = 0 
        self.at_bus_stop = []
        self.finished_agents = []
        self.sp_finished_counter = 0
        self.total_vehicle_steps = 0

        # ===== Set up ContinuousSpace =====
        buffer = 10000
        minx, miny, maxx, maxy = road_gdf.total_bounds
        self.space = ContinuousSpace(x_min=minx - buffer, x_max=maxx + buffer, y_min=miny - buffer, y_max=maxy + buffer, torus=False)

        # set up road segments with a helper
        self.road_segments = ih.init_road_segments(
            model=self,
            road_gdf=road_gdf
        )

        for agent, (x, y) in zip(self.road_segments, self.rs_pos):
            self.space.place_agent(agent, (x, y))

        # Persistent vehicle store (array kernel)
        from traffic.model.vehicle_store import VehicleStore
        self.vs = VehicleStore(max_concurrent_vehicles)
        self.max_concurrent_vehicles = max_concurrent_vehicles
        self.vid_to_vehicle: dict[int, object] = {}  # vid → VehicleAgent shell

        # set up the hybrid data collector using the provided config or fall back to defaults
        collector_config = hybrid_collector_config or HybridCollectorConfig(
            max_steps=max_steps,
        )
        self.datacollector = HybridDataCollector(collector_config)

        # agent lists
        self.vehicles_list = []
        self.blockers_list = []
        self.traffic_persons_list = []


    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ END OF INIT -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    def max_persons_check(self):
    # Stop model when entire population has completed a trip 
        if not self.season_person_pool and not self.traffic_persons_list:
            if not self.silent:
                print(f"{self.person_counter} people arrived stopping model.")
            self.running = False
    
    def max_steps_check(self):
        # Stop model at hard cap of steps
        if self.steps >= self.max_steps:
            if not self.silent:
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
    
        meters_this_step = float(num_cars) * float(avg_speed_mps)
        lambda_step = (crashes_per_100k_vmt / METERS_PER_100K_MILES) * meters_this_step

        total = remainder + lambda_step # this continuiously increases until a crash happens then resets 
        base = int(total)
        frac = total - base
        crashes = base + (1 if rng.random() < frac else 0)
        new_remainder = total - crashes  # keep only the fractional part
        return crashes, new_remainder
    
    def update_tolls(self):
        tolling.update_tolls(self)
    
     # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ Agent method loopers  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    # blockers
    def do_tick(self, agents):
            for a in agents:
                a.tick()

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ THE STEP  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    def step(self):
        self.total_vehicle_steps += len(self.vehicles_list)
        # Toll update
        self.update_tolls()

        # establish what p_generate is going to be for that step
        if self.traffic_percentile:
            if self.season_person_pool:
                idx = min(self.start_step, len(self.expected_counts_seconds) - 1)
                self.p_generate = self.expected_counts_seconds.iloc[idx, self.traffic_percentile]
                self.start_step += 1
            elif not self._p_generate_frozen:
                # Clamp at the trailing average so background cars don't escalate
                col = self.expected_counts_seconds.iloc[:, self.traffic_percentile]
                lookback = min(3600, self.start_step)
                self.p_generate = float(col.iloc[self.start_step - lookback:self.start_step].mean())
                self._p_generate_frozen = True
        
        # generate functions
        gen.generate_person(self)
        gen.generate_new_bus(self)

        # Vehicle kernel (replaces update_next_agents, adjust_status/speed, move, time_lost)
        from traffic.model import vehicle_kernel
        arrived = vehicle_kernel.step(self)
        for v in arrived:
            v.end_of_road()

        # Compute driving vehicle stats from store for crash generation
        vs = self.vs
        _n = vs.n_active
        if _n > 0:
            _active_mask = (vs.status[:_n] == 0) | (vs.status[:_n] == 1)
            n_driving = int(_active_mask.sum())
            if n_driving > 0:
                avg_speed_mps = float(vs.speed[:_n][_active_mask].mean())
            else:
                avg_speed_mps = 1.0
        else:
            n_driving = 0
            avg_speed_mps = 1.0

        self.crashes, self.remainder = self.should_crash_randomized_rounding(
            crashes_per_100k_vmt=self.crashes_per_100k_vmt,
            num_cars=n_driving,
            avg_speed_mps=avg_speed_mps,
            remainder=self.remainder,
            rng=self.rng
        )

        self.total_crashes += self.crashes

        # will generate a blocker if crashes > 0
        gen.generate_crash(self)
        gen.generate_canyon_closure(self)
        
        self.do_tick(self.blockers_list)

        # Collect data (interval checking is internal for Tier 2)
        self.datacollector.collect(self)

        # end of step house keeping            
        self.max_steps_check()
        self.max_persons_check()

    def run_model(self):
        for _ in tqdm(range(self.max_steps), desc="Simulating", unit="step", disable=self.silent):
            if not getattr(self, "running", True):
                break
            self.step()
   


