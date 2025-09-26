# genreal imports
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
import geopandas as gpd
import numpy as np
from tqdm import tqdm

# Import my agents
from agents.vehicle_agent import VehicleAgent
from agents.bus_agent import BusAgent
from agents.car_agent import CarAgent
from agents.road_segment_agent import RoadSegmentAgent

# import my utils
from utils import unit_conversion_utils as uc  # for get_mph, etc.

# import other parts of model
import model.reporting as rep
import model.generate as gen 
import model.init_helpers as ih 


#from model.reporting import agent_reporters, model_reporters
#from model.generate import generate_new_bus, generate_person

class TrafficModel(Model):
    """Mesa model simulating traffic on the canyon road with a car cap."""

    def __init__(self, road_gdf, ecs_df, max_steps=50000, seed=123, batchrun=False, collect_every_n=1, 
                 start_hr=5, traffic_percentile=None, p_generate=None, max_persons=50,
                 canyon_open_hr=5, 
                 bus_interval=30, car_preference=1, bus_capacity=30, 
                 crashes_per_100k_vmt_input=22
                 ):
        super().__init__(seed=seed)
        #model perams
        self.road_points_gdf = road_gdf
        self.expected_counts_seconds = ecs_df
        self.max_steps = max_steps
        self.batchrun = batchrun
        self.collect_every_n = collect_every_n
        self.initial_start_point = road_gdf.iloc[0].geometry.coords[0] # this one will go unchanged through out the model run 
        self.start_point = road_gdf.iloc[0].geometry.coords[0] # this one might change depending on if too_close is triggered

        
        # car centric perams
        self.start_step = uc.sec_after_five(start_hr)
        self.traffic_percentile = traffic_percentile
        self.p_generate = p_generate  # Probability of new car each step
        self.max_persons = max_persons  # Maximum number of persons allowed

        # canyon open peram
        self.canyon_open_step = max(uc.sec_after_five(canyon_open_hr) - uc.sec_after_five(start_hr), 1 ) # make sure the canyon is at least opened on step 0 
        self.canyon_closed_section = [2] 
        
        # bus centric perams
        self.bus_interval = bus_interval
        if self.bus_interval == 0: 
            self.car_preference = 1 
        else: 
            self.car_preference = car_preference
        self.bus_capacity = bus_capacity
        self.bus_first_departure = self.random.randint(0, 5 * 60)  # Random step between 0 and 5 mins
        
        # crash perams
        self.crashes_per_100k_vmt = crashes_per_100k_vmt_input
        self.remainder = 0 
        self.crashes = 0
        self.total_crashes = 0

        # verious trackers
        self.too_close_counter = 0 
        self.person_counter = 0 
        self.bus_counter = 0 
        self.car_counter = 0 
        self.bus_riders = 0 
        self.at_bus_stop = 0 
        self.finished_agents = []
        self.active_vehicles = [] # thus will be used for the crash model
        
        # Set up ContinuousSpace
        buffer = 1000
        minx, miny, maxx, maxy = road_gdf.total_bounds
        self.space = ContinuousSpace(x_min=minx - buffer, x_max=maxx + buffer, y_min=miny - buffer, y_max=maxy + buffer, torus=False)

        # set up road segments with a helper
        self.road_segments = ih.init_road_segments(
            model=self,
            road_gdf=self.road_points_gdf,
            canyon_open_step=self.canyon_open_step,
            closed_sections=set(self.canyon_closed_section)
        )

        # place the roadsegments on the space
        for agent, point in zip(self.road_segments, road_gdf.geometry):
            self.space.place_agent(agent, (point.x, point.y))

        # set up the reporters
        if self.batchrun: 
            self.datacollector = DataCollector(model_reporters = rep.model_reporters)
        else:
            self.datacollector = DataCollector(model_reporters = rep.model_reporters, agent_reporters = rep.agent_reporters)

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ END OF INIT -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    def model_stop_process(self):
        # add agent summary data to the datacollector
        if not self.batchrun:
            self.datacollector.model_vars["FinishedAgentsSummary"][-1] = self.finished_agents
        self.running = False
    
    def max_persons_check(self):
    # Stop model when all generated Vehiclea have been removed
        if self.person_counter == self.max_persons:
            if len(self.active_vehicles) == 0:
                print(f"{self.person_counter} people generated stopping model.")
                self.model_stop_process()
    
    def max_steps_check(self):
        # Stop model at hard cap of steps
        if self.steps >= self.max_steps:
            print(f"Reached max step count ({self.max_steps}). Stopping model.")
            self.model_stop_process()

    def maybe_reopen_canyon(self):
        #if self.canyon_open_step is not None and self.steps == self.canyon_open_step:
        if self.steps == self.canyon_open_step:
            #print(f'canyon open at {self.steps}')
            for agent in self.road_segments:
                if agent.road_section in self.canyon_closed_section:
                    agent.road_closed = False
        
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

    
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~ THE STEP  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    def step(self):
        #clear vehicles here from the roads - necessary for road segment analysis
        for segment in self.road_segments:
            segment.vehicles_here.clear()

        # establish what p_generate is going to be for that step
        if self.traffic_percentile:
            self.p_generate = self.expected_counts_seconds.iloc[self.start_step, self.traffic_percentile]
            self.start_step += 1

        # save the active vehicles
        self.active_vehicles = self.agents.select(agent_type=VehicleAgent)
#         print(f"Step: {self.steps}, Active Vehicles: {len(self.active_vehicles)}")
# +       print([i for i in self.active_vehicles])

        # maybe reopen the canyon 
        self.maybe_reopen_canyon()
        
        # generate functions
        gen.generate_person(self)
        gen.generate_new_bus(self)

        # crash function
        self.crashes, self.remainder = self.should_crash_randomized_rounding(
            crashes_per_100k_vmt=self.crashes_per_100k_vmt,
            num_cars=len(self.active_vehicles),
            avg_speed_mps=np.mean([veh.speed for veh in self.active_vehicles]) if len(self.active_vehicles) > 0 else 1,
            remainder=self.remainder,
            rng=self.random
        )
        self.total_crashes += self.crashes
        
        if self.crashes > 0:
            print(f"Step: {self.steps}, Crashes this step: {self.crashes}, Total Crashes: {self.total_crashes}")
            # # randomly select a vehicle to crash
            # vehicles_to_crash = self.random.sample(self.active_vehicles, min(1, len(self.active_vehicles)))
            # for vehicle in vehicles_to_crash:
            #     print (f"Crashing vehicle ID: {vehicle.unique_id}")
            #     # place holdervehicle.crash() # call the crash method on the vehicle agent
        
        # action functions
        self.agents.do("adjust_speed")
        self.agents.do("move_along_path")

        # Collect data
        if (self.steps % self.collect_every_n) == 0:
            self.datacollector.collect(self)

        # potentially end the model
        self.max_steps_check()
        self.max_persons_check()

    def run_model(self):
        for _ in tqdm(range(self.max_steps), desc="Simulating", unit="step"):
            if not self.running:
                break
        #while self.running:
            self.step()
    
    
   



