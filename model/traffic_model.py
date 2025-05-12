# genreal imports
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
import geopandas as gpd
import numpy as np

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


#from model.reporting import agent_reporters, model_reporters
#from model.generate import generate_new_bus, generate_person

class TrafficModel(Model):
    """Mesa model simulating traffic on the canyon road with a car cap."""

    def __init__(self, road_gdf, max_steps=50000, seed=123, batchrun=False,
                 p_generate=.001, max_persons=50,
                 bus_interval=30, car_preference=1, bus_capacity=30):
        super().__init__(seed=seed)
        #model perams
        self.batchrun = batchrun
        self.road_points_gdf = road_gdf
        self.start_point = road_gdf.iloc[0].geometry.coords[0]  
        self.max_steps = max_steps
    
        # car perams
        self.p_generate = p_generate  # Probability of new car each step
        self.max_persons = max_persons  # Maximum number of persons allowed
        
        
        # bus perams
        self.bus_interval = bus_interval
        if self.bus_interval == 0: 
            self.car_preference = 1 
        else: 
            self.car_preference = car_preference
        
        self.bus_capacity = bus_capacity
        self.bus_riders = 0 
        self.at_bus_stop = 0 
        self.bus_first_departure = self.random.randint(0, 5 * 60)  # Random step between 0 and 5 mins
        self.bus_generation_started = False
        
        
        # verious trackers
        self.too_close_counter = 0 
        self.person_counter = 0 
        self.bus_counter = 0 
        self.car_counter = 0 
        self.finished_agents = [] 
            
        # Set up ContinuousSpace
        buffer = .0001
        minx, miny, maxx, maxy = road_gdf.total_bounds
        self.space = ContinuousSpace(x_min=minx - buffer, x_max=maxx + buffer, y_min=miny - buffer, y_max=maxy + buffer, torus=False)

        # Create road segment agents - this just creates them in a loop setting the position via the gdf point
        self.road_segments = RoadSegmentAgent.create_agents( 
            model=self, 
            n=len(self.road_points_gdf), 
            position=[(point.x, point.y) for point in self.road_points_gdf.geometry], # need to be passed as a list
            speed_limit=[speed_limit for speed_limit in self.road_points_gdf.speed_limit],
            road_section = [road_section for road_section in self.road_points_gdf.road_section],
            curvature = [curvature for curvature in self.road_points_gdf.curvature],
            linked_coord=[linked_coord for linked_coord in self.road_points_gdf.linked_coord]
        )
        # place all the road segments in space - goes hand in hand with read point creation 
        for agent, point in zip(self.road_segments, self.road_points_gdf.geometry):self.space.place_agent(agent, (point.x, point.y))
        
        if not self.batchrun: # this triggers if batch run is set to false
            self.datacollector = DataCollector(
                model_reporters = rep.model_reporters
                , agent_reporters = rep.agent_reporters
            )
   
    def model_stop_process(self):
        # add agent summary data to the datacollector
        self.datacollector.model_vars["FinishedAgentsSummary"][-1] = self.finished_agents
        self.running = False
        
    def step(self):
        #clear vehicles here from the roads
        for segment in self.road_segments:
            segment.vehicles_here.clear()
        
        # generate functions
        gen.generate_person(self)
        gen.generate_new_bus(self)
        
        # action functions
        self.agents.do("adjust_speed")
        self.agents.do("move_along_path")

        # Collect data
        if not self.batchrun:
            self.datacollector.collect(self)

        # Stop model when all generated Vehiclea have been removed
        if self.person_counter == self.max_persons:
            remaining_vehicles = self.agents.select(agent_type=VehicleAgent)
            if len(remaining_vehicles) == 0:
                print(f"{self.person_counter} people generated stopping model.")
                self.model_stop_process()
        
        # Stop model at hard cap of steps
        if self.steps >= self.max_steps:
            print(f"Reached max step count ({self.max_steps}). Stopping model.")
            self.model_stop_process()

    def run_model(self):
        while self.running:
            self.step()




        ##
             
