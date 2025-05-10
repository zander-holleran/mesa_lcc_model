from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

# Import my agents
from agents.vehicle_agent import VehicleAgent
from agents.bus_agent import BusAgent
from agents.car_agent import CarAgent
from agents.road_segment_agent import RoadSegmentAgent

# import my utils
from utils import unit_conversion_utils as uc  # for get_mph, etc.

# import other parts of model
import model.reporting as rep

import geopandas as gpd
import numpy as np


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
            curvature = [curvature for curvature in self.road_points_gdf.curvature],
            linked_coord=[linked_coord for linked_coord in self.road_points_gdf.linked_coord]
        )
        # place all the road segments in space - goes hand in hand with read point reation 
        for agent, point in zip(self.road_segments, self.road_points_gdf.geometry):self.space.place_agent(agent, (point.x, point.y))
        
        if not self.batchrun: # this triggers if batch run is set to false
            self.datacollector = DataCollector(
                model_reporters = rep.model_reporters
                , agent_reporters = rep.agent_reporters
            )

    
    def generate_new_bus(self):
        """
        Generate a new bus:
        - First bus is generated at a random step (0–15 mins).
        - Then follow a fixed interval based on bus_interval (in minutes).
        - Never exceed max_buses on the road.
        """
        if self.person_counter >= self.max_persons:
            return

        if self.bus_interval == 0:
            return 
        current_step = self.steps
        steps_per_interval = self.bus_interval * 60
    
        # First departure check
        if not self.bus_generation_started:
            if current_step >= self.bus_first_departure:
                self.bus_generation_started = True
            else:
                return  # Still waiting for the randomized first departure
    
        # After the first departure
        if (current_step - self.bus_first_departure) % steps_per_interval == 0:    
            new_bus = BusAgent.create_agents(model=self, n=1, road_points_gdf=self.road_points_gdf)
            self.agents.add(*new_bus)
            #self.agents.add(new_bus)  # Use .add(), not .add_agents() or model-level methods
            self.bus_counter += 1
            self.person_counter += self.at_bus_stop
            self.bus_riders += self.at_bus_stop 
            self.at_bus_stop = 0 


    def generate_person(self):
        if self.person_counter >= self.max_persons:
            return

    
        vehicles = self.agents.select(agent_type=VehicleAgent)
        if self.person_counter > 0:
            if not vehicles:
                return  # No cars yet to check
            if self.space.get_distance(vehicles[-1].pos, self.start_point) < 1:
                self.too_close_counter += 1
                return
    
        if self.random.random() < self.p_generate:
            if (self.random.random() < self.car_preference) or (self.at_bus_stop >= self.bus_capacity):
                new_car = CarAgent.create_agents(model=self, n=1, road_points_gdf=self.road_points_gdf)
                self.agents.add(*new_car) 
                self.person_counter += 1
            else:
                self.at_bus_stop+=1 
   
    def model_stop_process(self):
        # add agent summary data to the datacollector
        self.datacollector.model_vars["FinishedAgentsSummary"][-1] = self.finished_agents
        self.running = False
        
    def step(self):
        # generate bus 
        self.generate_person()
        # generate a new car based on a simple probability 
        self.generate_new_bus()
        
        # Shuffle agent execution and step them - this calls the step functions of the agents
        self.agents.do("adjust_speed")
        self.agents.do("move_along_path")

        # Collect data before stepping
        if not self.batchrun:
            self.datacollector.collect(self)

        # Stop model when all generated cars have been removed
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
             
