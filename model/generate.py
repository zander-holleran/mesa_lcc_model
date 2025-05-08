from agents.vehicle_agent import VehicleAgent
from agents.bus_agent     import BusAgent
from agents.car_agent     import CarAgent


# my generate functions
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
            BusAgent.create_agents(model=self, n=1, road_points_gdf=self.road_points_gdf)
            #self.agents.add(new_bus)  # Use .add(), not .add_agents() or model-level methods
            self.bus_counter += 1
            self.person_counter += self.at_bus_stop
            self.bus_riders += self.at_bus_stop 
            self.at_bus_stop = 0 


def generate_person(self):
    # Only generate if under max limit
    if self.person_counter >= self.max_persons:
        return

    # vehicles = self.agents.select(agent_type=VehicleAgent)
    # if self.person_counter > 0:
    #     if not vehicles:
    #         return  # No cars yet to check
    #     if self.space.get_distance(vehicles[-1].pos, self.start_point) < 1:
    #         self.too_close_counter += 1
    #         return

    if self.random.random() < self.p_generate:
        if (self.random.random() < self.car_preference) or (self.at_bus_stop >= self.bus_capacity):
            CarAgent.create_agents(model=self, n=1, road_points_gdf=self.road_points_gdf)
            #self.agents.add(new_car) 
            self.person_counter += 1
        else:
            self.at_bus_stop+=1 