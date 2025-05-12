from agents.vehicle_agent import VehicleAgent
from agents.bus_agent     import BusAgent
from agents.car_agent     import CarAgent

# my generate functions
def generate_new_bus(model):
        """
        Generate a new bus:
        - First bus is generated at a random step (0–15 mins).
        - Then follow a fixed interval based on bus_interval (in minutes).
       # - Never exceed max_buses on the road.
        """
        if model.bus_interval == 0: 
            return 
    
        if model.person_counter >= model.max_persons:  
            return
 
        current_step = model.steps
        steps_per_interval = model.bus_interval * 60
    
        # First departure check
        if current_step < model.bus_first_departure:
            return  # Still waiting for the randomized first departure dont send a bus 
        elif (current_step - model.bus_first_departure) % steps_per_interval == 0:  # this triggers after first departure and only on the bus interval step
            new_bus = BusAgent.create_agents(model=model, n=1, road_points_gdf=model.road_points_gdf)
            model.agents.add(*new_bus)
            model.bus_counter += 1
            model.person_counter += model.at_bus_stop
            model.bus_riders += model.at_bus_stop 
            model.at_bus_stop = 0 


def generate_person(model):
        if model.person_counter >= model.max_persons: # dont generate people if you hit the limit
            return 
    
        vehicles = model.agents.select(agent_type=VehicleAgent)
        if vehicles and model.space.get_distance(vehicles[-1].pos, model.start_point) < 1: # check if there exists other vehicles and the last one is withen a meter of the start
            model.too_close_counter += 1 
            return # the last vehicle is too close pass on generation for now
    
        if model.random.random() < model.p_generate:
            # congrats you made a person
            # now determine if that person goes to the bus stop or in a car
            if (model.random.random() < model.car_preference) or (model.at_bus_stop >= model.bus_capacity):
                new_car = CarAgent.create_agents(model=model, n=1, road_points_gdf=model.road_points_gdf)
                model.agents.add(*new_car) 
                # this person got in a car
                model.person_counter += 1
                model.car_counter += 1 
            else:
                # if the person ends up going to the bus stop they will not be counted until the bus leaves
                model.at_bus_stop+=1 


                        
