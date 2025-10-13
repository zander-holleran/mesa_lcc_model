
def too_close(model): 
    vehicles = model.agents.select(agent_type=model.agent_cls['vehicle']
)
    closeness_threshold = 1 
    if vehicles and model.space.get_distance(vehicles[-1].pos, model.start_point) < closeness_threshold: # check if the last vehicle is withen x m of the start point
        model.start_point = (model.start_point[0], model.start_point[1]+1) # if the car is too close move the start point north
        model.too_close_counter += 1

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
            too_close(model)
            new_bus = model.agent_cls['bus'].create_agents(model=model, n=1)
            model.agents.add(*new_bus)
            model.bus_counter += 1
            model.person_counter += model.at_bus_stop
            model.bus_riders += model.at_bus_stop 
            model.at_bus_stop = 0 

def generate_person(model):
        if model.person_counter >= model.max_persons: # dont generate people if you hit the limit
            return             
        if model.random.random() < model.p_generate:
            too_close(model)
            # congrats you made a person
            # now determine if that person goes to the bus stop or in a car
            if (model.random.random() < model.car_preference) or (model.at_bus_stop >= model.bus_capacity):
                new_car = model.agent_cls['car'].create_agents(model=model, n=1)
                model.agents.add(*new_car) 
                # this person got in a car
                model.person_counter += 1
                model.car_counter += 1 
            else:
                # if the person ends up going to the bus stop they will not be counted until the bus leaves
                model.at_bus_stop+=1 


def generate_blocker(model, blocker_type, self_distruct_timer, road_segment):
        print(f"{blocker_type} at step {model.steps} on segment {road_segment.pos}")
        new_blocker = model.agent_cls['blocker'].create_agents(model=model, n=1, blocker_type=blocker_type, self_distruct_timer=self_distruct_timer, road_segment=road_segment)
        model.agents.add(*new_blocker)

def generate_crash(model): 
    if model.crashes == 0:
        # no crashes to generate end the function
        return                     

    # Step 1: filter to road segments that have a non-empty `vehicles_here`
    occupied_segments = model.agents.select(
            lambda s: hasattr(s, "vehicles_here") and len(getattr(s, "vehicles_here", [])) > 0,
            agent_type=model.agent_cls['road'],
        )

    # Step 2: pick one at random (or None if none exist)
    segment = model.random.choice(list(occupied_segments)) if len(occupied_segments) else None
    
    if segment:
        generate_blocker(model=model, blocker_type="crash", self_distruct_timer=model.random.randint(60, 300), road_segment=segment) # this is where blocker duration is set, currently between 1 and 5 mins
    else:
        print(f"No occupied segments available for crash at step {model.steps}.")

def generate_canyon_closure(model):
    if len(model.canyon_closures) == 0:
        return
    
    closure = model.canyon_closures.iloc[0]
    if closure.closure_step == model.steps:
        # Find the road segment agent with the matching unique_id
        segment = next(
            (agent for agent in model.agents.select(agent_type=model.agent_cls['road'])
             if agent.unique_id == closure.road_section),
            None
        )

        if segment is not None:
            generate_blocker(
                model=model,
                blocker_type="canyon_closure",
                self_distruct_timer=closure.duration,
                road_segment=segment
            )

            if not model.canyon_closures.empty:
                model.canyon_closures = model.canyon_closures.iloc[1:].reset_index(drop=True)
        

