import random


def too_close(model): 
    vehicles = model.agents.select(agent_type=model.agent_cls['vehicle'])

    closeness_threshold = 1 
    if vehicles and model.space.get_distance(vehicles[-1].pos, model.start_point) < closeness_threshold: # check if the last vehicle is withen x m of the start point
        model.start_point = (model.start_point[0], model.start_point[1]+1) # if the car is too close move the start point north
        model.too_close_counter += 1

def generate_new_bus(model):
        # 1. Early exit conditions
        if model.bus_interval == 0: 
            return 
    
        if model.person_counter >= model.max_persons and not model.at_bus_stop:  
            return
                
        if model.steps < model.next_bus_step:
            return
 
        # 2. Generate the bus
        too_close(model)
        new_bus = model.agent_cls['bus'].create_agents(model=model, n=1)[0]
        model.agents.add(new_bus)
        model.vehicles_list.append(new_bus)

        # FCFS boarding from the single queue
        capacity = model.bus_capacity
        n_board = min(capacity, len(model.at_bus_stop))
        boarding_passengers = model.at_bus_stop[:n_board]
        model.at_bus_stop = model.at_bus_stop[n_board:]     # remove them from the queue

        # link each passenger <-> bus
        for tp in boarding_passengers:
            tp.board_step = model.steps
            tp.vehicle = new_bus
            new_bus.passengers.append(tp)

        # increment counters
        model.bus_counter += 1
        model.bus_riders += n_board
        model.next_bus_step += model.bus_interval * 60

def pick_season_person_for_trip(model):
    """
    Return the next SeasonPerson in the precomputed draw order,
    or None if everyone has been used this day.
    """
    if not model.season_person_pool:
        return None

    pick = random.choices(model.season_person_pool, k=1)[0]
    model.season_person_pool.remove(pick)

    return pick


def generate_person(model):
    """
    Generate a person + vehicle or an empty car, depending on:
    - per-step Bernoulli arrival (p_generate)
    - person capacity (max_persons)
    - whether there are bus riders waiting at the stop
    """

    # 0. Bernoulli check for a potential arrival this step
    if model.random.random() >= model.p_generate:
        return
    
    season_person = pick_season_person_for_trip(model)
    tp_list = model.traffic_persons_list
    any_unfinished = len(tp_list) > 0
    
    if season_person is None and not any_unfinished:
        return

    if season_person is None and any_unfinished: # out of new persons to generate BUT we still have persons in the system
        # keep generating empty cars to maintain traffic
        too_close(model) 
        new_car = model.agent_cls['car'].create_agents(
            model=model,
            n=1,
        )[0]
        model.agents.add(new_car)
        model.vehicles_list.append(new_car)
        # empty car: no passengers to link
        model.car_counter += 1
        return  


    # 2. Normal case: there are persons left to generate
    new_tp = model.agent_cls['traffic_person'].create_agents(
        model=model,
        n=1,
        season_person=season_person,
    )[0]
  
    model.agents.add(new_tp)
    tp_list.append(new_tp)
    model.person_counter += 1
    new_tp.created_step = model.steps
    
    if new_tp.mode == "car":
        too_close(model)

        new_car = model.agent_cls['car'].create_agents(
            model=model,
            n=1,
        )[0]
        model.agents.add(new_car)
        model.vehicles_list.append(new_car)

        # double link: person <-> car
        new_tp.vehicle = new_car
        new_car.passengers.append(new_tp)
        # incroment tickers
        model.car_counter += 1

    else:
        # BUS CASE: add person to global bus-stop queue
        model.at_bus_stop.append(new_tp)
        

def generate_blocker(model, blocker_type, self_distruct_timer, road_segment):
        print(f"{blocker_type} at step {model.steps}")
        new_blocker = model.agent_cls['blocker'].create_agents(model=model, n=1, blocker_type=blocker_type, self_distruct_timer=self_distruct_timer, road_segment=road_segment)
        model.agents.add(*new_blocker)
        model.blockers_list.append(new_blocker[0])

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
    #else:
        #print(f"No occupied segments available for crash at step {model.steps}.")

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