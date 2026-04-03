_MPH_TO_MPS = 1609.34 / 3600  # consistent with unit_conversion_utils.get_mps


def too_close(model):
    if not model.vehicles_list:
        return
    v = model.vehicles_list[-1]
    vs = model.vs
    slot = vs.vid_to_slot.get(v.unique_id)
    if slot is None:
        return
    vx, vy = float(vs.pos_x[slot]), float(vs.pos_y[slot])
    sx, sy = model.start_point
    dist = ((vx - sx)**2 + (vy - sy)**2)**0.5
    closeness_threshold = 1
    if dist < closeness_threshold:
        model.start_point = (model.start_point[0], model.start_point[1]+1)
        model.too_close_counter += 1

def _spawn_offset(model) -> float:
    """Negative distance from initial_start_point to current start_point.

    Mirrors old VehicleAgent behaviour where distance_traveled started
    negative so that vehicles spawned behind the road origin don't
    share dist=0 and deadlock in the sorted-rank gap computation.
    """
    ix, iy = model.initial_start_point
    sx, sy = model.start_point
    return -((ix - sx)**2 + (iy - sy)**2)**0.5

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

        # Register in VehicleStore
        model.vs.add(
            vid=new_bus.unique_id,
            veh_type=1,
            pos_x=float(model.start_point[0]),
            pos_y=float(model.start_point[1]),
            speed=float(model.rs_speed_limit[0] * _MPH_TO_MPS),
            route_end_dist=float(model.rs_cumulative_s[-1]),
            acceptable_ov=float(new_bus.acceptable_over),
            ideal_dm=float(new_bus.ideal_distance_multiplier),
            curve_resp=float(new_bus.curve_responce),
            performance=float(new_bus.performance),
            created_step=int(model.steps),
            toll_paid=float(new_bus.toll_paid),
            init_dist=_spawn_offset(model),
        )
        model.vid_to_vehicle[new_bus.unique_id] = new_bus

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

    pick = model.rng.choice(model.season_person_pool)
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

    if season_person is None and any_unfinished:
        # out of new persons to generate BUT we still have persons in the system
        # Scale by the observed car fraction so we don't inflate car generation
        # relative to normal state (where only car-choosing persons generate vehicles).
        # Freeze car_fraction at the moment we enter exception state so that
        # empty-car generation doesn't inflate the ratio over time.
        if model._exception_car_fraction is None:
            model._exception_car_fraction = model.car_counter / model.person_counter
        if model.random.random() >= model._exception_car_fraction:
            return
        # keep generating empty cars to maintain traffic
        too_close(model)
        new_car = model.agent_cls['car'].create_agents(
            model=model,
            n=1,
        )[0]
        model.agents.add(new_car)
        model.vehicles_list.append(new_car)

        # Register in VehicleStore
        model.vs.add(
            vid=new_car.unique_id,
            veh_type=0,
            pos_x=float(model.start_point[0]),
            pos_y=float(model.start_point[1]),
            speed=float(model.rs_speed_limit[0] * _MPH_TO_MPS),
            route_end_dist=float(model.rs_cumulative_s[-1]),
            acceptable_ov=float(new_car.acceptable_over),
            ideal_dm=float(new_car.ideal_distance_multiplier),
            curve_resp=float(new_car.curve_responce),
            performance=float(new_car.performance),
            created_step=int(model.steps),
            toll_paid=float(new_car.toll_paid),
            init_dist=_spawn_offset(model),
        )
        model.vid_to_vehicle[new_car.unique_id] = new_car

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

        # Register in VehicleStore
        model.vs.add(
            vid=new_car.unique_id,
            veh_type=0,
            pos_x=float(model.start_point[0]),
            pos_y=float(model.start_point[1]),
            speed=float(model.rs_speed_limit[0] * _MPH_TO_MPS),
            route_end_dist=float(model.rs_cumulative_s[-1]),
            acceptable_ov=float(new_car.acceptable_over),
            ideal_dm=float(new_car.ideal_distance_multiplier),
            curve_resp=float(new_car.curve_responce),
            performance=float(new_car.performance),
            created_step=int(model.steps),
            toll_paid=float(new_car.toll_paid),
            init_dist=_spawn_offset(model),
        )
        model.vid_to_vehicle[new_car.unique_id] = new_car

        # double link: person <-> car
        new_tp.vehicle = new_car
        new_car.passengers.append(new_tp)
        # incroment tickers
        model.car_counter += 1

    else:
        # BUS CASE: add person to global bus-stop queue
        model.at_bus_stop.append(new_tp)
        

def generate_blocker(model, blocker_type, self_distruct_timer, seg_i):
        # Structured event logging
        if blocker_type == "crash":
            model.datacollector.log_crash(model, seg_i, self_distruct_timer)
        elif blocker_type == "canyon_closure":
            model.datacollector.log_canyon_closure(model, seg_i, self_distruct_timer)
        new_blocker = model.agent_cls['blocker'].create_agents(model=model, n=1, blocker_type=blocker_type, self_distruct_timer=self_distruct_timer, seg_i=seg_i)
        model.agents.add(*new_blocker)
        model.blockers_list.append(new_blocker[0])

def pick_occupied_segment(model):
    if not model.vehicles_list:
        return None
    v = model.random.choice(model.vehicles_list)
    slot = model.vs.vid_to_slot.get(v.unique_id)
    if slot is None:
        return None
    return int(model.vs.path_idx[slot])

def generate_crash(model): 
    if model.crashes == 0:
        return                     

    seg_i = pick_occupied_segment(model)
    if seg_i:
        generate_blocker(model=model, blocker_type="crash", self_distruct_timer=model.random.randint(60, 300), seg_i=seg_i) # this is where blocker duration is set, currently between 1 and 5 mins

def generate_canyon_closure(model):
    if len(model.canyon_closures) == 0:
        return
    
    closure = model.canyon_closures.iloc[0]
    if closure.closure_step <= model.steps:
            generate_blocker(
                model=model,
                blocker_type="canyon_closure",
                self_distruct_timer=closure.duration,
                seg_i=closure.road_section
            )

            if not model.canyon_closures.empty:
                model.canyon_closures = model.canyon_closures.iloc[1:].reset_index(drop=True)