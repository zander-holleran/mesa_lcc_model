
# -----------------------
# transforms (pure numeric)
# -----------------------

def piecewise_linear_tx(signal, min_threshold, slope, base):
    if signal <= min_threshold:
        return 0.0
    return base + slope * (signal - min_threshold)

# -----------------------
# signals (touch the model)
# -----------------------

def get_volume_signal(model, **kwargs):
    return len(model.vehicles_list)


def get_flow_signal(model, update_toll_every_n):
    """
    cars spawned successfully per step since last toll update.
    returns None if it's not time to update yet.
    """
    step = model.schedule.steps
    car_counter = model.car_counter

    if not hasattr(model, "last_toll_update_step"):
        model.last_toll_update_step = step
        model.last_toll_update_car_counter = car_counter
        return None

    elapsed = step - model.last_toll_update_step
    if elapsed != update_toll_every_n:
        return None

    spawned = car_counter - model.last_toll_update_car_counter

    model.last_toll_update_step = step
    model.last_toll_update_car_counter = car_counter

    return spawned / elapsed

# -----------------------
# mechanism specs (wire signal -> tx + params)
# -----------------------

MECHANISMS = {
    "static": None,

    "volume": dict(
        signal=get_volume_signal,
        signal_args=lambda p: dict(),   
        tx=piecewise_linear_tx,
        tx_args=lambda p: dict(
            min_threshold=p.get("min_threshold", 100),
            slope=p.get("slope", 0.05),
            base=p.get("base_price", 5.0),
        ),
    ),

    "flow": dict(
        signal=get_flow_signal,
        signal_args=lambda p: dict(    
            update_toll_every_n=p.get("update_toll_every_n", 1)
        ),
        tx=piecewise_linear_tx,
        tx_args=lambda p: dict(
            min_threshold=p.get("min_threshold", 1.0),
            slope=p.get("slope", 10.0),
            base=p.get("base_price", 0.0),
        ),
    ),
}

def update_tolls(model):
    mech = model.toll_mechanism
    spec = MECHANISMS.get(mech)

    if spec is None:
        return model.current_toll_car

    params = model.toll_params

    signal = spec["signal"](model, **spec["signal_args"](params))
    if signal is None:
        return model.current_toll_car

    car_toll = spec["tx"](signal, **spec["tx_args"](params))
    model.current_toll_car = car_toll
    return car_toll


   


