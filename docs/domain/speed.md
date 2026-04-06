# Speed

The **actual speed** of a vehicle at a given simulation step, in meters per second (m/s). This is the core state variable that determines how far a vehicle moves each step.

**Source:** `traffic/model/vehicle_kernel.py` -- section 7 (speed update) and `vehicle_store.py` (`vs.speed`)

**Related:** [Implicit Speed Limit](implicit-speed-limit.md) | [Cumulative Time Lost](cumulative-time-lost.md) | [Vehicle Physics](vehicle-physics.md)

---

## Definition

Speed is updated once per step (1 second) through a priority-based decision:

```python
# Priority 1: too close to leader -> brake
gap_brake_mask = active & (gap < max(speed * idm, 5.0))

# Priority 2: over implicit speed limit -> brake
sl_brake_mask  = active & ~gap_brake_mask & (speed > implicit_sl_mps)

# Priority 3: neither -> accelerate (subject to cooldown)
accel_mask     = active & ~gap_brake_mask & ~sl_brake_mask
```

Speed is clamped to zero at the floor:

```python
speed = max(speed, 0.0)
```

---

## How Speed Becomes Distance

Each step is 1 second. Movement is simply:

```python
new_dist = dist + speed   # meters traveled = meters/sec * 1 sec
```

So `speed` in m/s directly equals meters traveled per step.

---

## Units

Speed is stored internally in **m/s** but most driver-facing logic (speed limits, acceptable-over, acceleration curves) works in **mph**. Conversions:

```python
MPS_TO_MPH = 2.23694
MPH_TO_MPS = 1.0 / MPS_TO_MPH
```

---

## Example

| Scenario | Speed (m/s) | Speed (mph) | Distance per step |
|----------|-------------|-------------|-------------------|
| Cruising at limit on highway | 22.4 | 50 | 22.4 m |
| Slowed by traffic | 8.9 | 20 | 8.9 m |
| Nearly stopped behind crash | 0.4 | ~1 | 0.4 m |
