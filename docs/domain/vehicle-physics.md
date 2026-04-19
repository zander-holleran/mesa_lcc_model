# Vehicle Physics

Vehicle behavior in the simulation is governed by individualized acceleration curves, multiple braking mechanisms, and a two-phase speed-then-move execution model.

**Related:** [Speed](speed.md) | [Implicit Speed Limit](implicit-speed-limit.md) | [Cumulative Time Lost](cumulative-time-lost.md) | [Start Speed Ramp](start-speed-ramp.md)

---

## Acceleration Curves

> **Full article:** [Acceleration Curves](acceleration-curves.md)

Each vehicle gets a unique acceleration function built by `build_empirical_accel_function()` in `traffic/agents/vehicle_agent.py`. The function maps current speed (mph) to acceleration (m/s^2).

The curves are derived from **real-world stop-sign acceleration data**, modeled as 4 piecewise segments with skew-normal distributions:

| Segment | Speed range | Typical acceleration |
|---------|-------------|---------------------|
| 0--2 sec | 0 to ~5 mph | Moderate (initial pull-away) |
| 2--4 sec | ~5 to ~15 mph | Highest (main acceleration) |
| 4--6 sec | ~15 to ~25 mph | Moderate (sustaining) |
| 6--8 sec | ~25 to ~35+ mph | Lower (approaching cruising) |

A vehicle's `performance` percentile (0--1) determines where it falls within each segment's distribution. The model pre-computes **101 acceleration curves** (one per integer percentile 0--100) in `accel_curve_cache` at initialization.

Beyond the last segment boundary, acceleration tapers linearly toward zero as the vehicle approaches higher speeds.

---

## Braking Types

The simulation implements three distinct braking mechanisms, checked in priority order within `adjust_speed()`:

### 1. Gap-Based Braking (Smooth Brake)

> **Full article:** [Gap Braking](gap-braking.md)

Triggered when `gap < ideal_gap`. The braking force is computed by `less_smooth_brake()`:

```
force = max((ideal_gap - gap) / ideal_gap, 0)
base = force ** 2          # squared to overreact when very close
noise = rng.normal(0, 0.1) # human-like variation
deceleration = clip(base + noise, 0, 1) * 8  # max 8 m/s² decel
```

The `ideal_gap` is `max(speed * ideal_distance_multiplier, 5)` meters -- it scales with speed and the driver's personal following-distance preference.

### 2. Speed-Limit Braking

> **Full article:** [Speed-Limit Braking](speed-limit-braking.md)

Triggered when the vehicle exceeds its **implicit speed limit** (`posted_speed_limit + acceptable_over`). Deceleration is tiered:

| Over by | Decel (m/s²) |
|---------|-------------|
| > 7 mph | 1.1 |
| > 2 mph | 0.5 |
| > 0 mph | 0.2 |

### 3. Curve-Based Braking

> **Full article:** [Curve Braking](curve-braking.md)

Built into the speed limit calculation via `get_speed_limit()`. The implicit speed limit is reduced by curvature:

```
curve_effect = clip(avg_curvature / 90, 0, 1)
speed_effect = clip((avg_sl - 10) / (60 - 10), 0, 1)
curve_speed_limit = avg_sl * (1 - curve_responce * curve_effect * speed_effect)
implicit_speed_limit = curve_speed_limit + acceptable_over
```

The look-ahead averages the speed limit and curvature of the next `N_ahead=5` segments with distance-decaying weights, so drivers begin slowing before reaching the curve.

---

## Prevent-Pass Mechanism

After all other speed calculations, a final check prevents vehicles from overtaking:

```python
if next_agent is not None and (speed - next_agent.speed) > gap:
    driving_action = 'prevent_pass'
    speed = next_agent.speed - 1
```

This ensures no vehicle passes through the vehicle ahead in a single step, even if the gap calculation alone wouldn't have slowed it enough. It sets `break_cooldown = 5`, which suppresses full acceleration for the next several steps.

---

## Close-Spawn Handling

> **Full article:** [Close-Spawn Handling](close-spawn-handling.md)

When a new vehicle spawns, `too_close()` in `traffic/model/generate.py` checks whether the most recent vehicle is within 1 meter of the start point. If so, the start point is shifted north by 1 meter:

```python
if distance(last_vehicle.pos, start_point) < 1:
    start_point = (start_point[0], start_point[1] + 1)
    too_close_counter += 1
```

This prevents two vehicles from occupying the exact same position at spawn. The vehicle's initial `distance_traveled` is set to the negative of the distance from the original start point, so distance bookkeeping remains consistent.

---

## Driver Decision-Making

Each step, `adjust_speed()` evaluates conditions in priority order and assigns a `driving_action`:

| Priority | Condition | Action | Effect |
|----------|-----------|--------|--------|
| 1 | `gap < ideal_gap` | `smooth_break` | Gap-based deceleration |
| 2 | `speed > implicit_sl` | `speed_limit_break` | Tiered deceleration |
| 3 | `break_cooldown >= 4` | `coast` | No speed change |
| 4 | `break_cooldown >= 1` | `slow_accelerate` | Partial acceleration |
| 5 | None of the above | `accelerate` | Full acceleration |
| Override | Would pass next agent | `prevent_pass` | Match next agent's speed |

The `break_cooldown` counter (set to 3--5 after braking) creates a realistic delay before full acceleration resumes -- drivers don't instantly floor it after braking.

---

## Two-Step Speed-Then-Move Process

Vehicle physics executes in **two separated phases** each step:

1. **Adjust speed**: All vehicles compute their new speed simultaneously based on the state at the start of the step
2. **Move**: All vehicles advance along the road by their computed speed

This separation is critical. If speed and movement were interleaved (compute-then-move per vehicle), earlier vehicles in the loop would move first and change the gaps that later vehicles see. The two-phase approach ensures all vehicles make decisions based on the **same snapshot** of the world, preventing order-dependent race conditions.

In `TrafficModel.step()`:

```python
self.do_adjust_speed(driving_vehicles)   # phase 1: all compute speed
self.do_move_along_path(driving_vehicles) # phase 2: all move
```

---

## Vehicle Removal / End-of-Road

When `move_along_path()` advances a vehicle past the final road segment, `end_of_road()` is called:

1. **Log trip data**: Appends a summary dict to `model.finished_agents` with agent ID, type, steps taken, distance, average speed, driver traits, toll paid, and cumulative delay
2. **Hand off to passengers**: Calls `vehicle_to_tp_info_pass()` (see below)
3. **Set status**: `status = "arrived"`
4. **Cleanup**: Removes the vehicle from the road segment's `vehicles_here` list, from `model.space`, from `model.vehicles_list`, and from the Mesa agent set

---

## Vehicle-to-Person Handoff at Arrival

When a vehicle reaches the end of the road, `vehicle_to_tp_info_pass()` transfers trip data to each passenger (`TrafficPersonAgent`):

- `toll_paid` -- the vehicle's toll (for cars) or the `bus_user_fee` (for buses, overridden in `BusAgent`)
- `board_step` -- step when the person boarded (vehicle creation step for cars)
- `arrive_step` -- current model step
- `cumtime_lost_sec` -- cumulative delay from the vehicle

Each passenger then calls `tp_to_sp_info_pass()`, which computes derived metrics (`wait_time`, `onboard_time`, `total_travel_time`, `realized_cost`) and forwards the complete trip record to `SeasonPerson.record_experience()` for belief updating.

---

## Driver Parameters

All driver parameters are drawn from model-level truncated normal distributions at vehicle creation:

| Parameter | Car default | Bus value | Distribution |
|-----------|-------------|-----------|-------------|
| `acceptable_over` | ~3 mph mean | 0 | `truncnorm(15, -2, 4, mean=3)` |
| `ideal_distance_multiplier` | ~1.0 mean | 2.0 | `truncnorm(2, 0.6, 0.3, mean=1)` |
| `curve_responce` | ~0.95 mean | 0.9 | `truncnorm(0.95, 0.6, 0.1)` |
| `performance` | uniform 0--1 | 0.1 | `rng.random()` |
