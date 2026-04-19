# Cumulative Time Lost

**Cumulative time lost** measures the total real-world delay a vehicle experiences over its trip due to congestion -- time spent traveling slower than its [implicit speed limit](implicit-speed-limit.md). The final value is in **seconds** of actual delay.

**Source:** `traffic/model/vehicle_kernel.py` -- section 9 (time lost)

**Related:** [Speed](speed.md) | [Implicit Speed Limit](implicit-speed-limit.md) | [Vehicle Physics](vehicle-physics.md)

---

## Definition

Each simulation step (1 second), the kernel computes:

```python
time_lost = max((implicit_sl - speed) / implicit_sl, 0)
cumtime_lost += time_lost
```

If the vehicle is at or above its implicit speed limit, `time_lost = 0`. If below, the fractional deficit is accumulated.

---

## Why This Measures Real Delay

The per-step formula looks like a dimensionless ratio, not seconds. It seems like it would overweight slow zones. Here is why it actually produces real seconds of delay at 1-second resolution.

### The logic

In a 1-second step, the vehicle covers `speed * 1s` meters of distance. At the speed limit, that same distance would have taken:

```
ideal_time = (speed * 1s) / implicit_sl
```

The delay for that step is:

```
delay = 1s - ideal_time
     = 1 - speed / implicit_sl
     = (implicit_sl - speed) / implicit_sl     <-- the formula in the code
```

Each step contributes real seconds of delay. Summed over the trip, `cumtime_lost` is total seconds lost.

### Worked example

A vehicle travels 1 mile through a 60 mph zone at 30 mph:

```
Per step:   time_lost = (60 - 30) / 60 = 0.5 seconds of delay

Trip duration at 30 mph:  1 mile / 30 mph = 120 seconds (120 steps)
Trip duration at 60 mph:  1 mile / 60 mph = 60 seconds

Accumulated:  120 steps * 0.5 = 60 seconds
Actual delay: 120 - 60       = 60 seconds   <-- matches
```

### The "5 under" concern

Going 5 mph under the limit produces a larger per-step value in slow zones:

| Zone | Per-step `time_lost` |
|------|---------------------|
| 30 mph zone, going 25 | 5/30 = **0.167** |
| 60 mph zone, going 55 | 5/60 = **0.083** |

This looks biased. But slower vehicles spend **more steps** covering the same distance -- and the two effects cancel exactly:

| | 30 zone, going 25 | 60 zone, going 55 |
|---|---|---|
| Per-step delay | 0.167s | 0.083s |
| Steps for 1 mile | 144 | 65 |
| **Total delay** | **24.0 seconds** | **5.4 seconds** |

Cross-check with the exact formula `distance * (1/actual - 1/limit)`:

- 30 zone: `1mi * (1/25 - 1/30) = 24.0 sec/mile`
- 60 zone: `1mi * (1/55 - 1/60) = 5.5 sec/mile`

The results match. Going 5 under in a 30 zone genuinely costs more real time per mile than in a 60 zone -- the metric is correct, not biased.

---

## Downstream Usage

At end-of-trip, `cumtime_lost` (in seconds) is transferred to the person agent and converted to minutes:

```python
cumtime_lost_min = self.cumtime_lost_sec / 60
```

This flows into day summaries as:

| Metric | Meaning |
|--------|---------|
| `avg_cumlost_bus` | Mean minutes of congestion delay for bus riders that day |
| `avg_cumlost_car` | Mean minutes of congestion delay for car drivers that day |

---

## What It Does NOT Capture

- **Waiting time** (bus riders waiting at the stop) -- tracked separately as `wait_time`
- **Free-flow travel time** -- a trip at the speed limit has `cumtime_lost = 0`, but still takes time
- **Toll cost** -- monetary, not temporal; combined with travel time in `realized_cost`
