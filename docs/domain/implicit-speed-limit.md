# Implicit Speed Limit

The **implicit speed limit** is the effective maximum speed a particular driver will maintain on a given road segment. It combines the posted speed limit, road curvature, and the driver's personal tendencies into a single value.

**Source:** `traffic/model/vehicle_kernel.py` -- section 6 (speed limit lookup)

**Related:** [Speed](speed.md) | [Cumulative Time Lost](cumulative-time-lost.md) | [Vehicle Physics](vehicle-physics.md)

---

## Definition

```python
# 1. Look ahead at the next 5 segments, weighted by proximity
weights = [1/1, 1/2, 1/3, 1/4, 1/5]  # normalized
avg_sl_mph  = weighted_avg(speed_limits_ahead, weights)   # posted limits, mph
avg_curv    = weighted_avg(curvatures_ahead,   weights)

# 2. Reduce for curvature (driver personality determines how much)
curve_effect = clip(avg_curv / 90, 0, 1)
speed_effect = clip((avg_sl_mph - 10) / (60 - 10), 0, 1)   # no curve penalty below 15 mph
curve_sl_mph = avg_sl_mph * (1 - curve_responce * curve_effect * speed_effect)

# 3. Add driver's personal over-limit tendency
implicit_sl_mph = curve_sl_mph + acceptable_over
```

The implicit speed limit is **per-driver, per-location, per-step** -- it changes as the vehicle moves through different road geometry.

---

## Inputs

| Input | Source | Typical range | Description |
|-------|--------|---------------|-------------|
| `speed_limit` | Road segments (UDOT data) | 15--50 mph | Posted speed limit per segment |
| `curvature` | Road segments (derived from geometry) | 0--90+ degrees | Turn angle at each segment |
| `acceptable_over` | Driver trait | -2 to 15 mph (mean ~3) | How much this driver exceeds limits |
| `curve_responce` | Driver trait | 0.6--1.0 (mean ~0.95) | How aggressively driver slows for curves |

---

## Example

A driver with `acceptable_over = 3 mph` and `curve_responce = 0.95` approaches a curve:

| | Straight road (curv=0) | Moderate curve (curv=45) | Sharp curve (curv=90) |
|---|---|---|---|
| Posted limit | 40 mph | 40 mph | 40 mph |
| Curve reduction | 0 mph | -11.4 mph | -22.8 mph |
| + acceptable_over | +3 mph | +3 mph | +3 mph |
| **Implicit SL** | **43 mph** | **31.6 mph** | **20.2 mph** |

---

## Role in the Model

The implicit speed limit governs two things:

1. **Speed-limit braking** -- when `speed > implicit_sl`, the vehicle decelerates (see [Vehicle Physics](vehicle-physics.md))
2. **[Cumulative time lost](cumulative-time-lost.md)** -- the delay metric compares actual speed against implicit speed limit to measure congestion
