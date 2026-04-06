# Curve-Based Braking

**Curve-based braking** is not a separate braking mechanism -- it works by **reducing the implicit speed limit** based on upcoming road curvature. The [speed-limit brake](speed-limit-braking.md) then enforces that reduced limit. The net effect is that drivers slow down before curves, with the amount of slowing depending on curvature severity, posted speed, and the driver's personal curve sensitivity.

**Source:** `traffic/model/vehicle_kernel.py` -- lines 95--123

**Related:** [Implicit Speed Limit](implicit-speed-limit.md) | [Speed-Limit Braking](speed-limit-braking.md) | [Vehicle Physics](vehicle-physics.md)

---

## How It Works

### 1. Look ahead at upcoming road geometry

Each step, the kernel reads speed limits and curvatures from the next **5 road segments** ahead of the vehicle:

```python
N_ahead = 5
lookahead_idx = path_idx + [0, 1, 2, 3, 4]  # clamped to valid range
sl_ahead   = model.rs_speed_limit[lookahead_idx]   # mph
curv_ahead = model.rs_curvature[lookahead_idx]      # degrees
```

These are combined via distance-decaying weights:

```python
weights = [1/1, 1/2, 1/3, 1/4, 1/5]  # normalized
avg_sl   = weighted_average(sl_ahead, weights)
avg_curv = weighted_average(curv_ahead, weights)
```

The nearest segment has the most influence. At ~50m spacing, this creates a lookahead window of ~250m, giving drivers time to begin slowing before reaching the curve.

### 2. Compute curve reduction

```python
curve_effect = clip(avg_curv / 90.0, 0, 1)
speed_effect = clip((avg_sl - 10) / (60 - 10), 0, 1)

if avg_sl <= 15:
    speed_effect = 0    # no curve penalty at low speed limits

curve_sl = avg_sl * (1 - curve_resp * curve_effect * speed_effect)
```

### 3. Add driver's personal overage

```python
implicit_sl = curve_sl + acceptable_over
```

---

## Component Breakdown

### `curve_effect` -- how sharp is the curve

Curvature (in degrees) normalized to [0, 1] by dividing by 90:

| Curvature | curve_effect | Meaning |
|-----------|-------------|---------|
| 0° | 0.0 | Straight road |
| 30° | 0.33 | Gentle curve |
| 60° | 0.67 | Moderate curve |
| 90°+ | 1.0 | Sharp curve (capped) |

### `speed_effect` -- is this a fast enough road to care

Normalized speed limit, mapping the 10--60 mph range to [0, 1]:

| Posted limit | speed_effect | Meaning |
|-------------|-------------|---------|
| <= 15 mph | 0.0 | Too slow to worry about curves |
| 25 mph | 0.3 | Mild sensitivity |
| 40 mph | 0.6 | Moderate sensitivity |
| 60+ mph | 1.0 | Full sensitivity |

The 15 mph cutoff prevents curve reductions in parking lots or very slow zones where curvature is already navigable at the posted limit.

### `curve_resp` -- driver personality

| Vehicle type | Distribution | Typical range |
|-------------|-------------|---------------|
| Cars | `truncnorm(upper=0.95, lower=0.6, std=0.1)` | 0.6--0.95 |
| Buses | Fixed | 0.9 |

A `curve_resp` of 0.95 means the driver applies 95% of the maximum possible curve reduction. Lower values mean more aggressive cornering.

---

## Worked Example

A driver (`curve_resp = 0.95`, `acceptable_over = 3 mph`) approaches segments with the following lookahead:

| Segment | Speed limit | Curvature | Weight |
|---------|------------|-----------|--------|
| Current | 40 mph | 10° | 1.0 |
| +1 | 40 mph | 45° | 0.5 |
| +2 | 40 mph | 70° | 0.33 |
| +3 | 35 mph | 50° | 0.25 |
| +4 | 35 mph | 20° | 0.2 |

Weighted averages (after normalizing weights to sum to 1):

```
avg_sl   = (40×1 + 40×0.5 + 40×0.33 + 35×0.25 + 35×0.2) / 2.28 = 38.5 mph
avg_curv = (10×1 + 45×0.5 + 70×0.33 + 50×0.25 + 20×0.2) / 2.28 = 30.5°
```

Reduction:

```
curve_effect = 30.5 / 90 = 0.34
speed_effect = (38.5 - 10) / 50 = 0.57
curve_sl = 38.5 * (1 - 0.95 * 0.34 * 0.57) = 38.5 * 0.816 = 31.4 mph
implicit_sl = 31.4 + 3 = 34.4 mph
```

The vehicle will brake if traveling faster than 34.4 mph, even though the posted limit is 40.

---

## Figure: Speed Limit Reduction Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt

curve_resp = 0.95  # typical driver

speed_limits = np.arange(15, 65, 5)   # posted limits (mph)
curvatures = np.arange(0, 95, 5)      # degrees

reduction_pct = np.zeros((len(curvatures), len(speed_limits)))

for i, curv in enumerate(curvatures):
    for j, sl in enumerate(speed_limits):
        curve_effect = np.clip(curv / 90.0, 0, 1)
        speed_effect = np.clip((sl - 10) / 50.0, 0, 1)
        if sl <= 15:
            speed_effect = 0
        reduction_pct[i, j] = curve_resp * curve_effect * speed_effect * 100

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.imshow(reduction_pct, aspect="auto", origin="lower", cmap="YlOrRd",
               extent=[speed_limits[0], speed_limits[-1], curvatures[0], curvatures[-1]])

ax.set_xlabel("Posted Speed Limit (mph)")
ax.set_ylabel("Average Curvature (degrees)")
ax.set_title(f"Speed Limit Reduction % Due to Curves (curve_resp={curve_resp})")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Reduction (%)")

# Add contour lines
cs = ax.contour(speed_limits, curvatures, reduction_pct,
                levels=[10, 20, 30, 40, 50], colors="black", linewidths=0.8)
ax.clabel(cs, fmt="%.0f%%", fontsize=9)

plt.tight_layout()
plt.show()
```
