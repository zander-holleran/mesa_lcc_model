# Acceleration Curves

Each vehicle gets a unique acceleration function that maps current speed (mph) to acceleration (m/s²). The curves are derived from **real-world stop-sign acceleration data**, modeled as 4 piecewise segments with skew-normal distributions. A vehicle's `performance` percentile determines where it falls within each segment's distribution.

**Source:** `traffic/agents/vehicle_agent.py` -- `build_empirical_accel_function()` (lines 10--58)

**Related:** [Speed](speed.md) | [Vehicle Physics](vehicle-physics.md) | [Speed-Limit Braking](speed-limit-braking.md)

---

## How It Works

The function `build_empirical_accel_function(pctile)` builds a closure that returns acceleration for a given speed. The construction has three stages:

### 1. Sample acceleration from distributions

Four skew-normal distributions (one per time segment) define plausible acceleration values in m/s²:

| Segment | Time window | Base mean | Base variance | Shifted mean | Stretched var |
|---------|-------------|-----------|---------------|--------------|---------------|
| 0 | 0--2 sec | 1.0 | 0.35 | 0.8 | 0.455 |
| 1 | 2--4 sec | 2.5 | 0.40 | 2.3 | 0.520 |
| 2 | 4--6 sec | 2.0 | 0.40 | 1.8 | 0.520 |
| 3 | 6--8 sec | 1.5 | 0.30 | 1.3 | 0.390 |

Global tuning parameters shift and stretch all distributions:
- `mean_shift = -0.2` -- downward shift (vehicles are slightly slower than raw study data)
- `var_streach = 1.3` -- widens distributions for more inter-vehicle variation

Each distribution is `skewnorm(loc=shifted_mean, scale=stretched_var, a=1)`. The vehicle's percentile selects a specific value via the inverse CDF: `dist.ppf(trimmed_pctile)`.

The percentile is clipped to **[0.07, 0.95]** to avoid extreme distribution tails.

### 2. Convert time segments to speed ranges

Each segment's acceleration × duration gives velocity gained (m/s), converted to mph:

```
delta_v_mph = accel_mps2 * 2_seconds * 2.2
```

Speed bounds accumulate: segment 0 covers [0, bound₁], segment 1 covers [bound₁, bound₂], etc. The exact bounds depend on the percentile -- faster vehicles have wider speed ranges per segment.

### 3. Build the lookup function

The returned `accel(speed_mph)` function:
- If speed falls within a segment's range, return that segment's acceleration
- Beyond the last segment boundary, acceleration tapers linearly toward zero:

```python
accel = accels[-1] * (1 - (speed_mph - last_bound) / 70)
# clipped to [0, 10] m/s²
```

This means vehicles approaching high speeds gradually lose acceleration, never abruptly dropping to zero.

---

## Pre-Computed Cache

At model initialization (`TrafficModel.__init__`), 101 acceleration functions are built and cached:

```python
self.accel_curve_cache = [build_empirical_accel_function(p / 100) for p in range(101)]
```

During the kernel's acceleration phase, each vehicle's `performance` (uniform 0--1) is mapped to an integer index 0--100 to select its curve:

```python
perf_group = (vs.performance[:n] * 100).astype(np.int32)
perf_group = np.clip(perf_group, 0, 100)
accel_fn = model.accel_curve_cache[perf_group[i]]
```

**Source:** `traffic/model/traffic_model.py:83` (cache), `traffic/model/vehicle_kernel.py:162-171` (usage)

---

## Typical Values

| Percentile | Seg 0 (m/s²) | Seg 1 (m/s²) | Seg 2 (m/s²) | Seg 3 (m/s²) | Character |
|------------|--------------|--------------|--------------|--------------|-----------|
| 0.10 | ~0.5 | ~1.8 | ~1.2 | ~0.9 | Sluggish (old sedan) |
| 0.50 | ~0.9 | ~2.5 | ~2.0 | ~1.4 | Average (typical car) |
| 0.90 | ~1.4 | ~3.4 | ~2.8 | ~2.0 | Quick (sports car) |

Buses are fixed at `performance = 0.1` (slow end of the spectrum).

---

## Design Rationale

The empirical approach was chosen over parametric models (exponential decay, piecewise linear) after testing several alternatives in `notebooks/other_notebooks/car_performance_analysis.ipynb`. The skew-normal distributions approximate the acceleration profiles observed in a [naturalistic study of vehicle acceleration at intersections](https://www.jsheld.com/insights/articles/a-naturalistic-study-of-vehicle-acceleration-and-deceleration-at-an-intersection).

The 4-segment structure captures realistic behavior: moderate initial pull-away, peak acceleration in the mid-range, then tapering as the vehicle approaches cruising speed.

---

## Figure: Acceleration vs Speed by Performance Level

```python
import numpy as np
import matplotlib.pyplot as plt
from traffic.agents.vehicle_agent import build_empirical_accel_function

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: Accel vs Speed for multiple percentiles ---
percentiles = [0.1, 0.3, 0.5, 0.7, 0.9]
speeds = np.linspace(0, 80, 300)

for pctile in percentiles:
    accel_fn = build_empirical_accel_function(pctile)
    accels = [accel_fn(s) for s in speeds]
    axes[0].plot(speeds, accels, label=f"perf={pctile}")

axes[0].set_xlabel("Speed (mph)")
axes[0].set_ylabel("Acceleration (m/s²)")
axes[0].set_title("Acceleration Curves by Performance Percentile")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 4)

# --- Right: Simulated 0-to-cruise speed profiles ---
for pctile in percentiles:
    accel_fn = build_empirical_accel_function(pctile)
    speed_mph = 0.0
    times, speed_trace = [0], [0]
    for t in range(1, 40):
        a_mps2 = accel_fn(speed_mph)
        speed_mph += a_mps2 * 2.23694  # 1 sec step, m/s² -> mph
        speed_mph = max(speed_mph, 0)
        times.append(t)
        speed_trace.append(speed_mph)
    axes[1].plot(times, speed_trace, label=f"perf={pctile}")

axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Speed (mph)")
axes[1].set_title("Simulated Acceleration from Stop")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
