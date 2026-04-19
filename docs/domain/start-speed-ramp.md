# Start Speed Ramp

The first 3 road segments have **manually overridden speed limits** of 10, 20, and 35 mph, creating a gradual ramp-up from the canyon entrance. This prevents vehicles from spawning at 0 mph directly into a 40+ mph zone.

**Source:** `collect_external_data/road_geom.py` -- line 161

**Related:** [Implicit Speed Limit](implicit-speed-limit.md) | [Speed-Limit Braking](speed-limit-braking.md) | [Cumulative Time Lost](cumulative-time-lost.md) | [Acceleration Curves](acceleration-curves.md)

---

## The Override

After all UDOT data processing and 50-meter densification, a single line overrides the first 3 segments:

```python
# -------------- manual tweak to first few points --------------
road_gdf.loc[0:2, "speed_limit"] = [10, 20, 35]
```

| Segment index | Distance from start | Original limit | Overridden limit |
|--------------|--------------------:|---------------:|-----------------:|
| 0 | 0 m | ~40+ mph | **10 mph** |
| 1 | ~50 m | ~40+ mph | **20 mph** |
| 2 | ~100 m | ~40+ mph | **35 mph** |
| 3 | ~150 m | -- | Original UDOT value |

---

## Why It Exists

### Problem: artificial delay at spawn

Vehicles start at speed 0. Without the ramp, the implicit speed limit at the spawn point would be the full UDOT-posted limit (typically 40--50 mph). The [cumulative time lost](cumulative-time-lost.md) metric computes:

```
time_lost = (implicit_sl - speed) / implicit_sl
```

A vehicle at 0 mph in a 50 mph zone accumulates `time_lost = 1.0` per step -- maximum delay -- for every step it takes to accelerate up to speed. This would add ~10--15 seconds of artificial delay to every single trip, inflating the congestion metric.

### Solution: match the ramp to acceleration capability

With the ramp, a vehicle at 0 mph only needs to reach 10 mph to satisfy segment 0's limit. By segment 1 (~50m later), the vehicle should be near 20 mph through natural acceleration. By segment 2, near 35 mph. The ramp roughly tracks what a typical vehicle can achieve via its [acceleration curve](acceleration-curves.md), so cumulative time lost stays near zero during the initial acceleration phase.

---

## Downstream Flow

1. `road_geom.py` saves the modified limits to `data/road/road_gdf.parquet`
2. `traffic/model/init_helpers.py` loads the parquet and stores limits in `model.rs_speed_limit` (a NumPy array)
3. The kernel's lookahead (`vehicle_kernel.py:104`) reads `model.rs_speed_limit[lookahead_idx]` every step
4. The ramp values are treated identically to any other posted limit -- they participate in the 5-segment weighted average and curve adjustment

---

## Figure: Speed Limit Profile at Road Start

```python
import numpy as np
import matplotlib.pyplot as plt

# First 15 segments: 3 ramped + 12 at typical UDOT value
n_segments = 15
segment_spacing = 50  # meters
distances = np.arange(n_segments) * segment_spacing

# Simulated speed limits (replace index 3+ with actual UDOT value for your road)
speed_limits = np.full(n_segments, 40.0)  # assume 40 mph UDOT default
speed_limits[0:3] = [10, 20, 35]         # the manual ramp

fig, ax = plt.subplots(figsize=(10, 5))

# Speed limit profile
ax.step(distances, speed_limits, where="mid", color="blue", linewidth=2, label="Posted speed limit")
ax.fill_between(distances, speed_limits, step="mid", alpha=0.1, color="blue")

# Highlight the ramp zone
ax.axvspan(0, 100, alpha=0.15, color="orange", label="Manual ramp zone (segments 0-2)")

# Simulated vehicle speed during acceleration (typical performance)
times = distances / 1  # approximate: 1 step per ~segment at low speed
from traffic.agents.vehicle_agent import build_empirical_accel_function
accel_fn = build_empirical_accel_function(0.5)  # median performance
speed_mph = 0.0
vehicle_speeds = []
for d in distances:
    vehicle_speeds.append(speed_mph)
    a_mps2 = accel_fn(speed_mph)
    speed_mph += a_mps2 * 2.23694  # 1s step

ax.plot(distances, vehicle_speeds, color="red", linewidth=2, linestyle="--",
        marker="o", markersize=4, label="Typical vehicle speed (accel from 0)")

ax.set_xlabel("Distance from start (meters)")
ax.set_ylabel("Speed (mph)")
ax.set_title("Speed Limit Ramp vs Vehicle Acceleration at Canyon Entrance")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, distances[-1])
ax.set_ylim(0, 55)

plt.tight_layout()
plt.show()
```
