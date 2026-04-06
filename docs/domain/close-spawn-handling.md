# Close-Spawn Handling

When vehicles spawn in rapid succession, they can end up at the same position on the road. This causes deadlock in the gap computation (which relies on sorted distance rankings). The **close-spawn handler** detects this and shifts the spawn point to create separation.

**Source:** `traffic/model/generate.py` -- `too_close()` (lines 4--18), `_spawn_offset()` (lines 20--29)

**Related:** [Vehicle Physics](vehicle-physics.md) | [Gap Braking](gap-braking.md)

---

## The Problem

Vehicles are spawned at `model.start_point` -- the road's origin. If two vehicles spawn on consecutive steps and the first hasn't moved far enough, both end up at nearly the same position. The kernel computes gaps by sorting vehicles by distance traveled:

```
gap[i] = dist[i+1] - dist[i]
```

If two vehicles share `dist = 0`, gap is zero, both gap-brake to a stop, and neither can ever pass the other. Deadlock.

---

## Detection: `too_close()`

After each vehicle spawn, `too_close()` checks the most recently spawned vehicle:

```python
def too_close(model):
    v = model.vehicles_list[-1]
    vx, vy = vs.pos_x[slot], vs.pos_y[slot]
    sx, sy = model.start_point
    dist = sqrt((vx - sx)² + (vy - sy)²)

    if dist < 1:  # meters
        model.start_point = (sx, sy + 1)  # shift north by 1m
        model.too_close_counter += 1
```

The threshold is **1 meter** -- if the last vehicle is within 1m of the spawn point, the spawn point shifts.

---

## The Fix: Shift Start Point

Each time `too_close()` triggers, the start point moves 1 meter north (increasing y-coordinate). This accumulates -- if 3 vehicles spawn back-to-back, the start point may shift 3 meters from its original position.

```
Step 1: Vehicle A spawns at (x, y)      ← original start point
Step 2: Vehicle A hasn't moved far       → shift to (x, y+1)
Step 3: Vehicle B spawns at (x, y+1)
Step 4: Vehicle B hasn't moved far       → shift to (x, y+2)
Step 5: Vehicle C spawns at (x, y+2)
```

---

## Distance Bookkeeping: `_spawn_offset()`

Shifting the start point creates a problem: new vehicles spawn _behind_ the original road origin. If they started at `distance_traveled = 0`, they'd appear ahead of where they actually are in the sorted ranking.

The fix: vehicles spawned at a shifted start point get a **negative initial distance**:

```python
def _spawn_offset(model) -> float:
    ix, iy = model.initial_start_point   # original, never changes
    sx, sy = model.start_point           # current, possibly shifted
    return -sqrt((ix - sx)² + (iy - sy)²)
```

If the start point has shifted 3 meters north, new vehicles start at `distance_traveled = -3.0`. They then progress through 0 as they reach the original start point and continue into positive territory along the road.

This preserves correct ordering in the gap computation: vehicles further behind have smaller (more negative) distances.

---

## Tracking

`model.too_close_counter` records how many times the start point was shifted during the simulation. High values indicate heavy traffic at the spawn point -- useful for diagnosing congestion near the canyon entrance.

---

## Figure: Spawn Point Shifting

```
Road direction →

Original start (0m)          Road continues...
    |
    ▼
    ●─────────────────────────────────→

After 3 close-spawn shifts:

(-3m)  (-2m)  (-1m)  (0m)           Road continues...
  |      |      |      |
  ▼      ▼      ▼      ▼
  C      B      A      ●─────────────→
  │      │      │
  └──────┴──────┴── vehicles spawned behind origin
                     with negative distance_traveled

  dist_C = -3    dist_B = -2    dist_A = -1
  gap(C→B) = 1m  gap(B→A) = 1m
```

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 4))

# Road line
ax.plot([0, 10], [1, 1], color="gray", linewidth=3, zorder=1)
ax.annotate("Road →", xy=(10, 1), fontsize=12, ha="right", va="bottom")

# Original start point
ax.plot(3, 1, "s", color="green", markersize=12, zorder=3, label="Original start")

# Shifted spawn points and vehicles
shifts = [
    (3.0, "A", "Step 1: spawns at origin", "blue"),
    (2.0, "B", "Step 3: start shifted -1m", "orange"),
    (1.0, "C", "Step 5: start shifted -2m", "red"),
]

for x, label, desc, color in shifts:
    ax.plot(x, 1, "o", color=color, markersize=15, zorder=4)
    ax.text(x, 1.3, f"Vehicle {label}", ha="center", fontsize=10, fontweight="bold", color=color)
    ax.text(x, 0.6, desc, ha="center", fontsize=8, color="gray")

# Distance labels
ax.text(3, 0.3, "dist=0", ha="center", fontsize=9)
ax.text(2, 0.3, "dist=-1", ha="center", fontsize=9)
ax.text(1, 0.3, "dist=-2", ha="center", fontsize=9)

ax.set_xlim(0, 11)
ax.set_ylim(0, 2)
ax.set_yticks([])
ax.set_xlabel("Position along road (meters from origin)")
ax.set_title("Close-Spawn Handling: Start Point Shifting")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
```
