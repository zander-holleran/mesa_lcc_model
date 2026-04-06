# Gap-Based Braking

**Gap-based braking** (smooth brake) is the highest-priority braking mechanism. It activates when a vehicle is closer to the vehicle ahead than its **ideal following gap**. The braking force increases quadratically as the gap shrinks, producing human-like overreaction when dangerously close.

**Source:** `traffic/model/vehicle_kernel.py` -- lines 134, 138--145

**Related:** [Speed](speed.md) | [Speed-Limit Braking](speed-limit-braking.md) | [Vehicle Physics](vehicle-physics.md)

---

## Trigger Condition

```python
gap_brake_mask = active & (gap < max(speed * ideal_distance_multiplier, 5.0))
```

The **ideal gap** scales with speed and the driver's personal following-distance preference:

```
ideal_gap = max(speed_mps * idm, 5.0)
```

| Parameter | Source | Distribution | Typical range |
|-----------|--------|-------------|---------------|
| `speed` | Vehicle state | -- | 0--25 m/s |
| `ideal_distance_multiplier` | Driver trait | `truncnorm(upper=2, lower=0.6, std=0.3, mean=1)` | 0.6--2.0 |
| Minimum gap | Hardcoded | -- | 5.0 meters |

At 50 mph (~22.4 m/s) with `idm = 1.0`, the ideal gap is 22.4 meters (~1 second of following distance). A cautious driver (`idm = 2.0`) would want 44.8 meters. The 5-meter floor ensures braking even at very low speeds.

---

## Braking Formula

```python
force = clip((ideal_gap - gap) / ideal_gap, 0, 1)
noise = rng.normal(0, 0.1)
decel = force² * 8.0 + noise
```

Breaking this down:

1. **Force ratio** -- how much of the ideal gap has been violated (0 = at ideal, 1 = gap is zero)
2. **Quadratic scaling** -- `force²` makes braking gentle when slightly close but aggressive when very close
3. **Human noise** -- Gaussian noise (std=0.1) adds realistic variation in reaction
4. **Max deceleration** -- the `8.0` multiplier caps effective braking at ~8 m/s² (comfortable hard braking; emergency braking is ~10 m/s²)

### Deceleration at different gap fractions

| Gap as % of ideal | Force | Decel (m/s²) |
|-------------------|-------|-------------|
| 90% (slightly close) | 0.10 | 0.08 |
| 70% | 0.30 | 0.72 |
| 50% | 0.50 | 2.00 |
| 30% | 0.70 | 3.92 |
| 10% (very close) | 0.90 | 6.48 |
| 0% (bumper to bumper) | 1.00 | 8.00 |

---

## Side Effects

- **Brake cooldown** set to **5 steps** -- the longest cooldown of any braking type. This prevents immediate full acceleration after gap braking; the vehicle coasts or slowly accelerates for several steps.
- **Car interactions counter** incremented: `vs.car_interactions += 1`. This tracks how many steps a vehicle spent braking due to traffic, used in downstream analysis.

---

## Priority

Gap braking is checked **first** in the priority chain. If a vehicle is gap-braking, it will NOT also speed-limit brake in the same step:

```
1. Gap brake     (gap < ideal_gap)      → decel, cooldown=5
2. Speed-limit brake  (speed > implicit_sl)  → decel, cooldown=3
3. Accelerate    (neither)               → accel (subject to cooldown)
```

---

## Figure: Deceleration vs Gap

```python
import numpy as np
import matplotlib.pyplot as plt

ideal_gap = 100  # meters (example at highway speed)
gaps = np.linspace(0, ideal_gap, 500)

# Deterministic curve (no noise)
force = np.clip((ideal_gap - gaps) / ideal_gap, 0, 1)
decel_mean = force ** 2 * 8.0

# Envelope with noise (std=0.1)
decel_upper = decel_mean + 0.3  # ~3 sigma
decel_lower = np.maximum(decel_mean - 0.3, 0)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gaps, decel_mean, color="red", linewidth=2, label="Mean deceleration")
ax.fill_between(gaps, decel_lower, decel_upper, color="red", alpha=0.15, label="Noise band (±3σ)")
ax.axvline(ideal_gap, color="green", linestyle="--", alpha=0.7, label=f"Ideal gap ({ideal_gap}m)")

ax.set_xlabel("Actual gap (meters)")
ax.set_ylabel("Deceleration (m/s²)")
ax.set_title("Gap-Based Braking Response")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, ideal_gap * 1.1)
ax.set_ylim(0, 9)
ax.invert_xaxis()  # smaller gap = more danger, shown on right

plt.tight_layout()
plt.show()
```
