# Speed-Limit Braking

**Speed-limit braking** slows a vehicle when it exceeds its [implicit speed limit](implicit-speed-limit.md). The deceleration is **proportional** to how far over the limit the vehicle is, with a steeper coefficient when significantly over.

**Source:** `traffic/model/vehicle_kernel.py` -- lines 147--157

**Related:** [Implicit Speed Limit](implicit-speed-limit.md) | [Gap Braking](gap-braking.md) | [Curve Braking](curve-braking.md) | [Vehicle Physics](vehicle-physics.md)

---

## Trigger Condition

```python
sl_brake_mask = active & ~gap_brake_mask & (speed > implicit_sl_mps + 1e-6)
```

Speed-limit braking only fires when:
1. The vehicle is **not already gap-braking** (gap braking has higher priority)
2. Speed exceeds the implicit speed limit by more than a floating-point epsilon

---

## Deceleration Formula

```python
over_mph = (speed - implicit_sl_mps) * MPS_TO_MPH

sl_decel_mps = where(
    over_mph < 5.0,
    over_mph * MPH_TO_MPS * 0.3,   # gentle tier
    over_mph * MPH_TO_MPS * 0.6,   # firm tier
)

speed -= sl_decel_mps
```

The deceleration is proportional to the overage -- not a fixed step-down. This produces smooth, realistic corrections rather than jarring speed jumps.

### Two tiers

| Over by | Coefficient | Example: 3 mph over | Example: 8 mph over |
|---------|-------------|---------------------|---------------------|
| < 5 mph | 0.3 | 3 × 0.447 × 0.3 = **0.40 m/s²** | -- |
| >= 5 mph | 0.6 | -- | 8 × 0.447 × 0.6 = **2.15 m/s²** |

The gentle tier handles minor drift above the limit (a few mph). The firm tier activates when the vehicle is significantly over -- for example, cresting a hill into a lower speed zone.

---

## Brake Cooldown

Sets `break_cooldown = 3`, shorter than gap braking's cooldown of 5. This means the vehicle resumes normal acceleration sooner after a speed-limit correction, reflecting that speed-limit adjustments are minor course corrections rather than emergency maneuvers.

Cooldown effect on subsequent steps:

| Cooldown value | Behavior |
|----------------|----------|
| >= 4 | Coast (no speed change) |
| 1--3 | Slow accelerate (partial acceleration) |
| 0 | Full acceleration |

Since speed-limit braking sets cooldown to 3, the vehicle immediately enters slow-accelerate mode on the next step (not coasting first).

---

## Relationship to Implicit Speed Limit

The implicit speed limit already accounts for:
- **Road curvature** via the [curve braking](curve-braking.md) reduction
- **Driver personality** via `acceptable_over` (how much this driver exceeds posted limits)

So speed-limit braking fires relative to the driver's _personal_ limit, not the posted limit. A driver with `acceptable_over = 5 mph` in a 40 mph zone won't brake until exceeding 45 mph (plus any curve adjustment).

---

## Kernel vs Legacy Agent Implementation

The kernel version (documented here) uses **proportional** deceleration that scales with overage. The earlier `VehicleAgent` version used fixed deceleration tiers:

| Over by | Legacy (fixed) | Kernel (proportional) |
|---------|---------------|----------------------|
| > 0 mph | 0.2 m/s² | `over * 0.447 * 0.3` |
| > 2 mph | 0.5 m/s² | `over * 0.447 * 0.3` |
| > 5 mph | -- | `over * 0.447 * 0.6` |
| > 7 mph | 1.1 m/s² | `over * 0.447 * 0.6` |

The proportional approach produces smoother speed corrections and avoids oscillation at tier boundaries.

---

## Figure: Deceleration vs Speed Over Limit

```python
import numpy as np
import matplotlib.pyplot as plt

MPH_TO_MPS = 0.44704

over_mph = np.linspace(0, 15, 300)
decel = np.where(
    over_mph < 5.0,
    over_mph * MPH_TO_MPS * 0.3,
    over_mph * MPH_TO_MPS * 0.6,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(over_mph, decel, color="orange", linewidth=2)
ax.axvline(5.0, color="gray", linestyle="--", alpha=0.5, label="Tier boundary (5 mph)")

# Annotate the two regimes
ax.text(2.5, 0.5, "Gentle tier\n(coeff = 0.3)", ha="center", fontsize=10, color="green")
ax.text(10, 2.0, "Firm tier\n(coeff = 0.6)", ha="center", fontsize=10, color="red")

ax.set_xlabel("Speed over implicit limit (mph)")
ax.set_ylabel("Deceleration (m/s²)")
ax.set_title("Speed-Limit Braking Response")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 15)
ax.set_ylim(0, 5)

plt.tight_layout()
plt.show()
```
