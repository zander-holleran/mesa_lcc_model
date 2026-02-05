"""
Tolling system with composable Signal -> Transform -> Toll architecture.

Signals read model state and return a numeric value (or None if not ready).
Transforms map signal values to raw toll amounts.
TollConfig ties everything together with universal wrappers (rounding, cap, floor).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Optional


# -----------------------
# Signals
# -----------------------

@dataclass
class VolumeSignal:
    """Current number of vehicles on the road."""

    def __call__(self, model) -> float:
        return len(model.vehicles_list)


@dataclass
class FlowSignal:
    """Rolling average vehicle arrival rate (vehicles per step).

    Returns None until the window is full.
    """
    window_steps: int = 300  # 5 minutes at 1 step/sec

    def __post_init__(self):
        self._window: deque = deque(maxlen=self.window_steps)
        self._last_vehicle_count: Optional[int] = None

    def __call__(self, model) -> Optional[float]:
        current_count = model.car_counter + model.bus_counter

        if self._last_vehicle_count is None:
            self._last_vehicle_count = current_count
            return None

        spawned = current_count - self._last_vehicle_count
        self._last_vehicle_count = current_count
        self._window.append(spawned)

        if len(self._window) < self.window_steps:
            return None

        return sum(self._window) / len(self._window)


# -----------------------
# Transforms
# -----------------------

@dataclass
class PiecewiseLinearTransform:
    """Toll = base + slope * (signal - threshold) when signal > threshold, else 0."""
    threshold: float = 100.0
    slope: float = 0.05
    base: float = 5.0

    def __call__(self, signal: float) -> float:
        if signal <= self.threshold:
            return 0.0
        return self.base + self.slope * (signal - self.threshold)


@dataclass
class StepTransform:
    """Flat toll when signal exceeds threshold, else 0."""
    threshold: float = 100.0
    toll: float = 10.0

    def __call__(self, signal: float) -> float:
        return self.toll if signal > self.threshold else 0.0


@dataclass
class PITransform:
    """Proportional-integral feedback controller.

    Drives signal toward a target. Toll accumulates when signal stays
    above target and decreases when signal drops below.

    When reset_integral_on_target=True (default), the integral resets to 0
    whenever the signal is at or below the target. This prevents the toll
    from persisting at elevated levels once congestion is controlled.
    """
    target: float = 30.0
    kp: float = 0.5
    ki: float = 0.05
    toll_min: float = 0.0
    toll_max: float = 50.0
    reset_integral_on_target: bool = True

    def __post_init__(self):
        self._integral: float = 0.0

    def __call__(self, signal: float) -> float:
        error = signal - self.target

        # Reset integral when signal is at or below target
        if self.reset_integral_on_target and error <= 0:
            self._integral = 0.0
            return self.toll_min

        self._integral += error

        # Anti-windup: clamp integral so it can't accumulate beyond
        # what would produce toll_min/toll_max by itself
        if self.ki != 0:
            max_integral = self.toll_max / self.ki
            self._integral = max(min(self._integral, max_integral), -max_integral)

        toll = self.kp * error + self.ki * self._integral

        return max(self.toll_min, min(self.toll_max, toll))


# -----------------------
# Utilities
# -----------------------

def round_toll(toll: float, increment: float) -> float:
    """Round toll to nearest increment (e.g., 0.10 or 0.25)."""
    if increment <= 0:
        return toll
    return round(toll / increment) * increment


# -----------------------
# TollConfig
# -----------------------

@dataclass
class TollConfig:
    """Complete toll specification: signal + transform + wrappers.

    For static tolls, use TollConfig.static(car=10.0).
    For dynamic tolls, provide signal and transform.
    """
    signal: Any = None           # callable(model) -> float | None
    transform: Any = None        # callable(signal) -> float
    update_every_n_steps: int = 1      # how often to recalculate
    rounding: Optional[float] = None   # round to nearest increment (e.g., 0.10)
    cap: Optional[float] = None        # maximum toll
    floor: Optional[float] = None      # minimum nonzero toll (once threshold crossed)
    _static_toll: Optional[float] = None  # internal: for static toll case

    @staticmethod
    def static(car: float = 0.0) -> "TollConfig":
        """Convenience: fixed toll, no signal or transform needed."""
        return TollConfig(
            signal=None,
            transform=None,
            _static_toll=car,
        )

    def reset(self):
        """Reset signal/transform state between days."""
        if self.signal is not None and hasattr(self.signal, '__post_init__'):
            self.signal.__post_init__()
        if self.transform is not None and hasattr(self.transform, '__post_init__'):
            self.transform.__post_init__()

    def get_initial_toll(self) -> float:
        """Get the initial toll value (static toll or 0.0 for dynamic)."""
        if self._static_toll is not None:
            return self._static_toll
        return 0.0


# -----------------------
# Main update function
# -----------------------

def update_tolls(model) -> float:
    """Update and return the current car toll based on model's toll_config."""
    toll_config = model.toll_config

    # Static toll: return fixed value
    if toll_config.signal is None:
        return model.current_toll_car

    # Cadence check
    if model.steps % toll_config.update_every_n_steps != 0:
        return model.current_toll_car

    # Get signal value
    signal = toll_config.signal(model)
    if signal is None:
        return model.current_toll_car

    # Apply transform
    raw_toll = toll_config.transform(signal)

    # Apply universal wrappers
    toll = raw_toll
    if toll_config.floor is not None and toll > 0:
        toll = max(toll, toll_config.floor)
    if toll_config.cap is not None:
        toll = min(toll, toll_config.cap)
    if toll_config.rounding is not None:
        toll = round_toll(toll, toll_config.rounding)

    model.current_toll_car = toll
    return toll
