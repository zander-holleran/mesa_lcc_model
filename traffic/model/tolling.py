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


# -----------------------
# Interactive toll explorer
# -----------------------

def interactive_toll_plot(
    toll_configs: "TollConfig | list[TollConfig]",
    labels: "str | list[str] | None" = None,
    signal_range: tuple[float, float] = (0, 600),
    signal_label: str = "Vehicle Count",
    window_seconds: int = 7200,
    step_seconds: int = 300,
):
    """Interactive toll explorer using inline matplotlib (VSCode-compatible).

    Set the signal slider, click 'Step' to advance simulation time, and watch
    how each toll config responds. Change the signal between steps to simulate
    demand ramps, spikes, or drops.

    Parameters
    ----------
    toll_configs : One or more TollConfig objects to compare.
    labels : Display names for each config. Auto-generated if None.
    signal_range : Slider min/max for the signal value.
        Volume signal: (0, 600). Flow signal: (0.0, 1.0).
    signal_label : Axis / slider label for the signal.
    window_seconds : Rolling display window width in simulated seconds.
    step_seconds : How many seconds of sim time each 'Step' click advances.
    """
    import copy
    import io
    from bisect import bisect_left

    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output, Image

    # --- normalise inputs ---
    if isinstance(toll_configs, TollConfig):
        toll_configs = [toll_configs]
    n = len(toll_configs)
    if labels is None:
        labels = [f"Toll {i}" for i in range(n)]
    elif isinstance(labels, str):
        labels = [labels]
    configs = [copy.deepcopy(tc) for tc in toll_configs]

    # --- mutable state ---
    state = dict(
        step=0,
        times=[],
        signals=[],
        tolls={lb: [] for lb in labels},
        current={lb: 0.0 for lb in labels},
    )

    # --- widgets ---
    sig_step = 1.0 if (signal_range[1] - signal_range[0]) > 10 else 0.01
    signal_slider = widgets.FloatSlider(
        value=signal_range[0],
        min=signal_range[0],
        max=signal_range[1],
        step=sig_step,
        description=signal_label[:18],
        layout=widgets.Layout(width="80%"),
        style={"description_width": "130px"},
        readout_format=".1f" if sig_step < 1 else ".0f",
    )
    step_input = widgets.BoundedIntText(
        value=step_seconds,
        min=1,
        max=7200,
        description="Step (sec)",
        layout=widgets.Layout(width="200px"),
        style={"description_width": "80px"},
    )
    step_btn = widgets.Button(description="Step", button_style="success", icon="step-forward")
    reset_btn = widgets.Button(description="Reset", button_style="warning", icon="refresh")
    info = widgets.Label(value="Step: 0 | 0:00:00")
    plot_out = widgets.Output()

    # --- helpers ---
    def _apply_wrappers(cfg, raw):
        toll = raw
        if cfg.floor is not None and toll > 0:
            toll = max(toll, cfg.floor)
        if cfg.cap is not None:
            toll = min(toll, cfg.cap)
        if cfg.rounding is not None:
            toll = round_toll(toll, cfg.rounding)
        return toll

    def _advance(n_steps, signal_val):
        for _ in range(n_steps):
            step = state["step"]
            state["times"].append(step)
            state["signals"].append(signal_val)

            for lb, cfg in zip(labels, configs):
                if cfg._static_toll is not None:
                    state["current"][lb] = cfg._static_toll
                elif cfg.transform is not None and step % max(cfg.update_every_n_steps, 1) == 0:
                    raw = cfg.transform(signal_val)
                    state["current"][lb] = _apply_wrappers(cfg, raw)
                state["tolls"][lb].append(state["current"][lb])

            state["step"] += 1

        # trim to prevent unbounded growth
        keep = window_seconds * 3
        if len(state["times"]) > keep:
            drop = len(state["times"]) - keep
            state["times"] = state["times"][drop:]
            state["signals"] = state["signals"][drop:]
            for lb in labels:
                state["tolls"][lb] = state["tolls"][lb][drop:]

    def _draw():
        t = state["times"]
        with plot_out:
            clear_output(wait=True)
            if not t:
                print("No data yet — click Step.")
                return

            fig, ax_sig = plt.subplots(figsize=(12, 4.5))
            ax_toll = ax_sig.twinx()

            t_max = t[-1]
            t_min = max(0, t_max - window_seconds)
            idx = bisect_left(t, t_min)

            tw = [s / 3600 for s in t[idx:]]
            sw = state["signals"][idx:]

            ax_sig.plot(tw, sw, "k-", lw=1.5, alpha=0.6, label=signal_label)
            ax_sig.fill_between(tw, sw, alpha=0.08, color="black")

            colors = plt.cm.tab10.colors
            peak_toll = 1.0
            for i, lb in enumerate(labels):
                tw_toll = state["tolls"][lb][idx:]
                ax_toll.plot(tw, tw_toll, color=colors[i % 10], lw=1.8, label=lb)
                if tw_toll:
                    peak_toll = max(peak_toll, max(tw_toll))

            ax_sig.set_xlabel("Simulated time (hours)")
            ax_sig.set_ylabel(signal_label, color="black")
            ax_toll.set_ylabel("Toll ($)")
            ax_sig.set_xlim(t_min / 3600, max(t_max, t_min + 60) / 3600)
            ax_sig.set_ylim(signal_range[0], signal_range[1] * 1.05)
            ax_toll.set_ylim(0, peak_toll * 1.15)
            ax_sig.grid(True, alpha=0.3)

            # combined legend
            h1, l1 = ax_sig.get_legend_handles_labels()
            h2, l2 = ax_toll.get_legend_handles_labels()
            ax_sig.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)

            s = state["step"]
            ax_sig.set_title(
                f"Step {s:,} | {s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}",
                fontsize=10, loc="right",
            )
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            display(Image(data=buf.read()))

        s = state["step"]
        info.value = f"Step: {s:,} | {s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    # --- button callbacks ---
    def _on_step(_):
        _advance(step_input.value, signal_slider.value)
        _draw()

    def _on_reset(_):
        state["step"] = 0
        state["times"].clear()
        state["signals"].clear()
        for lb in labels:
            state["tolls"][lb].clear()
            state["current"][lb] = 0.0
        for cfg in configs:
            cfg.reset()
        _draw()

    step_btn.on_click(_on_step)
    reset_btn.on_click(_on_reset)

    controls = widgets.VBox([
        signal_slider,
        widgets.HBox([step_input, step_btn, reset_btn, info]),
        plot_out,
    ])
    display(controls)
