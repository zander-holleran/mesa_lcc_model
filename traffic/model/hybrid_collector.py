"""
Hybrid Data Collection System for Mesa Traffic Simulation

A 4-tier data collection system optimized for performance:
- Tier 1: Aggregate metrics every step (pre-allocated numpy arrays)
- Tier 2: Sampled spatial data for animations (configurable interval)
- Tier 3: Event-based logging (crashes, canyon closures)
- Tier 4: Full snapshots at key moments (optional, disabled by default)
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import numpy as np
import pandas as pd

from traffic.utils import unit_conversion_utils as uc


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class HybridCollectorConfig:
    """Configuration for HybridDataCollector."""
    max_steps: int = 50000

    # Tier 1 enable + cadence
    tier1_enabled: bool = True
    tier1_interval: int = 1  # collect every Nth step

    # Tier 1: Scalar metrics to collect (keys from TIER1_SCALARS)
    tier1_scalars: List[str] = field(default_factory=lambda: [
        'step', 'current_toll', 'vehicle_count', 'active_cars', 'active_buses',
        'bus_riders_waiting', 'bus_mode_share_recent', 'total_finished',
        'p_generate',
    ])

    # Tier 1: Window-based scalars (use tier1_window_seconds)
    tier1_window_scalars: List[str] = field(default_factory=lambda: [
        'recent_travel_time_avg',
        'rolling_count_vehicles_generated',
        'rolling_count_persons_generated',
    ])

    # Tier 1: Histogram metrics to collect (keys from TIER1_HISTOGRAMS)
    tier1_histograms: List[str] = field(default_factory=lambda: [
        'implicit_sl_delta', 'speed_mps'
    ])

    # Tier 2: Spatial sampling
    tier2_sample_interval: int = 10
    tier2_enabled: bool = True
    tier2_max_samples: int = 5000
    tier2_max_agents_per_sample: int = 200

    # Tier 3: Event logging
    tier3_enabled: bool = True

    # Tier 4: Full snapshots (disabled by default)
    tier4_enabled: bool = False
    tier4_snapshot_interval: int = 0
    tier4_snapshot_on_crash: bool = False
    tier4_max_snapshots: int = 100

    # Window size (seconds) for all tier1 window-based scalars
    tier1_window_seconds: int = 300


# =============================================================================
# Tier 1: Aggregate Metrics
# =============================================================================

def _compute_bus_mode_share(model) -> float:
    """Compute % of recent persons who chose bus."""
    persons = model.traffic_persons_list
    if not persons:
        return float('nan')

    window = model.collect_every_n if model.collect_every_n >= 1 else 1
    upper = model.steps
    lower = max(0, upper - window + 1)

    recent_n = 0
    bus_n = 0
    for p in persons:
        if lower <= p.created_step <= upper:
            recent_n += 1
            if p.mode == "bus":
                bus_n += 1

    return (100.0 * bus_n / recent_n) if recent_n else float('nan')


def _compute_recent_travel_time(model) -> float:
    """Compute mean travel time of trips completed within tier1_window_seconds."""
    if not model.finished_agents:
        return float('nan')

    window = model.datacollector.config.tier1_window_seconds
    current_step = model.steps
    cutoff = current_step - window

    recent_times = []
    for agent in model.finished_agents:
        # Check if agent finished within window
        created = agent.get('created_at_step', 0)
        steps_taken = agent.get('steps_taken', 0)
        finished_step = created + steps_taken

        if finished_step >= cutoff:
            recent_times.append(steps_taken / 60.0)  # Convert to minutes

    return np.mean(recent_times) if recent_times else float('nan')


# Scalar metric registry
TIER1_SCALARS: Dict[str, Dict[str, Any]] = {
    'step': {
        'dtype': np.int32,
        'fn': lambda m: m.steps,
    },
    'current_toll': {
        'dtype': np.float32,
        'fn': lambda m: m.current_toll_car,
    },
    'vehicle_count': {
        'dtype': np.int16,
        'fn': lambda m: len(m.vehicles_list),
    },
    'active_cars': {
        'dtype': np.int16,
        'fn': lambda m: sum(1 for v in m.vehicles_list if v.__class__.__name__ == 'CarAgent'),
    },
    'active_buses': {
        'dtype': np.int16,
        'fn': lambda m: sum(1 for v in m.vehicles_list if v.__class__.__name__ == 'BusAgent'),
    },
    'bus_riders_waiting': {
        'dtype': np.int16,
        'fn': lambda m: len(m.at_bus_stop),
    },
    'bus_mode_share_recent': {
        'dtype': np.float32,
        'fn': _compute_bus_mode_share,
    },
    'total_finished': {
        'dtype': np.int32,
        'fn': lambda m: len(m.finished_agents),
    },
    'p_generate': {
        'dtype': np.float32,
        'fn': lambda m: m.p_generate,
    },
}

# Window-based scalar registry
# 'fn' entries are for scalars computed directly (like recent_travel_time_avg).
# Rolling count scalars use 'cumulative_fn' — the collector handles the lookback diff.
TIER1_WINDOW_SCALARS: Dict[str, Dict[str, Any]] = {
    'recent_travel_time_avg': {
        'dtype': np.float32,
        'fn': _compute_recent_travel_time,
    },
    'rolling_count_vehicles_generated': {
        'dtype': np.int32,
        'cumulative_fn': lambda m: m.car_counter + m.bus_counter,
    },
    'rolling_count_persons_generated': {
        'dtype': np.int32,
        'cumulative_fn': lambda m: m.person_counter,
    },
}

# Histogram metric registry — functions take the model and return an array of values
TIER1_HISTOGRAMS: Dict[str, Dict[str, Any]] = {
    'implicit_sl_delta': {
        'bins': np.array([-np.inf, -30, -20, -10, 0, np.inf]),
        'dtype': np.int16,
        'fn': lambda m: m.vs.speed_delta[:m.vs.n_active].copy() if m.vs.n_active > 0 else np.array([]),
    },
    'speed_mps': {
        'bins': np.array([0, 10, 20, 30, 40, np.inf]),
        'dtype': np.int16,
        'fn': lambda m: m.vs.speed[:m.vs.n_active].copy() if m.vs.n_active > 0 else np.array([]),
    },
}


class Tier1Collector:
    """Collects aggregate metrics every step using pre-allocated numpy arrays."""

    def __init__(self, config: HybridCollectorConfig):
        self.config = config
        self._write_idx = 0

        # Pre-allocate scalar arrays
        self.scalar_arrays: Dict[str, np.ndarray] = {}
        for name in config.tier1_scalars:
            if name in TIER1_SCALARS:
                dtype = TIER1_SCALARS[name]['dtype']
                self.scalar_arrays[name] = np.zeros(config.max_steps, dtype=dtype)

        # Pre-allocate window scalar output arrays + internal cumulative arrays
        self.window_scalar_arrays: Dict[str, np.ndarray] = {}
        self._cumulative_arrays: Dict[str, np.ndarray] = {}
        self._lookback = config.tier1_window_seconds // max(1, config.tier1_interval)
        for name in config.tier1_window_scalars:
            if name in TIER1_WINDOW_SCALARS:
                spec = TIER1_WINDOW_SCALARS[name]
                dtype = spec['dtype']
                self.window_scalar_arrays[name] = np.zeros(config.max_steps, dtype=dtype)
                if 'cumulative_fn' in spec:
                    self._cumulative_arrays[name] = np.zeros(config.max_steps, dtype=dtype)

        # Pre-allocate histogram arrays
        self.histogram_arrays: Dict[str, np.ndarray] = {}
        for name in config.tier1_histograms:
            if name in TIER1_HISTOGRAMS:
                n_bins = len(TIER1_HISTOGRAMS[name]['bins']) - 1
                dtype = TIER1_HISTOGRAMS[name]['dtype']
                # Shape: (max_steps, n_bins)
                self.histogram_arrays[name] = np.zeros(
                    (config.max_steps, n_bins), dtype=dtype
                )

    def collect(self, model) -> None:
        """Collect all configured metrics for current step."""
        interval = max(1, self.config.tier1_interval)
        if interval > 1 and (model.steps % interval != 0):
            return
        if self._write_idx >= self.config.max_steps:
            return

        idx = self._write_idx

        # Collect scalars
        for name, arr in self.scalar_arrays.items():
            fn = TIER1_SCALARS[name]['fn']
            arr[idx] = fn(model)

        # Collect window scalars
        lookback_idx = max(0, idx - self._lookback)
        for name, arr in self.window_scalar_arrays.items():
            spec = TIER1_WINDOW_SCALARS[name]
            if 'cumulative_fn' in spec:
                cum_arr = self._cumulative_arrays[name]
                cum_arr[idx] = spec['cumulative_fn'](model)
                arr[idx] = cum_arr[idx] - cum_arr[lookback_idx]
            else:
                arr[idx] = spec['fn'](model)

        # Collect histograms (vectorized reads from store)
        for name, arr in self.histogram_arrays.items():
            hist_config = TIER1_HISTOGRAMS[name]
            bins = hist_config['bins']
            fn = hist_config['fn']
            values = fn(model)
            if len(values) > 0:
                counts, _ = np.histogram(values, bins=bins)
                arr[idx] = counts.astype(np.int16)
            else:
                arr[idx] = 0

        self._write_idx += 1

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected data to DataFrame."""
        n = self._write_idx
        if n == 0:
            return pd.DataFrame()

        data = {}

        # Add scalar columns
        for name, arr in self.scalar_arrays.items():
            data[name] = arr[:n]

        # Add window scalar columns
        for name, arr in self.window_scalar_arrays.items():
            data[name] = arr[:n]

        # Add histogram columns (one column per bin)
        for name, arr in self.histogram_arrays.items():
            bins = TIER1_HISTOGRAMS[name]['bins']
            for i in range(arr.shape[1]):
                low = bins[i]
                high = bins[i + 1]
                col_name = f"{name}_bin_{i}"
                data[col_name] = arr[:n, i]

        return pd.DataFrame(data)


# =============================================================================
# Tier 2: Sampled Spatial Data
# =============================================================================

class Tier2Collector:
    """Collects spatial agent data at configurable intervals for animations."""

    # Encoding maps (matches kernel constants)
    STATUS_MAP = {'driving': 0, 'slowing': 1, 'crash': 2, 'canyon_closure': 3, 'arrived': 4}
    STATUS_DECODE = {v: k for k, v in STATUS_MAP.items()}

    ACTION_MAP = {
        'accelerate': 0, 'coast': 1, 'slow_accelerate': 2,
        'speed_limit_break': 3, 'smooth_break': 4, 'prevent_pass': 5,
    }
    ACTION_DECODE = {v: k for k, v in ACTION_MAP.items()}

    def __init__(self, config: HybridCollectorConfig):
        self.config = config
        self._write_idx = 0
        self._sample_interval = config.tier2_sample_interval

        # Pre-allocate arrays
        max_records = config.tier2_max_samples * config.tier2_max_agents_per_sample

        self.step = np.zeros(max_records, dtype=np.int32)
        self.agent_id = np.zeros(max_records, dtype=np.int32)
        self.agent_type = np.zeros(max_records, dtype=np.int8)  # 0=Car, 1=Bus
        self.pos_x = np.zeros(max_records, dtype=np.float32)
        self.pos_y = np.zeros(max_records, dtype=np.float32)
        self.status = np.zeros(max_records, dtype=np.int8)
        self.distance_traveled = np.zeros(max_records, dtype=np.float32)
        self.gap_m = np.zeros(max_records, dtype=np.float32)
        self.ideal_gap_m = np.zeros(max_records, dtype=np.float32)
        self.driving_action = np.zeros(max_records, dtype=np.int8)
        self.speed_mps = np.zeros(max_records, dtype=np.float32)

    def collect(self, model) -> None:
        """Collect spatial data if at sample interval. Reads from VehicleStore."""
        if model.steps % self._sample_interval != 0:
            return

        vs = model.vs
        n = vs.n_active
        if n == 0:
            return

        step_val = model.steps
        space_left = len(self.step) - self._write_idx
        n_write = min(n, space_left)
        if n_write <= 0:
            return

        i = self._write_idx
        j = i + n_write

        self.step[i:j] = step_val
        self.agent_id[i:j] = vs.slot_to_vid[:n_write]
        self.agent_type[i:j] = vs.veh_type[:n_write]
        self.pos_x[i:j] = vs.pos_x[:n_write].astype(np.float32)
        self.pos_y[i:j] = vs.pos_y[:n_write].astype(np.float32)
        self.status[i:j] = vs.status[:n_write]
        self.distance_traveled[i:j] = vs.dist[:n_write].astype(np.float32)
        self.gap_m[i:j] = np.minimum(vs.gap[:n_write], 9999.0).astype(np.float32)
        self.ideal_gap_m[i:j] = vs.ideal_gap[:n_write].astype(np.float32)
        self.driving_action[i:j] = vs.driving_action[:n_write]
        self.speed_mps[i:j] = vs.speed[:n_write].astype(np.float32)

        self._write_idx = j

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to animation-compatible DataFrame format."""
        n = self._write_idx
        if n == 0:
            return pd.DataFrame()

        # Create pos tuples
        pos_tuples = list(zip(self.pos_x[:n], self.pos_y[:n]))

        # Decode status and action back to strings
        status_strs = [self.STATUS_DECODE.get(s, 'driving') for s in self.status[:n]]
        action_strs = [self.ACTION_DECODE.get(a, 'coast') for a in self.driving_action[:n]]
        agent_types = ['BusAgent' if t == 1 else 'CarAgent' for t in self.agent_type[:n]]

        df = pd.DataFrame({
            'Step': self.step[:n],
            'AgentID': self.agent_id[:n],
            'AgentType': agent_types,
            'pos': pos_tuples,
            'status': status_strs,
            'distance_traveled': self.distance_traveled[:n],
            'gap_m': self.gap_m[:n],
            'ideal_gap_m': self.ideal_gap_m[:n],
            'driving_action': action_strs,
            'speed': self.speed_mps[:n] * 2.237,  # Convert to mph
            'speed_mps': self.speed_mps[:n],
        })

        return df


# =============================================================================
# Tier 3: Event Logging
# =============================================================================

class EventType(Enum):
    CRASH = "crash"
    CANYON_CLOSURE = "canyon_closure"


@dataclass
class TrafficEvent:
    """Represents a discrete event (crash, closure, etc.)."""
    event_type: str
    step: int

    # Location
    segment_index: Optional[int] = None
    distance_m: Optional[float] = None
    pos_x: Optional[float] = None
    pos_y: Optional[float] = None

    # Event-specific
    duration_sec: Optional[int] = None
    vehicles_on_road: Optional[int] = None


class Tier3Collector:
    """Collects discrete events like crashes and closures."""

    def __init__(self, config: HybridCollectorConfig):
        self.config = config
        self.events: List[TrafficEvent] = []

    def log_crash(self, model, segment_index: int, duration: int) -> None:
        """Log a crash event."""
        pos = model.rs_pos[segment_index] if segment_index < len(model.rs_pos) else (0, 0)
        distance = model.rs_distance[segment_index] if segment_index < len(model.rs_distance) else 0

        self.events.append(TrafficEvent(
            event_type=EventType.CRASH.value,
            step=model.steps,
            segment_index=segment_index,
            distance_m=float(distance),
            pos_x=float(pos[0]),
            pos_y=float(pos[1]),
            duration_sec=duration,
            vehicles_on_road=len(model.vehicles_list),
        ))

    def log_canyon_closure(self, model, segment_index: int, duration: int) -> None:
        """Log a canyon closure event."""
        self.events.append(TrafficEvent(
            event_type=EventType.CANYON_CLOSURE.value,
            step=model.steps,
            segment_index=segment_index,
            duration_sec=duration,
        ))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert events to DataFrame."""
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self.events])


# =============================================================================
# Tier 4: Full Snapshots (Optional)
# =============================================================================

class Tier4Collector:
    """Captures complete agent state at key moments for debugging."""

    def __init__(self, config: HybridCollectorConfig):
        self.config = config
        self.snapshots: List[Dict[str, Any]] = []

    def maybe_collect(self, model, trigger: str = "interval") -> None:
        """Collect snapshot if conditions are met."""
        if not self.config.tier4_enabled:
            return
        if len(self.snapshots) >= self.config.tier4_max_snapshots:
            return

        should_snap = False
        if trigger == "interval" and self.config.tier4_snapshot_interval > 0:
            should_snap = model.steps % self.config.tier4_snapshot_interval == 0
        elif trigger == "crash" and self.config.tier4_snapshot_on_crash:
            should_snap = True

        if not should_snap:
            return

        vs = model.vs
        n = vs.n_active
        status_decode = {0: 'driving', 1: 'slowing', 2: 'crash', 3: 'canyon_closure'}
        snapshot = {
            'step': model.steps,
            'trigger': trigger,
            'model_state': {
                'current_toll_car': model.current_toll_car,
                'bus_user_fee': model.bus_user_fee,
                'person_counter': model.person_counter,
                'car_counter': model.car_counter,
                'bus_counter': model.bus_counter,
                'total_crashes': model.total_crashes,
                'vehicles_count': len(model.vehicles_list),
            },
            'agents': [
                {
                    'agent_id': int(vs.slot_to_vid[s]),
                    'agent_type': 'CarAgent' if vs.veh_type[s] == 0 else 'BusAgent',
                    'pos': (float(vs.pos_x[s]), float(vs.pos_y[s])),
                    'speed': float(vs.speed[s]),
                    'status': status_decode.get(int(vs.status[s]), 'driving'),
                    'distance_traveled': float(vs.dist[s]),
                }
                for s in range(n)
            ],
        }
        self.snapshots.append(snapshot)


# =============================================================================
# Main HybridDataCollector
# =============================================================================

class HybridDataCollector:
    """
    4-tier hybrid data collection system for Mesa traffic simulation.

    Replaces Mesa's built-in DataCollector with a more efficient,
    pre-allocated numpy array approach.

    Usage:
        config = HybridCollectorConfig(tier2_sample_interval=10)
        collector = HybridDataCollector(config)

        # In model step:
        collector.collect(model)

        # For events:
        collector.log_crash(model, segment_index, duration)

        # Post-simulation:
        tier1_df = collector.get_tier1_dataframe()
        tier2_df = collector.get_tier2_dataframe()
        events_df = collector.get_events_dataframe()
    """

    def __init__(self, config: HybridCollectorConfig):
        self.config = config

        # Initialize tier collectors
        self.tier1 = Tier1Collector(config) if config.tier1_enabled else None
        self.tier2 = Tier2Collector(config) if config.tier2_enabled else None
        self.tier3 = Tier3Collector(config) if config.tier3_enabled else None
        self.tier4 = Tier4Collector(config) if config.tier4_enabled else None

    def collect(self, model) -> None:
        """Main collection method - call every step."""
        if self.tier1:
            self.tier1.collect(model)
        if self.tier2:
            self.tier2.collect(model)
        if self.tier4:
            self.tier4.maybe_collect(model, trigger="interval")

    def log_crash(self, model, segment_index: int, duration: int) -> None:
        """Log a crash event."""
        if self.tier3:
            self.tier3.log_crash(model, segment_index, duration)
        if self.tier4:
            self.tier4.maybe_collect(model, trigger="crash")

    def log_canyon_closure(self, model, segment_index: int, duration: int) -> None:
        """Log a canyon closure event."""
        if self.tier3:
            self.tier3.log_canyon_closure(model, segment_index, duration)

    # === Data Access Methods ===

    def get_tier1_dataframe(self) -> pd.DataFrame:
        """Get aggregate metrics as DataFrame."""
        if self.tier1:
            return self.tier1.to_dataframe()
        return pd.DataFrame()

    def get_tier2_dataframe(self) -> pd.DataFrame:
        """Get sampled spatial data (animation-compatible format)."""
        if self.tier2:
            return self.tier2.to_dataframe()
        return pd.DataFrame()

    def get_events_dataframe(self) -> pd.DataFrame:
        """Get event log as DataFrame."""
        if self.tier3:
            return self.tier3.to_dataframe()
        return pd.DataFrame()

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get full snapshots (Tier 4)."""
        if self.tier4:
            return self.tier4.snapshots
        return []

    # === Compatibility Methods (for existing code) ===

    def get_model_vars_dataframe(self) -> pd.DataFrame:
        """Compatibility: returns Tier 1 data in Mesa DataCollector format."""
        df = self.get_tier1_dataframe()
        if df.empty:
            return df

        # Rename columns for compatibility
        rename_map = {
            'vehicle_count': 'volume',
            'step': 'Step',
        }
        df = df.rename(columns=rename_map)
        return df

    def get_agent_vars_dataframe(self) -> pd.DataFrame:
        """Compatibility: returns Tier 2 data in Mesa DataCollector format."""
        df = self.get_tier2_dataframe()
        if df.empty:
            return df

        # Add implicit_sl_delta and posted_sl_delta for compatibility
        # These would need vehicle speed limit data, which we don't store in tier2
        # For now, just return what we have
        return df
