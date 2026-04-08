"""
Bus service cost estimation with TE-standard formulas.

All calculation is post-hoc — zero simulation overhead.
Uses only data already captured by the model (realized_tt, bus_interval, steps).

References:
    - Fleet sizing: TCRP Report 30, Human Transit (humantransit.org/02box)
    - Capital: FTA Default ULB 14 years, FTA Capital Cost Database
    - Labor: BLS OES 53-3052 (May 2023), TCRP scheduling guidance
    - Operations: NTD 2024 National Transit Summaries
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil


@dataclass
class BusCostConfig:
    """Bus service cost estimation parameters.

    All defaults are research-backed for a mountain canyon shuttle
    (Little Cottonwood Canyon / UTA ski bus context).
    """
    # Fleet sizing
    layover_recovery_pct: float = 0.10      # 10% of running time (TCRP Report 30)
    return_trip_factor: float = 1.0         # return leg as multiplier on one-way time
    spare_ratio: float = 0.20              # 20% spare fleet (FTA guidance)

    # Capital
    bus_purchase_price: float = 550_000.0   # midpoint 40-ft heavy-duty (FTA Capital Cost DB)
    useful_life_years: float = 14.0         # FTA Default ULB for heavy-duty bus

    # Labor
    driver_hourly_rate: float = 35.0        # fully loaded: BLS median $24 + ~45% fringe
    relief_factor: float = 1.17            # breaks, sick, vacation (TCRP scheduling)

    # Operations (non-labor: fuel + maintenance)
    ops_cost_per_hour: float = 40.0         # NTD non-labor portion of cost/revenue-hr

    # Overhead
    overhead_multiplier: float = 1.10       # 10% admin/insurance/facility

    # Time scaling
    service_hours_per_day: float = 12.0     # 6am-6pm standard ski bus window
    service_days_per_year: float = 150.0    # ~5 month ski season

    @staticmethod
    def default() -> "BusCostConfig":
        """Convenience: all research-backed defaults, no args needed."""
        return BusCostConfig()


def compute_bus_costs(
    config: BusCostConfig,
    avg_one_way_tt_min: float,
    headway_min: int,
    model_steps: int,
) -> dict | None:
    """Compute bus service cost metrics from post-run data.

    Returns dict with fleet sizing, hourly rates, and three cost metrics.
    Returns None if headway_min <= 0 (no bus service).
    """
    if headway_min <= 0:
        return None

    # Step 1: Fleet sizing
    cycle_time = avg_one_way_tt_min * (1 + config.return_trip_factor) * (1 + config.layover_recovery_pct)
    active_buses = ceil(cycle_time / headway_min)
    total_fleet = ceil(active_buses * (1 + config.spare_ratio))

    # Step 2: Hourly cost rates
    labor_per_hour = active_buses * config.relief_factor * config.driver_hourly_rate
    ops_per_hour = active_buses * config.ops_cost_per_hour
    capital_per_hour = (config.bus_purchase_price * total_fleet) / (
        config.useful_life_years * config.service_days_per_year * config.service_hours_per_day
    )
    total_per_hour = (labor_per_hour + ops_per_hour + capital_per_hour) * config.overhead_multiplier

    # Step 3: Three output metrics
    model_hours = model_steps / 3600.0
    model_run_cost = total_per_hour * model_hours
    daily_cost = total_per_hour * config.service_hours_per_day
    annual_cost = daily_cost * config.service_days_per_year

    return {
        # fleet
        "bus_cost_cycle_time_min": round(cycle_time, 1),
        "bus_cost_active_buses": active_buses,
        "bus_cost_total_fleet": total_fleet,
        # rates
        "bus_cost_per_hour": round(total_per_hour, 2),
        "bus_cost_labor_per_hour": round(labor_per_hour, 2),
        "bus_cost_ops_per_hour": round(ops_per_hour, 2),
        "bus_cost_capital_per_hour": round(capital_per_hour, 2),
        # three metrics
        "bus_cost_model_run": round(model_run_cost, 2),
        "bus_cost_daily": round(daily_cost, 2),
        "bus_cost_annual": round(annual_cost, 2),
    }
