from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Callable
import numpy as np
from season.persons import SeasonPerson
from scipy.stats import lognorm, skewnorm, norm
from traffic.model.hybrid_collector import HybridCollectorConfig
from traffic.model.tolling import TollConfig

# ===================== The following are imputs to SeasonConfig ====================== #
@dataclass
class PopulationParams:
    population_size: int = 500

    # each of these can be a scalar or a frozen scipy dist
    value_of_time: Any = lognorm(s=0.64 , scale=(40/60) ) # $/minute
    experience_weight_car: Any = 1.0
    experience_weight_bus: Any = skewnorm(6, loc=1.15, scale=0.3)

    prior_car: Any = 22.0
    prior_bus: Any = 60.0

    time_decay_rate: Any = 0.1
    prior_weight: Any = 1.0
    uncertainty_multiplier: Any = 1.0
    travel_propensity: Any = 1.0

    def _draw_field(self, spec, rng: np.random.Generator, n: int):
        """If spec has .rvs, draw n samples; else repeat scalar n times."""
        if hasattr(spec, "rvs"):
            draws = spec.rvs(size=n, random_state=rng)
            return list(draws)
        else:
            return [spec] * n

    def create_season_persons(self, season_id: str, seed: Optional[int] = None) -> List[SeasonPerson]:
        """
        Generate a list of SeasonPerson objects according to this population spec.
        """
        rng = np.random.default_rng(seed)

        n = self.population_size

        vots = self._draw_field(self.value_of_time, rng, n)
        w_car = self._draw_field(self.experience_weight_car, rng, n)
        w_bus = self._draw_field(self.experience_weight_bus, rng, n)

        prior_car_vals = self._draw_field(self.prior_car, rng, n)
        prior_bus_vals = self._draw_field(self.prior_bus, rng, n)

        t_decay = self._draw_field(self.time_decay_rate, rng, n)
        p_weight = self._draw_field(self.prior_weight, rng, n)
        u_mult = self._draw_field(self.uncertainty_multiplier, rng, n)
        travel_prop = self._draw_field(self.travel_propensity, rng, n)

        persons: List[SeasonPerson] = []

        for i in range(n):
            persons.append(
                SeasonPerson(
                    person_id=i,
                    season_id=season_id,
                    value_of_time=float(vots[i]),
                    experience_weight_car=float(w_car[i]),
                    experience_weight_bus=float(w_bus[i]),
                    prior_car=float(prior_car_vals[i]),
                    prior_bus=float(prior_bus_vals[i]),
                    time_decay_rate=float(t_decay[i]),
                    prior_weight=float(p_weight[i]),
                    uncertainty_multiplier=float(u_mult[i]),
                    travel_propensity=float(travel_prop[i])
                )
            )
        return persons

@dataclass
class DayParams:
    day_index: int
    day_seed: int
    # these map directly to TrafficModel's day-varying args
    traffic_percentile: float
    bus_interval: int
    crashes_per_100k_vmt_input: float
    canyon_closures: Optional[dict] = None  # keep simple for now

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Args you pass into TrafficModel for this day."""
        return dict(
            traffic_percentile=self.traffic_percentile,
            canyon_closures=self.canyon_closures or {},
            bus_interval=self.bus_interval,
            crashes_per_100k_vmt_input=self.crashes_per_100k_vmt_input,
        )


ScheduleMode = Literal["static", "dist", "list"] # makes it such that ScheduleSpecs only accepts these strings as mode

@dataclass
class ScheduleSpecs:
    mode: ScheduleMode
    value: Optional[Any] = None       # used when mode == "static" (scalar) or "list" (list of values)
    dist: Optional[Any] = None        # used when mode == "dist" (e.g. scipy.stats.norm)
    round_to_int: bool = True         # whether to round the draws to integers

    def realize(self, rng: np.random.Generator = np.random.default_rng(123), n_days: int=1):
        if self.mode == "static":
            return [self.value] * n_days

        if self.mode == "list":
            if len(self.value) != n_days:
                raise ValueError(f"ScheduleSpecs list has {len(self.value)} values but n_days={n_days}")
            return list(self.value)

        # happy path: mode == "dist", dist is a scipy frozen dist with .rvs
        draws = self.dist.rvs(size=n_days, random_state=rng)
        if self.round_to_int:
            rounded = np.round(draws).astype(int)
            return list(rounded)
        else:
            return list(draws)


#===================== Data Classes used by  make_season_config ====================== #
# SeasonConfig is the main config object for a Season run - it is the output of make_season_config
@dataclass
class SeasonConfig:
    season_id: str
    run_description: str
    seed: int
    n_days: int
    batch_run: bool = True

    # TrafficModel season-level args
    max_steps: int = 50000
    max_persons: int = 50
    collect_every_n: int = 10  
    start_hr: int = 5
    bus_capacity: int = 30

    # we keep road/ecs as *references* here (paths, version names, etc.),
    # not the full GeoDataFrames
    road_path: Optional[str] = None
    ecs_path: Optional[str] = None

    # toll configuration for the whole season
    toll_config: TollConfig = field(default_factory=lambda: TollConfig.static(car=0.0))
    bus_user_fee: float = 0.0

    # the actual per-day parameters
    day_params: List[DayParams] = field(default_factory=list)
    
    # population parameters
    population_params: PopulationParams = field(default_factory=PopulationParams)
    
    # optional hybrid collector config for TrafficModel
    hybrid_collector_config: Optional[HybridCollectorConfig] = None


# ====================== Factory function to create SeasonConfig objects ====================== #
# This is the final user-facing function. It creates a SeasonConfig object from high-level specs.
def make_season_config(
    *,
    season_id: str,
    run_description: str,
    seed: int,
    n_days: int,
    batch_run: bool = True,

    # season-level TrafficModel settings
    max_steps: int = 99999,
    max_persons: int = 99999,
    collect_every_n: int = 10,
    start_hr: int = 7,
    bus_capacity: int = 60,

    # references to GeoDataFrames (paths or identifiers)
    road_path: str = 'data/roads/hw210_sl_and_curvs.parquet',
    ecs_path: str= 'data/vehicle_counts/expected_counts_seconds.csv',

    # schedules for day-varying TrafficModel args
    traffic_percentile_schedule: ScheduleSpecs = ScheduleSpecs("static", 50),
    bus_interval_schedule: ScheduleSpecs = ScheduleSpecs("static", 30),
    crashes_schedule: ScheduleSpecs = ScheduleSpecs("static", 0),
    canyon_closures_schedule: Optional[Any] = None,

    # toll configuration
    toll: TollConfig = None,
    bus_user_fee: float = 0.0,

    population_params: PopulationParams = None,

    hybrid_collector_config: HybridCollectorConfig = None,

) -> SeasonConfig:
    rng = np.random.default_rng(seed)

    # create the per-day realizations 
    traffic_percentiles = traffic_percentile_schedule.realize(rng, n_days)
    bus_intervals = bus_interval_schedule.realize(rng, n_days)
    crashes = crashes_schedule.realize(rng, n_days)
    canyon_closures = canyon_closures_schedule if canyon_closures_schedule is not None else [None]*n_days
    day_seeds = [seed + i for i in range(n_days)]

    # assemble per-day parameters: this is a list of DayParams objects
    day_params = []
    for d in range(n_days):
        day_params.append(
            DayParams(
                day_index=d,
                day_seed=day_seeds[d], 
                traffic_percentile=traffic_percentiles[d],
                bus_interval=int(bus_intervals[d]),
                crashes_per_100k_vmt_input=float(crashes[d]),
                canyon_closures=canyon_closures[d]
            )
        )

    return SeasonConfig(
        # season-level settings
        season_id=season_id,
        run_description=run_description,
        seed=seed,
        n_days=n_days,
        batch_run=batch_run,
        # season level TrafficModel args
        max_steps=max_steps,
        max_persons=max_persons,
        collect_every_n=collect_every_n,
        start_hr=start_hr,
        bus_capacity=bus_capacity,
        road_path=road_path,
        ecs_path=ecs_path,
        toll_config=toll if toll is not None else TollConfig.static(car=0.0),
        bus_user_fee=bus_user_fee,
        # per-day parameters sent to TrafficModel
        day_params=day_params,
        population_params=population_params,
        hybrid_collector_config=hybrid_collector_config
    )


def summarize_config(config: SeasonConfig, high_only: bool = True) -> None:
    """Print a readable summary of a SeasonConfig."""

    lines = []
    b = lambda s: f"\033[1m{s}\033[0m"  # bold

    # ── High priority ──
    lines.append(b("── Season Config ──"))

    days = config.n_days
    pop = config.population_params.population_size if config.population_params else "?"
    lines.append(f"  {b('Days:')} {days}   {b('Population:')} {pop:,}")

    # traffic percentile per day
    percs = [dp.traffic_percentile for dp in config.day_params]
    if len(set(percs)) == 1:
        lines.append(f"  {b('Traffic percentile:')} {percs[0]} (all days)")
    else:
        lines.append(f"  {b('Traffic percentile:')} {percs}")

    # bus interval per day
    bus_ints = [dp.bus_interval for dp in config.day_params]
    if len(set(bus_ints)) == 1:
        val = bus_ints[0]
        lines.append(f"  {b('Bus interval:')} {'off' if val == 0 else f'{val} min'}   {b('Bus capacity:')} {config.bus_capacity}")
    else:
        lines.append(f"  {b('Bus interval:')} {bus_ints} min   {b('Bus capacity:')} {config.bus_capacity}")

    # toll summary
    tc = config.toll_config
    if tc._static_toll is not None:
        toll_desc = f"static ${tc._static_toll:.2f}"
    elif tc.transform is not None:
        t = tc.transform
        tname = type(t).__name__.replace("Transform", "")
        if hasattr(t, 'target'):
            toll_desc = f"{tname} → target {t.target}, range ${t.toll_min}–${t.toll_max}"
        elif hasattr(t, 'threshold'):
            toll_desc = f"{tname}, threshold={t.threshold}, slope={getattr(t, 'slope', '?')}"
        else:
            toll_desc = tname
        if tc.rounding:
            toll_desc += f", rounded to ${tc.rounding:.2f}"
    else:
        toll_desc = "none"
    lines.append(f"  {b('Toll:')} {toll_desc}")

    # ── Conditional high ──
    crashes = [dp.crashes_per_100k_vmt_input for dp in config.day_params]
    if any(c > 0 for c in crashes):
        if len(set(crashes)) == 1:
            lines.append(f"  {b('Crashes:')} {crashes[0]} per 100k VMT")
        else:
            lines.append(f"  {b('Crashes:')} {crashes} per 100k VMT")

    closures = [dp.canyon_closures for dp in config.day_params]
    if any(c is not None for c in closures):
        n_with = sum(1 for c in closures if c is not None)
        lines.append(f"  {b('Canyon closures:')} configured on {n_with}/{days} days")

    if config.bus_user_fee > 0:
        lines.append(f"  {b('Bus user fee:')} ${config.bus_user_fee:.2f}")

    # ── Low priority ──
    if not high_only:
        lines.append("")
        lines.append(b("── Details ──"))
        lines.append(f"  {b('Start hour:')} {config.start_hr}:00")

        pp = config.population_params
        if pp:
            vot = pp.value_of_time
            vot_str = f"lognorm (median ~${vot.median():.2f}/min)" if hasattr(vot, 'median') else f"${vot:.2f}/min"
            lines.append(f"  {b('Value of time:')} {vot_str}")
            lines.append(f"  {b('Priors:')} car={pp.prior_car}, bus={pp.prior_bus}")
            lines.append(f"  {b('Belief params:')} decay={pp.time_decay_rate}, prior_wt={pp.prior_weight}, uncertainty_mult={pp.uncertainty_multiplier}")

        hc = config.hybrid_collector_config
        if hc:
            parts = []
            if hc.tier1_enabled:
                parts.append(f"Tier 1 (scalars+histograms every {hc.tier1_interval} steps)")
            if hc.tier2_enabled:
                parts.append(f"Tier 2 (spatial snapshots every {hc.tier2_sample_interval} steps, up to {hc.tier2_max_agents_per_sample} agents)")
            if hc.tier3_enabled:
                parts.append("Tier 3 (crash & closure event log)")
            if hc.tier4_enabled:
                t4 = []
                if hc.tier4_snapshot_interval > 0:
                    t4.append(f"every {hc.tier4_snapshot_interval} steps")
                if hc.tier4_snapshot_on_crash:
                    t4.append("on crash")
                parts.append(f"Tier 4 (full snapshots {' & '.join(t4)}, max {hc.tier4_max_snapshots})")
            if parts:
                lines.append(f"  {b('Data collection:')}")
                for p in parts:
                    lines.append(f"    • {p}")
            else:
                lines.append(f"  {b('Data collection:')} all tiers disabled")

    print("\n".join(lines))
