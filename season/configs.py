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
    max_steps: int = 500,
    max_persons: int = 50,
    collect_every_n: int = 10,
    start_hr: int = 5,
    bus_capacity: int = 30,

    # references to GeoDataFrames (paths or identifiers)
    road_path: str = None,
    ecs_path: str= None,

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
