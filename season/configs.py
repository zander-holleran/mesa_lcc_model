from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Callable
import numpy as np


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

@dataclass
class SeasonConfig:
    season_id: str
    run_description: str
    seed: int
    n_days: int

    # TrafficModel season-level args
    max_steps: int = 50000
    max_persons: int = 50
    start_hr: int = 5
    bus_capacity: int = 30

    # we keep road/ecs as *references* here (paths, version names, etc.),
    # not the full GeoDataFrames
    road_path: Optional[str] = None
    ecs_path: Optional[str] = None

    # toll mechanism for the whole season (can encode pigouvian vs static here)
    toll_mechanism: Optional[str] = None
    toll_params: Dict[str, Any] = field(default_factory=dict)

    # the actual per-day parameters
    day_params: List[DayParams] = field(default_factory=list)



ScheduleMode = Literal["static", "dist"] # makes it such that ScheduleSpecs only accepts these two strings as mode

@dataclass
class ScheduleSpecs:
    mode: ScheduleMode
    value:Optional[int] = None       # used when mode == "static"
    dist: Optional[Any] = None        # used when mode == "dist" (e.g. scipy.stats.norm)
    round_to_int: bool = True              # whether to round the draws to integers

    def realize(self, rng: np.random.Generator, n_days: int):
        if self.mode == "static":
            return [self.value] * n_days

        # happy path: mode == "dist", dist is a scipy frozen dist with .rvs
        draws = self.dist.rvs(size=n_days, random_state=rng)
        if self.round_to_int:
            rounded = np.round(draws).astype(int)
        return list(rounded)



def make_season_config(
    *,
    season_id: str,
    run_description: str,
    seed: int,
    n_days: int,

    # season-level TrafficModel settings
    max_steps: int = 50000,
    max_persons: int = 50,
    start_hr: int = 5,
    bus_capacity: int = 30,

    # references to GeoDataFrames (paths or identifiers)
    road_path: str = None,
    ecs_path: str= None,

    # schedules for day-varying TrafficModel args
    traffic_percentile_schedule: ScheduleSpecs = ScheduleSpecs("static", 50),
    bus_interval_schedule: ScheduleSpecs = ScheduleSpecs("static", 30),
    crashes_schedule: ScheduleSpecs = ScheduleSpecs("static", 4),
    canyon_closures_schedule: Optional[Any] = None,

    # toll mechanism
    toll_mechanism: Optional[str] = None,
    toll_params: Optional[Dict[str, Any]] = {"car": 0.0, "bus": 0.0},
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
        # season level TrafficModel args
        max_steps=max_steps,
        max_persons=max_persons,
        start_hr=start_hr,
        bus_capacity=bus_capacity,
        road_path=road_path,
        ecs_path=ecs_path,
        toll_mechanism=toll_mechanism,
        toll_params=toll_params or {},
        # per-day parameters sent to TrafficModel
        day_params=day_params
    )








