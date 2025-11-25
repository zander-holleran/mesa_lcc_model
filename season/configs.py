from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Callable
import numpy as np


@dataclass
class DayParams:
    day_index: int

    # these map directly to TrafficModel's day-varying args
    traffic_percentile: Optional[float]
    bus_interval: int
    crashes_per_100k_vmt_input: float
    canyon_closures: Optional[dict] = None  # keep simple for now

    # optional: for tolls once you wire them into TrafficModel
    toll_mechanism: Optional[str] = None
    toll_params: Optional[Dict[str, Any]] = None

    def to_model_kwargs(self) -> Dict[str, Any]:
        """Args you pass into TrafficModel for this day."""
        return dict(
            traffic_percentile=self.traffic_percentile,
            canyon_closures=self.canyon_closures or {},
            bus_interval=self.bus_interval,
            crashes_per_100k_vmt_input=self.crashes_per_100k_vmt_input,
            # later: toll_mechanism / toll_params if you add them to __init__
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
    road_source: Optional[str] = None
    ecs_source: Optional[str] = None

    # toll mechanism for the whole season (can encode pigouvian vs static here)
    toll_mechanism: Optional[str] = None
    toll_params: Dict[str, Any] = field(default_factory=dict)

    # placeholder: canyon_closures defined at season-level if you want
    canyon_closures: Any = None  # as you said, more complex type later

    # the actual per-day parameters
    day_params: List[DayParams] = field(default_factory=list)



ScheduleMode = Literal["static", "dist"]

@dataclass
class ScheduleSpecs:
    mode: ScheduleMode
    value: Optional[Any] = None       # used when mode == "static"
    dist: Optional[Any] = None        # used when mode == "dist" (e.g. scipy.stats.norm)

    def realize(self, rng: np.random.Generator, n_days: int):
        if self.mode == "static":
            return [self.value] * n_days

        # happy path: mode == "dist", dist is a scipy frozen dist with .rvs
        draws = self.dist.rvs(size=n_days, random_state=rng)
        return list(draws)




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
    road_source: str = None,
    ecs_source: str= None,

    # schedules for day-varying TrafficModel args
    traffic_percentile_schedule: ScheduleSpecs = ScheduleSpecs("static", 50),
    bus_interval_schedule: ScheduleSpecs = ScheduleSpecs("static", 30),
    crashes_schedule: ScheduleSpecs = ScheduleSpecs("static", 4),

    # canyon closures: leave as None for now as you requested
    canyon_closures=None,

    # toll mechanism
    toll_mechanism: Optional[str] = None,
    toll_params: Optional[Dict[str, Any]] = {"car": 0.0, "bus": 0.0},
) -> SeasonConfig:
    rng = np.random.default_rng(seed)

    # realize schedules
    traffic_percentiles = traffic_percentile_schedule.realize(rng, n_days)
    bus_intervals = bus_interval_schedule.realize(rng, n_days)
    crashes = crashes_schedule.realize(rng, n_days)

    # assemble per-day parameters: this is a list of DayParams objects
    day_params: List[DayParams] = []
    for d in range(n_days):
        day_params.append(
            DayParams(
                day_index=d,
                traffic_percentile=traffic_percentiles[d],
                bus_interval=int(bus_intervals[d]),
                crashes_per_100k_vmt_input=float(crashes[d]),
                canyon_closures=None,  # placeholder; later override per day if needed
                toll_mechanism=toll_mechanism,
                toll_params=toll_params or {},
            )
        )

    return SeasonConfig(
        season_id=season_id,
        run_description=run_description,
        seed=seed,
        n_days=n_days,
        max_steps=max_steps,
        max_persons=max_persons,
        start_hr=start_hr,
        bus_capacity=bus_capacity,
        road_source=road_source,
        ecs_source=ecs_source,
        toll_mechanism=toll_mechanism,
        toll_params=toll_params or {},
        canyon_closures=canyon_closures,
        day_params=day_params,
    )








