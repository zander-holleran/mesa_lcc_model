"""Utilities for orchestrating multi-day traffic model runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd

from traffic.model.traffic_model import TrafficModel
from traffic.utils import analysis_utils as au

from season.configs import SeasonConfig, DayParams


class SeasonOrchestrator:
    """Run a series of daily simulations and persist their outputs."""

    def __init__(self, season_config: SeasonConfig,  output_dir: str = "data/season_run"):
        # define output directory for this season
        self.output_dir = Path(output_dir) / season_config.season_id
        print(f"Season outputs will be saved to: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # store season config
        self.config = season_config

        # load road and ecs data once for the season
        self.road_gdf = gpd.read_parquet(season_config.road_path)
        self.ecs_df = pd.read_csv(season_config.ecs_path)

        # set up RNG
        self.rng = np.random.default_rng(self.config.seed)

    def run_season(self):    
        # this is the main run season command for now there are no season function other than running days in sequence

        for day_cfg in self.config.day_params:
            # used for writing correct names of output files 
            day_index = day_cfg.day_index
           
            # define output paths for each file
            prefix = f"day_{day_index}"
            vehicles_path = self.output_dir / f"{prefix}_vehicles_full.parquet"
            model_ts_path = self.output_dir / f"{prefix}_model_ts.parquet"
            finished_path = self.output_dir / f"{prefix}_finished_agents.parquet"

            # define and run the model for this day
            tm = self._build_model(day_cfg=day_cfg)
            tm.run_model()

            #collect and save outputs 
            #vehicles_full = au.vehicle_agent_data_time_series(tm, plots=False)
            model_ts = au.model_data_time_series(tm)
            #finished_agents = au.finished_agents_summary_df(tm, plots=False)

            #vehicles_full.to_parquet(vehicles_path)
            model_ts.to_parquet(model_ts_path)
            #finished_agents.to_parquet(finished_path)

          


    def _build_model(self, day_cfg) -> TrafficModel:
        """Create a ``TrafficModel`` instance for a single day."""
        return TrafficModel(
            # perams from season orchestrator init
            road_gdf=self.road_gdf, #road_gdf ecs_df dont come from day config because the data is looaded in the season orcestrator, the path is fassed from config
            ecs_df=self.ecs_df,
            # season level parameters from config
            max_steps=self.config.max_steps,
            max_persons=self.config.max_persons,
            start_hr=self.config.start_hr,
            bus_capacity=self.config.bus_capacity,

            # day specific parameters
            seed=day_cfg.day_seed,
            traffic_percentile=day_cfg.traffic_percentile,
            bus_interval=day_cfg.bus_interval,
            crashes_per_100k_vmt_input=day_cfg.crashes_per_100k_vmt_input,
            canyon_closures=day_cfg.canyon_closures,
            
            # irrelevant for season runs
            p_generate=None,
            batchrun=True,
            collect_every_n=999999,  # effectively disable intermediate collection
            car_preference=1,

            # will return to this concept
            season_persons=None, # none for now, can be added later
        )

    # def _load_road_gdf(self):
    #     """Load the road geometry using the path embedded in the season config."""
    #     road_path = Path(self.config.road_path)
    #     return gpd.read_parquet(road_path)

    # def _load_ecs_df(self):
    #     """Load the expected counts data using the path embedded in the season config."""
    #     ecs_path = self.config.ecs_path
    #     return pd.read_csv(ecs_path)

    