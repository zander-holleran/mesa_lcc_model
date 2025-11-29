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

from season.configs import SeasonConfig


class SeasonOrchestrator:
    """Run a series of daily simulations and persist their outputs."""

    def __init__(self, season_config: SeasonConfig,  output_dir: str = "data/season_run"):
        # store season config
        self.config = season_config
        
        # define output directory for this season
        self.output_dir = Path(output_dir) / self.config.season_id
        print(f"Season outputs will be saved to: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # load road and ecs data once for the season
        self.road_gdf = gpd.read_parquet(self.config.road_path)
        self.ecs_df = pd.read_csv(self.config.ecs_path)

        # set up RNG
        self.rng = np.random.default_rng(self.config.seed)

        self.season_persons = self.config.population_params.create_season_persons(
            season_id=self.config.season_id,
            seed=self.config.seed,   
        )

    def run_season(self):    
        # this is the main run season command for now there are no season function other than running days in sequence



        for day_cfg in self.config.day_params:
            # 0) set up paths and day index
            day_index = day_cfg.day_index
            prefix = f"day_{day_index}"
            model_ts_path = self.output_dir / f"{prefix}_model_ts.parquet"

           
             # UPDATE BELIEFS: for all persons at the start of the day
            for person in self.season_persons:
                person.update_beliefs_from_history(current_day=day_index)

            # MODEL RUN: define and run the model for this day
            tm = self._build_model(day_cfg=day_cfg)
            tm.run_model()

            #DATA HANDLING:
            # after model run, update each person's history with realized experiences from this day
            # for agent in tm.person_agents:
            #     season_person = self.season_persons[agent.person_id]
            #     season_person.record_experience(day_index=day_index,  mode=agent.chosen_mode, realized_tt=agent.total_travel_time)


            #collect and save outputs 
            model_ts = au.model_data_time_series(tm)
            model_ts.to_parquet(model_ts_path)


            

          


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
            # person data from season orchestrator
            season_persons=self.season_persons,
            # day specific parameters
            seed=day_cfg.day_seed,
            traffic_percentile=day_cfg.traffic_percentile,
            bus_interval=day_cfg.bus_interval,
            crashes_per_100k_vmt_input=day_cfg.crashes_per_100k_vmt_input,
            canyon_closures=day_cfg.canyon_closures,
            current_day= day_cfg.day_index,

            # irrelevant for season runs
            p_generate=None,
            batchrun=True,
            collect_every_n=999999,  # effectively disable intermediate collection
            car_preference=1,
        )

   