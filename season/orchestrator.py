# season/orchestrator.py

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

import numpy as np
import pandas as pd

from season.configs import SeasonConfig
from traffic.model.traffic_model import TrafficModel  # adjust import
import traffic.utils.analysis_utils as au             # adjust name to your module



@dataclass
class SeasonRunResult:
    season_config: SeasonConfig
    day_file_log: pd.DataFrame  # one row per day, with file paths


class SeasonOrchestrator:
    def __init__(
        self,
        season_config: SeasonConfig,
        road_gdf,
        ecs_df,
        season_persons=None,
        output_dir: str = "season_runs",
    ):
        self.config = season_config
        self.road_gdf = road_gdf
        self.ecs_df = ecs_df
        self.season_persons = season_persons

        self.output_dir = Path(output_dir) / season_config.season_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(self.config.seed)

    def run(self) -> SeasonRunResult:
        """Run all days and save per-day outputs to disk."""
        day_rows: List[Dict[str, Any]] = []

        for day_cfg in self.config.day_params:
            day_index = day_cfg.day_index
            day_seed = self.config.seed + day_index

            # --- build TrafficModel for this day ---
            tm = TrafficModel(
                road_gdf=self.road_gdf,
                ecs_df=self.ecs_df,
                season_persons=self.season_persons,
                max_steps=self.config.max_steps,
                seed=day_seed,
                batchrun=False,
                collect_every_n=1,
                start_hr=self.config.start_hr,
                traffic_percentile=day_cfg.traffic_percentile,
                p_generate=None,
                max_persons=self.config.max_persons,
                canyon_closures=day_cfg.canyon_closures or {},
                bus_interval=day_cfg.bus_interval,
                car_preference=1,
                bus_capacity=self.config.bus_capacity,
                crashes_per_100k_vmt_input=day_cfg.crashes_per_100k_vmt_input,
            )

            # --- run the model for this day ---
            if hasattr(tm, "run_model"):
                tm.run_model()
            else:
                while tm.schedule.steps < self.config.max_steps:
                    tm.step()

            # --- collect outputs with plotting turned OFF ---
            vehicles_full = au.vehicle_agent_data_time_series(tm, plots=False)
            model_ts = au.model_data_time_series(tm)
            finished_agents = au.finished_agents_summary_df(tm, plots=False)

            # --- save per-day files ---
            prefix = f"day_{day_index:03d}"
            vehicles_path = self.output_dir / f"{prefix}_vehicles_full.parquet"
            model_ts_path = self.output_dir / f"{prefix}_model_ts.parquet"
            finished_path = self.output_dir / f"{prefix}_finished_agents.parquet"

            vehicles_full.to_parquet(vehicles_path)
            model_ts.to_parquet(model_ts_path)
            finished_agents.to_parquet(finished_path)

            # optional: log paths + key inputs
            day_rows.append(
                dict(
                    season_id=self.config.season_id,
                    day_index=day_index,
                    traffic_percentile=day_cfg.traffic_percentile,
                    bus_interval=day_cfg.bus_interval,
                    crashes_per_100k_vmt_input=day_cfg.crashes_per_100k_vmt_input,
                    vehicles_full_path=str(vehicles_path),
                    model_ts_path=str(model_ts_path),
                    finished_agents_path=str(finished_path),
                )
            )

        day_file_log = pd.DataFrame(day_rows)

        return SeasonRunResult(
            season_config=self.config,
            day_file_log=day_file_log,
        )
