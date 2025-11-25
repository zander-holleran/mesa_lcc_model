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


class DayConfig(Protocol):
    """Interface describing the per-day configuration already produced elsewhere."""

    day_index: int
    traffic_percentile: int | None
    bus_interval: int
    canyon_closures: Optional[Dict[str, Any]]
    crashes_per_100k_vmt_input: float | None


class SeasonConfig(Protocol):
    """Lightweight interface for existing season configs, preserving their builders.

    ``Protocol`` lets us type-hint the attributes the orchestrator relies on without
    redefining (or constraining) the real config objects that are created elsewhere.
    """

    season_id: str
    seed: int
    max_steps: int
    start_hr: int
    max_persons: int
    bus_capacity: int
    road_gdf_path: str | Path
    ecs_df_path: str | Path
    day_params: List[DayConfig]

    def validate(self) -> None:
        ...


@dataclass
class SeasonRunResult:
    season_config: SeasonConfig
    day_file_log: pd.DataFrame  # one row per day, with file paths


class SeasonOrchestrator:
    """Run a series of daily simulations and persist their outputs."""

    def __init__(
        self,
        season_config: SeasonConfig,
        season_persons=None,
        output_dir: str = "season_runs",
    ):
        if hasattr(season_config, "validate"):
            season_config.validate()

        self.config = season_config
        self.road_gdf = self._load_road_gdf()
        self.ecs_df = self._load_ecs_df()
        self.season_persons = season_persons

        self.output_dir = Path(output_dir) / season_config.season_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(self.config.seed)

    def run(self, overwrite: bool = True, progress: bool = False) -> SeasonRunResult:
        """Run all days and save per-day outputs to disk.

        Args:
            overwrite: If False, skip days whose parquet outputs already exist.
            progress: If True, wrap the day iterator with ``tqdm`` for a progress bar.
        """
        day_rows: List[Dict[str, Any]] = []

        day_iterator: Iterable[DayConfig] = self.config.day_params
        if progress:
            try:
                from tqdm import tqdm

                day_iterator = tqdm(day_iterator, desc="Running season days")
            except ImportError:
                # Fall back gracefully if tqdm isn't installed
                pass

        for day_cfg in day_iterator:
            day_index = day_cfg.day_index
            day_seed = self.config.seed + day_index

            prefix = f"day_{day_index:03d}"
            vehicles_path = self.output_dir / f"{prefix}_vehicles_full.parquet"
            model_ts_path = self.output_dir / f"{prefix}_model_ts.parquet"
            finished_path = self.output_dir / f"{prefix}_finished_agents.parquet"

            if not overwrite and vehicles_path.exists() and model_ts_path.exists() and finished_path.exists():
                # When ``overwrite`` is False we re-use existing parquet files as-is;
                # no new data is appended and nothing is rewritten.
                day_rows.append(
                    self._log_day_row(
                        day_cfg=day_cfg,
                        vehicles_path=vehicles_path,
                        model_ts_path=model_ts_path,
                        finished_path=finished_path,
                    )
                )
                continue

            tm = self._build_model(day_cfg=day_cfg, seed=day_seed)
            self._run_model(tm)

            vehicles_full = au.vehicle_agent_data_time_series(tm, plots=False)
            model_ts = au.model_data_time_series(tm)
            finished_agents = au.finished_agents_summary_df(tm, plots=False)

            vehicles_full.to_parquet(vehicles_path)
            model_ts.to_parquet(model_ts_path)
            finished_agents.to_parquet(finished_path)

            day_rows.append(
                self._log_day_row(
                    day_cfg=day_cfg,
                    vehicles_path=vehicles_path,
                    model_ts_path=model_ts_path,
                    finished_path=finished_path,
                )
            )

        day_file_log = pd.DataFrame(day_rows)

        return SeasonRunResult(
            season_config=self.config,
            day_file_log=day_file_log,
        )

    def _build_model(self, day_cfg: DayConfig, seed: int) -> TrafficModel:
        """Create a ``TrafficModel`` instance for a single day."""
        return TrafficModel(
            road_gdf=self.road_gdf,
            ecs_df=self.ecs_df,
            season_persons=self.season_persons,
            max_steps=self.config.max_steps,
            seed=seed,
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

    def _load_road_gdf(self):
        """Load the road geometry using the path embedded in the season config."""
        road_path = Path(self.config.road_gdf_path)
        return gpd.read_parquet(road_path)

    def _load_ecs_df(self):
        """Load the expected counts data using the path embedded in the season config."""
        ecs_path = Path(self.config.ecs_df_path)
        if ecs_path.suffix.lower() == ".csv":
            return pd.read_csv(ecs_path)
        return pd.read_parquet(ecs_path)

    def _run_model(self, tm: TrafficModel) -> None:
        """Advance the model until completion."""
        if hasattr(tm, "run_model"):
            tm.run_model()
        else:
            while tm.schedule.steps < self.config.max_steps:
                tm.step()

    def _log_day_row(
        self,
        day_cfg: DayConfig,
        vehicles_path: Path,
        model_ts_path: Path,
        finished_path: Path,
    ) -> Dict[str, Any]:
        """Build a row summarizing a day's configuration and output paths."""
        return dict(
            season_id=self.config.season_id,
            day_index=day_cfg.day_index,
            traffic_percentile=day_cfg.traffic_percentile,
            bus_interval=day_cfg.bus_interval,
            crashes_per_100k_vmt_input=day_cfg.crashes_per_100k_vmt_input,
            vehicles_full_path=str(vehicles_path),
            model_ts_path=str(model_ts_path),
            finished_agents_path=str(finished_path),
        )
