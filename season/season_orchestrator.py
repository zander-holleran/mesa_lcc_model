"""Utilities for orchestrating multi-day traffic model runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import warnings


from traffic.model.traffic_model import TrafficModel
from traffic.utils import analysis_utils as au

from season.configs import SeasonConfig


class SeasonOrchestrator:
    """Run a series of daily simulations and persist their outputs."""

    def __init__(self, season_config: SeasonConfig,  store_data: bool = False , output_dir: str = "data/season_outputs"):
        self.config = season_config

        self.road_gdf = gpd.read_parquet(self.config.road_path)
        self.ecs_df = pd.read_csv(self.config.ecs_path)

        self.season_persons = self.config.population_params.create_season_persons(
            season_id=self.config.season_id,
            seed=self.config.seed,   
        )
        
        self.rng = np.random.default_rng(self.config.seed)

        # define output directory for this season
        self.store_data = store_data
        self.output_dir = None
        if store_data:
            self.output_dir = Path(output_dir) / self.config.season_id
            print(f"Season outputs will be saved to: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    
        self.last_model_run = None
        self._next_day_ix = 0

        self.trip_log_rows = []   # list of dicts; one row per trip
        self.day_summaries = []   # list of per-day summary dicts

    def run_season(self):
        """Run all days in the season config, in order."""
        # could also just: for _ in self.config.day_params: self.run_day()
        for day_cfg in self.config.day_params:
            self.run_day(day_cfg)

    def run_day(self, day_cfg=None):
        """
        Run a single day.
        - If day_cfg is provided: use that.
        - If day_cfg is None: run the next day in config.day_params.
        """
        if day_cfg is None:
            if self._next_day_ix >= len(self.config.day_params):
                print("All days in season_config have already been run.")
                return None
            day_cfg = self.config.day_params[self._next_day_ix]
            self._next_day_ix += 1

        day_index = day_cfg.day_index

        # 1) belief update
        self._update_beliefs(day_index)

        # 2) build and run traffic model
        tm = self._build_model(day_cfg=day_cfg)
        tm.run_model()
        self.last_model_run = tm

        self._append_day_trip_log(day_index)
        summary = self.compute_day_summary(day_index)

        # simple print for now
        if summary is not None:
            print(
                f"Day {day_index}: "
                f"N={summary['total_persons']}, "
                f"bus_share={summary['share_bus']:.2f}, "
                f"avg_tt_bus={summary['avg_tt_bus']:.1f} min, "
                f"avg_tt_car={summary['avg_tt_car']:.1f} min, "
                f"avg_toll_car=${summary['avg_toll_car']:.2f}"
            )

        if self.store_data:
            self._save_day_outputs(day_index, tm)

        return tm
    
    def _update_beliefs(self, day_index):
        for person in self.season_persons:
            person.update_beliefs_from_history(current_day=day_index)

    def _save_day_outputs(self, day_index, tm):
        prefix = f"day_{day_index}"

        model_ts = au.model_data_time_series(tm)
        finished_agents = au.finished_agents_summary_df(tm, plots=True)

        model_ts_path = self.output_dir / f"{prefix}_model_ts.parquet"
        finished_path = self.output_dir / f"{prefix}_finished_agents.parquet"

        model_ts.to_parquet(model_ts_path)
        finished_agents.to_parquet(finished_path)
         
    def _build_model(self, day_cfg) -> TrafficModel:
        """Create a ``TrafficModel`` instance for a single day."""

        return TrafficModel(
            # perams from season orchestrator init
            road_gdf=self.road_gdf, #road_gdf ecs_df dont come from day config because the data is looaded in the season orcestrator, the path is fassed from config
            ecs_df=self.ecs_df,

            # season level parameters from config
            batchrun=self.config.batch_run,
            max_steps=self.config.max_steps,
            max_persons=self.config.max_persons,
            start_hr=self.config.start_hr,
            bus_capacity=self.config.bus_capacity,
            collect_every_n=self.config.collect_every_n,  
            toll_mechanism=self.config.toll_mechanism,
            toll_params=self.config.toll_params,

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
            car_preference=1,
        )

    def _append_day_trip_log(self, day_index):
        """
        Collect all SeasonPerson history entries for this day
        and append them to the season-level trip_log_rows.

        Assumes SeasonPerson.history is a list of dicts with at least:
        - day_index
        - mode
        - toll_paid
        - realized_tt
        - wait_time
        - onboard_time
        - cumtime_lost_min
        - realized_cost
        """
        for sp in self.season_persons:
            # adjust "pid" to whatever your SeasonPerson id attribute is
            person_id = getattr(sp, "person_id", None)

            for rec in getattr(sp, "history", []):
                if rec.get("day_index") == day_index:
                    row = {
                        "season_person_id": person_id,
                        **rec,   # day_index, mode, toll_paid, realized_tt, etc.
                    }
                    self.trip_log_rows.append(row)

    def compute_day_summary(self, day_index):
        """
        Compute summary stats for a single day from trip_log_rows.
        Returns a dict; also appends it to self.day_summaries.
        """
        if not self.trip_log_rows:
            print(f"No trip log rows yet; day {day_index} summary is empty.")
            return None

        df = pd.DataFrame(self.trip_log_rows)
        day_df = df[df["day_index"] == day_index]

        if day_df.empty:
            print(f"No trips recorded for day {day_index}.")
            return None

        # persons / modes
        total_persons = day_df["season_person_id"].nunique()
        bus_df = day_df[day_df["mode"] == "bus"]
        car_df = day_df[day_df["mode"] == "car"]

        share_bus = len(bus_df) / total_persons if total_persons > 0 else 0.0

        # bus metrics
        avg_wait_bus      = bus_df["wait_time"].mean()
        avg_tt_bus        = bus_df["realized_tt"].mean()
        avg_cumlost_bus   = bus_df["cumtime_lost_min"].mean()

        # car metrics
        avg_tt_car        = car_df["realized_tt"].mean()
        avg_toll_car      = car_df["toll_paid"].mean()
        total_toll_car    = car_df["toll_paid"].sum()
        avg_cumlost_car   = car_df["cumtime_lost_min"].mean()

        summary = {
            "day_index": day_index,
            "total_persons": total_persons,
            "share_bus": share_bus,
            "avg_wait_bus": avg_wait_bus,
            "avg_tt_bus": avg_tt_bus,
            "avg_tt_car": avg_tt_car,
            "avg_toll_car": avg_toll_car,
            "total_toll_car": total_toll_car,
            "avg_cumlost_bus": avg_cumlost_bus,
            "avg_cumlost_car": avg_cumlost_car,
        }

        self.day_summaries.append(summary)
        return summary

    def get_trip_log_df(self):
        """
        Return trip log as a DataFrame, with consistent column order
        and sorted by day_index then season_person_id.
        """
        if not self.trip_log_rows:
            warnings.warn(
                "No trip log rows recorded yet; run at least one day before "
                "calling get_trip_log_df().",
                UserWarning,
            )
            return None

        df = pd.DataFrame(self.trip_log_rows)

        col_order = [
            "season_person_id",
            "day_index",
            "mode",
            "toll_paid",
            "realized_tt",
            "wait_time",
            "onboard_time",
            "cumtime_lost_min",
            "realized_cost",
        ]
        # keep known columns first, then any extras
        first = [c for c in col_order if c in df.columns]
        rest = [c for c in df.columns if c not in first]
        df = df[first + rest]

        if "day_index" in df.columns and "season_person_id" in df.columns:
            df = df.sort_values(["day_index", "season_person_id"])

        return df.reset_index(drop=True)

    def get_day_summary_df(self):
        """
        Return per-day summaries as a DataFrame, or None if none exist yet.
        """
        if not self.day_summaries:
            warnings.warn(
                "No day summaries recorded yet; make sure run_day() or "
                "run_season() has been called with summary logging enabled.",
                UserWarning,
            )
            return None

        df = pd.DataFrame(self.day_summaries)

        col_order = [
            "day_index",
            "total_persons",
            "share_bus",
            "avg_wait_bus",
            "avg_tt_bus",
            "avg_tt_car",
            "avg_toll_car",
            "total_toll_car",
            "avg_cumlost_bus",
            "avg_cumlost_car",
        ]
        first = [c for c in col_order if c in df.columns]
        rest = [c for c in df.columns if c not in first]
        df = df[first + rest]

        if "day_index" in df.columns:
            df = df.sort_values("day_index")

        return df.reset_index(drop=True)

    def run_day_temp(self):
        """
        Run a single-day simulation, compute avg cumtime_lost_sec,
        and append one summary row to results_df.

        results_df is a pandas DataFrame that you pass in.
        """
        day_cfg = self.config.day_params[0]

        tm = self._build_model(day_cfg=day_cfg)
        tm.run_model()

        # --- collect cumtime_lost_sec from all SeasonPersons ---
        cumtimes = []
        realized_tts = []
        n_car = 0
        n_bus = 0

        for sp in self.season_persons:
            hist = sp.history  # list of dicts
            if not hist:
                continue

            # assume cumtime_lost_sec in the *last* record is the cumulative value
            last = hist[-1]
            if "cumtime_lost_sec" in last:
                cumtimes.append(last["cumtime_lost_sec"])
            if "realized_tt" in last:
                realized_tts.append(last["realized_tt"])
            
            # mode counts from final record
            mode = last.get("mode")
            if mode == "car":
                n_car += 1
            elif mode == "bus":
                n_bus += 1

        if cumtimes:
            avg_cumtime_lost_sec = sum(cumtimes) / len(cumtimes)
        else:
            avg_cumtime_lost_sec = float("nan")

        if realized_tts:
            avg_realized_tt = sum(realized_tts) / len(realized_tts)
        else:
            avg_realized_tt = float("nan")

        # --- pull metadata for this run ---
        # tweak these attribute names to match your config
        seed = getattr(self.config, "seed", None)
        traffic_percentile = getattr(day_cfg, "traffic_percentile", None)
        bus_interval  = getattr(day_cfg, "bus_interval", None)

        car_toll = self.config.toll_params['car']
        bus_prior = getattr(self.config.population_params, "prior_bus", None)
        car_prior = getattr(self.config.population_params, "prior_car", None)

        row = {
            "seed": seed,
            "traffic_percentile": traffic_percentile,
            "car_toll": car_toll,
            "avg_realized_tt": avg_realized_tt,
            "avg_cumtime_lost_sec": avg_cumtime_lost_sec,
            "bus_interval": bus_interval,
            "bus_prior": bus_prior,
            "n_car": n_car,
            "n_bus": n_bus,
        }

        return row




   