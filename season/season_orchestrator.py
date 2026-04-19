"""Utilities for orchestrating multi-day traffic model runs."""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional, Protocol

import json
import subprocess
import geopandas as gpd
import numpy as np
import pandas as pd
import pickle
import warnings


from traffic.model.traffic_model import TrafficModel
from traffic.utils import analysis_utils as au

from season.configs import SeasonConfig


def _get_git_commit_short() -> str:
    """Return the short git commit hash, or 'unknown' on failure."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"




class SeasonOrchestrator:
    """Run a series of daily simulations and persist their outputs."""

    def __init__(self, season_config: SeasonConfig,
                 output_root_dir: str = "data/outputs/seasons", silent: bool = False):
        self.config = season_config
        self.silent = silent

        self.road_gdf = gpd.read_parquet(self.config.road_path)
        self.ecs_df = pd.read_csv(self.config.ecs_path)

        self.season_persons = self.config.population_params.create_season_persons(
            season_id=self.config.season_id,
            seed=self.config.seed,
        )

        self.rng = np.random.default_rng(self.config.seed)

        # always create output directory for this season
        self.output_dir = Path(output_root_dir) / self.config.season_id
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        if not self.silent:
            print(f"Season outputs will be saved to: {self.output_dir}")

        self.last_model_run = None
        self._next_day_ix = 0

        self.trip_log_rows = []   # list of dicts; one row per trip
        self.day_summaries = []   # list of per-day summary dicts
        self.season_person_log_rows = []  # list of dicts; snapshot per person per day
        self.sp_day_summaries = []        # list of per-day SP summaries

        self._total_steps = 0
        self._total_vehicle_steps = 0
        self._wall_time = 0.0
        self.season_summary = None  # populated by run_season()

    def run_season(self):
        """Run all days in the season config, in order."""
        import time
        t0 = time.perf_counter()

        for day_cfg in self.config.day_params:
            self.run_day(day_cfg)

        self._wall_time = time.perf_counter() - t0
        self.season_summary = self._compute_season_summary()

        self._save_config()
        self._save_season_summary(self.season_summary)
        self._save_df_if_exists(self.get_trip_log_df(), "trip_log.parquet")
        self._save_df_if_exists(self.get_day_summary_df(), "day_summary.parquet")
        self._save_df_if_exists(self.get_season_person_log_df(), "season_person_log.parquet")
        self._save_df_if_exists(self.get_sp_day_summary_df(), "sp_day_summary.parquet")

        

    def run_day(self, day_cfg=None):
        """
        Run a single day.
        - If day_cfg is provided: use that.
        - If day_cfg is None: run the next day in config.day_params.
        """
        import time

        if day_cfg is None:
            if self._next_day_ix >= len(self.config.day_params):
                print("All days in season_config have already been run.")
                return None
            day_cfg = self.config.day_params[self._next_day_ix]
            self._next_day_ix += 1

        day_index = day_cfg.day_index

        # 0) reset toll state for new day (clears signal windows, PI integrals, etc.)
        self.config.toll_config.reset()

        # 1) belief update
        self._update_beliefs(day_index)

        # 2) build and run traffic model
        tm = self._build_model(day_cfg=day_cfg)
        t0 = time.perf_counter()
        tm.run_model()
        day_wall_time = time.perf_counter() - t0

        self.last_model_run = tm
        self._total_steps += tm.steps
        self._total_vehicle_steps += tm.total_vehicle_steps
        self._day_wall_time = day_wall_time
        self._day_steps = tm.steps
        self._day_vehicle_steps = tm.total_vehicle_steps

        if not self.silent:
            print(dict(tm.created_counts)) # temp


        self._append_day_trip_log(day_index)
        self._append_season_person_log(day_index)
        self._compute_day_summary(day_index)
        _ = self.compute_sp_day_summary(day_index)            

        self._save_datacollector_outputs(day_index, tm)

        return tm
    
    def _update_beliefs(self, day_index):
        cp = self.config.car_preference
        if cp is not None:
            median_vot = self.config.population_params.value_of_time.median()
            for person in self.season_persons:
                person.forced_mode = True
                person.assigned_mode = "car" if self.rng.random() < cp else "bus"
                person.value_of_time = median_vot
        else:
            for person in self.season_persons:
                person.forced_mode = False
                person.assigned_mode = None
                person.update_beliefs_from_history(current_day=day_index)

         
    def _build_model(self, day_cfg) -> TrafficModel:
        """Create a ``TrafficModel`` instance for a single day."""

        return TrafficModel(
            # perams from season orchestrator init
            road_gdf=self.road_gdf, #road_gdf ecs_df dont come from day config because the data is looaded in the season orcestrator, the path is fassed from config
            ecs_df=self.ecs_df,

            # season level parameters from config
            max_steps=self.config.max_steps,
            max_persons=self.config.max_persons,
            max_concurrent_vehicles=self.config.max_concurrent_vehicles,
            start_hr=self.config.start_hr,
            bus_capacity=self.config.bus_capacity,
            toll_config=self.config.toll_config,
            bus_user_fee=self.config.bus_user_fee,

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
            data_collection_config=self.config.data_collection,
            silent=self.silent,
        )
#-----------------------------------------------------------------------------  
# ------------------------- Compute logs + summaries -------------------------
#-----------------------------------------------------------------------------
    def _save_datacollector_outputs(self, day_index, tm):
        prefix = f"day_{day_index}"
        dc = tm.datacollector

        if dc.tier1:
            tier1_df = dc.get_tier1_dataframe()
            if not tier1_df.empty:
                tier1_df.to_parquet(self.output_dir / f"{prefix}_model_ts.parquet")

        if dc.tier2:
            tier2_df = dc.get_tier2_dataframe()
            if not tier2_df.empty:
                tier2_df.to_parquet(self.output_dir / f"{prefix}_spatial.parquet")

        if dc.tier3:
            events_df = dc.get_events_dataframe()
            if not events_df.empty:
                events_df.to_parquet(self.output_dir / f"{prefix}_events.parquet")



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
            person_id = getattr(sp, "person_id", None)

            for rec in getattr(sp, "history", []):
                if rec.get("day_index") == day_index:
                    row = {
                        "season_person_id": person_id,
                        **rec,   # day_index, mode, toll_paid, realized_tt, etc.
                    }
                    self.trip_log_rows.append(row)

    def _append_season_person_log(self, day_index):
        """
        Capture a snapshot of every SeasonPerson state for the given day.
        Stores a deep copy of all public attributes so future mutations
        do not change the logged record.
        """
        _NULL_BELIEF_FIELDS = {
            "expected_tt_car", "expected_tt_bus",
            "prior_car", "prior_bus",
            "travel_time_uncertainty_car", "travel_time_uncertainty_bus",
            "experience_weight_car", "experience_weight_bus",
            "uncertainty_multiplier",
        }
        null_beliefs = self.config.car_preference is not None

        for sp in self.season_persons:
            snapshot = {"day_index": day_index}
            for attr, val in vars(sp).items():
                if attr.startswith("_"):
                    continue
                if null_beliefs and attr in _NULL_BELIEF_FIELDS:
                    snapshot[attr] = None
                else:
                    snapshot[attr] = copy.deepcopy(val)
            self.season_person_log_rows.append(snapshot)



    @staticmethod
    def _aggregate_trips(df, median_vot):
        """
        Compute the canonical metric dict from a trip-log DataFrame.

        Returns a dict with all target columns except identifiers
        (season_id, days_run, day_index) which callers add themselves.
        """
        n = len(df)
        if n == 0:
            return None

        bus_df = df[df["mode"] == "bus"]
        car_df = df[df["mode"] == "car"]
        has_bus = not bus_df.empty
        has_car = not car_df.empty
        nan = float("nan")

        # per-trip VOT-standardized cost column (compute once, reuse)
        vot_cost = df["realized_tt"] * median_vot + df["toll_paid"]

        # VOT-standardized by mode
        vot_std_bus = (bus_df["realized_tt"] * median_vot + bus_df["toll_paid"]).mean() if has_bus else nan
        vot_std_car = (car_df["realized_tt"] * median_vot + car_df["toll_paid"]).mean() if has_car else nan

        result = {
            # overall
            "total_persons": df["season_person_id"].nunique(),
            "total_trips": n,
            "car_trips": len(car_df),
            "bus_trips": len(bus_df),
            # travel time
            "avg_tt": df["realized_tt"].mean(),
            "avg_tt_bus": bus_df["realized_tt"].mean() if has_bus else nan,
            "avg_tt_car": car_df["realized_tt"].mean() if has_car else nan,
            "avg_cum_time_lost": df["cumtime_lost_min"].mean(),
            "avg_cum_time_lost_bus": bus_df["cumtime_lost_min"].mean() if has_bus else nan,
            "avg_cum_time_lost_car": car_df["cumtime_lost_min"].mean() if has_car else nan,
            # bus
            "share_bus": len(bus_df) / n,
            "avg_wait_bus": bus_df["wait_time"].mean() if has_bus else nan,
            "avg_onboard_time_bus": bus_df["onboard_time"].mean() if has_bus else nan,
            # revenue
            "total_rev": df["toll_paid"].sum(),
            "avg_toll_car": car_df["toll_paid"].mean() if has_car else nan,
            "avg_toll_bus": bus_df["toll_paid"].mean() if has_bus else nan,
            "total_toll_car": car_df["toll_paid"].sum() if has_car else 0.0,
            "total_toll_bus": bus_df["toll_paid"].sum() if has_bus else 0.0,
            # VOT-standardized cost
            "avg_cost_vot_standardized": vot_cost.mean(),
            "avg_cost_vot_standardized_bus": vot_std_bus,
            "avg_cost_vot_standardized_car": vot_std_car,
            "avg_cost_vot_standardized_bus_car_delta": abs(vot_std_bus - vot_std_car),
            # realized cost
            "avg_realized_cost": df["realized_cost"].mean(),
            "avg_realized_cost_bus": bus_df["realized_cost"].mean() if has_bus else nan,
            "avg_realized_cost_car": car_df["realized_cost"].mean() if has_car else nan,
        }
        return {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in result.items()
        }

    def _compute_day_summary(self, day_index):
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

        median_vot = self.config.population_params.value_of_time.median()
        metrics = self._aggregate_trips(day_df, median_vot)
        if metrics is None:
            return None

        # Bus cost calculation (post-hoc, zero sim overhead)
        bc_config = self.config.bus_cost_config
        tm = self.last_model_run
        if bc_config is not None and tm.bus_interval > 0 and metrics.get("bus_trips", 0) > 0:
            from traffic.model.bus_system_cost import compute_bus_costs
            bus_costs = compute_bus_costs(
                config=bc_config,
                avg_one_way_tt_min=metrics["avg_tt_bus"],
                headway_min=tm.bus_interval,
                model_steps=tm.steps,
            )
            if bus_costs is not None:
                metrics.update(bus_costs)

        # run metadata for this day
        day_wt = max(self._day_wall_time, 0.001)
        metrics["wall_time_seconds"] = round(self._day_wall_time, 2)
        metrics["steps"] = self._day_steps
        metrics["vehicle_steps"] = self._day_vehicle_steps
        metrics["steps_per_second"] = round(self._day_steps / day_wt, 1)
        metrics["vehicle_steps_per_second"] = round(self._day_vehicle_steps / day_wt, 1)

        summary = {"day_index": day_index, **metrics}
        self.day_summaries.append(summary)

        if not self.silent:
            bus_cost_str = f", bus_cost_daily:${summary['bus_cost_daily']:,.0f}" if "bus_cost_daily" in summary else ""
            print(
                f"Day:{day_index}: "
                f"N:{summary['total_persons']}, "
                f"Avg TT:{summary['avg_tt']:.1f} min, "
                f"Avg CumTimeLost:{summary['avg_cum_time_lost']:.1f} min, "
                f"VOT Std Cost:${summary['avg_cost_vot_standardized']:.1f}, "
                f"Realized Cost:${summary['avg_realized_cost']:.1f}, "
                f"bus_share:{summary['share_bus']:.2f}, "
                f"avg_tt_bus:{summary['avg_tt_bus']:.1f} min, "
                f"avg_tt_car:{summary['avg_tt_car']:.1f} min, "
                f"avg_toll_car:${summary['avg_toll_car']:.2f}, "
                f"total_toll_car:${summary['total_toll_car']:.2f}"
                f"{bus_cost_str}"
            )
        return summary

    def compute_sp_day_summary(self, day_index):
        """
        Compute summary stats for SeasonPerson states for a single day.
        Returns a dict; also appends it to self.sp_day_summaries.
        """
        if not self.season_person_log_rows:
            print(f"No season person logs yet; day {day_index} SP summary is empty.")
            return None

        df = pd.DataFrame(self.season_person_log_rows)
        day_df = df[df["day_index"] == day_index]

        if day_df.empty:
            print(f"No SeasonPerson snapshots recorded for day {day_index}.")
            return None

        history_lengths = (
            day_df["history"].apply(len) if "history" in day_df else pd.Series(dtype=float)
        )

        null_beliefs = self.config.car_preference is not None
        sp_summary = {
            "day_index": day_index,
            "total_persons": len(day_df),
            "avg_expected_tt_car": None if null_beliefs else day_df["expected_tt_car"].mean(),
            "avg_expected_tt_bus": None if null_beliefs else day_df["expected_tt_bus"].mean(),
            "avg_prior_car": None if null_beliefs else day_df["prior_car"].mean(),
            "avg_prior_bus": None if null_beliefs else day_df["prior_bus"].mean(),
            "avg_travel_time_uncertainty_car": None if null_beliefs else day_df["travel_time_uncertainty_car"].mean(),
            "avg_travel_time_uncertainty_bus": None if null_beliefs else day_df["travel_time_uncertainty_bus"].mean(),
            "avg_value_of_time": day_df["value_of_time"].mean(),
            "avg_travel_propensity": day_df["travel_propensity"].mean(),
            "avg_history_len": history_lengths.mean() if not history_lengths.empty else 0.0,
        }
        sp_summary = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in sp_summary.items()
        }

        self.sp_day_summaries.append(sp_summary)
        return sp_summary
    
    def _compute_season_summary(self):
        """
        Aggregate multi-day metrics and print a season summary.
        """
        if not self.trip_log_rows:
            print("No trips recorded; season summary is empty.")
            return None

        df = pd.DataFrame(self.trip_log_rows)
        median_vot = self.config.population_params.value_of_time.median()
        metrics = self._aggregate_trips(df, median_vot)
        if metrics is None:
            return None

        # Bus cost aggregation from day summaries
        bc_config = self.config.bus_cost_config
        if bc_config is not None and self.day_summaries:
            run_costs = [d["bus_cost_model_run"] for d in self.day_summaries if d.get("bus_cost_model_run") is not None]
            daily_costs = [d["bus_cost_daily"] for d in self.day_summaries if d.get("bus_cost_daily") is not None]
            if run_costs:
                metrics["bus_cost_season_total"] = round(sum(run_costs), 2)
            if daily_costs:
                avg_daily = sum(daily_costs) / len(daily_costs)
                metrics["bus_cost_avg_daily"] = round(avg_daily, 2)
                metrics["bus_cost_annual"] = round(avg_daily * bc_config.service_days_per_year, 2)

        wt = max(self._wall_time, 0.001)
        summary = {
            "season_id": self.config.season_id,
            "days_run": df["day_index"].nunique(),
            "commit": _get_git_commit_short(),
            "wall_time_seconds": round(self._wall_time, 2),
            "total_steps": self._total_steps,
            "total_vehicle_steps": self._total_vehicle_steps,
            "avg_steps_per_second": round(self._total_steps / wt, 1),
            "avg_vehicle_steps_per_second": round(self._total_vehicle_steps / wt, 1),
            **metrics,
        }

        if not self.silent:
            bus_cost_str = ""
            if "bus_cost_season_total" in summary:
                bus_cost_str = (
                    f"\n  Bus cost (sim period): ${summary['bus_cost_season_total']:,.0f}, "
                    f"Bus cost (annual est): ${summary.get('bus_cost_annual', 0):,.0f}"
                )

            print(
                f"Season Summary - {summary['season_id']}\n"
                f"  Days: {summary['days_run']}, Trips: {summary['total_trips']} "
                f"(car: {summary['car_trips']}, bus: {summary['bus_trips']})\n"
                f"  Avg TT: {summary['avg_tt']:.1f} min, "
                f"CumTimeLost: {summary['avg_cum_time_lost']:.1f} min\n"
                f"  VOT Std Cost: ${summary['avg_cost_vot_standardized']:.1f}, "
                f"Realized Cost: ${summary['avg_realized_cost']:.1f}\n"
                f"  Bus share: {summary['share_bus']:.2f}, "
                f"Avg toll car: ${summary['avg_toll_car']:.2f}, "
                f"Total rev: ${summary['total_rev']:.2f}"
                f"{bus_cost_str}"
            )

        return summary

    def _save_season_summary(self, season_summary: dict):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = self.output_dir / "season_summary.parquet"
        pd.DataFrame([season_summary]).to_parquet(parquet_path, index=False)

    def _save_config(self):
        """Persist the SeasonConfig object next to other outputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.output_dir / "season_config.json"

        def json_serializer(obj):
            """Handle non-JSON-serializable objects."""
            if hasattr(obj, '__dict__'):
                return {
                    '_type': type(obj).__name__,
                    **{k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                }
            return str(obj)

        config_dict = asdict(self.config)
        with config_path.open("w") as fh:
            json.dump(config_dict, fh, indent=2, default=json_serializer)

    def _save_df_if_exists(self, df, filename: str):
        if df is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_dir / filename)

# ----------------------------------------------------------------------
# -------------------------  get df functions  -------------------------
# ----------------------------------------------------------------------

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

        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].round(3)
        return df.reset_index(drop=True)

    def get_season_person_log_df(self):
        """
        Return SeasonPerson state snapshots as a DataFrame.
        """
        if not self.season_person_log_rows:
            warnings.warn(
                "No season person logs recorded yet; run at least one day before "
                "calling get_season_person_log_df().",
                UserWarning,
            )
            return None

        df = pd.DataFrame(self.season_person_log_rows)

        col_order = [
            "day_index",
            "person_id",
            "season_id",
            "value_of_time",
            "expected_tt_car",
            "expected_tt_bus",
            "travel_time_uncertainty_car",
            "travel_time_uncertainty_bus",
            "prior_car",
            "prior_bus",
            "travel_propensity",
        ]
        first = [c for c in col_order if c in df.columns]
        rest = [c for c in df.columns if c not in first]
        df = df[first + rest]

        if "day_index" in df.columns and "person_id" in df.columns:
            df = df.sort_values(["day_index", "person_id"])

        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].round(3)
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
            # overall
            "total_persons", "total_trips", "car_trips", "bus_trips",
            # travel time
            "avg_tt", "avg_tt_bus", "avg_tt_car",
            "avg_cum_time_lost", "avg_cum_time_lost_bus", "avg_cum_time_lost_car",
            # bus
            "share_bus", "avg_wait_bus", "avg_onboard_time_bus",
            # revenue
            "total_rev", "avg_toll_car", "avg_toll_bus", "total_toll_car", "total_toll_bus",
            # VOT-standardized cost
            "avg_cost_vot_standardized", "avg_cost_vot_standardized_bus_car_delta",
            "avg_cost_vot_standardized_bus", "avg_cost_vot_standardized_car",
            # realized cost
            "avg_realized_cost", "avg_realized_cost_bus", "avg_realized_cost_car",
            # bus cost
            "bus_cost_active_buses", "bus_cost_total_fleet", "bus_cost_cycle_time_min",
            "bus_cost_per_hour", "bus_cost_model_run", "bus_cost_daily", "bus_cost_annual",
            # run metadata
            "wall_time_seconds", "steps", "vehicle_steps",
            "steps_per_second", "vehicle_steps_per_second",
        ]
        first = [c for c in col_order if c in df.columns]
        rest = [c for c in df.columns if c not in first]
        df = df[first + rest]

        if "day_index" in df.columns:
            df = df.sort_values("day_index")

        return df.reset_index(drop=True)

    def get_sp_day_summary_df(self):
        """
        Return per-day SeasonPerson summaries as a DataFrame, or None if none exist yet.
        """
        if not self.sp_day_summaries:
            warnings.warn(
                "No SeasonPerson day summaries recorded yet; make sure run_day() or "
                "run_season() has been called to populate SP summaries.",
                UserWarning,
            )
            return None

        df = pd.DataFrame(self.sp_day_summaries)

        col_order = [
            "day_index",
            "total_persons",
            "avg_expected_tt_car",
            "avg_expected_tt_bus",
            "avg_prior_car",
            "avg_prior_bus",
            "avg_travel_time_uncertainty_car",
            "avg_travel_time_uncertainty_bus",
            "avg_value_of_time",
            "avg_travel_propensity",
            "avg_history_len",
        ]
        first = [c for c in col_order if c in df.columns]
        rest = [c for c in df.columns if c not in first]
        df = df[first + rest]

        if "day_index" in df.columns:
            df = df.sort_values("day_index")

        return df.reset_index(drop=True)




