"""
Parallel parameter-sweep runner for SeasonOrchestrator.

Worker function, sweep grid builder, collector configs, and ParallelSweepRunner.
All worker functions are defined at module level (required for macOS spawn method).
"""
import os
import json
import inspect
import itertools
import dataclasses
import subprocess
import typing
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from season.configs import SeasonConfig, PopulationParams, make_season_config
from traffic.model.hybrid_collector import HybridCollectorConfig


# ---------------------------------------------------------------------------
# Sweep collector config constants
# ---------------------------------------------------------------------------

SWEEP_SUMMARIES_ONLY = HybridCollectorConfig(
    max_steps=100000,
    tier1_enabled=False,
    tier1_scalars=[],
    tier1_window_scalars=[],
    tier1_histograms=[],
    tier2_enabled=False,
    tier3_enabled=False,
    tier4_enabled=False,
)

SWEEP_FULL_TIER1 = HybridCollectorConfig(
    max_steps=100000,
    tier1_enabled=True,
    tier1_interval=60,
    tier1_scalars=[
        "step", "current_toll", "vehicle_count", "active_cars",
        "active_buses", "persons_at_bus_stop",
        "persons_finished", "persons_pool_remaining", "persons_in_transit", "p_generate",
    ],
    tier1_window_scalars=[
        "recent_travel_time_avg",
        "rolling_count_vehicles_generated",
        "rolling_count_persons_generated",
    ],
    tier1_histograms=["implicit_sl_delta"],
    tier1_window_seconds=600,
    tier2_enabled=False,
    tier3_enabled=False,
    tier4_enabled=False,
)


# ---------------------------------------------------------------------------
# Worker function (must be top-level for pickling via macOS spawn)
# ---------------------------------------------------------------------------

def run_one_season(config: SeasonConfig) -> dict:
    """Worker: accepts picklable SeasonConfig, returns season_summary dict."""
    from season.season_orchestrator import SeasonOrchestrator

    orch = SeasonOrchestrator(season_config=config, store_data=False, silent=True)
    orch.run_season()
    summary = orch.season_summary

    # Merge sweep params into summary so the DataFrame has columns for each swept dim
    if hasattr(config, "_sweep_params"):
        summary.update(config._sweep_params)

    return summary


# ---------------------------------------------------------------------------
# Sub-param registry (built dynamically from SeasonConfig + make_season_config)
# ---------------------------------------------------------------------------

def _get_dataclass_type(annotation: Any) -> type | None:
    """Extract dataclass type from annotation, handling Optional[X]."""
    if dataclasses.is_dataclass(annotation):
        return annotation
    origin = typing.get_origin(annotation)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1 and dataclasses.is_dataclass(args[0]):
            return args[0]
    return None


# Direct params of make_season_config — these are NEVER treated as sub-params
_MAKE_SEASON_CONFIG_PARAMS: frozenset[str] = frozenset(
    inspect.signature(make_season_config).parameters
)

# Discover wrappable dataclasses: SeasonConfig fields whose type is a dataclass
# AND whose field name is a direct param of make_season_config.
# Using get_type_hints() handles any string annotations / forward refs.
_hints = typing.get_type_hints(SeasonConfig)
_SUB_PARAM_CLASSES: list[tuple[type, str]] = []  # (dataclass_type, kwarg_name)
for _f in dataclasses.fields(SeasonConfig):
    _cls = _get_dataclass_type(_hints[_f.name])
    if _cls is not None and _f.name in _MAKE_SEASON_CONFIG_PARAMS:
        _SUB_PARAM_CLASSES.append((_cls, _f.name))

# Build registry: sub-param field_name -> [(dataclass_type, kwarg_name), ...]
# Fields that are already direct make_season_config params (e.g. max_steps) are excluded —
# they pass through unchanged so the direct param takes precedence.
_SUB_PARAM_REGISTRY: dict[str, list[tuple[type, str]]] = {}
for _cls, _kwarg in _SUB_PARAM_CLASSES:
    for _field in dataclasses.fields(_cls):
        if _field.name not in _MAKE_SEASON_CONFIG_PARAMS:
            _SUB_PARAM_REGISTRY.setdefault(_field.name, []).append((_cls, _kwarg))

try:
    _collisions = {k: v for k, v in _SUB_PARAM_REGISTRY.items() if len(v) > 1}
    if _collisions:
        names = {k: [c.__name__ for c, _ in v] for k, v in _collisions.items()}
        raise RuntimeError(
            f"Sub-param field name collision in season/parallel.py: {names}. "
            "Two sub-param dataclasses share a field name. Pass full objects "
            "(e.g. population_params=PopulationParams(...)) to resolve."
        )
except Exception as _e:
    raise ImportError(f"season.parallel failed to build sub-param registry: {_e}") from _e


# ---------------------------------------------------------------------------
# Sweep grid builder
# ---------------------------------------------------------------------------


def _resolve_sub_params(kwargs: dict) -> dict:
    """
    Detect sub-param fields in kwargs, validate uniqueness, and wrap them
    into their parent dataclass objects.

    Direct make_season_config params pass through unchanged.
    Fields belonging to exactly one sub-param dataclass are auto-wrapped.
    Unknown keys pass through (make_season_config will raise on bad keys).

    If a base wrapper object already exists in kwargs (e.g. population_params=...),
    sub-param overrides are merged on top of it via shallow copy — safe for
    scipy frozen distributions stored in PopulationParams.
    """
    direct: dict = {}
    sub_params: dict[str, dict] = {}  # kwarg_name -> {field: value}

    for key, val in kwargs.items():
        if key in _MAKE_SEASON_CONFIG_PARAMS:
            direct[key] = val
        elif key in _SUB_PARAM_REGISTRY:
            _, kwarg_name = _SUB_PARAM_REGISTRY[key][0]
            sub_params.setdefault(kwarg_name, {})[key] = val
        else:
            direct[key] = val

    for cls, kwarg_name in _SUB_PARAM_CLASSES:
        if kwarg_name not in sub_params:
            continue
        overrides = sub_params[kwarg_name]
        base = direct.get(kwarg_name)
        base_obj = base if base is not None else cls()
        # Shallow copy — avoids deepcopy of scipy frozen dists
        base_dict = {f.name: getattr(base_obj, f.name) for f in dataclasses.fields(base_obj)}
        base_dict.update(overrides)
        direct[kwarg_name] = cls(**base_dict)

    return direct


def _make_season_id(combo: dict, index: int) -> str:
    """Auto-generate a human-readable season_id from swept param values."""
    parts = [f"sweep_{index:03d}"]
    for key, val in combo.items():
        short_key = key.replace("_schedule", "").replace("_", "")
        # For ScheduleSpecs, use the value
        if hasattr(val, "value") and hasattr(val, "mode"):
            short_val = val.value
        else:
            short_val = val
        parts.append(f"{short_key}{short_val}")
    return "_".join(parts)


def build_sweep_configs(
    sweep_space: dict[str, list],
    fixed: dict,
    base_seed: int = 42,
) -> list[SeasonConfig]:
    """
    Generate a list of SeasonConfig objects from the cross-product of sweep_space.

    Parameters
    ----------
    sweep_space : dict mapping param names to lists of values to sweep
    fixed : dict of kwargs passed to make_season_config for every combo
    base_seed : each config gets seed = base_seed + sweep_index

    Returns
    -------
    list[SeasonConfig] with unique season_ids and seeds
    """
    keys = list(sweep_space.keys())
    value_lists = [sweep_space[k] for k in keys]

    configs: list[SeasonConfig] = []

    for i, values in enumerate(itertools.product(*value_lists)):
        combo = dict(zip(keys, values))

        # combo overrides fixed; _resolve_sub_params wraps any sub-param fields
        resolved = _resolve_sub_params({**fixed, **combo})
        if "seed" not in resolved:
            resolved["seed"] = base_seed + i
        resolved["season_id"] = _make_season_id(combo, i)
        if "run_description" not in resolved:
            resolved["run_description"] = f"sweep config {i}"

        cfg = make_season_config(**resolved)

        # Attach sweep params as a non-field attribute (survives pickle, ignored by asdict)
        sweep_params = {}
        for k, v in combo.items():
            if hasattr(v, "value") and hasattr(v, "mode"):
                sweep_params[k] = v.value
            else:
                sweep_params[k] = v
        cfg._sweep_params = sweep_params

        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Git info helper
# ---------------------------------------------------------------------------

def _get_git_info() -> dict:
    """Return current git commit, branch, and dirty status."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip())
        return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": None}


# ---------------------------------------------------------------------------
# Custom JSON encoder for SeasonConfig serialization
# ---------------------------------------------------------------------------

class _ConfigEncoder(json.JSONEncoder):
    """Handle scipy frozen dists, numpy types, and other non-serializable objects."""

    def default(self, obj):
        if hasattr(obj, "rvs") and hasattr(obj, "mean"):
            # scipy frozen distribution — store as repr string
            return repr(obj)
        if hasattr(obj, "item"):
            # numpy scalar
            return obj.item()
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# ParallelSweepRunner
# ---------------------------------------------------------------------------

class ParallelSweepRunner:
    """Run many SeasonConfigs in parallel and collect results into a DataFrame."""

    def __init__(
        self,
        configs: list[SeasonConfig],
        max_workers: int | None = None,
        output_root: str = "data/sweep_outputs",
    ):
        self.configs = configs
        self.max_workers = max_workers or min(os.cpu_count() - 2, len(configs))
        self.output_root = Path(output_root)
        self.results: list[dict] = []
        self.failures: list[dict] = []
        self.output_dir: Path | None = None

    def run(self) -> pd.DataFrame:
        """Execute all configs in parallel, return combined season_summary DataFrame."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm.auto import tqdm

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(run_one_season, cfg): cfg.season_id
                for cfg in self.configs
            }
            with tqdm(total=len(futures), desc="Sweep", unit="season") as pbar:
                for future in as_completed(futures):
                    sid = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        self.failures.append({"season_id": sid, "error": str(e)})
                    pbar.update(1)
                    pbar.set_postfix(last=sid)

        if self.failures:
            print(f"\n{len(self.failures)} failure(s):")
            for f in self.failures:
                print(f"  {f['season_id']}: {f['error']}")

        df = pd.DataFrame(self.results)
        self._save_outputs(df)
        return df

    def _save_outputs(self, df: pd.DataFrame) -> None:
        """Persist sweep results and metadata to data/sweep_outputs/sweep_YYYYMMDD_HHMMSS/."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_root / f"sweep_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1) Results parquet
        if not df.empty:
            df.to_parquet(self.output_dir / "sweep_results.parquet", index=False)

        # 2) Config JSON (reproducibility metadata)
        meta = {
            "created_at": datetime.now().isoformat(),
            **_get_git_info(),
            "max_workers": self.max_workers,
            "n_configs": len(self.configs),
            "n_completed": len(self.results),
            "n_failed": len(self.failures),
            "configs": [dataclasses.asdict(c) for c in self.configs],
        }
        with open(self.output_dir / "sweep_config.json", "w") as f:
            json.dump(meta, f, indent=2, cls=_ConfigEncoder)

        # 3) Failures (if any)
        if self.failures:
            with open(self.output_dir / "failures.json", "w") as f:
                json.dump(self.failures, f, indent=2)

        print(f"Sweep outputs saved to: {self.output_dir}")
