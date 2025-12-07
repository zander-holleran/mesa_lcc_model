from pathlib import Path
import subprocess
import requests
import pandas as pd
import numpy as np

# ------------------- repo + paths -------------------
PROJECT_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

DATA_DIR = PROJECT_ROOT / "data"
VEHICLE_COUNTS_DIR = DATA_DIR / "vehicle_counts"
VEHICLE_COUNTS_DIR.mkdir(parents=True, exist_ok=True)

raw_csv_path = VEHICLE_COUNTS_DIR / "210_real_count_data.csv"
expected_seconds_path = VEHICLE_COUNTS_DIR / "expected_counts_seconds.csv"

if expected_seconds_path.exists():
    print(f"expected_counts_seconds.csv already exists, skip re-processing.")
else:
    print("expected_counts_seconds.csv does not exist")
    # ------------------- download Google Sheet (sheet '0317') -------------------
    FILE_ID = "1NroJmNFLNE_GiaSNb0lqkqSnywu_BVfIA232WEaK6xw"
    SHEET_NAME = "0317"

    url = f"https://docs.google.com/spreadsheets/d/{FILE_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}"

    print(f"Downloading UDOT 2023 Hw 210 data to {raw_csv_path} ...")
    r = requests.get(url)
    r.raise_for_status()
    raw_csv_path.write_bytes(r.content)
    print("Raw data download complete.")

    # ------------------- load + aggregate to daily hour counts -------------------
    print("Begining transformations:")

    count_df = pd.read_csv(raw_csv_path)
    count_df.columns = [c.lower() for c in count_df.columns]
    count_df["date"] = pd.to_datetime(count_df["date"])


    # hour columns are the ones like "h0000", "h0500", etc.
    hour_cols = [c for c in count_df.columns if c.startswith("h")]

    # sum counts by date for each hour
    day_counts = count_df.groupby("date", as_index=False)[hour_cols].sum()

    # keep some simple calendar features (optional, but light)
    day_counts["month"] = day_counts["date"].dt.month
    day_counts["week"] = day_counts["date"].dt.isocalendar().week
    day_counts["weekday"] = day_counts["date"].dt.weekday  # 0=Mon, 6=Sun

    # ------------------- long format + hourly percentiles -------------------
    df_long = day_counts.melt(
        id_vars=["date", "month", "week", "weekday"],
        value_vars=hour_cols,
        var_name="hour_str",
        value_name="counts",
    )

    # "h0500" → 5, "h2100" → 21
    df_long["hour"] = df_long["hour_str"].str[1:3].astype(int)

    # keep only hours between 05:00 and 21:00
    slim_hours = df_long.loc[df_long["hour"].between(5, 21), ["hour", "counts"]]

    # within each hour, rank days by volume → hourly percentile (1–100)
    slim_hours["hourly_percentile"] = (
        slim_hours
        .groupby("hour")["counts"]
        .transform(lambda x: np.ceil(x.rank(pct=True) * 100))
        .astype(int)
    )

    # rows: hour (5–21), cols: percentile (1–100), values: expected counts for that bin
    expected_counts = slim_hours.pivot_table(
        index="hour",
        columns="hourly_percentile",
        values="counts",
    )

    # ------------------- smooth gaps with a rolling average -------------------
    expected_counts_filled = expected_counts.apply(
        lambda row: row.fillna(
            row.rolling(window=3, center=True, min_periods=1).mean()
        ),
        axis=1,
    )

    # ------------------- convert to per-second expected counts -------------------
    hourly_df = expected_counts_filled.sort_index()
    n_hours = len(hourly_df)

    # index in seconds: 0, 3600, 7200, ...
    hourly_df.index = np.arange(0, n_hours * 3600, 3600)

    # full second-by-second index: 0, 1, 2, ..., (n_hours * 3600 - 1)
    second_index = np.arange(0, n_hours * 3600)

    df_interp = (
        hourly_df
        .reindex(second_index)
        .interpolate(method="linear", axis=0)
    )

    # per-second generation rate
    expected_counts_seconds = df_interp / 3600

    # clean integer percentile labels on the columns (1–100)
    expected_counts_seconds.columns = range(1, expected_counts_seconds.shape[1] + 1)

    # ------------------- save final table -------------------
    expected_counts_seconds.to_csv(expected_seconds_path)
    print(f"Transformation complete: expected_counts_seconds.csv saved to {VEHICLE_COUNTS_DIR}")
