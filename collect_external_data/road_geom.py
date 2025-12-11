import os
from pathlib import Path
import subprocess

import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

# --------------  set PROJECT_ROOT / pwd and paths --------------
PROJECT_ROOT = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
)

os.chdir(PROJECT_ROOT)

DATA_DIR = PROJECT_ROOT / "data"
ROADS_DIR = DATA_DIR / "roads"
ROADS_DIR.mkdir(parents=True, exist_ok=True)

ROAD_PATH = ROADS_DIR / "hw210_sl_and_curvs.parquet"


def get_road_geometry():
    # --------------  short-circuit if file already exists --------------
    if ROAD_PATH.exists():
        print("hw210_sl_and_curvs.parquet exists, skip re-processing")
    else:
        print("road file does not exist, making request...")

        # --------------  define helper functions --------------
        def densify_lines_to_points(gdf_lines, spacing_meters=50):
            """
            Convert LineStrings into evenly spaced Points with inherited attributes.
            """
            points_data = []

            for _, row in gdf_lines.iterrows():
                line = row.geometry
                length = line.length
                num_points = max(int(length // spacing_meters), 1)
                distances = [i * spacing_meters for i in range(num_points + 1)]
                distances = [d if d <= length else length for d in distances]

                for d in distances:
                    point = line.interpolate(d)
                    data = row.drop("geometry").to_dict()
                    data["geometry"] = point
                    points_data.append(data)

            return gpd.GeoDataFrame(points_data, crs=gdf_lines.crs)

        def calculate_heading(point1, point2):
            if point1 is None or point2 is None:
                return np.nan
            dx = point2.x - point1.x
            dy = point2.y - point1.y
            return np.degrees(np.arctan2(dy, dx))

        # --------------  Make Request --------------
        base_url = (
            "https://maps.udot.utah.gov/central/rest/services/"
            "TrafficAndSafety/UDOT_Speed_Limits/MapServer/0/query"
        )
        params = {
            "where": "Name='0210'",
            "outFields": "*",
            "f": "json",
        }

        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        features = response.json()["features"]

        print("request complete, starting road transformations...")

        # --------------  convert request to gdf  --------------
        geoms = []
        speed_limits = []

        for feature in features:
            sl = feature["attributes"]["Speed_Limit"]
            for path in feature["geometry"]["paths"]:
                geoms.append(LineString(path))
                speed_limits.append(sl)

        gdf = gpd.GeoDataFrame(
            {"speed_limit": speed_limits},
            geometry=geoms,
            crs="EPSG:26912",
        ).to_crs(epsg=32612)

        # drop Snowbird loop
        gdf = gdf.drop(1).reset_index(drop=True)

        # reorder the rows so that the segments are in road order
        gdf = gdf.reindex([3, 2, 1, 0]).reset_index(drop=True)

        # --------------  densify_lines_to_points  --------------
        road_gdf = densify_lines_to_points(gdf, spacing_meters=50)

        # --------------  calculate distance and heading cols --------------
        geom = road_gdf.geometry

        road_gdf["meters_to_next"] = geom.distance(geom.shift(-1)).fillna(0)
        road_gdf["distance_traveled"] = (
            road_gdf["meters_to_next"].shift(fill_value=0).cumsum()
        )

        road_gdf["heading"] = geom.combine(geom.shift(-1), calculate_heading)
        road_gdf["delta_heading"] = (
            road_gdf["heading"].shift(-1) - road_gdf["heading"]
        ).fillna(0)

        curvature = (road_gdf["delta_heading"].abs() / road_gdf["meters_to_next"]) * 100
        road_gdf["curvature"] = curvature.fillna(0).round(3)

        road_gdf["radius_ft"] = 59055.12 / (np.pi * road_gdf["curvature"])

        # placeholder for later linkage
        road_gdf["linked_coord"] = None

        # --------------  calculate road segment --------------
        gdf_reset = road_gdf.reset_index()

        speed_ranges = (
            gdf_reset.groupby("speed_limit")["index"]
            .agg(min_index="min", max_index="max")
            .sort_values("min_index")
        )

        sections = []
        section_id = 1

        for speed, row in speed_ranges.iterrows():
            start = int(row["min_index"])
            end = int(row["max_index"])

            if speed == 40:
                mid = (start + end) // 2
                sections.append((section_id, start, mid))
                section_id += 1
                sections.append((section_id, mid + 1, end))
                section_id += 1
            else:
                sections.append((section_id, start, end))
                section_id += 1

        conditions = [
            road_gdf.index.to_series().between(start, end)
            for (_, start, end) in sections
        ]
        labels = [sec_id for (sec_id, _, _) in sections]

        road_gdf["road_section"] = np.select(conditions, labels, default=0)

        # --------------  manual tweak to first few points --------------
        road_gdf.loc[0:2, "speed_limit"] = [10, 20, 35]

        # --------------  save to parquet --------------
        road_gdf.to_parquet(ROAD_PATH, index=True)
        print(f"transformations complete, road saved to {ROAD_PATH}")
