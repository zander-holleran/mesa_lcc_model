# Road Network

The road is represented as a sequence of evenly spaced waypoints derived from real GIS data. Vehicles traverse these waypoints using continuous positioning -- segments are directional targets, not exclusive occupancy zones.

---

## Road GeoDataFrame

The road geometry is stored as a GeoDataFrame (`road_gdf`) loaded from `data/roads/hw210_sl_and_curvs.parquet`. Each row represents a point on the road with the following columns:

| Column | Description |
|--------|-------------|
| `geometry` | Point coordinate (x, y) in the road's CRS |
| `speed_limit` | Posted speed limit at that point (mph, converted to m/s internally) |
| `curvature` | Road curvature value at that point |
| `road_section` | Segment index |
| `distance_traveled` | Cumulative distance from the canyon mouth (meters) |

Points are spaced approximately **50 meters apart**.

---

## Waypoint Model

Each point in the GeoDataFrame becomes a `RoadSegmentAgent`. Vehicles use these as **directional waypoints** for navigation:

- A vehicle tracks its current segment index (`path_index`) and progress within that segment (`_s` meters)
- Movement advances `_s` by the vehicle's speed each step; when `_s` exceeds the segment length, the vehicle transitions to the next segment
- The vehicle's position is interpolated: `start_of_segment + segment_direction * _s`

Multiple vehicles can be near the same segment simultaneously. The `vehicles_here` list on `RoadSegmentAgent` tracks proximity for estimation purposes, not exclusive occupancy.

---

## Pre-Computed Arrays

`init_helpers.py` converts the GeoDataFrame into **numpy arrays** at model initialization for hot-path performance. These are stored as model attributes:

| Array | Contents |
|-------|----------|
| `rs_pos` | (x, y) coordinates of each segment |
| `rs_distance` | Cumulative distance at each segment |
| `rs_speed_limit` | Posted speed limit at each segment |
| `rs_curvature` | Curvature at each segment |
| `rs_seg_len` | Length of each segment (distance to next point) |
| `rs_seg_dir` | Unit direction vector of each segment |
| `rs_cumulative_s` | Cumulative distance for end-of-road detection |

Each `VehicleAgent` also builds its own copies of segment vectors and lengths at creation for use in `move_along_path()`.

---

## Speed Limit Look-Ahead

Vehicles don't just read the speed limit of their current segment. `get_speed_limit()` averages the speed limit and curvature of the **next N segments** (`N_ahead = 5`) with distance-decaying weights:

```python
weights = [1/(1+i) for i in range(N_ahead)]  # [1.0, 0.5, 0.33, 0.25, 0.2]
avg_speed_limit = dot(segment_speeds[i:i+N], normalized_weights)
avg_curvature = dot(segment_curves[i:i+N], normalized_weights)
```

This allows drivers to begin braking **before** reaching a low-speed-limit or high-curvature section of road.

---

## Data Source

The road geometry is produced by `collect_external_data/road_geom.py`, which downloads and processes GIS data for Highway 210. The `get_road_geometry()` function checks if the parquet file exists locally and creates it if not.

---

## Continuous Space

The model uses Mesa's `ContinuousSpace` for spatial placement of all agents. The space bounds are derived from the road GeoDataFrame's extent with a 10,000-unit buffer on all sides:

```python
minx, miny, maxx, maxy = road_gdf.total_bounds
self.space = ContinuousSpace(
    x_min=minx - buffer, x_max=maxx + buffer,
    y_min=miny - buffer, y_max=maxy + buffer,
    torus=False
)
```

All road segment agents, vehicles, and blockers are placed in this space.
